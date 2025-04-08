import torch
import torch.nn.functional as F
from scipy.ndimage import label
from utils.mie_utils import mie
import numpy as np

def get_gaussian_kernel(kernel_size=5, sigma=1.5, device='cpu'):
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2

    # Calculating gaussian distribution
    gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel

def gaussian_blur(img, kernel_size=5, sigma=1.5):
    device = img.device
    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0) 
    img = img.permute(2, 0, 1).unsqueeze(0)  # [w, h, 1] -> [1, 1, w, h]
    blurred_img = F.conv2d(img, gaussian_kernel, padding=kernel_size // 2)

    return blurred_img.squeeze(0).permute(1, 2, 0)  # [1, 1, w, h] -> [w, h, 1]



def gaussian_expand_depth(depth, kernel_size=5):
    structuring_element = torch.ones((1, 1, kernel_size, kernel_size), device=depth.device)
    depth_expanded = F.conv2d(depth.unsqueeze(0), structuring_element, padding=kernel_size // 2)
    mask = (depth > 0).float()
    depth_expanded = mask * depth + (1 - mask) * depth_expanded.squeeze(0)

    return depth_expanded


def restore_depth(snow_mask, snow_depth):
    if snow_depth.dim() == 3 and snow_depth.size(0) == 1:
        snow_depth = snow_depth.squeeze(0)
    mask = (snow_depth > 0).cpu().numpy()
    labeled_array, num_features = label(mask)
    labeled_array = torch.tensor(labeled_array, device=snow_depth.device)
    restored_depth = snow_depth.clone()

    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        region_max_depth = snow_depth[region_mask].max()
        restored_depth[region_mask] = region_max_depth

    return restored_depth


def restore_depth_fast(snow_depth, kernel_size=3):
    if snow_depth.dim() == 3 and snow_depth.size(0) == 1:
        snow_depth = snow_depth.squeeze(0)

    mask = (snow_depth > 0).float()
    max_depth = snow_depth.max()
    min_depth = F.max_pool2d(snow_depth.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    restored_depth = mask * min_depth.squeeze(0) + (1 - mask) * snow_depth
    return restored_depth

def extinction_coefficient(radius, concentration, Qex):
    return np.pi * radius**2 * concentration * Qex

def cal_luminance_factor(luminance_image, sky_brightness, style_type):
    assert style_type in ["fog", "sandstorm", "snow", "rain"]
    
    if style_type == "snow":
        luminance_difference = torch.clamp(sky_brightness - luminance_image, max=0.55, min=-0.20)
        luminance_factor = luminance_difference
    elif style_type == "rain":
        luminance_difference = torch.clamp(sky_brightness - luminance_image, max=0.35, min=-0.05)
        # luminance_factor = 0.1 + luminance_difference - luminance_difference.min()
    return luminance_factor.unsqueeze(-1).repeat(1,1,3)
    

def stylization(img, r_depth, abs_depth, sky_color=None, style_type="fog", style_intensity=1.0, fog_method="exp"):
    """ input:
        img: [3,h,w]
        depth: [1,h,w]

        output:
        style_rendering: [3,h,w]
    """
    img = img.permute(1, 2, 0)
    sky_mask = r_depth > 0.9
    sky_mask = sky_mask.squeeze(0)
    object_mask = r_depth != 1.0
    rendering_style = torch.zeros(img.shape)
    # r_depth = torch.clamp(r_depth-0.1, 0.0, 1.0)

    assert style_type in ["fog", "sandstorm", "snow", "rain", "haze"]
    style_RGB = torch.zeros([1, 3]).to("cuda")
    if style_type == "fog":
        # style_RGB = torch.tensor([180, 180, 195]).to("cuda")/255.0
        style_RGB = torch.tensor([189, 189, 189]).to("cuda")/255.0
        if fog_method == "exp":
            style = torch.ones(img.shape).to("cuda")
            fog_density = style_intensity
            for i in range(3):
                style[:, :, i] = torch.exp(-fog_density * r_depth)
                # style[sky_mask] = 0.0
            mean_brightness = 1
            rendering_style = (img * style + (1 - style) * style_RGB) * mean_brightness
        elif fog_method=="mie":
            # Mie scattering
            r = 5.1  # particle radius, μm, 0.1 ~ 100
            c = style_intensity * 10**(-10)  # concentration, μm^-3, 10e-10 ~ 10e-8
            # wavelength of light, μm
            r_lambda = 0.650
            g_lambda = 0.550
            b_lambda = 0.450
            # dimensionless size parameter for each wavelength
            r_x = 2 * np.pi * r / r_lambda
            g_x = 2 * np.pi * r / g_lambda
            b_x = 2 * np.pi * r / b_lambda
            n = 1.33-0.1j  # refraction of the water sphere
            r_Qex, _, _, _ = mie(m=n, x=r_x)
            g_Qex, _, _, _ = mie(m=n, x=g_x)
            b_Qex, _, _, _ = mie(m=n, x=b_x)
            Q_ex = np.array([r_Qex, g_Qex, b_Qex])
            # print(Q_ex)
            alpha_ex = torch.tensor(extinction_coefficient(r, c, Q_ex), device="cuda") * 1e6   # μm^(-1) to m^(-1)
            sky_color = torch.mean(img[sky_mask], dim=0)
            sky_color = style_RGB
            # print(alpha_ex.shape,depth.shape,sky_color.shape,img.shape)
            abs_depth = abs_depth.permute(1,2,0)
            alpha_ex = alpha_ex.view(1, 1, 3)
            sky_color = sky_color.view(1, 1, 3)
            exp_term = torch.exp(-alpha_ex * abs_depth)
            rendering_style = img * exp_term + sky_color * (1 - exp_term)

    elif style_type == "sandstorm":
        style_RGB = torch.tensor([0.925, 0.906, 0.758]).to("cuda")
        # style_RGB = (torch.tensor([182, 164, 141]).to("cuda"))/255.0
        style = torch.ones(img.shape).to("cuda")
        sand_density = style_intensity
        for i in range(3):
            style[:, :, i] = torch.exp(-sand_density * r_depth)
        mean_brightness = 1
        rendering_style = (img * style + (1 - style) * style_RGB) * mean_brightness
    elif style_type == "haze":
        style_RGB = torch.tensor([230, 230, 220]).to("cuda")/255.0
        style = torch.ones(img.shape).to("cuda")
        sand_density = style_intensity
        for i in range(3):
            style[:, :, i] = torch.exp(-sand_density * r_depth)
        mean_brightness = 1
        rendering_style = (img * style + (1 - style) * style_RGB) * mean_brightness
    elif style_type == "snow":
        # TODO extract sky color for the fading color
        if sky_color is not None:
            style_RGB = sky_color
        else:
            style_RGB = torch.tensor([190, 190, 200]).to("cuda")/255.0
        
        style = torch.ones(img.shape).to("cuda")
        fading_density = 3.0
        # applied_depth = torch.clamp(r_depth - 0.8, min = 0.0)
        for i in range(3):
            # style[:, :, i] = torch.clamp(1 - fading_density * (torch.exp(applied_depth+1)-np.exp(1)), min=0.2)
            style[:, :, i] = torch.exp(-fading_density * r_depth)
            # style[sky_mask] = 0.0
        mean_brightness = 1
        rendering_style = (img * style + (1 - style) * style_RGB) * mean_brightness
        rendering_style[:,:,0] = 0.8 * rendering_style[:,:,0]
        rendering_style[:, :, 1] = 0.8 * rendering_style[:,:,1]
        rendering_style[:, :, 2] = 0.84 * rendering_style[:,:,2]

    elif style_type == "rain":
        if sky_color is not None:
            style_RGB = sky_color
        else:
            style_RGB = torch.tensor([160, 160, 160]).to("cuda")/255.0
        # sky_color = torch.mean(img[sky_mask], dim=0)
        # style_RGB = sky_color

        style = torch.ones(img.shape).to("cuda")
        fading_density = 1.0
        # applied_depth = torch.clamp(r_depth - 0.6, min = 0.0)
        # mask = (applied_depth>0.0)
        # applied_depth[mask] = (applied_depth[mask]-0.2)*5/2*3 +0.5
        for i in range(3):
            # style[:, :, i] = torch.clamp(1 - fading_density * (torch.exp(applied_depth+1)-np.exp(1)), min=0.2)
            style[:, :, i] = torch.exp(-fading_density * r_depth)
            # style[sky_mask] = 0.0
        mean_brightness = 1
        rendering_style = (img * style + (1 - style) * style_RGB) * mean_brightness

        rendering_style[:,:,0] = 0.75 * rendering_style[:,:,0]
        rendering_style[:, :, 1] = 0.75 * rendering_style[:,:,1]
        rendering_style[:, :, 2] = 0.78 * rendering_style[:,:,2]


    return rendering_style.permute(2,0,1)