import numpy as np

def init_flake_position(n, radi):
    # Generating n points 
    points = np.random.uniform(-1, 1, (n, 3))

    distances = np.linalg.norm(points, axis=1)
    inside_sphere = points[distances <= 1]
    while inside_sphere.shape[0] < n:
        extra_points = np.random.uniform(-1, 1, (n, 3))
        extra_distances = np.linalg.norm(extra_points, axis=1)
        extra_inside = extra_points[extra_distances <= 1]
        inside_sphere = np.vstack((inside_sphere, extra_inside))

    positions = inside_sphere[:n] * radi

    return positions


def random_quaternion(max_angle_degrees=10):
    angle_radians = np.radians(max_angle_degrees)
    random_axis = np.random.normal(size=3)
    random_axis /= np.linalg.norm(random_axis)

    w = np.cos(angle_radians / 2)
    x, y, z = random_axis * np.sin(angle_radians / 2)

    return np.array([w, x, y, z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    ])

def apply_random_rotation(quaternions):
    n = quaternions.shape[0]
    sub = n/3
    sub_quaterions_1 = quaternions[sub:sub*2, :]
    sub_quaterions_2 = quaternions[sub*2:, :]
    rotated_quaternions = np.zeros_like(quaternions)



    for i in range(n/3):
        random_q = random_quaternion()
        rotated_quaternions[i] = quaternion_multiply(random_q, quaternions[i])

    return rotated_quaternions


def small_rotation_quaternion(axis, max_angle_degrees):
    max_angle_radians = np.radians(max_angle_degrees)
    angle = np.random.uniform(-max_angle_radians, max_angle_radians)
    w = np.cos(angle / 2)
    x, y, z = axis * np.sin(angle / 2)

    return np.array([w, x, y, z])


def x_rotation_quaternion(angle = None):
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    w = np.cos(angle / 2)
    x = np.sin(angle / 2)

    return np.array([w, x, 0, 0])


def y_rotation_quaternion(angle = None):
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    w = np.cos(angle / 2)
    y = np.sin(angle / 2)

    return np.array([w, 0, y, 0])


def z_rotation_quaternion(angle = None):
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    w = np.cos(angle / 2)
    z = np.sin(angle / 2)

    return np.array([w, 0, 0, z])


def apply_rotation_old(quaternions, if_snow, max_xy_angle_degrees=180):
    n = quaternions.shape[0]
    if if_snow:
        sub = int(n / 3)
        rotated_quaternions = quaternions

        for i in range(sub):
            x_rotation_q = z_rotation_quaternion(np.pi/3)

            # combined_rotation_q = quaternion_multiply(y_rotation_q, quaternion_multiply(z_rotation_q, x_rotation_q))
            combined_rotation_q = x_rotation_q
            rotated_quaternions[i] = quaternion_multiply(combined_rotation_q, quaternions[i])
            rotated_quaternions[i] = x_rotation_q
        for i in range(sub, sub*2):
            x_rotation_q = z_rotation_quaternion(np.pi*2/3)
            # combined_rotation_q = quaternion_multiply(y_rotation_q, quaternion_multiply(z_rotation_q, x_rotation_q))
            combined_rotation_q = x_rotation_q
            rotated_quaternions[i] = quaternion_multiply(combined_rotation_q, quaternions[i])
            rotated_quaternions[i] = x_rotation_q
        for i in range(sub):
            y_rotation_q = small_rotation_quaternion(np.array([0, 1, 0]), max_xy_angle_degrees)
            x_rotation_q = small_rotation_quaternion(np.array([1, 0, 0]), max_xy_angle_degrees)

            combined_rotation_q = quaternion_multiply(x_rotation_q, y_rotation_q)
            rotated_quaternions[i] = quaternion_multiply(combined_rotation_q, rotated_quaternions[i])
            rotated_quaternions[i+sub] = quaternion_multiply(combined_rotation_q, rotated_quaternions[i+sub])
            rotated_quaternions[i+2*sub] = quaternion_multiply(combined_rotation_q, rotated_quaternions[i+2*sub])
    else:
        rotated_quaternions = quaternions
        for i in range(n):
            y_rotation_q = small_rotation_quaternion(np.array([0, 1, 0]), max_xy_angle_degrees)
            x_rotation_q = small_rotation_quaternion(np.array([1, 0, 0]), max_xy_angle_degrees)
            # y_rotation_q = y_rotation_quaternion(np.pi/2)
            # z_rotation_q = z_rotation_quaternion(np.pi/2)

            combined_rotation_q = quaternion_multiply(x_rotation_q, y_rotation_q)
            rotated_quaternions[i] = quaternion_multiply(combined_rotation_q, quaternions[i])

    return rotated_quaternions

def rotate_quaternions_to_gravity(np_rots, g):
    default_gravity = np.array([0, -1, 0], dtype=np.float32)
    g = g / np.linalg.norm(g)
    
    axis = np.cross(default_gravity, g) 
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np_rots
    axis = axis / axis_norm
    angle = np.arccos(np.dot(default_gravity, g))
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    
    rotation_quaternion = np.array([cos_half_angle, 
                                     sin_half_angle * axis[0],
                                     sin_half_angle * axis[1],
                                     sin_half_angle * axis[2]])
    
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    
    updated_np_rots = np.zeros_like(np_rots)
    for i, quat in enumerate(np_rots):
        updated_np_rots[i] = quaternion_multiply(rotation_quaternion, quat)
    
    return updated_np_rots

def apply_rotation(quaternions, if_snow, max_xy_angle_degrees=90, num_rotations=30):
    def random_rotation_quaternions(num_rotations, max_angle):
        rotations = []
        for _ in range(num_rotations):
            axis = np.random.randn(2)
            axis = np.append(axis, 0)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.radians(max_angle))
            rotations.append(small_rotation_quaternion(axis, np.degrees(angle)))
        return rotations

    n = quaternions.shape[0]
    rotation_quaternions = random_rotation_quaternions(num_rotations, max_xy_angle_degrees)
    rotated_quaternions = quaternions
    if if_snow:
        # rotate 2 flakes to create one snowflake
        sub = int(n / 3)
        z_rotation_q_1 = z_rotation_quaternion(np.pi / 3)
        z_rotation_q_2 = z_rotation_quaternion(np.pi * 2 / 3)
        for i in range(sub):
            random_index = np.random.randint(0, num_rotations)
            random_rotation = rotation_quaternions[random_index]
            # rotated_quaternions[i] = z_rotation_q_1
            # rotated_quaternions[i+sub] = z_rotation_q_2
            rotated_quaternions[i] = quaternion_multiply(random_rotation, rotated_quaternions[i])
            rotated_quaternions[i+sub] = quaternion_multiply(random_rotation, z_rotation_q_1)
            rotated_quaternions[i+sub*2] = quaternion_multiply(random_rotation, z_rotation_q_2)
    else:
        for i in range(n):
            random_index = np.random.randint(0, num_rotations)
            random_rotation = rotation_quaternions[random_index]
            rotated_quaternions[i] = quaternion_multiply(random_rotation, quaternions[i])

    return rotated_quaternions


def quaternion_to_rotation_matrix_batch(quaternions):
    n = quaternions.shape[0]
    rotation_matrices = np.zeros((n, 3, 3))

    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    rotation_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rotation_matrices[:, 0, 1] = 2 * (x * y - w * z)
    rotation_matrices[:, 0, 2] = 2 * (x * z + w * y)

    rotation_matrices[:, 1, 0] = 2 * (x * y + w * z)
    rotation_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rotation_matrices[:, 1, 2] = 2 * (y * z - w * x)

    rotation_matrices[:, 2, 0] = 2 * (x * z - w * y)
    rotation_matrices[:, 2, 1] = 2 * (y * z + w * x)
    rotation_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return rotation_matrices