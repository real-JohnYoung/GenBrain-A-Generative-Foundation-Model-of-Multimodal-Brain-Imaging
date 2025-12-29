import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


def generate_random_motion(num_movements, rotation_range=(-30, 30), translation_range=(-5, 5), rotation_lam=15, translation_lam=2 ):

    rotations = np.random.poisson(rotation_lam, size=(num_movements, 3)) *  np.random.choice([-1, 1], size=(num_movements, 3))
    rotations = np.clip(rotations, rotation_range[0], rotation_range[1])

    translations = np.random.poisson(translation_lam, size=(num_movements, 3)) * np.random.choice([-1, 1], size=(num_movements, 3))
    translations = np.clip(translations, translation_range[0], translation_range[1])

    return rotations, translations

def create_affine_matrix(rotation_angles, translation_vector):
    rotation_angles = np.deg2rad(rotation_angles)

    # Rotation matrix around X-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                    [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    # Rotation matrix around Y-axis
    R_y = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                    [0, 1, 0],
                    [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    # Rotation matrix around Z-axis
    R_z = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                    [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                    [0, 0, 1]])

    R = R_z @ R_y @ R_x

    affine_matrix = np.vstack([np.hstack([R, translation_vector.reshape(-1, 1)]), [0, 0, 0, 1]])

    return affine_matrix

def demean_movement_transform(image,movement_matrices,masks,num_movements):
    
    w_all = []
    k_space = np.fft.fftn(image)
    k_space_centered = np.fft.fftshift(image)

    for i in range(num_movements+1):
        k_mask_i = masks[i] * k_space_centered
        image_i = np.fft.ifftn(k_mask_i)
        w_i = np.sum(abs(image_i))
        w_all.append(w_i)
    w_all = w_all/np.sum(w_all)

    movement_matrices = [ matrix+0j for matrix in movement_matrices]

    A_avg = np.real(np.exp(np.sum(w_all[1, np.newaxis, np.newaxis] * np.log(movement_matrices),axis=0)))

    return A_avg


def resample_image(image, transform_matrix):

    image_shape = image.shape
    coords = np.indices(image_shape).reshape(3, -1).T  # (num_voxels, 3)

    homogeneous_coords = np.hstack([coords, np.ones((coords.shape[0], 1))]) 
    transformed_coords = homogeneous_coords @ transform_matrix.T  

    new_coords = transformed_coords[:, :3]
    new_coords = np.round(new_coords).astype(int)
    new_coords = np.clip(new_coords, 0, np.array(image_shape) - 1)

    transformed_image = map_coordinates(image, new_coords.T, order=3, mode='constant')

    return transformed_image.reshape(image_shape)


def generate_masks(image_shape, phase_direction='X', num_movements=2):
    # Initialize the list to store the masks
    masks = []
    
    total_len = image_shape[0] * image_shape[1] * image_shape[2]
    
    # Randomly pick time indices for the movements
    time_index = sorted( np.random.uniform(1, total_len, num_movements))
    time_index = list(np.floor(time_index).astype(int))
    time_index = [0] + time_index + [total_len]  # Include the start and end indices
    
    k_mask = np.zeros(image_shape)
    x, y, z = np.indices(image_shape)
    center = np.array([image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2])

    if phase_direction == 'X':
        distance = np.abs(x - center[0])
    elif phase_direction == 'Y':
        distance = np.abs(y - center[1])
    elif phase_direction == 'Z':
        distance = np.abs(z - center[2])

    sorted_indices = np.argsort(distance.ravel())
    sorted_values = np.arange(0, total_len)
    k_mask.ravel()[sorted_indices] = sorted_values   # stimulate the K-space sequence scan 
    
    for i in range(num_movements + 1):
        mask_i = np.zeros(image_shape)
        if i == 0 :
            mask_i[ k_mask<time_index[1]] = 1
        else:
            mask_i[ (k_mask>time_index[i]) & (k_mask<time_index[i+1])] = 1
        masks.append(mask_i)

    return masks


def add_motion_artifacts(image, num_movements, rotation_range=(-30, 30), translation_range=(-5, 5), rotation_lam=1, translation_lam=1,phase_direction='Y'):
    
    rotations, translations = generate_random_motion(num_movements, rotation_range, translation_range,rotation_lam, translation_lam)
    movement_matrices = [create_affine_matrix(rotation, translation) for rotation, translation in zip(rotations, translations)]       # A matrix set

    masks = generate_masks(image.shape, phase_direction ,num_movements)

    A_avg = demean_movement_transform(image,movement_matrices,masks,num_movements)

    try:
        A_avg_inv = np.linalg.inv(A_avg)
    except np.linalg.LinAlgError:
        A_avg_inv = np.linalg.inv(A_avg + 1e-6 * np.eye(A_avg.shape[0]))

    A_d_i_minus_1 = A_avg_inv.copy()
    K_set = []

    for i in range(num_movements):
        A_d_i = np.real(np.exp(np.log(A_d_i_minus_1+0j)+np.log(A_avg_inv+0j)+ np.log(movement_matrices[i]+0j)))
        image_i = resample_image(image.copy(),A_d_i)
        K_i =  np.fft.ifftshift(np.fft.fftn(image_i))
        K_set.append(K_i)
        A_d_i_minus_1 = A_d_i

    K_0 = np.fft.ifftshift(np.fft.fftn(image))
    K_set = [K_0] + K_set

    composite_k_space = np.sum(np.array(K_set)*np.array(masks),axis=0)

    corrupted_image = np.fft.ifftn(composite_k_space)
    
    corrupted_image_abs = np.abs(corrupted_image)
    
    return corrupted_image_abs


