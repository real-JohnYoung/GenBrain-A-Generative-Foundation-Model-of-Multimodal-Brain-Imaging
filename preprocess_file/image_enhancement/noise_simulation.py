import numpy as np

def add_gaussian_noise(image, noise_level=1):
    sigma_list = [20, 30, 50]
    sigma = sigma_list[noise_level-1]
    non_zero_data = image[image!=0]
    noise = np.random.normal(0, sigma, image.shape)
    image = noise + image
    return image

def add_rician_noise(image, noise_level):
    sigma_list = [20, 30, 50]
    sigma = sigma_list[noise_level-1]
    real_noise = np.random.normal(0, sigma, image.shape)
    imag_noise = np.random.normal(0, sigma, image.shape)
    image = image + real_noise + imag_noise*1j
    image = np.abs(image)
    return image


def add_gaussian_noise_more(image, noise_level):
    sigma_list = [5*i for i in range(0,11)]
    sigma = sigma_list[noise_level]
    non_zero_data = image[image!=0]
    noise = np.random.normal(0, sigma, image.shape)
    image = noise + image
    return image

def add_rician_noise_more(image, noise_level):
    sigma_list = [5*i for i in range(0,11)]
    sigma = sigma_list[noise_level]
    real_noise = np.random.normal(0, sigma, image.shape)
    imag_noise = np.random.normal(0, sigma, image.shape)
    image = image + real_noise + imag_noise*1j
    image = np.abs(image)
    return image