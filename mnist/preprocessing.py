# deskewing
from scipy.ndimage import interpolation
import scipy as sp
import numpy as np

# Acknowledgement: modified from https://bit.ly/3bBtKNg
def _moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def _deskew(image):
    """ Straighten an image that has been scanned or written crookedly using affine transformation

        Parameters
        ----------
        image: an image
        Returns: image deskewed and min/max standardized
    """
    c,v = _moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    img = interpolation.affine_transform(image,affine,offset=offset)
    
    return img

def deskew(X):
    """ Straighten all images in X

    Parameters
    ----------
    X : an N by D numpy array, representing N flatten images each with D pixels
    Returns: an N by D numpy array, representing N flatten and deskewed images with D pixels
    """
    currents = []
    for i in range(len(X)):
        currents.append(_deskew(X[i].reshape(28, 28)).flatten())
    return np.array(currents)


# elastic distortion
# Acknowledgement: modified from https://bit.ly/3581IGF
import keras
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def _elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

   # Arguments
       image: Numpy array with shape (height, width, channels).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def augment_with_elastic_distortions(samples, labels, alpha_range=0, sigma=0,
                                     width_shift_range=0, height_shift_range=0, zoom_range=0.0,
                                     iterations=1):
    """
    add elastic distortion results to selected sample of x

    Parameters
    ----------
    samples: an N by D array of flatten images
    alpha_range, sigma: arguments for `elastic_transform()`
    width_shift_range, height_shift_range, zoom_range: arguments for Keras `ImageDataGenerator()`
    iterations: Int, number of times to randomly augment the examples

    Returns
    -------
    augmented samples: a (iterations*batch_size+N) by D array of flatten images. Default batch_size is 32.

    """
    samples_reshaped = samples.reshape(len(samples), 28, 28, 1)
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        preprocessing_function=lambda x: _elastic_transform(x, alpha_range=alpha_range, sigma=sigma)
    )
    augs = [datagen.flow(samples_reshaped, labels, shuffle=False).next() for i in
            range(iterations)]  # list of augmentations for each sample
    X_aug = np.concatenate([aug[0] for aug in augs])  # augmentation array for X
    y_aug = np.concatenate([aug[1] for aug in augs])  # corresponding labels
    return np.concatenate((samples, X_aug.reshape(-1, 784))), np.concatenate((y, y_aug))