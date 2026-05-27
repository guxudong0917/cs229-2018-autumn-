from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('init_centroids function not implemented')

    #初始化为(k,3)
    pixels=image.reshape(-1,3)

    index=np.random.choice(pixels.shape[0],num_clusters,replace=False)
    centroids_init=pixels[index].astype("float")

    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
    # Usually expected to converge long before `max_iter` iterations
    # Initialize `dist` vector to keep track of distance to every centroid

    k=centroids.shape[0]
    m=image.shape[0]*image.shape[1]
    image_flat=image.reshape(-1,3)
    dist=np.zeros((m,k))
    # Loop over all centroids and store distances in `dist`
    for it in range(max_iter):
        for i in range(m):
            #用(n,1)-(n,k)广播，然后沿着列去求范数
            dist[i]=np.linalg.norm(image_flat[i].reshape(-1,1)-centroids.T,ord=2,axis=0)**2
           
    # Find closest centroid and update `new_centroids`
        closest=np.argmin(dist,axis=1).flatten()
        for j in range(k):

            if (image_flat[closest==j].shape[0]>0):
                centroids[j]=np.mean(image_flat[closest==j],axis=0)
    # Update `new_centroids`
        if it%print_every==0:
            print(centroids)
    # *** END YOUR CODE ***
    new_centroids=centroids.astype(int)
    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
    # Initialize `dist` vector to keep track of distance to every centroid

    H=image.shape[0]
    W=image.shape[1]

    k=centroids.shape[0]
    image_flat=image.reshape(-1,3).astype(float)
    m=image_flat.shape[0]
    # Loop over all centroids and store distances in `dist`
    dist=np.zeros((m,k))
    for i in range(m):
        dist[i]=np.linalg.norm(image_flat[i].reshape(-1,1)-centroids.T,ord=2,axis=0)**2

    # Find closest centroid and update pixel value in `image`
    closest_ind=np.argmin(dist,axis=1)
    image_flat=centroids[closest_ind]
    image=image_flat.reshape(H,W,-1)
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)
    colors = np.unique(image_clustered.reshape(-1, image_clustered.shape[-1]), axis=0)
    print(colors.shape[0])
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='../data/peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='../data/peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
