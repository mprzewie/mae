"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
and later from TokenCut: https://github.com/YangtaoWANG95/TokenCut/blob/master/object_discovery.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def batch_ncut(patch_tokens, tau=0, eps=1e-5, im_name='', no_binary_graph=True):
    patch_tokens = F.normalize(patch_tokens, p=2, dim=2)

    A = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))


    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = (A > tau).float()
        A = A + ((A==0)*eps)

    d_i = A.sum(dim=2)
    D = torch.diag_embed(d_i)

    Ae = D - A
    Be = D
    L = torch.linalg.cholesky(Be)
    L_inv = torch.linalg.inv(L)
    L_inv_T = L_inv.transpose(1, 2)
    transformed_A = L_inv @ Ae @ L_inv_T
    with torch.cuda.amp.autocast(enabled=False):
        eigenvalues, eigenvectors_w = torch.linalg.eigh(transformed_A.float())

    eigenvectors_v = torch.linalg.solve(L.transpose(1, 2), eigenvectors_w)

    second_smallest_eigenvec = eigenvectors_v[:, :, 1]

    avg = second_smallest_eigenvec.mean(dim=1, keepdim=True)

    bipartition = ((second_smallest_eigenvec - avg) > 0).float()

    return bipartition, second_smallest_eigenvec





def ncut(feats, dims, scales, init_image_size, tau=0, eps=1e-5, im_name='', no_binary_graph=True, method: str = "tcut"):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    # cls_token = feats[0, 0:1, :].cpu().numpy()

    assert feats.shape[1] == 196, feats.shape
    feats = feats[0, :, :]
    # assert False, feats.shape

    feats = F.normalize(feats, p=2)
    # print(feats[0, :5])

    A = (feats @ feats.transpose(1, 0))
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    assert False, d_i[:5]
    D = np.diag(d_i)
    # print("A", A[:1, :10])
    # print("d_i", d_i[:10])

    # Print second and third smallest eigenvector
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])

    second_smallest_vec = eigenvectors[:, 0]
    seed = np.argmax(np.abs(second_smallest_vec))
    # print("eig", eigenvectors[:4])

    if method == "tcut":
        # Using average point to compute bipartition
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)

    # method3 EM algo
    elif method == "gmm":
        second_smallest_vec = eigenvectors[:, 0:1]
        bipartition = GMM(second_smallest_vec)

    elif method == "kmeans":
        # method 4 Kmeans
        second_smallest_vec = eigenvectors[:, 0:1]
        bipartition = Kmeans_partition(second_smallest_vec)

    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects, cc = detect_box(bipartition, seed, dims, scales=scales,
                                      initial_im_size=init_image_size[1:])  ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


def detect_box(bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError


def GMM(eigvec):
    gmm = GaussianMixture(n_components=2, max_iter=300)
    gmm.fit(eigvec)
    partition = gmm.predict(eigvec)
    return partition


def Kmeans_partition(eigvec):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(eigvec)
    return kmeans.labels_