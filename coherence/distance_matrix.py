import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse.linalg import norm
from node import Node


class DistanceMatrix():
    
    def __init__(self, playlist: list[Node]):
        self.distance_matrix = self.get_distance_matrix(playlist)

    def get_distance_matrix(self, playlist: list[Node]):
        n = len(playlist)
        distance_matrix = np.zeros((n, n))
        # embeddings = [ track.artists for track in playlist]

        for k, a_track in enumerate(playlist[:-1]):
            for l, b_track in enumerate(playlist[k + 1:]):
                i = a_track.id
                j = b_track.id
                d = a_track - b_track
                distance_matrix[i,j] = d
                distance_matrix[j, i] = d

        return distance_matrix
    
    def substract(self, a, b):
        return self.distance_matrix[a,b]