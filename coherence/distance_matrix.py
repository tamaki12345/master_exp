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

        for i, a_track in enumerate(playlist[:-1]):
            for j, b_track in enumerate(playlist[i+1:]):
                d = a_track - b_track
                distance_matrix[i, i+j+1] = d
                distance_matrix[i+j+1, i] = d

        distance_matrix = (distance_matrix*1000000000).astype(int)

        return distance_matrix
    
    def substract(self, a, b):
        return int(self.distance_matrix[a,b])