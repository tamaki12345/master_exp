from distance_matrix import DistanceMatrix
from node import Node

import numpy as np

class Calculator():

    def __init__(self, playlist: list[Node]):
        self.distance_matrix = DistanceMatrix(playlist)
        self.playlist = playlist

    def playlist_variance(self, nodes: list[Node]):
        sum_ = 0
        n = len(nodes)
        for k, a_track in enumerate(nodes[:-1]):
            i = a_track.id
            for l, b_track in enumerate(nodes[k + 1:]):
                j = b_track.id
                sum_ += self.distance_matrix.substract(i, j) ** 2
        div = n * (n - 1)
        
        # 分散を返す
        return sum_ / div

    def get_pl_variance(self, nodes: list[Node]):
        artists = []
        for node in nodes:
            artists += [f for f in node.artists]
        pl_variances = self.playlist_variance(artists)
        
        return np.array(pl_variances)
    
    def sequential_variance(self, nodes: list[Node]):
        sum_ = 0
        n = len(self.playlist)
        for k, a_track in enumerate(nodes[:-1]):
            i = a_track.id
            j = self.playlist[k + 1].id
            sum_ += self.distance_matrix.substract(i, j) ** 2

        # 分散を返す
        return sum_ / (n - 1) / 2
    
    def get_sq_variance(self, nodes: list[Node]):
        artists = []
        for node in nodes:
            artists += [f for f in node.artists]
        sq_variance = self.sequential_variance(artists)
        return sq_variance
    
    def substract(self, a, b):
        return self.distance_matrix.substract(a,b)