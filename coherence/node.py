import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

def embedding_similarity(a_embeddings, b_embeddings):
    return np.mean(cosine_distances(a_embeddings.T, b_embeddings.T))


class Node:
    def __init__(self, name, tracks: list[int], artists: list):
        self.name = name 
        self.tracks = tracks
        self.artists = artists

    @staticmethod
    def create_node(name, tracks: list[int],matrix, track_to_ids):
        artists = [matrix[:, track_to_ids[t]] for t in tracks]
        return Node(name, tracks, artists)


    def __sub__(self, other):
        assert isinstance(other, Node)

        artist_a = self.artists[-1]
        artist_b = self.artists[0]
        sq_distances =embedding_similarity(artist_a, artist_b)
        return sq_distances