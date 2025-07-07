import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

def embedding_similarity(a_embeddings, b_embeddings):
    return np.mean(cosine_distances(a_embeddings.T, b_embeddings.T))


class Node:
    def __init__(self, id, name, tracks: list[int], artists: list):
        self.id = id
        self.name = name 
        self.tracks = tracks
        self.artists = artists

    @staticmethod
    def create_node(id, name, tracks: list[int],matrix, track_to_ids):
        artists = [matrix[:, track_to_ids[t]] for t in tracks]
        return Node(id, name, tracks, artists)


    def __sub__(self, other):
        assert isinstance(other, Node)

        sq_distances = 0
        for artist_a in self.artists:
            for artist_b in other.artists:
                # print(artist_a.ndim)
                # print(artist_b.ndim)
                sq_distances += cosine_distances(artist_a.T, artist_b.T)
        return np.mean(sq_distances)