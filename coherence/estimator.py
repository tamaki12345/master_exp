import concurrent
import os
from more_itertools import batched

import numpy as np
from sklearn.linear_model import LinearRegression

from algo import embedding_similarity

from tqdm import tqdm

def calc_distances(embeddings):
    average_distances = []

    for i, a_embedding in enumerate(embeddings[:-1]):
        if a_embedding is None:
            continue

        b_embedding = embeddings[i+1]
        if b_embedding is None:
            continue

        distance = embedding_similarity(a_embedding, b_embedding)
        average_distances.append(distance)
    average_distances = np.array(average_distances)
    return np.mean(average_distances)


class Estimator():
    def __init__(self):
        self.coherence_estimator = LinearRegression()
        self.transition_estimator = LinearRegression()
        self.attributes = ['length', 'log_length', 'popularity']
        # super().__init__(filepath)

    def train_transitions(self, playlist_list, artist_matirx, track_to_ids, coherence_table):

        artist_matirx = artist_matirx.tocsc()

        X = []
        Y = []

        for _, row in tqdm( coherence_table.iterrows() ,total = len(coherence_table)):
            futures = []
            pid = int(row['id'])
            # with (concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor):
            playlist = playlist_list[pid]

            embeddings = [artist_matirx[:, track_to_ids[int(tid)]].toarray().flatten() for tid in playlist]

            Y.append(calc_distances(embeddings))
            X.append(row[self.attributes].values)

        X = np.array(X)
        Y = np.array(Y)
        print("training transition")
        self.transition_estimator.fit(X, Y)
        print("finished training transition")

    def load_from_data(self, playlist_list, artist_matirx, track_to_ids, coherence_table):
        self.train_transitions(playlist_list, artist_matirx, track_to_ids, coherence_table)

        data = coherence_table.dropna()
        X = data[self.attributes].values
        Y = data['coherence'].values
        print("training coherence")
        self.coherence_estimator.fit(X, Y)
        print("finished training coherence")