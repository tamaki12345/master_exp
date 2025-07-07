import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

from bayesian import BayesianService

from node import Node
from coherence_calculator import Calculator

def embedding_similarity(a_embeddings, b_embeddings):
    return np.mean(cosine_distances(a_embeddings.T, b_embeddings.T))


# プレイリスト内分散
def playlist_variance(embeddings):
    sum_ = 0
    n = len(embeddings)
    for i, a_track in enumerate(embeddings[:-1]):
        for j, b_track in enumerate(embeddings[i + 1:]):
            sum_ += embedding_similarity(a_track, b_track) ** 2
    div = n * (n - 1)
    
    # 分散を返す
    return sum_ / div

# プレイリスト内連続分散
def sequential_variance(embeddings):
    sum_ = 0
    n = len(embeddings)
    for i, a_artists in enumerate(embeddings[:-1]):
        b_artists = embeddings[i + 1]
        sum_ += embedding_similarity(a_artists, b_artists) ** 2

    # 分散を返す
    return sum_ / (n - 1) / 2

def get_pl_variance(nodes: list[Node]):
    artists = []
    for node in nodes:
        artists += [f for f in node.artists]
    pl_variances = playlist_variance(artists)
    
    return np.array(pl_variances)

def get_sq_variance(nodes: list[Node]):
    artists = []
    for node in nodes:
        artists += [f for f in node.artists]
    sq_variance = sequential_variance(artists)
    return sq_variance

class GreedyArrangeAlgo():

    def __init__(self, pid, playlist, bayesian_threshold=0.70):
        self.pid = pid
        self.bayesian_threshold = bayesian_threshold
        # filepath = os.path.join(cached_path, f'algorithm_{pid}.pk')

        self.playlist = playlist
        self.nodes = []

        self.target_coherence = np.zeros(11, dtype=float)
        self.target_transition = np.zeros(11, dtype=float)
        self.pl_variance = np.zeros(11, dtype=float)

        # super().__init__(filepath)

    def coherence_error(self, nodes: list[Node]):
        return np.sum(np.square(self.target_coherence - self.get_coherence(nodes)))
    
    def transition_error(self, i, current: list[Node], previous: list[Node]):
        current_dist = self.calculator.substract( current[i].id, current[i+1].id )
        previous_dist = self.calculator.substract( previous[i].id, previous[i+1].id )

        error = np.square(current_dist - self.target_transition) - np.square(previous_dist - self.target_transition)
        return np.sum(error)
    
    def get_coherence(self, nodes: list[Node]):
        return 1.0 - self.calculator.sequential_variance(nodes) / self.pl_variance
    
    def get_new_transitions(self, i, dist, current: list[Node]):
        if dist == 1:
            if len(current[i].tracks) == 1 and len(current[i+1].tracks) == 1:
                return [i-1, i+1]
            else:
                return [i - 1, i, i + 1]
        else:
            return [i-1, i, i+dist-1, i+dist]
        
    def _create_node_system(self, bayesian_service: BayesianService, matrix, track_to_ids):
        matrix = matrix.tocsc()
        i = 0
        id = 0
        self.nodes = []
        tracks = self.playlist

        while i < len(tracks):
            a_track = int(tracks[i])
            node_tracks = [a_track]
            node_name = [i+1]
            for j in range(i + 1, len(tracks)):
                b_track = int(tracks[j])
                try:
                    score, counts = bayesian_service.mapping[a_track][b_track]
                    if score > self.bayesian_threshold:
                        node_tracks.append(b_track)
                        node_name.append(j+1)
                        i = j
                        continue
                except KeyError:
                    break
            node = Node.create_node(id, node_name, node_tracks, matrix, track_to_ids)
            id += 1
            self.nodes.append(node)
            i+=1

    def train(self, output=False):
        playlist = self.nodes.copy()
        playlist_coh_error = self.coherence_error(playlist)
        if output:
            print(f"{0:3d}: dist={0:3d}, coh_error={playlist_coh_error:.4f}")

        iteration = 0
        total_trans_error = 0
        dist = 1
        N = len(playlist)
        while dist < N-2:
            min_trans_error = float('inf')
            playlist_best = None
            for i in range(1, N-dist-1):
                playlist_prime = playlist.copy()
                playlist_prime[i], playlist_prime[i+dist] = playlist_prime[i+dist], playlist_prime[i]
                playlist_prime_coh_error = self.coherence_error(playlist_prime)
                if playlist_prime_coh_error >= playlist_coh_error:
                    continue

                trans_error = sum(self.transition_error(j, playlist_prime, playlist) for j in
                                  self.get_new_transitions(i, dist, playlist_prime))
                if trans_error < min_trans_error:
                    min_trans_error = trans_error
                    playlist_best = playlist_prime
            if playlist_best is not None:
                iteration += 1
                total_trans_error += min_trans_error
                playlist = playlist_best
                playlist_coh_error = self.coherence_error(playlist_best)
                if output:
                    print(f"{iteration:3d}: dist={dist:3d}, coh_error={playlist_coh_error:.4f}, trans_error={min_trans_error: .4f}")
                dist = 1
            else:
                dist += 1
        if output:
            print("Total trans error:", total_trans_error / iteration)
        return playlist
    
    def _set_targets(self, estimator, coherence_table):
        X = coherence_table.loc[[self.pid]][estimator.attributes].values
        self.target_coherence = estimator.coherence_estimator.predict(X)[0]
        self.target_transition = estimator.transition_estimator.predict(X)[0]

    def load_from_data(self, playlist, estimator, coherence_table, bayesian_service, matrix, track_to_ids):
        self.playlist = playlist
        self._set_targets(estimator, coherence_table)
        self._create_node_system(bayesian_service, matrix, track_to_ids)

        self.calculator = Calculator(self.nodes)
        if len(self.nodes) > 1:
            self.pl_variance = self.calculator.playlist_variance(self.nodes)
        else:
            self.pl_variance = 0