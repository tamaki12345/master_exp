import os

class BayesianService():
    def __init__(self, alpha=1e-5, threshold=0.1):
        self.alpha = alpha
        self.threshold = threshold
        self.mapping = {}

        # filepath = os.path.join(cached_path, f'bayesian_service.pk')
        # super().__init__(filepath)

    @staticmethod
    def _create_neighbors_dict(playlist_list):
        neighbors_dict = {}
        for i, (pid, playlist) in enumerate(playlist_list):
            if i % 10000 == 0:
                print(f'creating neighboring dict {i}')

            for i in range(len(playlist)-1):
                track = playlist[i]
                next_track = playlist[i+1]
                neighbors = neighbors_dict.setdefault(track, {})
                neighbors.setdefault(next_track, []).append(pid)
        return neighbors_dict

    def load_from_data(self, playlist_list):
        neighbors_dict = self._create_neighbors_dict(playlist_list)
        M = len(neighbors_dict)
        for i, (a_track, neighbors) in enumerate(neighbors_dict.items()):
            if i % 10000 == 0:
                print(f'searching for track sequences {i}')
            denominator = sum(len(pids) for pids in neighbors.values()) + self.alpha*M
            for b_track, pids in neighbors.items():
                likelihood = (self.alpha + len(pids)) / denominator
                if likelihood > self.threshold:
                    self.mapping.setdefault(a_track, {})[b_track] = likelihood, len(pids)

    def save(self):
        self._save(self.mapping)

    def load_from_cache(self):
        self.mapping = self._load_from_cache()