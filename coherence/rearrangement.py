from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from scipy.sparse.linalg import norm

import numpy as np

from tqdm import tqdm

import joblib

from concurrent.futures import ProcessPoolExecutor, as_completed

class Rearrangement():
    def __init__(self, track_artist_id, playlist_artist_matrix):
        self.track_artist_id = track_artist_id
        self.matrix = playlist_artist_matrix
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

    def rearrange(self, playlist):

        if len(playlist) < 3:
            return playlist
        
        n, self.distance_matrix = self.get_distance_matrix(playlist)

        self.manager = pywrapcp.RoutingIndexManager( n, 1, [0], [n-1] )
        self.routing = pywrapcp.RoutingModel(self.manager)
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        self.solution = self.routing.SolveWithParameters(self.search_parameters)

        if self.solution == None:
            return playlist

        arranged = self.get_routes()[0]

        return arranged

    def get_distance_matrix(self, playlist):
        n = len(playlist)
        distance_matrix = np.zeros((n, n))
        embeddings = [self.matrix[:, self.track_artist_id[int(tid)]] for tid in playlist]

        for i, a_track in enumerate(embeddings[:-1]):
            for j, b_track in enumerate(embeddings[i+1:]):
                d = 1 - a_track.T.dot(b_track)[0,0] / ( norm(a_track) * norm(b_track) ) 
                distance_matrix[i, i+j+1] = d
                distance_matrix[i+j+1, i] = d

        distance_matrix = (distance_matrix*1000000000).astype(int)

        return n, distance_matrix
    
    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node, to_node]
    
    def get_routes(self):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(self.routing.vehicles()):
            index = self.routing.Start(route_nbr)
            route = [self.manager.IndexToNode(index)]
            while not self.routing.IsEnd(index):
                index = self.solution.Value(self.routing.NextVar(index))
                route.append(self.manager.IndexToNode(index))
            routes.append(route)
        return routes
    
    def get_route_parallel(self, playlist_list, threshold = 100):

        # calc_ids = []
        # for p_id, playlist in enumerate(playlist_list):
        #     if len(playlist) <= threshold:
        #         calc_ids.append( p_id )

        rearranged = joblib.Parallel(n_jobs=-1)(joblib.delayed( self.rearrange)( playlist_list[p_id][:min(threshold, len(playlist_list[p_id]))] ) for p_id in tqdm(range(len(playlist_list))) )

        result = []
        for p_id, playlist in enumerate(rearranged):
            result.append( (p_id, tuple(playlist)) )

        return result
    
    def get_route_ProcessExe(self, playlist_list, threshold = 100):

        calc_ids = []
        calc_playlists = []
        for p_id, playlist in enumerate(playlist_list):
            if len(playlist) <= threshold:
                calc_ids.append( p_id )
                calc_playlists.append( playlist )

        result = []
        with tqdm(total = len(calc_ids)) as pbar:
            with ProcessPoolExecutor(max_workers=None) as executor:
                rearranged = {executor.submit(self.get_routes, playlist_list[p_id]): p_id for p_id in calc_ids}

                for id, playlist in enumerate(as_completed(rearranged)):
                    result.append(calc_ids[id], playlist)
                    pbar.update(1)

        return result