import os
import json
import random
import numpy as np
from voyager import Index, Space, StorageDataType

class RecommendationAgent:
    def __init__(
            self,
            alpha=0.1,
            min_alpha=0.05,
            alpha_decay=0.99,
            epsilon=0.3,
            min_epsilon=0.1,
            epsilon_decay=0.99,
            n_statistics=4,
            domains=None,
            visualizations=None,
            q_table_path=None,
            batch_size=10,
        ):
        """
        Initializes the RecommendationAgent.

        Parameters:
        alpha (float): Learning rate.
        min_alpha (float): Minimum learning rate after decay.
        alpha_decay (float): Decay rate of the learning rate.
        epsilon (float): Exploration rate.
        min_epsilon (float): Minimum exploration rate after decay.
        epsilon_decay (float): Decay rate of the exploration rate.
        n_statistics (int): Number of statistics for state representation.
        domains (list): List of domains for recommendations.
        visualizations (list): List of visualization options.
        q_table_path (str): Path to the Q-table JSON file for loading/saving state-action values.
        batch_size (int): Batch size for updating Q-values.
        """
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.alpha_decay = alpha_decay
        
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.batch_updates = []

        self.n_statistics = n_statistics
        self.domains = domains if domains is not None else []
        self.visualizations = visualizations if visualizations is not None else []
        self.scores = {domain: {} for domain in self.domains}
        
        self.indexes = {domain: Index(Space.Cosine, num_dimensions=n_statistics, storage_data_type=StorageDataType.Float32) for domain in self.domains}
        self.index_lookup = {}

        self.q_table_path = q_table_path if q_table_path else '../model/database.json'
        if q_table_path and os.path.isfile(self.q_table_path):
            self.load_scores()

    def load_scores(self):
        """
        Loads the scores from an existing Q-table.

        Raises:
        RuntimeError: If the Q-table file cannot be loaded.
        """
        try:
            with open(self.q_table_path, 'r') as file:
                log_data = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load Q-table from {self.q_table_path}: {e}")

        self.domains = list(log_data['q_table'].keys())
        self.visualizations = log_data['visualizations']
        self.scores = {}

        for domain in log_data['q_table']:
            self.scores[domain] = {}
            self.indexes[domain] = Index(Space.Cosine, num_dimensions=self.n_statistics, storage_data_type=StorageDataType.Float32)
            for state_id_str in log_data['q_table'][domain]:
                state_id = tuple(map(float, state_id_str.split(',')))
                self.scores[domain][state_id] = np.array(log_data['q_table'][domain][state_id_str])
                self.add_index(domain, state_id)
    
    def state_id(self, statistics):
        """
        Transforms a statistics dictionary to a tuple representing the state ID.

        Parameters:
        statistics (dict): A dictionary of statistics.

        Returns:
        tuple: Tuple representing the state ID, padded with zeros if necessary.
        """
        values = list(statistics.values())

        if len(values) < self.n_statistics:
            values.extend([0.0] * (self.n_statistics - len(values)))

        return tuple(values[:self.n_statistics])
    
    def format_state_id(self, state_id):
        """
        Formats a state ID tuple as a string.

        Parameters:
        state_id (tuple): The state ID tuple.

        Returns:
        str: The formatted state ID string.
        """
        return ','.join(f"{val:.1f}" for val in state_id)

    def recommend_visualization(self, domain, state_id, greedy=False):
        """
        Recommends a visualization for a given domain and state ID.

        Parameters:
        domain (str): The domain for which to recommend a visualization.
        state_id (tuple): The state ID representing the current state.

        Returns:
        int: The index of the recommended visualization.

        Raises:
        ValueError: If domains or visualizations are not defined.
        """
        if not self.domains or not self.visualizations:
            raise ValueError("Both domains and visualizations must be defined before choosing an action.")

        nearest_keys = self.find_nearest_index(domain, state_id, n_neighbours=10)
        if nearest_keys is None:
            q_table = np.zeros(len(self.visualizations))
        else:
            neighbour_scores = [self.scores[domain][self.format_state_id(neighbour)] for neighbour in nearest_keys]
            q_table = np.array([sum(col) / len(col) for col in zip(*neighbour_scores)])

        # Choose either an exploitation or exploration step
        if (random.uniform(0, 1) < self.epsilon) and not greedy:
            return random.randint(0, len(q_table) - 1)
        else:
            return np.argmax(q_table)

    def initialize_q_table(self, domain, state_id):
        """
        Initializes the Q-table for a new state ID.
        
        Parameters:
        domain (str): The domain.
        state_id (tuple): The state ID.
        
        Returns:
        np.ndarray: The initialized Q-table.
        """
        nearest_keys = self.find_nearest_index(domain, state_id, n_neighbours=10)

        if nearest_keys is None:
            q_table = np.zeros(len(self.visualizations))
        else:
            neighbour_scores = [self.scores[domain][self.format_state_id(neighbour)] for neighbour in nearest_keys]
            q_table = np.array([sum(col) / len(col) for col in zip(*neighbour_scores)])

        self.add_index(domain, state_id)
        self.scores[domain][self.format_state_id(state_id)] = q_table
        return q_table
    
    def update_q_value(self, domain, state_id, action, reward):
        if not self.domains or not self.visualizations:
            raise ValueError("Both domains and visualizations must be defined before choosing an action.")

        formatted_state_id = self.format_state_id(state_id)
        q_table = self.scores[domain].get(formatted_state_id)

        if q_table is None:
            q_table = self.initialize_q_table(domain, state_id)

        q_table[action] += self.alpha * (reward - q_table[action])
        q_table[action] = np.clip(q_table[action], -3, 3)
        
        self.batch_updates.append((domain, state_id, action, reward))

        if len(self.batch_updates) >= self.batch_size:
            self.process_batch_updates()

    def process_batch_updates(self):
        """
        Processes batch updates for the Q-values.
        """
        for domain, state_id, action, reward in self.batch_updates:
            self.update_nearest_neighbors(domain, state_id, action, reward)
        
        self.batch_updates = []
        self.log_update()
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def update_nearest_neighbors(self, domain, state_id, action, reward):
        nearest_keys = self.find_nearest_index(domain, state_id, n_neighbours=11)

        if nearest_keys:
            for nearest_key in nearest_keys:
                formatted_neighbor = self.format_state_id(nearest_key)
                neighbor_q_table = self.scores[domain].get(formatted_neighbor)
                if neighbor_q_table is not None:
                    neighbor_q_table[action] += self.alpha * (reward - neighbor_q_table[action])
                    neighbor_q_table[action] = np.clip(neighbor_q_table[action], -3, 3)
    
    def log_update(self):
        """
        Logs the updated Q-table to the JSON file.

        Raises:
        RuntimeError: If the Q-table file cannot be written.
        """
        scores_serializable = {
            domain: {
                state_id: q_table.tolist() for state_id, q_table in domain_q.items()
            } for domain, domain_q in self.scores.items()
        }
        
        try:
            with open(self.q_table_path, 'w') as file:
                json.dump({'visualizations': self.visualizations, 'q_table': scores_serializable}, file, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        except Exception as e:
            raise RuntimeError(f"Failed to write Q-table to {self.q_table_path}: {e}")


    def add_domain(self, domain):
        """
        Adds a domain to the list of domains if it doesn't already exist.

        Parameters:
        domain (str): The domain to add.
        """
        if not self.has_domain(domain):
            self.domains.append(domain)
            self.scores[domain] = {}
            self.indexes[domain] = Index(Space.Cosine, num_dimensions=self.n_statistics, storage_data_type=StorageDataType.Float32)

    def has_domain(self, domain):
        """
        Checks if a domain exists.

        Parameters:
        domain (str): The domain to check.

        Returns:
        bool: True if the domain exists, False otherwise.
        """
        return domain in self.domains

    def add_visualization(self, new_option):
        """
        Adds a visualization to the list of visualizations if it doesn't already exist.

        Parameters:
        new_option (str): The visualization to add.

        Raises:
        ValueError: If domains are not defined.
        """
        if not self.domains:
            raise ValueError("Domain must be defined in order to add visualizations")
        
        if new_option not in self.visualizations:
            self.visualizations.append(new_option)
            
            for domain in self.domains:
                for state_id in self.scores[domain]:
                    self.scores[domain][state_id] = np.append(self.scores[domain][state_id], 0.0)
            
            self.log_update()

    def add_index(self, domain, state_id):
        """
        Adds an index for a state ID to the domain index.

        Parameters:
        domain (str): The domain.
        state_id (tuple): The state ID.
        """
        formatted_id = self.format_state_id(state_id)
        id = self.indexes[domain].add_item(list(state_id))
        vec = self.indexes[domain].get_vector(id)
        self.index_lookup[formatted_id] = vec


    def find_nearest_index(self, domain, state_id, n_neighbours=1):
        """
        Finds the nearest neighbor for a given state ID.

        Parameters:
        domain (str): The domain.
        state_id (tuple): The state ID.
        n_neighbours (int): The number of nearest neighbors to find.

        Returns:
        list: List of tuples representing the nearest state IDs.
        """
        try:
            neighbors, _ = self.indexes[domain].query(np.array(list(state_id), dtype=np.float32), k=n_neighbours)

            if neighbors.any():
                value_array = self.indexes[domain].get_vectors(neighbors)
                
                neighbour_vectors = []
                for neighbour in value_array:
                    neighbour_vec = next((key for key, value in self.index_lookup.items() if np.array_equal(value, neighbour)), None)
                    vec = [float(numeric_string) for numeric_string in neighbour_vec.split(',')]
                    neighbour_vectors.append(vec)
                
                return [tuple(val[:self.n_statistics]) for val in neighbour_vectors]
            else:
                return None
        except RuntimeError or TypeError:
            return None

    def export_index(self, folder='model/indices/'):
        """
        Exports indices to a .voy file.

        Parameters:
        folder (str): The folder to save the indices.
        """
        for domain, index in self.indexes.items():
            index.save(folder + domain + '.voy')