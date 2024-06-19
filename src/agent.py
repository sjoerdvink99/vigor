import os
import json
import random
import numpy as np
from voyager import Index, Space

class RecommendationAgent:
    def __init__(
            self,
            alpha=0.1,
            epsilon=0.1,
            n_statistics=4,
            domains=None,
            visualizations=None,
            q_table_path=None,
            batch_size=10,
            decay_rate=0.99,
            min_alpha=0.05
        ):
        """
        Initializes the RecommendationAgent.

        Parameters:
        alpha (float): Learning rate.
        epsilon (float): Exploration rate.
        n_statistics (int): Number of statistics for state representation.
        domains (list): List of domains.
        visualizations (list): List of visualizations.
        q_table_path (str): Path to the Q-table JSON file.
        """
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.decay_rate = decay_rate
        
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.batch_updates = []

        self.n_statistics = n_statistics
        self.q_table_path = q_table_path if q_table_path else '../model/database.json'
        self.domains = domains if domains is not None else []
        self.visualizations = visualizations if visualizations is not None else []
        self.scores = {domain: {} for domain in self.domains}
        
        self.indexes = {domain: Index(Space.Euclidean, num_dimensions=n_statistics) for domain in self.domains}

        if q_table_path and os.path.isfile(self.q_table_path):
            self.load_scores()

    def load_scores(self):
        """
        Loads the scores from an existing Q-table.
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
            self.indexes[domain] = Index(Space.Euclidean, num_dimensions=self.n_statistics)
            for state_id_str in log_data['q_table'][domain]:
                state_id = tuple(map(float, state_id_str.split(',')))
                self.scores[domain][id] = np.array(log_data['q_table'][domain][state_id])
                self.add_index(domain, state_id)
    
    def state_id(self, statistics):
        """
        Transforms a statistics object to a bytes array.

        Parameters:
        statistics (dict): A dictionary of statistics.

        Returns:
        bytes: Byte array representing the state ID.
        """
        values = list(statistics.values())

        # Add padding if not enough statistics
        if len(values) < self.n_statistics:
            values.extend([0.0] * (self.n_statistics - len(values)))

        return tuple(values[:self.n_statistics])
    
    def format_state_id(self, state_id):
        return ','.join(f"{val:.1f}" for val in state_id)

    def recommend_visualization(self, domain, state_id):
        if not self.domains or not self.visualizations:
            raise ValueError("Both domains and visualizations must be defined before choosing an action.")

        formatted_state_id = self.format_state_id(state_id)
        q_table = self.scores[domain].get(formatted_state_id)

        if q_table is None:
            nearest_key = self.find_nearest_index(domain, state_id, n_neighbours=20)

            if nearest_key is None:
                q_table = np.full((1, len(self.visualizations)), 1)[0]
            else:
                neighbour_scores = []
                for neighbour in nearest_key:
                    formatted_neighbour = self.format_state_id(neighbour)
                    score_set = self.scores[domain].get(formatted_neighbour)
                    if score_set is not None:
                        neighbour_scores.append(score_set)

                if not neighbour_scores:
                    q_table = np.full((1, len(self.visualizations)), 1)[0]
                else:
                    scores = []
                    for col in zip(*neighbour_scores):
                        scores.append(sum(col) / len(col))
                    q_table = np.array(scores)

            self.add_index(domain, state_id)
            self.scores[domain][formatted_state_id] = q_table
            self.log_update()

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(q_table) - 1), True
        else:
            return np.argmax(q_table), False

    def update_q_value(self, domain, state_id, action, reward, require_feedback):
        if not self.domains or not self.visualizations:
            raise ValueError("Both domains and visualizations must be defined before choosing an action.")

        if require_feedback:
            self.batch_updates.append((domain, state_id, action, reward))

            if len(self.batch_updates) >= self.batch_size:
                for domain, state_id, action, reward in self.batch_updates:
                    formatted_state_id = self.format_state_id(state_id)
                    q_table = self.scores[domain].get(formatted_state_id)

                    if q_table is None:
                        q_table = np.zeros(len(self.visualizations))
                        self.scores[domain][formatted_state_id] = q_table

                    q_table[action] += self.alpha * (reward - q_table[action])

                self.batch_updates = []
                self.log_update()
                self.alpha = max(self.min_alpha, self.alpha * self.decay_rate)

    def log_update(self):
        """
        Logs the updated values to the database (json file in this case)
        """
        scores_serializable = {
            domain: {
                ','.join(map(str, state_id)): q_table.tolist() for state_id, q_table in domain_q.items()
            } for domain, domain_q in self.scores.items()
        }
        
        try:
            with open(self.q_table_path, 'w') as file:
                json.dump({'visualizations': self.visualizations, 'q_table': scores_serializable}, file, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        except Exception as e:
            raise RuntimeError(f"Failed to write Q-table to {self.q_table_path}: {e}")


    def add_domain(self, domain):
        """
        Adds a domain to the list of domains.

        Parameters:
        domain (str): The domain to add.
        """
        if not self.has_domain(domain):
            self.domains.append(domain)
            self.scores[domain] = {}
            self.indexes[domain] = Index(Space.Euclidean, num_dimensions=self.n_statistics)
            self.log_update()

    def has_domain(self, domain):
        """
        Checks if a domain exists
        """
        return domain in self.domains

    def add_visualization(self, new_option):
        """
        Adds a visualization to the list of visualizations.

        Parameters:
        new_option (str): The visualization to add.
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
        state_id (bytes): The state ID.
        """
        self.indexes[domain].add_item(list(state_id))

    def find_nearest_index(self, domain, state_id, n_neighbours=1):
        """
        Finds the nearest neighbor for a given state ID.

        Parameters:
        domain (str): The domain.
        state_id (bytes): The state ID.

        Returns:
        bytes: The nearest state ID.
        """
        try:
            neighbors, _ = self.indexes[domain].query(np.array(list(state_id), dtype=np.float32), k=n_neighbours)

            if neighbors.any():
                value_array = self.indexes[domain].get_vectors(neighbors)
                return [tuple(val[:self.n_statistics]) for val in value_array]
            else:
                return None
        except RuntimeError or TypeError:
            return None

    def export_index(self, folder='model/indexes/'):
        """
        Exports indices to a .voy file.

        Parameters:
        folder (str): The folder to save the indices.
        """
        for domain, index in self.indexes.items():
            index.save(folder + domain + '.voy')