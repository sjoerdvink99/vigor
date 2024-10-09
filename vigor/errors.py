class GraphError(Exception):
    """Base class for errors in the Graph class."""
    def __init__(self, message: str):
        super().__init__(message)

class ModularityError(GraphError):
    """Raised when an error occurs while calculating the modularity."""
    def __init__(self, message="Failed to calculate modularity."):
        super().__init__(message)

class AssortativityError(GraphError):
    """Raised when an error occurs while calculating the assortativity coefficient."""
    def __init__(self, message="Failed to calculate assortativity coefficient."):
        super().__init__(message)

class SpatialAttributeError(GraphError):
    """Raised when an error occurs while detecting spatial attributes in the graph."""
    def __init__(self, message="Error in detecting spatial attributes."):
        super().__init__(message)

class TemporalAttributeError(GraphError):
    """Raised when an error occurs while detecting temporal attributes in the graph."""
    def __init__(self, message="Error in detecting temporal attributes."):
        super().__init__(message)

class NotConnectedError(GraphError):
    """Raised when attempting to calculate statistics for non-connected graphs."""
    def __init__(self, message="Graph is not connected. Cannot calculate diameter or average shortest path length."):
        super().__init__(message)

class BipartiteError(GraphError):
    """Raised when an error occurs during the bipartite graph check."""
    def __init__(self, message="Failed to check if the graph is bipartite."):
        super().__init__(message)

class DirectedGraphError(GraphError):
    """Raised when an error occurs while checking if the graph is directed."""
    def __init__(self, message="Error while checking if the graph is directed."):
        super().__init__(message)