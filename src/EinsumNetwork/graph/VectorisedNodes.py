from src.EinsumNetwork.graph.EiNetAddress import EiNetAddress
from itertools import count

class DistributionVector:
    """
    Represents either a vectorized leaf or a vectorized sum node in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """
    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope):
        """
        :param scope: the scope of this node
        """
        self.scope = tuple(sorted(scope))
        self.num_dist = None
        self.einet_address = EiNetAddress()
        self.id = next(self._id_counter)

    def __lt__(self, other):
        if type(other) == Product:
            return True
        else:
            return (self.scope, self.id) < (other.scope, other.id)
        
class Product:
    """
    Represents a (cross-)product in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """
    # we assign each object a unique id.
    _id_counter = count(0)

    def __init__(self, scope):
        self.scope = tuple(sorted(scope))
        self.id = next(self._id_counter)

    def __lt__(self, other):
        if type(other) == DistributionVector:
            return False
        else:
            return (self.scope, self.id) < (other.scope, other.id)