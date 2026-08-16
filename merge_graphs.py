import os
from graph_converter import Icoxx2Graph
#from test

g = Icoxx2Graph(bond_length_cutoff=5).get_graph(path_to_structure='CONTCAR.gz',
                path_to_icoxxlist='ICOHPLIST.lobster.gz',
            )    




