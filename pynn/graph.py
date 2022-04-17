from typing import List


class Graph(object):
    def __init__(self):
        # from pynn.node import Node
        self.nodes: List["Node"] = []
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    @property
    def node_count(self):
        return len(self.nodes)

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()


default_graph = Graph()
