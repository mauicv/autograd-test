class Edge:
    def __init__(self, from_node, to_node, dv):
        self.from_node = from_node
        self.to_node = to_node
        self.dv = dv


class Tape:
    def __init__(self):
        self.nodes = []

    def watch(self, nodes):
        self.nodes.extend(nodes)
        for node in nodes:
            node.tape = self

    def compute_grads(self):
        for node in self.nodes[::-1]:
            node.compute_df_dn()


class Var:
    def __init__(self, value, tape=None):
        self.in_edges = []
        self.out_edges = []
        self.df_dn = None
        self.value = value
        if tape:
            self.tape = tape
            self.tape.nodes.append(self)

    def new_node(self, other, value, dv_s, dv_o):
        node = Var(value, tape=self.tape)
        e1 = Edge(self, node, dv=dv_s)
        e2 = Edge(other, node, dv=dv_o)
        self.out_edges.append(e1)
        other.out_edges.append(e2)
        node.in_edges = [e1, e2]
        return node

    def compute_df_dn(self):
        if not self.out_edges:
            self.df_dn = 1
        else:
            self.df_dn = 0
            for edge in self.out_edges:
                self.df_dn += edge.to_node.df_dn * edge.dv
        return self.df_dn

    def __add__(self, other):
        return self.new_node(other, value=self.value + other.value,
                             dv_s=1, dv_o=1)

    def __neg__(self, other):
        return self.new_node(other, value=self.value - other.value,
                             dv_s=1, dv_o=-1)

    def __mul__(self, other):
        return self.new_node(other, value=self.value * other.value,
                             dv_s=other.value, dv_o=self.value)

    def __truediv__(self, other):
        return self.new_node(other, value=self.value / other.value,
                             dv_s=-1 / self.value ** 2, dv_o=1 / other.value)


if __name__ == "__main__":
    x, y, c1, c2 = (Var(2), Var(3), Var(3), Var(2))
    tape = Tape()
    tape.watch([x, y, c1, c2])
    fn = c2 * y * x * x + x * y + c1
    tape.compute_grads()
    print('fn output value =', fn.value)
    print('x.df_dn \t=', x.df_dn)
    print('y.df_dn \t=', y.df_dn)
