class Tape:
    def __init__(self, root):
        self.nodes = [root]
        self.seen = {root}

    def empty(self):
        self.nodes = []

    def add(self, nodes):
        for node in nodes:
            if node not in self.seen:
                self.nodes.append(node)
                self.seen.add(node)


class Var:
    def __init__(self, w, tape=None, seen=None, name=None):
        self.name = name
        self.tape = Tape(self) if not tape else tape
        self.val = w
        self.dep = []
        self.dot = None

    def new_node(self, w, dw, other):
        """new_node

        Note that this function assumes 2 - 1 operations such as addition,
        subtraction, multiplication, division. Will not work with functional
        application.
        """
        self.tape.add([self, other])
        dw_self, dw_other = dw
        name = f'({self.name}) o ({other.name})'
        new_var = Var(w, tape=self.tape, name=name)
        self.dep.append((new_var, dw_self))
        other.dep.append((new_var, dw_other))
        return new_var

    def __add__(self, other):
        w = self.val + other.val  # w = x + y
        dw = (1, 1)  # (dw/dx, dw/dy)
        return self.new_node(w, dw, other)

    def __neg__(self, other):
        w = self.val - other.val  # w = x - y
        dw = (1, -1)  # (dw/dx, dw/dy)
        return self.new_node(w, dw, other)

    def __mul__(self, other):
        w = self.val * other.val  # w = x * y
        dw = (other.val, self.val)  # (dw/dx, dw/dy)
        return self.new_node(w, dw, other)

    def __truediv__(self, other):
        w = self.val / other.val  # w = x / y
        dw = (1 / other.val, -1 / self.val ** 2)  # (dw/dx, dw/dy)
        return self.new_node(w, dw, other)

    def compute_grad(self):
        grads = []
        self.dot = 1
        for node in self.tape.nodes[::-1]:
            dot = 0
            for node_dep, dw in node.dep:
                dot += node_dep.dot * dw
            grads.append(dot)
            node.dot = dot
        return grads

    def __str__(self):
        return f'<name={self.name}, val: {str(self.val)}>'


if __name__ == "__main__":
    x = Var(2, name='x')
    y = Var(1, name='y')
    output = (x + y) * x
    print(output.compute_grad())
