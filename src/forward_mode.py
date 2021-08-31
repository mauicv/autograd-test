class Var:
    def __init__(self, val, dot):
        self.val = val
        self.dot = dot

    def __add__(self, other):
        dot = self.dot + other.dot
        return Var(self.val + other.val, dot)

    def __neg__(self, other):
        dot = self.dot - other.dot
        return Var(self.val - other.val, dot)

    def __mul__(self, other):
        dot = other.val * self.dot + self.val * other.dot
        return Var(self.val * other.val, dot)

    def __truediv__(self, other):
        dot = (other.val * self.dot - self.val * other.dot) / (other.val ** 2)
        return Var(self.val / other.val, dot)

    def __str__(self):
        return f'<val: {str(self.val)}, dot: {str(self.dot)}>'


if __name__ == "__main__":
    v1 = Var(1, 1)
    v2 = Var(2, 0)
    v3 = (v1 + v2) * v1
    print(v3)
