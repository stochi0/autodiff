class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._op = _op
        self._children = set(_children)
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._children}, _op={self._op}, label={self.label})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    def __truediv__(self, other):
        return Value(self.data / other.data, (self, other), '/')

    def __pow__(self, other):
        return Value(self.data ** other.data, (self, other), '**')

    def __sub__(self, other):
        return Value(self.data - other.data, (self, other), '-')