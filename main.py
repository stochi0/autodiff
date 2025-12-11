from src.nn import Value
from src.viz import draw_dot

def main():
    print("Hello from autodiff!")
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a + b
    print(c)
    dot = draw_dot(c)
    dot.render('graph', format='svg', cleanup=True)

if __name__ == "__main__":
    main()
