# Tiny Autograd & Neural Nets (Inspired by Micrograd)
Educational neural network and autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd)  by Andrej Karpathy, extended with my own experiments and implementations.

This project builds upon [**micrograd**](https://github.com/karpathy/micrograd) by **Andrej Karpathy**, a ~100-line autograd engine and tiny neural net library.  
I use it here as a foundation to learn, experiment, and extend with my own implementations.

---

## ✨ Features
- 🔢 Reverse-mode autodiff over a dynamically built DAG  
- 🧠 Minimal neural networks with a PyTorch-like API  
- 📝 Educational examples of training simple models from scratch  
- 📈 Graph visualization of forward/backward passes  
- 🚀 My extensions: *[list your additions here]*  

---

## ⚡ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
```

## 🚀 Usage Example
```python
from engine import Value
from nn import Neuron

# Example forward + backward pass
a, b = Value(-4.0), Value(2.0)
c = a + b
d = a * b + b**3
d.backward()

print(a.grad, b.grad)
```

## 📊 Visualizing the Computation Graph

This project supports graph tracing using Graphviz.
Here’s a minimal example:
```python
from engine import Value
from visualize import draw_dot  # make sure visualize.py has the helper

# simple neuron-like operation
a, b = Value(1.0), Value(-2.0)
c = a * b + a + b
dot = draw_dot(c)

# save visualization
dot.render("graph", format="png", cleanup=True)
```

## 🙏 Credits

Original idea and base implementation from [**micrograd**](https://github.com/karpathy/micrograd) by **Andrej Karpathy**
Extensions, experiments, and modifications by 

## 📄 License

This project follows the MIT License, in line with the original micrograd license.