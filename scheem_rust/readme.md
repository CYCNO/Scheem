Original Scheem was very slow, I initially thought, it was because of the language (which is kinda true) so i ported it in rust (wasted my 3 days for it), and it did made the training part 10x faster, but still it was very slow, I assume the underlying problem is `micrograd` value storing and with each computation it is creating lot of heap allocations, which is slowing down the process. Also I don't have enough skill for now to make it any better. Probably after some time, when I would gain more knowledge regarding this, I would make it faster

> It's not like it doesn't work, It does, but it is just very slow

But if you wanna try it out this rust part, here are the procedures:

# Procedure

### 1. Set Up the Virtual Environment
```bash
python3 -m venv .env
source .env/bin/activate
```

### 2. Install Dependencies
```bash
pip install maturin pandas scikit-learn jupyter
```

### 3. Compile the Rust Extension
```bash
maturin develop --release
```

### 4. Run the Code
```bash
PYTHONPATH=. python main.py
```
Or run the notebook for the demo:
```bash
jupyter notebook house_price.ipynb
```
