Only 10 times faster than pure python implementations, recommended to not use it

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
