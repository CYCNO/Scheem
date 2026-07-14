from scheem_rust import Mat, ValueRef, free_graph
import time

class Scheem:
    def __init__(self, layers, X, Y, activation_function="relu"):
        self.X = Mat(X)
        self.Y = Mat(Y)
        self.activation_function = activation_function

        self.W = []
        self.B = []

        for i in range(0, len(layers) - 1):
            w = Mat.empty(layers[i], layers[i + 1])
            b = Mat.empty(1, layers[i + 1])
            w.random()
            b.random()
            self.W.append(w)
            self.B.append(b)

    def forward(self, x_batch):
        activation = x_batch if isinstance(x_batch, Mat) else Mat(x_batch)
        if self.activation_function not in ["relu", "sigmoid"]:
            raise NotImplementedError(f"Activation function '{self.activation_function}' is not supported.")

        for i in range(len(self.W)):
            # z is computed for the entire batch at once!
            # self.B[i] is broadcasted to match the batch size of activation
            z = (activation * self.W[i]) + self.B[i]

            if i < len(self.W) - 1:
                if self.activation_function == "relu":
                    activation = z.relu()
                elif self.activation_function == "sigmoid":
                    activation = z.sigmoid()
            else: # last layer
                if self.activation_function == "relu":
                    activation = z
                elif self.activation_function == "sigmoid":
                    activation = z.sigmoid()

        return activation

    def cost(self):
        # Single vectorized call for the entire batch
        pred = self.forward(self.X)
        diff = self.Y - pred
        sq_diff = diff.pow(2)
        return sq_diff.sum() / self.X.rows

    def fit(self, iter=10000, lr=0.01, log_after=1000):
        start = time.time()
        params = [w for rows in self.W for l in rows for w in l] + [
            b for rows in self.B for l in rows for b in l
        ]

        for i in range(iter):
            # zero grad all the params
            for p in params:
                p.grad = 0.0

            # calculate loss and backprop
            loss = self.cost()
            loss.backward()

            # update the params in-place
            for p in params:
                p.update_params(lr)

            if i % log_after == 0:
                print(f"iter = {i} | cost = {loss.data}")

        total_time = time.time() - start
        print(f"Done, Total Time taken: {total_time:.4f}")

    def accuracy(self, X_test, Y_test):
        # We can also vectorize accuracy!
        pred = self.forward(X_test)
        diff_sum = 0.0
        for i in range(len(X_test)):
            diff_sum += abs(Y_test[i][0] - pred[i][0].data)
        return 1.0 - (diff_sum / len(X_test))

    def predict(self, X):
        return self.forward(X)
