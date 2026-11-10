inputs = list(map(int, input().split()))
weights = [list(map(float, input().split())) for _ in range(len(inputs))]
biases = list(map(float, input().split()))

def neuron_output(inputs, weights, biases):
    if len(inputs) != len(weights):
        raise ValueError(
            "Number of inputs must match number of weights"
        )
    return sum(i * w for i, w in zip(inputs, weights)) + biases

"""
[1,2,3]
[[0.3, 0.6], [-0.2, 0.4]]
[-0.5, 0.1]

(1,0.3), (2,0.5), (3,0.1)
"""

#Activation function (ReLU)
def relu(x):
    return max(0, x)

#Activation function (Sigmoid)
def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))

#dense layer = neuron_output(inputs, weights, bias)
def dense_layer(inputs, weights, bias):
    return neuron_output(inputs, weights, bias)


if(len(biases) != len(weights)):
    raise ValueError("Number of biases must match number of weight sets")

for weight, bias in zip(weights, biases):
    output = dense_layer(inputs, weight, bias)
    activated_output_relu = relu(output)
    activated_output_sigmoid = sigmoid(output)
    print(f"Output: {output}, ReLU Activated: {activated_output_relu}, Sigmoid Activated: {activated_output_sigmoid}")