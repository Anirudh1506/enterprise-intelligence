x = list(map(int, input().split()))
actual = list(map(int, input().split()))
weight = list(map(float, input().split()))
learning_rate = float(input())

def predict(x, weight):
    if len(x) != len(weight):
        raise ValueError("Length of input and weight lists must be the same")
    return sum(i * w for i, w in zip(x, weight))

def calculate_loss(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return sum((a - p)**2 for a, p in zip(actual, predicted)) / len(actual)

def calculate_gradient(x, actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return [-2 * (a - p) * i for a, p, i in zip(actual, predicted, x)]

def update_weights(weight, gradient, learning_rate):
    if len(weight) != len(gradient):
        raise ValueError("Length of weight and gradient lists must be the same")
    return [w - learning_rate * g for w, g in zip(weight, gradient)]

predicted = [predict([x_val], weight) for x_val in x]
loss = calculate_loss(actual, predicted)
gradient = calculate_gradient(x, actual, predicted)
new_weights = update_weights(weight, gradient, learning_rate)

print(f"Predicted: {predicted}")
print(f"New Weights: {new_weights}")
print(f"Loss: {loss}")
print(f"Gradient: {gradient}")