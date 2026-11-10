import math

actual = list(map(int, input().split()))
predicted = list(map(float, input().split()))

def binary_cross_entropy(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return -sum(a * math.log(p) + (1 - a) * math.log(1 - p) for a, p in zip(actual, predicted)) / len(actual)

def categorical_cross_entropy(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return -sum(a * math.log(p) for a, p in zip(actual, predicted)) / len(actual)

print(binary_cross_entropy(actual, predicted))
print(categorical_cross_entropy(actual, predicted))