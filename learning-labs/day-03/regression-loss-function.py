actual = list(map(int, input().split()))
predicted = list(map(float, input().split()))

def mean_squared_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)

def mean_absolute_error(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted lists must be the same")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


print(mean_squared_error(actual, predicted))
print(mean_absolute_error(actual, predicted))