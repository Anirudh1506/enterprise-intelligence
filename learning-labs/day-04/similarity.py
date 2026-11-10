import math

vector_a = [1, 2]
vector_b = [2, 4]
vector_c = [-2, 1]
vector_d = [1, 0]

def cosine_similarity(vector_a, vector_b):
    # Validate dimensions
    # Calculate dot product
    # Calculate magnitude of both vectors
    # Handle zero vectors
    # Return cosine similarity
    
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be of the same length")
    
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.hypot(*vector_a)
    magnitude_b = math.hypot(*vector_b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Vectors must not be zero vectors")
    
    return dot_product / (magnitude_a * magnitude_b)

print(cosine_similarity(vector_a, vector_b))
print(cosine_similarity(vector_a, vector_c))
print(cosine_similarity(vector_a, vector_d))