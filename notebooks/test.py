from itertools import combinations

n = 10
k = 3
elements = range(n)
combinations_n_k = list(combinations(elements, k))
print(combinations_n_k)
for delete in combinations_n_k:
    inside = []
    for j in range(n):
        if j not in delete:
            inside.append(j)
    print(inside)

print(len(combinations_n_k))