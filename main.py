import seaborn as sn
import numpy as np
import random
import pandas as pd
import math
import matplotlib.pyplot as plt

# prepare data

x_points = [random.uniform(0, 100) for _ in range(100)]
y_points = [random.uniform(0, 100) for _ in range(100)]

city_map = pd.DataFrame({'x': pd.Series(x_points)
                        , 'y': pd.Series(y_points)})

# what does the path look like initially?

plt.plot(city_map.x, city_map.y)

# generate initial path
sequence = list(range(0, 100))
random.shuffle(sequence)

# get distance
def get_distance(seq, city_map):
    distance = 0
    for i in range(len(seq) - 1):
        point_1 = city_map.loc[seq[i]]
        point_2 = city_map.loc[seq[i + 1]]
        distance += math.sqrt((point_2.x - point_1.x)**2 + (point_2.y - point_1.y)**2)
    return distance

get_distance(seq=sequence, city_map=city_map)


outputs = list()
sequence = list(range(0, 100))
for i in range(100):
    temp_sequence = sequence.copy()
    random.shuffle(temp_sequence)
    distance = get_distance(temp_sequence, city_map)
    outputs.append((temp_sequence, distance))

outputs.sort(key=lambda x: x[1])


def cross_over(seq1, seq2):
    output = list()
    cut_off_points = [random.randint(1, len(seq1)) for _ in range(20)]
    for point in cut_off_points:
        new_seq1 = seq1[0:point] + seq2[point:]
        new_seq2 = seq2[0:point] + seq1[point:]
        output.append(new_seq1)
        output.append(new_seq2)
    return output


def fix_seq(seq):
    seq_copy = seq.copy()
    missing = [x for x in range(len(seq)) if x not in seq]
    random.shuffle(missing)
    temp_sequence = list()
    for _ in range(len(seq)):
        temp = seq_copy.pop(0)
        if temp in temp_sequence:
            temp_sequence.append(missing.pop())
        else:
            temp_sequence.append(temp)
    return temp_sequence


def mutate(seq):
    x, y = random.choice(range(len(seq))), random.choice(range(len(seq)))
    seq[x], seq[y] = seq[y], seq[x]
    return seq


parents = [outputs[0][0], outputs[1][0]]
for i in range(1000):
    print('iteration {}'.format(i))
    parent_1, parent_2 = parents[0], parents[1]
    new_family = cross_over(parent_1, parent_2)
    new_family_fixed = [fix_seq(seq) for seq in new_family]
    new_family_mutated = [mutate(seq) for seq in new_family_fixed]
    new_family_mutated.append(parent_1)
    new_family_mutated.append(parent_2)
    new_family_evaluated = [(seq, get_distance(seq, city_map)) for seq in new_family_mutated]
    new_family_evaluated.sort(key=lambda x: x[1])
    parents = [new_family_evaluated[0][0], new_family_evaluated[1][0]]
    print('Current best: {}'.format(new_family_evaluated[0][1]))
    best_seq = new_family_evaluated[0][0]

print('done')
print(new_family_evaluated[0][1])

best_city_map = city_map.copy()
best_city_map = best_city_map.reindex(best_seq)
plt.plot(best_city_map.x, best_city_map.y)
best_city_map.shape


