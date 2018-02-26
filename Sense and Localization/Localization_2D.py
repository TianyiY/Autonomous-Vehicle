import numpy as np

def localize(colors, measurements, motions, sensor_right, p_move):
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]

    def sense(p, colors, M, sensor_right):
        q = []
        for i in range(len(p)):
            for j in range(len(p[i])):
                hit = (M == colors[i][j])
                q.append(p[i][j] * (hit * sensor_right + (1 - hit) * (1 - sensor_right)))
        s = np.sum(q)
        q = np.asarray(q).reshape((len(p), len(p[0])))
        for i in range(len(q)):
            for j in range(len(q[0])):
                q[i][j] = q[i][j] / s
        return q

    def move(p, U, p_move):
        q = []
        for i in range(len(p)):
            for j in range(len(p[i])):
                q.append(
                    p[(i-U[0])%len(p)][(j - U[1]) % len(p[i])] * p_move + p[i][j]*(1-p_move))
        q=np.asarray(q).reshape((len(p), len(p[0])))
        return q

    for i in range (len(measurements)):
        p = move(p, motions[i], p_move)
        p=sense(p, colors, measurements[i], sensor_right)
    return p


def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x), r)) + ']' for r in p]
    print ('[' + ',\n '.join(rows) + ']')


colors = [['R', 'G', 'G', 'R', 'R'],
          ['R', 'R', 'G', 'R', 'R'],
          ['R', 'R', 'G', 'G', 'R'],
          ['R', 'R', 'R', 'R', 'R']]
measurements = ['G', 'G', 'G', 'G', 'G']
motions = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1]]
p = localize(colors, measurements, motions, sensor_right=0.7, p_move=0.8)
show(p)  # displays

# output should be
# [[0.01105, 0.02464, 0.06799, 0.04472, 0.02465],
#  [0.00715, 0.01017, 0.08696, 0.07988, 0.00935],
#  [0.00739, 0.00894, 0.11272, 0.35350, 0.04065],
#  [0.00910, 0.00715, 0.01434, 0.04313, 0.03642]]
# (within a tolerance of +/- 0.001 for each entry)