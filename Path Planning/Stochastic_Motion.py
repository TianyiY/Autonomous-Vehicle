def stochastic_value(delta, delta_name, grid, goal, cost_step, collision_cost, success_prob):
    failure_prob = (1.0 - success_prob) / 2.0  # Probability(stepping left) = prob(stepping right) = failure_prob
    value = [[collision_cost for col in range(len(grid[0]))] for row in range(len(grid))]
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    change = True
    while change:
        change = False
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0
                        policy[x][y] = '*'
                        change = True
                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        v2=cost_step
                        for i in range(-1, 2):
                            a2=(a+i)%len(delta)
                            x2 = x + delta[a2][0]
                            y2 = y + delta[a2][1]
                            if i==0:
                                p2=success_prob
                            else:
                                p2=failure_prob
                            if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                                v2+= value[x2][y2] * p2
                            else:
                                v2 += collision_cost * p2
                        if v2 < value[x][y]:
                            change = True
                            value[x][y] = v2
                            policy[x][y]=delta_name[a]
    return value, policy

# test
delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']  # Use these when creating policy grid.

grid = [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]]

goal = [0, len(grid[0]) - 1]  # Goal is in top right corner
cost_step = 1
collision_cost = 100
success_prob = 0.5

value, policy = stochastic_value(delta, delta_name, grid, goal, cost_step, collision_cost, success_prob)

for row in value:
    print('values')
    print (row)
for row in policy:
    print('policy')
    print (row)

# Expected outputs:
# [57.9029, 40.2784, 26.0665,  0.0000]
# [47.0547, 36.5722, 29.9937, 27.2698]
# [53.1715, 42.0228, 37.7755, 45.0916]
# [77.5858, 100.00, 100.00, 73.5458]

# ['>', 'v', 'v', '*']
# ['>', '>', '^', '<']
# ['>', '^', '^', '<']
# ['^', ' ', ' ', '^']