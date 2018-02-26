# Localization function in Python

n=5
p_uniform=[1.0/n for i in range(n)]
# print(p_uniform)
p=[0, 1, 0, 0, 0]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1
motions = [1,1]

def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
    q = []
    for i in range(len(p)):
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)
    return q

for i in range(len(motions)):
    p = sense(p, measurements[i])
    p = move(p, motions[i])
print('initial p=[0, 1, 0, 0, 0]')
print (p)

print('************************************')

for i in range(len(motions)):
    p_uniform = sense(p_uniform, measurements[i])
    p_uniform = move(p_uniform, motions[i])
print('initial p=[0.2, 0.2, 0.2, 0.2, 0.2]')
print (p_uniform)