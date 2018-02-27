import numpy as np

measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
initial_xy = [4., 12.]

dt = 0.1

x = np.array([[initial_xy[0]], [initial_xy[1]], [0.], [0.]])  # initial state (location and velocity)
u = np.array([[0.], [0.], [0.], [0.]])  # no external motion

# initial uncertainty: 0 for positions x and y, 1000 for the two velocities
P = np.array([[0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1000.0, 0.0],
              [0.0, 0.0, 0.0, 1000.0]])

# next state function: generalize the 2d version to 4d
F = np.array([[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

# measurement function: reflect the fact that we observe x and y but not the two velocities
H = np.array([[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]])

# measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
R = np.array([[0.1, 0.0],
             [0.0, 0.1]])

# 4d identity matrix
I = np.array([[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]])

class matrix:
    def __init__(self, value):
        self.value = value
        self.dimx = len(value)
        self.dimy = len(value[0])
        if value == [[]]:
            self.dimx = 0

    def transpose(self):
        # compute transpose
        res = matrix([[]])
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res


def Kalman_filter1(x, P, u, F, H, R, I, measurements):
    for i in range(len(measurements)):
        # measurement update
        z = np.asarray(measurements[i])
        y = z.T - H.dot(x)
        s = H.dot(P).dot(H.T) + R
        k = P.dot(H.T).dot(np.linalg.inv(s))
        x = x + k.dot(y)
        P = (I - k.dot(H)).dot(P)
        # prediction
        x=F.dot(x)+u
        P=F.dot(P).dot(F.T)
    return x, P

def Kalman_filter2(x, P, u, F, H, R, I, measurements):
    for i in range(len(measurements)):
        # prediction
        x = F.dot(x) + u
        P = F.dot(P).dot(F.T)
        # measurement update
        z = np.asarray(measurements[i])
        y = np.mat(z).T - H.dot(x)
        s = H.dot(P).dot(H.T) + R
        k = P.dot(H.T).dot(np.mat(s).I)
        x = x + k.dot(y)
        P = (I - k.dot(H)).dot(P)
    return x, P

x1, P1=Kalman_filter1(x, P, u, F, H, R, I, measurements)
print ('x1= ', x1)
print ()
print ('P1= ', P1)

print('*******************************************************')

x2, P2=Kalman_filter2(x, P, u, F, H, R, I, measurements)
print ('x2= ', x2)
print ()
print ('P2= ', P2)

# x2 =
# [9.999340731787717]
# [0.001318536424568617]
# [9.998901219646193]
# [-19.997802439292386]
#
# P2 =
# [0.03955609273706198, 0.0, 0.06592682122843721, 0.0]
# [0.0, 0.03955609273706198, 0.0, 0.06592682122843721]
# [0.06592682122843718, 0.0, 0.10987803538073201, 0.0]
# [0.0, 0.06592682122843718, 0.0, 0.10987803538073201]