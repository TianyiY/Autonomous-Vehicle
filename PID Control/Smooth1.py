# Smoothing should be implemented by iteratively updating each entry in newpath until some desired level of accuracy is reached.
# The update should be done according to the gradient descent equations

def printpaths(path, newpath):
    for old, new in zip(path, newpath):
        print ('[' + ', '.join('%.3f' % x for x in old) + '] -> [' + ', '.join('%.3f' % x for x in new) + ']')

path = [[0, 0],
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [4, 3],
        [4, 4]]


def smooth(path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
    # Make a deep copy of path into newpath
    newpath = [[0 for row in range(len(path[0]))] for col in range(len(path))]
    for i in range(len(path)):
        for j in range(len(path[0])):
            newpath[i][j]=path[i][j]

    change=tolerance  # initialize change

    while change>=tolerance:
        change=0.0
        for i in range(1, len(path)-1):       # exclude the first and last poionts
            for j in range(len(path[0])):
                temp=newpath[i][j]
                newpath[i][j]+=weight_data*(path[i][j]-newpath[i][j])
                newpath[i][j]+=weight_smooth*(newpath[i-1][j]+newpath[i+1][j]-2.0*newpath[i][j])
                change+=abs(temp-newpath[i][j])

    return newpath

printpaths(path, smooth(path))
print ('*********************')
printpaths(path, smooth(path, weight_data=0, weight_smooth=0.1, tolerance=0.000001))
print ('*********************')
printpaths(path, smooth(path, weight_data=0.5, weight_smooth=0, tolerance=0.000001))
