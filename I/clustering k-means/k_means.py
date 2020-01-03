from math import sqrt


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Cluster(object):
    def __init__(self, centroid):
        self.centroid = Point(centroid.x, centroid.y)
        self.points = []

    def add(self, point):
        self.points.append(point)

    def update(self):
        x_cor = y_cor = 0
        for p in self.points:
            x_cor += p.x
            y_cor += p.y
        self.centroid.x = x_cor/len(self.points)
        self.centroid.y = y_cor/len(self.points)
        self.points = []

    def __str__(self):
        result = ''
        for c in self.points:
            result = result + str(c) + '\n'
        return result


A = Point(1, 1)
B = Point(2, 1)
C = Point(4, 3)
D = Point(6, 5)
E = Point(3, 5)
C1 = Cluster(A)
C2 = Cluster(B)
data = [A, B, C, D]
clusters = [C1, C2]
for i in range(2):
    for p in data:
        if p.distance_to(C1.centroid) < p.distance_to(C2.centroid):
            C1.add(p)
        else:
            C2.add(p)
    print("C1 is",C1)
    C1.update()
    C2.update()
    print("new A is:", C1.centroid)
    print("new B is:", C2.centroid)
