from math import sqrt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


p1 = Point(1, 1)
p2 = Point(1, 0)
p3 = Point(0, 2)