import unittest

from aStarAlgo import aStarAlgorithm


class TestAStar(unittest.TestCase):

    def test_case_1(self):
        graph = [[0,0,0,0],
                 [0,0,1,0],
                 [0,0,0,0]]
        startX, startY = 0, 3
        endX, endY = 1, 0
        path = aStarAlgorithm(startX, startY, endX, endY, graph)
        print(path)

    def test_case_2(self):
        graph = [[0,0,0,0],
                 [0,0,1,0],
                 [0,0,0,0]]
        startX, startY = 0, 0
        endX, endY = 0, 0
        path = aStarAlgorithm(startX, startY, endX, endY, graph)
        print(path)


if __name__ == '__main__':
    unittest.main()