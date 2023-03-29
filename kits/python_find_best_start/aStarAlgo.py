import logging
import math

class Node:
    def __init__(self, x, y, value):
        self.id = str(x) + "-" + str(y)
        self.x = x
        self.y = y
        self.value = value
        self.distanceFromStart = float("inf")
        self.estimatedDistanceToEnd = float("inf")
        self.cameFrom = None

def aStarAlgorithm(startX, startY, endX, endY, graph, power_cost_map = []):
    # Write your code here.
    if startX == endX and startY == endY:
        return [], 0

    nodes = initializeNodes(graph)  # return an array same dim as graph 

    startNode = nodes[startX][startY] 
    endNode = nodes[endX][endY]

    startNode.distanceFromStart = 0
    startNode.estimatedDistanceToEnd = calculateHeuristicCost(startNode, endNode, power_cost_map)

    nodesToVisit = MinHeap([startNode])

    while not nodesToVisit.isEmpty():
        currentMinDistanceNode = nodesToVisit.remove()

        if currentMinDistanceNode == endNode:
            break

        neighbors = getNeighboringNodes(currentMinDistanceNode, nodes)
        for neighbor in neighbors:
            if neighbor.value == 1: 
                continue

            tentativeDistanceToNeighbor = currentMinDistanceNode.distanceFromStart + math.floor(20+power_cost_map[neighbor.x][neighbor.y])  # 1 update made here for weighted A*star

            if tentativeDistanceToNeighbor >= neighbor.distanceFromStart:
                continue

            neighbor.cameFrom = currentMinDistanceNode
            neighbor.distanceFromStart = tentativeDistanceToNeighbor
            neighbor.estimatedDistanceToEnd = tentativeDistanceToNeighbor + calculateHeuristicCost(
                neighbor, endNode, power_cost_map
            )

            if not nodesToVisit.containsNode(neighbor):
                nodesToVisit.insert(neighbor)
            else:
                nodesToVisit.update(neighbor)
    
    final_path, total_cost = reconstructPath(endNode, power_cost_map)

    return final_path, total_cost

def initializeNodes(graph):
    nodes = []
    for i, xCol in enumerate(graph):
        nodes.append([])
        for j, value in enumerate(xCol):
            nodes[i].append(Node(i, j, value))   ## value shows which cell is blocked.
    return nodes

def calculateHeuristicCost(currentNode, endNode, power_cost_map):
    currentX = currentNode.x
    currentY = currentNode.y
    endX = endNode.x
    endY = endNode.y
    total_cost = 0
    # using pixel average cost as heuristic.
    minX, maxX, minY, maxY = min(currentX, endX), max(currentX, endX), min(currentY, endY), max(currentY, endY)
    total_tiles = (maxX + 1 - minX) * (maxY + 1 - minY)

    for x in range(minX, maxX + 1):
        for y in range(minY, maxY + 1):
            total_cost += power_cost_map[x][y]

    return (total_cost / total_tiles) # * calculateManhattanDistance(currentNode, endNode)

def calculateManhattanDistance(currentNode, endNode):
    currentX = currentNode.x
    currentY = currentNode.y
    endX = endNode.x
    endY = endNode.y

    return abs(currentX - endX) + abs(currentY- endY)

def getNeighboringNodes(node, nodes):
    neighbors = []

    numXs = len(nodes)
    numYs = len(nodes[0])

    x = node.x
    y = node.y

    # Go Right
    if x < numXs - 1:
        neighbors.append(nodes[x+1][y])
    # Go Left
    if x > 0:
        neighbors.append(nodes[x-1][y])
    # Go Down
    if y < numYs - 1:
        neighbors.append(nodes[x][y+1])
    # Go Up
    if y > 0:
        neighbors.append(nodes[x][y-1])

    return neighbors

def reconstructPath(endNode, power_cost_map):
    if not endNode.cameFrom:
        return []

    currentNode = endNode
    path = []

    total_cost = 0
    while currentNode is not None:
        path.append([currentNode.x, currentNode.y])
        ## if it's the first step, don't add 
        if currentNode.cameFrom is not None:
            total_cost += power_cost_map[currentNode.x][currentNode.y]
        currentNode = currentNode.cameFrom
        
    return path[::-1], total_cost
    
class MinHeap:
	def __init__(self, array):
		self.nodePositionsInHeap = {node.id: idx for idx, node in enumerate(array)}
		self.heap = self.buildHeap(array)
		
	def isEmpty(self):
		return len(self.heap) == 0
	
	def buildHeap(self, array):
		firstParentIdx = (len(array) - 2) // 2
		for currentIdx in reversed(range(firstParentIdx + 1)):
			self.siftDown(currentIdx, len(array) - 1, array)
		return array
	
	def siftDown(self, currentIdx, endIdx, heap):
		childOneIdx = currentIdx * 2 + 1
		while childOneIdx <= endIdx:
			childTwoIdx = currentIdx * 2 + 2 if currentIdx *2 + 2 <= endIdx else -1
			if (
				childTwoIdx != -1
				and heap[childTwoIdx].estimatedDistanceToEnd < heap[childOneIdx].estimatedDistanceToEnd
			) :
				idxToSwap = childTwoIdx
			else:
				idxToSwap = childOneIdx
			if heap[idxToSwap].estimatedDistanceToEnd < heap[currentIdx].estimatedDistanceToEnd:
				self.swap(currentIdx, idxToSwap, heap)
				currentIdx = idxToSwap
				childOneIdx = currentIdx *2 + 1
			else:
				return
	
	def siftUp(self, currentIdx, heap):
		parentIdx = (currentIdx - 1) // 2
		while currentIdx > 0 and heap[currentIdx].estimatedDistanceToEnd < heap[parentIdx].estimatedDistanceToEnd:
			self.swap(currentIdx, parentIdx, heap)
			currentIdx = parentIdx
			parentIdx = (currentIdx - 1)//2
			
	def remove(self):
		if self.isEmpty():
			return 
		self.swap(0, len(self.heap)-1, self.heap)
		node  = self.heap.pop()
		del self.nodePositionsInHeap[node.id]
		self.siftDown(0, len(self.heap) -1, self.heap)
		return node
		
	def insert(self, node):
		self.heap.append(node)
		self.nodePositionsInHeap[node.id] = len(self.heap) - 1
		self.siftUp(len(self.heap) - 1, self.heap)
		
	def swap(self, i, j, heap):
		self.nodePositionsInHeap[heap[i].id] = j
		self.nodePositionsInHeap[heap[j].id] = i
		heap[i], heap[j] = heap[j], heap[i]
		
	def containsNode(self, node):
		return node.id in self.nodePositionsInHeap

	def update(self, node):
		self.siftUp(self.nodePositionsInHeap[node.id], self.heap)