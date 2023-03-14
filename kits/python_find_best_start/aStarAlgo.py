import logging

class Node:
    def __init__(self, x, y, value):
        self.id = str(x) + "-" + str(y)
        self.x = x
        self.y = y
        self.value = value
        self.distanceFromStart = float("inf")
        self.estimatedDistanceToEnd = float("inf")
        self.cameFrom = None

def aStarAlgorithm(startX, startY, endX, endY, graph):
    # Write your code here.
    nodes = initializeNodes(graph)  # return an array same dim as graph 

    startNode = nodes[startX][startY]  ## TODO: notes from Mar8 th, very likely this part is flipped here.
    endNode = nodes[endX][endY]

    startNode.distanceFromStart = 0
    startNode.estimatedDistanceToEnd = calculateManhattanDistance(startNode, endNode)

    nodesToVisit = MinHeap([startNode])

    while not nodesToVisit.isEmpty():
        currentMinDistanceNode = nodesToVisit.remove()

        if currentMinDistanceNode == endNode:
            break

        neighbors = getNeighboringNodes(currentMinDistanceNode, nodes)
        for neighbor in neighbors:
            if neighbor.value == 1: 
                continue

            tentativeDistanceToNeighbor = currentMinDistanceNode.distanceFromStart + 1

            if tentativeDistanceToNeighbor >= neighbor.distanceFromStart:
                continue

            neighbor.cameFrom = currentMinDistanceNode
            neighbor.distanceFromStart = tentativeDistanceToNeighbor
            neighbor.estimatedDistanceToEnd = tentativeDistanceToNeighbor + calculateManhattanDistance(
                neighbor, endNode
            )

            if not nodesToVisit.containsNode(neighbor):
                nodesToVisit.insert(neighbor)
            else:
                nodesToVisit.update(neighbor)

    return reconstructPath(endNode)

def initializeNodes(graph):
    nodes = []
    for i, xCol in enumerate(graph):
        nodes.append([])
        for j, value in enumerate(xCol):
            nodes[i].append(Node(i, j, value))   ## value shows which cell is blocked.
    return nodes

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

def reconstructPath(endNode):
    if not endNode.cameFrom:
        return []

    currentNode = endNode
    path = []

    while currentNode is not None:
        path.append([currentNode.x, currentNode.y])
        currentNode = currentNode.cameFrom

    return path[::-1]
    
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