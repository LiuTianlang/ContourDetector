#include "MyContourDetector.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <unordered_map>

// 8-neighborhood directions (clockwise): E, NE, N, NW, W, SW, S, SE
const std::vector<cv::Point> dirs = {
	{1, 0}, {1, -1}, {0, -1}, {-1, -1},
	{-1, 0}, {-1, 1}, {0, 1}, {1, 1}
};

// Initialize with a dummy root node
MyContourDetector::MyContourDetector() {
	root = new ContourNode(std::vector<cv::Point>());
}

// Destructor: release all nodes
MyContourDetector::~MyContourDetector() {
	clear();
}

// Set the binary image (CV_8UC1) for processing
void MyContourDetector::setInputImage(const Mat& binary) {
	binImage = binary;
}

// Run the full detection and tree construction pipeline
void MyContourDetector::detect() {
	if (!binImage.data || binImage.channels() != 1)
		return;
	buildContourTree();
}

// Return the root of the contour tree
ContourNode* MyContourDetector::getContourTree() const {
	return root;
}

// Check if a point is within image bounds
bool isPointAvailable(int x, int y, int m, int n) {
	return x >= 0 && x < m && y >= 0 && y < n;
}

// A contour point must be white and have at least one black neighbor in 8-neighborhood
bool isContourPoint(int x, int y, int m, int n, Mat& img) {
	int temp = 0;
	for (int i = 0; i < 8; i++) {
		int newX = x + dirs[i].x, newY = y + dirs[i].y;
		if (!isPointAvailable(newX, newY, m, n)) continue;
		//The boundary must contains 1 and 0
		//Which make the & ->0
		if (!i)
			temp = img.at<uchar>(newY, newX);
		else
			temp &= img.at<uchar>(newY, newX);
	}
	return !temp;
}

// Extract all contour nodes from the binary image (flattened list)
std::vector<ContourNode*> MyContourDetector::getAllContoursFlat() {
	std::vector<ContourNode*> allNodes;

	// Add a synthetic root node to ensure tree structure
	// It is the largest ContourNode
	ContourNode* outMost = new ContourNode({});
	outMost->contour.area = INT_MAX;
	outMost->contour.points.push_back({ INT_MAX, INT_MAX });
	outMost->contour.isClosed = true;
	outMost->parent = outMost;
	allNodes.push_back(outMost);

	std::unordered_map<cv::Point, int, PointHash> allVisitedPoints;
	int n = binImage.rows, m = binImage.cols;

	for (int x = 0; x < m; x++) {
		for (int y = 0; y < n; y++) {
			if (!binImage.at<uchar>(y, x)) continue;
			if (!isContourPoint(x, y, m, n, binImage)) continue;
			if (allVisitedPoints.count({ x, y })) continue;
			ContourNode* node = traceContourFrom(x, y, allVisitedPoints);
			allNodes.push_back(node);
		}
	}
	return allNodes;
}

// Check if a point is valid for tracing: foreground, boundary, and unvisited
bool isValidBoundaryPoint(
	int x, int y, int m, int n,
	std::unordered_map<cv::Point, int, PointHash>& visitedPointsMap,
	std::unordered_map<cv::Point, int, PointHash>& allVisitedPoints,
	Mat& binImage) {

	if (!binImage.at<uchar>(y, x)) return false;
	if (!isContourPoint(x, y, m, n, binImage)) return false;
	if (visitedPointsMap.count({ x, y }) || allVisitedPoints.count({ x, y })) return false;
	return true;
}

// Trace a single contour using 8-neighborhood and direction priority (Moore's algorithm)
ContourNode* MyContourDetector::traceContourFrom(int x, int y,
	std::unordered_map<cv::Point, int, PointHash>& allVisitedPoints) {

	std::vector<cv::Point> visitedPoints;
	std::unordered_map<cv::Point, int, PointHash> visitedMap;

	cv::Point start(x, y), cur = start;
	int dir = 0;
	int rows = binImage.rows, cols = binImage.cols;
	bool isBackToStart = false;

	while (true) {
		visitedPoints.push_back(cur);
		visitedMap[cur] = 1;
		allVisitedPoints[cur] = 1;

		bool foundNext = false;
		for (int i = 0; i < 8; ++i) {
			//Core of Moore's algorithm
			//This time dir, next time (dir+6+i)%8, avoiding go back
			int checkDir = (dir + 6 + i) % 8;
			cv::Point next = cur + dirs[checkDir];

			if (!isPointAvailable(next.x, next.y, cols, rows)) continue;
			if (!isValidBoundaryPoint(next.x, next.y, cols, rows, visitedMap, allVisitedPoints, binImage)) continue;

			// Handle diagonal skips by marking midpoints
			// e.g., If you pick north east, the point of north maybe ignore by you if it is boundary of contours
			// Thus, it must be added to hashmap
			// Or it will be detected again as a small contour
			if (std::abs(dirs[checkDir].x) == 1 && std::abs(dirs[checkDir].y) == 1) {
				cv::Point mid1(cur.x + dirs[checkDir].x, cur.y);
				cv::Point mid2(cur.x, cur.y + dirs[checkDir].y);
				if (binImage.at<uchar>(mid1) > 0)
					visitedMap[mid1] = allVisitedPoints[mid1] = 1;
				if (binImage.at<uchar>(mid2) > 0)
					visitedMap[mid2] = allVisitedPoints[mid2] = 1;
			}
			//Break when it goes back to start
			if (next == start && visitedPoints.size() > 5) {
				isBackToStart = true;
				break;
			}

			dir = checkDir;
			cur = next;
			foundNext = true;
			break;
		}

		if (isBackToStart || !foundNext) break;
	}
	return new ContourNode(visitedPoints);
}

// Build a tree of contours by containment
// The parent of a node is the 1st contour which has larger area than the node
void MyContourDetector::buildContourTree() {
	std::vector<ContourNode*> allNodes = getAllContoursFlat();
	//Sort all ContourNodes by ascending area
	std::sort(allNodes.begin(), allNodes.end(), [](const ContourNode* a, const ContourNode* b) {
		return a->contour.area < b->contour.area;
	});

	int n = allNodes.size();
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			//1st larger area found
			if (allNodes[j]->doesContainContourNode(allNodes[i])) {
				allNodes[i]->setParent(allNodes[j]);
				allNodes[j]->addChildren(allNodes[i]);
				break;
			}
		}
	}
	root = allNodes.back();
}

// BFS traversal of the contour tree (prints node count by level)
void bfsPrint(ContourNode* node) {
	std::queue<ContourNode*> q;
	if (!node) {
		std::cout << "No contours found!" << std::endl;
		return;
	}
	q.push(node);
	int layer = 0;
	std::unordered_map<int, int> layerCount;

	while (!q.empty()) {
		std::vector<ContourNode*> nextLayer;
		while (!q.empty()) {
			ContourNode* front = q.front(); q.pop();
			layerCount[layer]++;
			for (auto child : front->children)
				if (child) nextLayer.push_back(child);
		}
		layer++;
		for (auto node : nextLayer)
			q.push(node);
	}
	for (const auto& pair : layerCount)
		std::cout << "Layer " << pair.first << ": " << pair.second << std::endl;
}

// Print the contour tree structure
void MyContourDetector::printContourTree() {
	bfsPrint(root);
}

// Release the entire tree structure
void MyContourDetector::clear() {
	delete root;
	root = nullptr;
}

// Optional: recursive deletion (if not using destructors in ContourNode)
void MyContourDetector::freeTree(ContourNode* node) {
	// Deprecated in favor of RAII
}
