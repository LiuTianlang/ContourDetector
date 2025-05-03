#include "ContourNode.h"

// Check if the contour has no points.
bool Contour::isEmpty()
{
	return points.empty();
}

// Compute area, closure status, and perimeter of the current contour.
// All properties are cached for efficiency.
// Assumes points are already populated.
void Contour::computeProperties()
{
	area = computeArea(points);
	isClosed = isContourClosed(points);
	perimeter = computePerimeter(points, isClosed);
}

// Compute the 2D cross product of vectors a and b.
// Equivalent to a.x * b.y - a.y * b.x.
// Used in geometric algorithms like point-in-polygon and convex hull.
int cross(const cv::Point& a, const cv::Point& b) {
	return a.x * b.y - a.y * b.x;
}

// Compute the winding number of a point relative to a polygon.
// A non-zero winding number means the point lies inside the polygon.
// This method is robust against concave and self-intersecting polygons.
int windingNumber(const std::vector<cv::Point>& polygon, const cv::Point& point) {
	int wn = 0;

	for (size_t i = 0; i < polygon.size(); ++i) {
		const cv::Point& a = polygon[i];
		const cv::Point& b = polygon[(i + 1) % polygon.size()];

		if (a.y <= point.y) {
			if (b.y > point.y && cross(b - a, point - a) > 0)
				++wn;  // upward crossing
		} else {
			if (b.y <= point.y && cross(b - a, point - a) < 0)
				--wn;  // downward crossing
		}
	}
	return wn;
}

// Check if a point lies inside a given contour using winding number method.
// The contour is treated as a polygon; closure is assumed if called here.
bool isPointInContour(cv::Point point, const Contour& contour)
{
	return windingNumber(contour.points, point) != 0;
}

// Determine whether the current contour fully contains another contour.
// If any point of 'another' lies outside this contour, return false.
// Special case: if this contour is a synthetic root (area = INT_MAX), always contains.
bool Contour::doesContainContour(Contour& another)
{
	if (isEmpty()) return false;
	if (!isClosed) return false;

	if (this->points.size() == 1 && this->points[0] == cv::Point(INT_MAX, INT_MAX))
		return true;  // synthetic root node

	for (auto point : another.points)
	{
		if (!isPointInContour(point, *this))
			return false;
	}
	return true;
}

// Check if this contour node geometrically contains another node's contour.
bool ContourNode::doesContainContourNode(ContourNode* another)
{
	return contour.doesContainContour(another->contour);
}

// Set the parent pointer of the node.
void ContourNode::setParent(ContourNode* another)
{
	parent = another;
}

// Add a child node and update its parent reference.
void ContourNode::addChildren(ContourNode* another)
{
	children.push_back(another);
	another->setParent(this);
}

// Compute the area of a simple polygon using the Shoelace formula.
// Formula: A = 0.5 * |∑(x_i * y_{i+1} - x_{i+1} * y_i)|
// 
// Assumes the polygon is:
// - simple (non-intersecting),
// - vertex-ordered (either clockwise or counterclockwise).
// 
// This method avoids triangulation and works efficiently with integer coordinates.
double computeArea(const std::vector<cv::Point>& points)
{
	double area = 0;
	int n = points.size();
	for (int i = 0; i < n; i++)
	{
		area += points[i].x * 1.0 * points[(i + 1) % n].y
		      - points[(i + 1) % n].x * 1.0 * points[i].y;
	}
	area = 0.5 * std::abs(area);
	return area;
}

// Check if a contour is closed by comparing first and last point.
// Allows 8-neighborhood connection (dx <= 1 && dy <= 1) but not identical point.
bool isContourClosed(const std::vector<cv::Point>& points)
{
	if (points.size() < 2) return false;

	cv::Point a = points.front();
	cv::Point b = points.back();

	int dx = std::abs(a.x - b.x);
	int dy = std::abs(a.y - b.y);

	return (dx <= 1 && dy <= 1) && !(dx == 0 && dy == 0);
}

// Compute the total length of the contour's edges.
// If the contour is closed, the last point connects back to the first.
double computePerimeter(const std::vector<cv::Point>& points, const bool isClosed)
{
	int n = points.size();
	double perimeter = 0;
	for (int i = 0; i < n - 1; i++)
		perimeter += norm(points[i + 1] - points[i]);

	if (isClosed)
		perimeter += norm(points.front() - points.back());

	return perimeter;
}
