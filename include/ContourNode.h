#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
/**
 * @brief Struct representing a geometric contour.
 * 
 * This structure stores the key properties of a 2D contour, including its point sequence,
 * area, perimeter, and whether it forms a closed loop.
 */
struct Contour {
	std::vector<cv::Point> points;

	double area = -1.0;
	double perimeter = -1.0;
	bool isClosed = false;

	Contour(const std::vector<cv::Point>& pts) :points(pts) { computeProperties(); };
	bool isEmpty();
	void computeProperties();  // ������ܳ�
	bool doesContainContour(Contour& another);
};

/**
 * @brief Struct representing a contour node.
 * 
 * 
 * This structure stores the properties of a contour node, which represents the relationship of 
 * different contours.
 * 
 * A node has a parent(nearest outside), some children (inside) and a geometrical contour
 */
struct ContourNode {
	Contour contour;
	std::vector<ContourNode*> children;
	ContourNode* parent;

	ContourNode(std::vector<cv::Point> inputContourPoints) :contour(inputContourPoints), parent(this), children({}) {};
	~ContourNode() {
		for (auto child : children)
			delete child;
	}
	bool doesContainContourNode(ContourNode* another);
	void setParent(ContourNode* another);
	void addChildren(ContourNode* another);
};
/**
 * @brief Calculate cross-multiplication of two 2D vector
 * @param a First 2D vector const cv::Point
 * @param b Second 2D vector const cv::Point
 * @return The scalar result of the cross production (a.x * b.y - a.y * b.x) int
 */
int cross(const cv::Point& a, const cv::Point& b);
/**
 * @brief Calculate Winding number of a point with contour points of a polygon. 
 * Widing number is used to identify whether point in a polygon
 * @param polygon Contour points of polygon const std::vector<cv::Point>
 * @param b Point const cv::Point
 * @return The result of Winding number int
 */
int windingNumber(const std::vector<cv::Point>& polygon, const cv::Point& point);
/**
 * @brief Identify whether a point is in a polygon
 * @param point A point cv::Point
 * @param contour A Contour object const Contour&
 * @return Whether the point is in a Contour bool
 */
bool isPointInContour(cv::Point point, const Contour& contour);
/**
 * @brief Calculate the area of a contour
 * @param points The vetor of points on contour const std::vector<cv::Point>&
 * @return Area of the contour double
 */
double computeArea(const std::vector<cv::Point>& points);
/**
 * @brief Identify is the contour closed
 * @param points The vetor of points on contour const std::vector<cv::Point>&
 * @return Whether close or not bool
 */
bool isContourClosed(const std::vector<cv::Point>& points);
/**
 * @brief Calculate contour perimeter
 * @param points The vetor of points on contour const std::vector<cv::Point>&
 * @param isClosed Whether the contour is closed const bool
 * @return Perimeter of the contour double
 */
double computePerimeter(const std::vector<cv::Point>& points, const bool isClosed);