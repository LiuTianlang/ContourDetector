#pragma once  
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>

#include "ContourNode.h"
using namespace cv;
/**
 * @brief Hash functor for cv::Point.
 *
 * This structure enables the use of cv::Point as a key in unordered containers
 * such as std::unordered_map or std::unordered_set by providing a custom hash function.
 *
 * The hash value is computed by combining the hashes of the x and y coordinates
 * using XOR and bit shifting to reduce collisions.
 */
struct PointHash {
	/**
     * @brief Computes the hash value for a cv::Point.
     * @param p The cv::Point to hash.
     * @return A size_t hash value combining p.x and p.y.
     */
	size_t operator()(const cv::Point& p) const {
		return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
	}
};

/**
 * @brief A class for detecting and organizing image contours into a hierarchical tree structure.
 *
 * This class processes a binary image, extracts all contours using 8-neighborhood tracing,
 * and builds a parent-child tree of contours based on containment. Each contour is wrapped in a ContourNode.
 */
class MyContourDetector
{
public:
    /**
     * @brief Constructor. Initializes the contour detector.
     */
    MyContourDetector();

    /**
     * @brief Destructor. Automatically releases all dynamically allocated contour nodes.
     */
    ~MyContourDetector();

    /**
     * @brief Set the input binary image to be processed.
     * @param binary A single-channel binary image (CV_8UC1) where white pixels are considered as foreground.
     */
    void setInputImage(const cv::Mat& binary);

    /**
     * @brief Perform contour detection and construct the contour tree.
     *
     * This function should be called after setInputImage(). It traces contours and builds
     * a hierarchy based on spatial containment.
     */
    void detect();

    /**
     * @brief Get the root node of the contour tree.
     * @return Pointer to the root ContourNode. May be nullptr if detect() hasn't been called.
     */
    ContourNode* getContourTree() const;

    /**
     * @brief Get all detected contours in a flat vector.
     * @return A list of all ContourNode pointers in no particular hierarchical order.
     */
    std::vector<ContourNode*> getAllContoursFlat();

    /**
     * @brief Print or visualize the current contour tree. (Optional: implementation-dependent)
     */
    void printContourTree();

    /**
     * @brief Clear all internal data and free memory.
     *
     * Recursively deletes all ContourNode objects from the tree and resets the root pointer.
     */
    void clear();

private:
    cv::Mat binImage;         ///< Binary input image (CV_8UC1)
    ContourNode* root = nullptr;  ///< Root of the contour tree

    /**
     * @brief Build the hierarchical tree from flat contours based on spatial containment.
     */
    void buildContourTree();

    /**
     * @brief Trace a single contour from a starting point using 8-neighborhood logic.
     * @param x Starting x-coordinate.
     * @param y Starting y-coordinate.
     * @param allVisitedPoints A hash map for recording already visited points globally.
     * @return A pointer to the newly created ContourNode representing the traced contour.
     */
    ContourNode* traceContourFrom(int x, int y, std::unordered_map<cv::Point, int, PointHash>& allVisitedPoints);

    /**
     * @brief Recursively delete all nodes in the contour tree.
     * @param node The root of the subtree to delete.
     */
    void freeTree(ContourNode* node);
};

/**
 * @brief Check whether a point (x, y) is within the image boundary.
 * 
 * @param x X-coordinate of the point.
 * @param y Y-coordinate of the point.
 * @param m Width of the image (number of columns).
 * @param n Height of the image (number of rows).
 * @return true if the point is inside the image boundary, false otherwise.
 */
bool isPointAvailable(int x, int y, int m, int n);

/**
 * @brief Determine whether a point is a contour point.
 * 
 * A point is considered a contour point if it is white (foreground)
 * and has at least one black (background) neighbor in its 8-neighborhood.
 *
 * @param x X-coordinate of the point.
 * @param y Y-coordinate of the point.
 * @param m Width of the image.
 * @param n Height of the image.
 * @param img The binary input image (CV_8UC1).
 * @return true if the point is on the boundary of a contour, false otherwise.
 */
bool isContourPoint(int x, int y, int m, int n, cv::Mat& img);

/**
 * @brief Check whether a point can be used as a valid tracing point in the contour detection process.
 * 
 * Conditions:
 * - The point must be white in the binary image.
 * - It must be a valid contour point (with black neighbors).
 * - It must not be visited in the current path (`visitedPointsMap`) or globally (`allVisitedPoints`).
 *
 * @param x X-coordinate of the point.
 * @param y Y-coordinate of the point.
 * @param m Width of the image.
 * @param n Height of the image.
 * @param visitedPointsMap Map of locally visited points during current contour tracing.
 * @param allVisitedPoints Map of globally visited points across all contours.
 * @param binImage The binary input image (CV_8UC1).
 * @return true if the point is a valid unvisited boundary point, false otherwise.
 */
bool isValidBoundaryPoint(
    int x,
    int y,
    int m,
    int n,
    std::unordered_map<cv::Point, int, PointHash>& visitedPointsMap,
    std::unordered_map<cv::Point, int, PointHash>& allVisitedPoints,
    cv::Mat& binImage
);

//double computeArea(const std::vector<cv::Point>& points);
//bool isContourClosed(const std::vector<cv::Point>& points);
//double computePerimeter(const std::vector<cv::Point>& points, const bool isClosed);
//int cross(const cv::Point& a, const cv::Point& b);
//int windingNumber(const std::vector<cv::Point>& polygon, const cv::Point& point);
//bool isPointInContour(cv::Point point, const Contour& contour, double threshold = 0.001);
