#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include "MyContourDetector.h"
using namespace cv;

int main() {
	Mat image = imread("test.png");
	//to binary image
	Mat greayImage,binaryImage;
	cvtColor(image, greayImage, cv::COLOR_BGRA2GRAY);
	threshold(greayImage, binaryImage, 0, 255,THRESH_BINARY);

	MyContourDetector* myContourDetector = new MyContourDetector();
	myContourDetector->setInputImage(binaryImage);
	myContourDetector->detect();
	myContourDetector->printContourTree();
	myContourDetector->clear();
	//binaryImage /= 255;
	//namedWindow("binaryImage");
	//imshow("binaryImage",binaryImage);
	//waitKey();

	//std::vector<std::vector<cv::Point>> contours;
	//std::vector<cv::Vec4i> hierarchy;

	//cv::findContours(
	//	binaryImage,          
	//	contours,        
	//	hierarchy,      
	//	cv::RETR_TREE,   
	//	cv::CHAIN_APPROX_NONE 
	//);
	//for(auto contour :contours)
	//	std::cout << cv::contourArea(contour) << std::endl;
	//namedWindow("binaryImage");
	//imshow("binaryImage", binaryImage);
	//waitKey();



	return 0;
}