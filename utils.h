/*
 * basic_operations.h
 *
 *  Created on: 4 mar 2019
 *      Author: alan
 */

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <opencv2/opencv.hpp>
#include <stdio.h>

typedef struct
{
	cv::Mat img;
	cv::Mat hist;
	cv::Mat channel[3];
} img_type;

cv::Mat getHistogram(cv::Mat src);
void drawHistogram(cv::Mat src, cv::Mat dst, cv::Scalar color);
cv::Mat getNearestBlob(cv::Mat src, int coordX, int coordY, int minArea);
cv::Mat removeShadows(cv::Mat src, img_type *img);
cv::Mat equalizeShadows(cv::Mat src);
cv::Mat getHistogram2(cv::Mat src);

#endif /* SRC_UTILS_H_ */
