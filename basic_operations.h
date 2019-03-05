/*
 * basic_operations.h
 *
 *  Created on: 4 mar 2019
 *      Author: alan
 */

#ifndef SRC_BASIC_OPERATIONS_H_
#define SRC_BASIC_OPERATIONS_H_

#include <opencv2/opencv.hpp>
#include <stdio.h>

void showImg(cv::Mat *img, char * window, int type, int time);
cv::Mat getHistogram(cv::Mat *src, int histSize);
void drawHistogram(cv::Mat *src, cv::Mat dst, int histSize, cv::Scalar color);



#endif /* SRC_BASIC_OPERATIONS_H_ */
