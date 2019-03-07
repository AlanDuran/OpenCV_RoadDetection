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

typedef struct
{
	cv::Mat img;
	cv::Mat img_planes[3];
	cv::Mat img_hist[3];
	uint8_t dominantChannel;
} img_type;

void showImg(cv::Mat *img, char * window, int type, int time);
cv::Mat getHistogram(cv::Mat *src, int histSize);
void getDominantHistogram(img_type *src, int type);
void drawHistogram(cv::Mat *src, cv::Mat dst, int histSize, cv::Scalar color);



#endif /* SRC_BASIC_OPERATIONS_H_ */
