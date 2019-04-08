/*
 * otsu.h
 *
 *  Created on: 25 feb 2019
 *      Author: alan
 */

#ifndef SRC_OTSU_H_
#define SRC_OTSU_H_

#include <opencv2/opencv.hpp>
#include "basic_operations.h"

uint8_t get_threshold(cv::Mat src, int limit);
uint16_t get_horizon(cv::Mat src);
cv::Mat get_roadImage(img_type *src);

#endif /* SRC_OTSU_H_ */
