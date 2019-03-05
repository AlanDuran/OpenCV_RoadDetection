/*
 * basic_operations.cpp
 *
 *  Created on: 4 mar 2019
 *      Author: alan
 */

#include "basic_operations.h"

#include <stdio.h>

using namespace cv;
using namespace std;

void showImg(Mat *img, char * window, int type, int time)
{
	namedWindow( window, type );
	imshow( window, *img );
	waitKey (time);
}

Mat getHistogram(Mat *src, int histSize)
{
	Mat img_planes = *src;
	Mat img_hist;

	float range[] = { 0, (float)histSize - 1}; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	calcHist( &img_planes, 1, CV_HIST_ARRAY, Mat(), img_hist, 1, &histSize, &histRange, uniform, accumulate );

	return img_hist;
}

void drawHistogram(Mat *src, Mat dst, int histSize, Scalar color)
{
	Mat img_hist = *src;
	Mat temp;

	normalize(img_hist, temp, 0, dst.rows, NORM_MINMAX, -1, Mat() );

	int bin_w = cvRound( (double) dst.cols/256 );
	uint16_t i;

	for(i = 1; i < histSize; i++ )
	{
		line( dst, Point( bin_w*(i-1), dst.rows - cvRound(temp.at<float>(i-1)) ),
			  Point( bin_w*(i), dst.rows - cvRound(temp.at<float>(i)) ), color, 2, 8, 0);
	}
}
