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

void getDominantHistogram(img_type *src, int type)
{
	Mat temp = src->img;
	split( temp, src->img_planes );

	if(type == true)
	{
		src->img_hist[0] = getHistogram(&src->img_planes[0],180);
		src->img_hist[1] = getHistogram(&src->img_planes[1],256);
		src->img_hist[2] = getHistogram(&src->img_planes[2],256);
		src->dominantChannel = 0; //Channel H
	}

	else
	{
		src->img_hist[0] = getHistogram(&src->img_planes[0],256);
		src->img_hist[1] = getHistogram(&src->img_planes[1],256);
		src->img_hist[2] = getHistogram(&src->img_planes[2],256);

		double areas[3] = {0,0,0};
		int i, histSize = 256;

		//Histogram area comparison
		for(i = 0; i < histSize - 1; i++ )
		{
		  areas[0] += i*(double)src->img_hist[0].at<float>(i);
		  areas[1] += i*(double)src->img_hist[1].at<float>(i);
		  areas[2] += i*(double)src->img_hist[2].at<float>(i);
		}

		printf("\n\narea r = %f\narea g = %f \narea b = %f",areas[0],areas[1],areas[2]);
		fflush(stdout);

		//Get index of max element
		src->dominantChannel = distance(areas,max_element(areas, areas + 3));
	}
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
