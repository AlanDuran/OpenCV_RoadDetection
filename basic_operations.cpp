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

void showImg(Mat img, const char * window, int type, int time)
{
	namedWindow( window, type );
	imshow( window, img );
	waitKey (time);
}

Mat getHistogram(Mat img_planes, int histSize)
{
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
		equalizeHist(src->img_planes[0],src->img_planes[0]);
		src->img_hist[0] = getHistogram(src->img_planes[0],180);
		src->img_hist[1] = getHistogram(src->img_planes[1],256);
		src->img_hist[2] = getHistogram(src->img_planes[2],256);
		src->dominantChannel = 0; //Channel H
	}

	else
	{
		equalizeHist(src->img_planes[0],src->img_planes[0]);
		equalizeHist(src->img_planes[1],src->img_planes[1]);
		equalizeHist(src->img_planes[2],src->img_planes[2]);
		src->img_hist[0] = getHistogram(src->img_planes[0],256);
		src->img_hist[1] = getHistogram(src->img_planes[1],256);
		src->img_hist[2] = getHistogram(src->img_planes[2],256);

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

void drawHistogram(Mat img_hist, Mat dst, int histSize, Scalar color)
{
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

Mat getNearestBlob(Mat src, int coordX, int coordY, int minArea)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours( src, contours, hierarchy ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	// get the moments of the contours
	vector<Moments> mu(contours.size());
	for( unsigned int i = 0; i<contours.size(); i++ )
	{
		mu[i] = moments( contours[i], false );
	}

	// get the centroids and calculate distances to bottom center of the image.
	double dist[contours.size()];
	int countourIndex;

	for( unsigned int i = 0; i<contours.size(); i++)
	{
		if(contourArea(contours[i]) > minArea) //Area threshold
		{
			double cx = mu[i].m10/mu[i].m00;
			double cy = mu[i].m01/mu[i].m00;
			dist[i] = ((coordY -  cy)*(coordY))
					+ (((coordX / 2) - cx) * ((coordX / 2) - cx));
		}

		else
		{
			dist[i] = 100000; //Arbitrary distance
		}
	}

	//select minimun distance of centroid
	countourIndex = distance(dist,min_element(dist, dist + contours.size()));

	//draw selected blob
	Mat edges = Mat::zeros(src.size(), CV_8UC1);
	drawContours( edges, contours, countourIndex, Scalar(255), CV_FILLED );

	return edges;
}
