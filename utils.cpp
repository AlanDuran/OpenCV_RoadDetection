/*
 * basic_operations.cpp
 *
 *  Created on: 4 mar 2019
 *      Author: alan
 */

#include "utils.h"

#include <stdio.h>
#include <termios.h>
#include <unistd.h>

using namespace cv;
using namespace std;

#define HIST_SIZE 256

Mat getHistogram(Mat src)
{
	Mat img_hist;
	int histSize = HIST_SIZE;

	float range[] = { 0, (float)histSize - 1}; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	calcHist( &src, 1, CV_HIST_ARRAY, Mat(), img_hist, 1, &histSize, &histRange, uniform, accumulate );

	return img_hist;
}

void drawHistogram(Mat img_hist, Mat dst, Scalar color)
{
	Mat temp;

	normalize(img_hist, temp, 0, dst.rows, NORM_MINMAX, -1, Mat() );

	int bin_w = cvRound( (double) dst.cols/HIST_SIZE );
	uint16_t i;

	for(i = 1; i < HIST_SIZE; i++ )
	{
		line( dst, Point( bin_w*(i-1), dst.rows - cvRound(temp.at<float>(i-1)) ),
			  Point( bin_w*(i), dst.rows - cvRound(temp.at<float>(i)) ), color, 2, 8, 0);
	}
}

Mat removeShadows(Mat src, img_type *img)
{
	//Convert to HSV
	cvtColor(src, src, CV_BGR2HSV);
	split(src, img->channel);

	//src = channel[0] * (255.0/180.0);

	//Set V channel to a fixed value
	//equalizeHist(channel[2],channel[2]);
	img->channel[2] = Mat(src.rows, src.cols, CV_8UC1, 128);//Set V

	//Merge channels
	merge(img->channel, 3, src);
	cvtColor(src, src, CV_HSV2BGR);

	//2. Convert to gray and normalize
	cvtColor(src, src, CV_BGR2GRAY);
	normalize(src, src, 0, 255, NORM_MINMAX, CV_8UC1);
	return src;
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
			dist[i] = 	1000000; //Arbitrary distance
		}
	}

	//select minimun distance of centroid
	countourIndex = distance(dist,min_element(dist, dist + contours.size()));

	//draw selected blob
	Mat edges = Mat::zeros(src.size(), CV_8UC1);
	drawContours( edges, contours, countourIndex, Scalar(255), CV_FILLED );

	return edges;
}

/***********************************************************************************/
/*********** KEYBOARD **************************************************************/

