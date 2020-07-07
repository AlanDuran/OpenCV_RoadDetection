/*
 * otsu.c
 *
 *  Created on: 25 feb 2019
 *      Author: alan
 */
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "otsu.h"

using namespace cv;
using namespace std;

#define N_DIV 	10
#define X_SEC	0.10
#define Y_SEC	0.50

#define HIST_SIZE 256

uint8_t get_threshold(Mat hist, int limit)
{
	double q[2] = {0,0}, w_mean[2] = {0,0}, cv[2] = {0,0};
	double *intra_cv = new double[limit];
	int threshold,i;

	for (threshold = 0; threshold < limit; threshold++)
	{
		int index;
		for(index = 0; index <= 1; index++)
		{
			q[index] = 0;
			w_mean[index] = 0;
			cv[index] = 0;
		}

		for(i = 0; i < limit; i++)
		{
			if(i <= threshold)
			{
				q[0] += (double)hist.at<float>(i);
			}

			else
			{
				q[1] += (double)hist.at<float>(i);
			}
		}

		//To avoid getting a division to zero
		if(q[0] == 0) q[0] = 1;
		if(q[1] == 0) q[1] = 1;

		for(i = 0; i < limit; i++)
		{
			if(i <= threshold)
			{
				w_mean[0] += (i * (double)hist.at<float>(i)) / (q[0]);
			}

			else
			{
				w_mean[1] += (i * (double)hist.at<float>(i)) / (q[1]);
			}
		}

		for(i = 0; i < limit; i++)
		{
			if(i <= threshold)
			{
				cv[0] += ((i - w_mean[0])*(i - w_mean[0]))*(hist.at<float>(i)/q[0]);
			}

			else
			{
				cv[1] += ((i - w_mean[1])*(i - w_mean[1]))*(hist.at<float>(i)/q[1]);
			}
		}

		intra_cv[threshold] = q[0]*cv[0] + q[1]*cv[1];
	}
	//Get index of min element
	return distance(intra_cv, min_element(intra_cv, intra_cv + limit));
}

uint16_t get_horizon(Mat image)
{
	Mat rec;
	Mat temp[N_DIV];

	uint8_t tholds[N_DIV];
	uint16_t accum[N_DIV] = {0};
	double pdiff[N_DIV];

	int i,j;
	int segment = image.rows / N_DIV;

	//Calculate thresholds for each segment
	for(i = 0; i < N_DIV; i++)
	{
		int limit = 255;
		Mat hist;
		Range rows(i*segment, (i + 1)*segment);
		Range cols(0,image.cols);
		temp[N_DIV - i - 1] = image(rows,cols);

		hist = getHistogram(temp[N_DIV - i - 1]);

		while(hist.at<float>(limit-1) == 0)
		{
			limit--;
		}

		tholds[N_DIV - i - 1] = get_threshold(hist, limit);
	}

	//Get the percentage of foreground (white) pixels for each threshold
	for(i = 0; i < N_DIV; i++)
	{
		for(j = 0; j < N_DIV; j++)
		{
			threshold( temp[i], temp[i], tholds[j], 255, THRESH_BINARY);
			accum[i] += 100*countNonZero(temp[i])/(temp[i].cols*temp[i].rows);
		}
	}

	//Calculate the percentage difference between segments
	pdiff[0] = 0;
 	for(i = 1; i < N_DIV; i++)
	{
		double prom = (accum[i-1] + accum[i]) / 2;
		pdiff[i] = abs((accum[i] - accum[i-1]) / prom);
	}

	return segment * distance(pdiff,max_element(pdiff, pdiff + N_DIV));
}

Mat get_roadImage(img_type *src)
{
	Mat road[2];
	uint8_t selected_img = 0;

	//Detect road with received image
	uint16_t limit = HIST_SIZE;

	while(src->hist.at<float>(limit-1) == 0)
	{
		limit--;
	}

	uint8_t thold = get_threshold(src->hist, limit);
	threshold( src->img, road[0], thold, 255, THRESH_BINARY);

	//Check if white pixels (road) are in the center bottom of the image
	Range rows(src->img.rows * (1 - Y_SEC), src->img.rows); //0.8
	Range cols(src->img.cols * (0.50 - X_SEC), src->img.cols * (0.50 + X_SEC)); //0.45 - 0.55

	if(countNonZero(road[0](rows,cols)) < (src->img.rows * Y_SEC)*(src->img.cols * X_SEC) * 0.65)
	{
		//Detect road with negated image
		Mat temp = src->img.clone();
		temp = limit - temp;

		Mat hist = getHistogram(temp);

		thold = get_threshold(hist, limit);
		threshold( temp, road[1], thold, 255, THRESH_BINARY);

		//Compare road pixels of both segmented images
		if(countNonZero(road[0](rows,cols)) > countNonZero(road[1](rows,cols)))
		{
			selected_img = 0;
		}

		{
			src->img = limit - src->img;
			selected_img = 1;
		}

		hist.release();
		temp.release();
	}

	//rectangle(road[selected_img],Point(cols.start,rows.start),Point(cols.end,rows.end),Scalar(0),1,1);
	return road[selected_img];
}
