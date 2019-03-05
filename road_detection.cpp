/*
 * Image.cpp
 *
 *  Created on: 20 feb 2019
 *      Author: Alan Duran
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "otsu.h"
#include "basic_operations.h"

#define xDEBUG
#define xHSV			0

#define KERNEL_WIDTH	41
#define KERNEL_HEIGHT	41
#define SIGMA_X 		7
#define SIGMA_Y			7
#define SIZE_X			320
#define SIZE_Y			200

using namespace std;
using namespace cv;

#ifdef DEBUG
	char original[] = "Imagen original";
	char window_gauss[] = "Filtro gaussiano";
	char window_rec[] = "Recorte cielo";
	char window_cc[] = "Canal seleccionado";
	char window_rec_s[] = "Recorte camino";
#endif

char sky[] = "Threshold cielo";
char window_horizon[] = "Horizonte";
char window_road[] = "Camino";

Mat src,dst,temp;

int main( int argc, char** argv )
{
	/*********** Load image and pre-processing *********************/
	const char* filename = argc >=2 ? argv[1] : "data/pixy5.png";
	src = imread( filename, IMREAD_COLOR );

	if(xHSV == true)
	{
		cvtColor(src, src, CV_BGR2HSV);
	}

	if(src.empty())
	{
	  printf(" Error opening image\n");
	  return -1;
	}

	#ifdef DEBUG
		//Display image
		showImg( &src, original, WINDOW_AUTOSIZE, 100);
	#endif

	//Gaussian filter and image reduction
	if(src.rows > SIZE_Y)
	{
	  float ratio = src.rows / (float)SIZE_Y;
	  uint16_t new_cols = src.cols / ratio;
	  resize(src, src, Size(new_cols,SIZE_Y), 0, 0, INTER_LANCZOS4);
	}

	GaussianBlur( src, dst, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);

	//Select image fraction to analyze
	Range rows(0, src.rows * 0.6);
	Range cols(0, src.cols);
	temp = dst(rows,cols);

	#ifdef DEBUG
		showImg(&dst, window_gauss, WINDOW_AUTOSIZE, 100);
		showImg(&temp, window_rec, WINDOW_AUTOSIZE, 100);
	#endif


	/************* Histogram calculation and selection **************************/

	uint8_t channel;
	vector<Mat> img_planes;
	split( temp, img_planes );
	Mat img_hist[3];

	if(xHSV == true)
	{
		img_hist[0] = getHistogram(&img_planes[0],180);
		img_hist[1] = getHistogram(&img_planes[1],256);
		img_hist[2] = getHistogram(&img_planes[2],256);
		channel = 0; //Channel H
	}

	else
	{
		img_hist[0] = getHistogram(&img_planes[0],256);
		img_hist[1] = getHistogram(&img_planes[1],256);
		img_hist[2] = getHistogram(&img_planes[2],256);

		double areas[3] = {0,0,0};
		int i, histSize = 256;

		//Histogram area comparison
		for(i = 0; i < histSize - 1; i++ )
		{
		  areas[0] += i*(double)img_hist[0].at<float>(i);
		  areas[1] += i*(double)img_hist[1].at<float>(i);
		  areas[2] += i*(double)img_hist[2].at<float>(i);
		}

		printf("\n\narea r = %f\narea g = %f \narea b = %f",areas[0],areas[1],areas[2]);
		fflush(stdout);

		//Get index of max element
		channel = distance(areas,max_element(areas, areas + 3));
	}

	#ifdef DEBUG

		int hist_w = 512, hist_h = 400;
		Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

		if(xHSV)
		{
			drawHistogram(&img_hist[0],histImage,180,Scalar(255,0,0));
		}

		else
		{
			drawHistogram(&img_hist[0],histImage,256,Scalar(255,0,0));
		}

		drawHistogram(&img_hist[1],histImage,256,Scalar(0,255,0));
		drawHistogram(&img_hist[2],histImage,256,Scalar(0,0,255));

		imshow("calcHist Demo", histImage );
		showImg(&img_planes[channel], window_cc, WINDOW_AUTOSIZE, 100);
	#endif


/**************** Horizon detection ********************************************/

	uint16_t limit = (xHSV) ? 180 : 256;
	uint8_t thold = get_threshold(&img_hist[channel], limit);
	threshold( img_planes[channel], temp, thold, 255, THRESH_BINARY);
	showImg(&temp, sky, WINDOW_AUTOSIZE, 100);

	uint16_t horizon = get_horizon(&img_planes[channel], limit);
	horizon = src.rows - horizon - src.rows*0.4;

	temp = src.clone();
	line( temp, Point(0,horizon), Point(src.cols,horizon),Scalar( 0, 0, 255 ), 2, 1);
	showImg(&temp, window_horizon, WINDOW_AUTOSIZE, 100);

/******************** Road detection ******************************************/

	rows.start = horizon;
	rows.end = src.rows;
	temp = dst(rows,cols);

	#ifdef DEBUG
		showImg(&temp, window_rec_s, WINDOW_AUTOSIZE, 100);
	#endif

	split( temp, img_planes );

	if(xHSV == true)
	{
		img_hist[0] = getHistogram(&img_planes[0],180);
		img_hist[1] = getHistogram(&img_planes[1],256);
		img_hist[2] = getHistogram(&img_planes[2],256);
		channel = 0; //Channel H
	}

	else
	{
		img_hist[0] = getHistogram(&img_planes[0],256);
		img_hist[1] = getHistogram(&img_planes[1],256);
		img_hist[2] = getHistogram(&img_planes[2],256);

		double areas[3] = {0,0,0};
		int i, histSize = 256;

		//Histogram area comparison
		for(i = 0; i < histSize - 1; i++ )
		{
		  areas[0] += i*(double)img_hist[0].at<float>(i);
		  areas[1] += i*(double)img_hist[1].at<float>(i);
		  areas[2] += i*(double)img_hist[2].at<float>(i);
		}

		printf("\n\narea r = %f\narea g = %f \narea b = %f",areas[0],areas[1],areas[2]);
		fflush(stdout);

		//Get index of max element
		channel = distance(areas,max_element(areas, areas + 3));
	}

	thold = get_threshold(&img_hist[channel], limit);
	threshold( img_planes[channel], temp, thold, 255, THRESH_BINARY);
	showImg(&temp, window_road, WINDOW_AUTOSIZE, 100);
	//cvReleaseHist();

/****************** Image derivative **********************************/

	GaussianBlur( src, dst, Size(3, 3), 0, 0);
	split( src, img_planes );

	char window_name[] = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	Sobel( img_planes[channel], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel( img_planes[channel], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	imshow( window_name, grad );
	waitKey(100000);
	return 0;
}
