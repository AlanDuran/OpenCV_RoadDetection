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
#define xHSV			1

#define KERNEL_WIDTH	41
#define KERNEL_HEIGHT	41
#define SIGMA_X 		7
#define SIGMA_Y			7
#define SIZE_X			320
#define SIZE_Y			200

using namespace std;
using namespace cv;

#ifdef DEBUG
	char window_full[] = "Imagen original";
	char window_gauss[] = "Filtro gaussiano";
	char window_rec[] = "Recorte cielo";
#endif

char window_horizon[] = "Horizonte";
char window_otsu[] = "Camino detectado con otsu";
char window_display[] = "Imagen original redimensionada";

Mat src,dst,temp, display;
img_type img;

int main( int argc, char** argv )
{
/***************** Load image and pre-processing ***************************/

	//Read Image
	const char* filename = argc >=2 ? argv[1] : "data/pixy5.png";
	src = imread( filename, IMREAD_COLOR );

	if(src.empty())
	{
	  printf(" Error opening image\n");
	  return -1;
	}

	#ifdef DEBUG
		//Display image
		showImg( &src, window_full, WINDOW_AUTOSIZE, 100);
	#endif

	//Image resizing
	if(src.rows > SIZE_Y)
	{
	  float ratio = src.rows / (float)SIZE_Y;
	  uint16_t new_cols = src.cols / ratio;
	  resize(src, src, Size(new_cols,SIZE_Y), 0, 0, INTER_LANCZOS4);
	}

	display = src.clone();

	//Change color space
	if(xHSV == true)
	{
		cvtColor(src, src, CV_BGR2HSV);
	}

	//Smoothing image with Gaussian filter
	GaussianBlur( src, dst, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);

	//Select image fraction to analyze
	Range rows(0, src.rows * 0.6);
	Range cols(0, src.cols);
	temp = dst(rows,cols);

	showImg(display, window_display, WINDOW_AUTOSIZE, 100);

	#ifdef DEBUG
		showImg(&dst, window_gauss, WINDOW_AUTOSIZE, 100);
	#endif


/************* Histogram calculation and selection **************************/

	img.img = temp;
	//Calculate an histogram for each channel, store it in img.img_hist[] and select dominant channel
	getDominantHistogram(&img, xHSV);

/**************** Horizon detection *************************************************/

	uint16_t limit = (xHSV) ? 180 : 256;

	uint16_t horizon = get_horizon(img.img_planes[img.dominantChannel], limit);
	horizon = src.rows - horizon - src.rows*0.4;

	temp = display.clone();
	line( temp, Point(0,horizon), Point(src.cols,horizon),Scalar( 0, 0, 255 ), 2, 1);
	showImg(temp, window_horizon, WINDOW_AUTOSIZE, 100);

/******************** Road detection with Otsu **************************************/

	Mat otsu_road;
	//Cut the image from horizon to the bottom
	rows.start = horizon;
	rows.end = src.rows;
	temp = dst(rows,cols);

	img.img = temp;
	otsu_road = get_roadImage(&img,xHSV);

	showImg(otsu_road, window_otsu, WINDOW_AUTOSIZE, 100);

/************ Improvements with Canny detector and Hough transform *****************/

	/*
	GaussianBlur( src, dst, Size(5, 5), 0, 0);
	split( src, img.img_planes );

	char window_name[] = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	Sobel( img.img_planes[img.dominantChannel], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel( img.img_planes[img.dominantChannel], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	imshow( window_name, grad );
	waitKey(100);
*/
	temp = src(rows,cols);
	GaussianBlur( temp, temp, Size(5, 5), 0, 0);
	split( temp, img.img_planes );

	char window_edge[] = "Canny Edge Detector";
	temp = dst(rows,cols);
	Mat edge;
	Canny(img.img_planes[img.dominantChannel],edge,15,25);
	imshow( window_edge, edge);
	waitKey(100000);

	return 0;
}
