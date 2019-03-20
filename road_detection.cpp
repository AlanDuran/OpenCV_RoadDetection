/*
 * Image.cpp
 *
 *  Created on: 20 feb 2019
 *      Author: Alan Duran
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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
	const char window_full[] = "Imagen original";
	const char window_gauss[] = "Filtro gaussiano";
	const char window_rec[] = "Recorte cielo";
#endif

const char window_horizon[] = "Horizonte";
const char window_otsu[] = "Camino detectado con otsu";
const char window_display[] = "Imagen original redimensionada";

Mat src,dst,temp, display;
img_type img;

int main( int argc, char** argv )
{
/********************** Load image and pre-processing ******************************/

	//Read Image
	const char* filename = argc >=2 ? argv[1] : "data/iteso0.png";
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


/******************* Histogram calculation and selection ****************************/

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

	#ifdef DEBUG
		Mat road_test;
		temp = dst(Range(dst.rows * 0.8, dst.rows), Range(dst.cols * .4, dst.cols*0.6));
		img.img = temp;
		showImg(display(Range(dst.rows * 0.8, dst.rows),
				Range(dst.cols * .4, dst.cols*0.6)), "Otsu test", WINDOW_AUTOSIZE, 100);
		showImg(get_roadImage(&img,xHSV), "Otsu test result", WINDOW_AUTOSIZE, 100);
	#endif
/******** Improvements with Canny detector and Hough Line transform *****************/

	Mat houghP;

	houghP = display(rows,cols).clone();
	//cvtColor(houghP,houghP,CV_BGR2GRAY);
	GaussianBlur( houghP, houghP, Size(5, 5), 0, 0);

	// Edge detection
	Canny(houghP, houghP, 150, 250, 3); //25, 100, 3

    imshow("Canny", houghP);

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection


    HoughLinesP(houghP, linesP, 10, CV_PI/180, 80, 20, 20 ); // runs the actual detection 10, CV_PI/360, 80, 20, 20

    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, 16);
    }

    // Show results
    imshow("Detected Lines - Probabilistic Line Transform", houghP);

/************************* Blob operations *****************************************/

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<int> small_blobs;
    double contour_area;

	findContours( houghP, contours, hierarchy ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	// Find indices of contours whose area is less than `threshold`
	for (size_t i=0; i<contours.size(); ++i) {
		contour_area = contourArea(contours[i]) ;
		if ( contour_area < 250)
			small_blobs.push_back(i);
	}

	Mat edges = Mat::zeros(houghP.size(), CV_8UC1);

	for (uint i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255);
		drawContours( edges, contours, i, color, CV_FILLED );
	}

	showImg(edges, "All contours", WINDOW_AUTOSIZE, 100);

	// fill-in all small contours with zeros
	for (size_t i=0; i < small_blobs.size(); ++i) {
	    drawContours(edges, contours, small_blobs[i], Scalar(0), CV_FILLED, 8);
	}

	//Check if white pixels (road) are in the center bottom of the image
	Range h_rows(houghP.rows * 0.8, houghP.rows);
	Range h_cols(houghP.cols * 0.45, houghP.cols * 0.55);
	temp = edges(h_rows, h_cols);

	if(countNonZero(temp) < (houghP.rows * 0.2)*(houghP.cols * 0.1) * 0.9)
	{
		Mat inv_edges = 255 - edges;

		//Compare road pixels of both segmented images
		if(countNonZero(inv_edges) > countNonZero(temp))
		{
			edges = inv_edges;
		}
	}

	showImg(edges, "Selected contours", WINDOW_AUTOSIZE, 100);

	/*********** weighted average of the images intensities ***********/

	Mat planes[3];
	Mat old_image, new_image;

	split( src(rows,cols), planes);
	old_image = (xHSV) ? planes[img.dominantChannel] * (255.0/180.0): planes[img.dominantChannel];

	imshow("old_image", old_image);

	addWeighted(old_image, 1.0/3.0, otsu_road, 1.0/3.0, 0, new_image);
	addWeighted(new_image, 1.0, edges, 1.0/3.0, 0, new_image);

	imshow("Result", new_image);
	waitKey(1000000);

	return 0;
}
