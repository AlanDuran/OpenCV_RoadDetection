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
	const char* filename = argc >=2 ? argv[1] : "data/pixy5.png";
	src = imread( filename, IMREAD_COLOR );

	if(src.empty())
	{
	  printf(" Error opening image\n");
	  return -1;
	}

	#ifdef DEBUG
		//Display image
		showImg( src, window_full, WINDOW_AUTOSIZE, 100);
	#endif

	//Image resizing
	if(src.rows > SIZE_Y)
	{
	  float ratio = src.rows / (float)SIZE_Y;
	  uint16_t new_cols = src.cols / ratio;
	  resize(src, src, Size(new_cols,SIZE_Y), 0, 0, INTER_LANCZOS4);
	  imwrite( "data/iteso_0.png", src );
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
		showImg(dst, window_gauss, WINDOW_AUTOSIZE, 100);
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

	/* For homogeneous roads 15, 75, 3*/
	/* For no homogeneous roads 150, 240, 3*/
	Canny(houghP, houghP, 15, 75, 3);

    imshow("Canny", houghP);

    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection


    /*
     *  Probabilistic Hough Line Transform arguments:
     *
     *
     * 	dst: Output of the edge detector. It should be a grayscale image
     * 	lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
     * 	rho : The resolution of the parameter r in pixels. We use 1 pixel.
     * 	theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
     * 	threshold: The minimum number of intersections to “detect” a line
     * 	minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
     * 	maxLineGap: The maximum gap between two points to be considered in the same line.
     */


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
	for (size_t i=0; i<contours.size(); ++i)
	{
		contour_area = contourArea(contours[i]) ;
		if ( contour_area < 250)
			small_blobs.push_back(i);
	}

	temp = Mat::zeros(houghP.size(), CV_8UC1);

	for (uint i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255);
		drawContours( temp, contours, i, color, CV_FILLED );
	}

	showImg(temp, "All contours", WINDOW_AUTOSIZE, 100);

	// fill-in all small contours with zeros
	for (size_t i=0; i < small_blobs.size(); ++i) {
	    drawContours(temp, contours, small_blobs[i], Scalar(0), CV_FILLED, 8);
	}

	showImg(temp, "no small", WINDOW_AUTOSIZE, 100);

	temp = 255 - temp;
	showImg(temp, "inv", WINDOW_AUTOSIZE, 100);
	findContours( temp, contours, hierarchy ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	// get the moments
	vector<Moments> mu(contours.size());
	for( int i = 0; i<contours.size(); i++ )
	{
		mu[i] = moments( contours[i], false );
	}

	// get the centroids and calculate distances to bottom center of the image.
	double dist[contours.size()];
	int countourIndex;

	for( int i = 0; i<contours.size(); i++)
	{
		if(contourArea(contours[i]) > 50)
		{
			double cx = mu[i].m10/mu[i].m00;
			double cy = mu[i].m01/mu[i].m00;
			dist[i] = ((SIZE_Y -  cy)*(SIZE_Y - cy))
					+ (((SIZE_X / 2) - cx) * ((SIZE_X / 2) - cx));
		}

		else
		{
			dist[i] = 100000;
		}
	}

	//select minimun distance of centroid
	countourIndex = distance(dist,min_element(dist, dist + contours.size()));

	//draw selected blob
	Mat edges = Mat::zeros(houghP.size(), CV_8UC1);
	drawContours( edges, contours, countourIndex, Scalar(255), CV_FILLED );

	showImg(edges, "Selected contour", WINDOW_AUTOSIZE, 100);

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
