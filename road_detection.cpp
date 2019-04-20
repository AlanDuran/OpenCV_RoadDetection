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
#include "utils.h"
#include <pthread.h>

#define DEBUG
#define FIND_HORIZON	0
#define HORIZON_SIZE	0.7

#define KERNEL_WIDTH	7
#define KERNEL_HEIGHT	7
#define SIGMA_X 		0
#define SIGMA_Y			0
#define SIZE_X			320
#define SIZE_Y			200

/*
 * iteso.mp4 	--> 15 75
 * base_aerea 	--> 75 200
 * atras_iteso 	--> 10 50
 */
#define CANNY_LOW		15 /* Homogeneous roads 15 - 25, no homogeneous roads 75 - 100*/
#define CANNY_HIGH		75 /* Homogeneous roads 75 - 100, no homogeneous roads 150 - 200*/

static void* userInput_thread(void*);

using namespace std;
using namespace cv;

static bool keep_running = true;
static bool rewind_frame = false;
static bool forward_frame = false;

//Read video
VideoCapture cap("data/iteso.mp4");

int main( int argc, char** argv )
{
    pthread_t tId;
    pthread_create(&tId, 0, userInput_thread, 0);

    /********************** Load image and pre-processing ******************************/


	if(!cap.isOpened()){
	    cout << "Error opening video stream or file" << endl;
	    return -1;
	}

	while(true)
	{
		Mat src,dst,temp, display;
		img_type img;

		// Capture frame-by-frame
		cap >> src;

		if (src.empty())
		      break;

		//Image resizing
		if(src.rows > SIZE_Y)
		{
		  float ratio = src.rows / (float)SIZE_Y;
		  uint16_t new_cols = src.cols / ratio;
		  resize(src, src, Size(new_cols,SIZE_Y), 0, 0, INTER_LANCZOS4);
		}

		//flip(src,src,-1);
		display = src.clone();

		#ifdef DEBUG
			Mat display_otsu = Mat::zeros(display.size(), CV_8UC1);
			Mat display_canny = Mat::zeros(display.size(), CV_8UC1);
			Mat display_lines = Mat::zeros(display.size(), CV_8UC1);
			Mat display_edges = Mat::zeros(display.size(), CV_8UC1);
		#endif

		//Change color space and remove shadows
		src = removeShadows(src,&img);

		//Smoothing image with Gaussian filter
		GaussianBlur( src, dst, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);

/**************** Horizon detection *************************************************/

		uint16_t horizon;

		//Select image fraction to analyze
		Range rows(0, src.rows * HORIZON_SIZE);
		Range cols(0, src.cols);

		if(FIND_HORIZON)
		{
			temp = img.channel[0](rows,cols).clone();
			GaussianBlur( temp, temp, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);
			horizon = get_horizon(temp);
			horizon = src.rows - horizon - src.rows * (1 - HORIZON_SIZE);
		}

		else
		{
			horizon = src.rows * 0.30;
		}

		temp = display.clone();
		line( temp, Point(0,horizon), Point(src.cols,horizon),Scalar( 0, 0, 255 ), 2, 1);
		imshow("original_image", temp);

/******************** Road detection with Otsu **************************************/

		Mat otsu_road;

		//Cut the image from horizon to the bottom
		rows.start = horizon;
		rows.end = src.rows;

		/*
		temp = img.channel[0](rows,cols).clone();
		GaussianBlur( temp, temp, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);
		equalizeHist(temp,temp);
		*/

		img.img = dst(rows,cols).clone();
		img.hist = getHistogram(img.img);
		otsu_road = get_roadImage(&img);
/******** Improvements with Canny detector and Hough Line transform *****************/

		Mat houghP;

		houghP = src(rows,cols).clone();
		GaussianBlur( houghP, houghP, Size(5, 5), 0, 0);

		// Edge detection
		Canny(houghP, houghP, CANNY_LOW, CANNY_HIGH, 3);

		#ifdef DEBUG
			houghP.copyTo(display_canny(rows,cols));
			imshow("Canny", display_canny);
		#endif

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

		HoughLinesP(houghP, linesP, 10, CV_PI/180, 80, 20, 30 ); // runs the actual detection 10, CV_PI/180, 80, 15, 30

		// Draw the lines
		for( size_t i = 0; i < linesP.size(); i++ )
		{
			Vec4i l = linesP[i];
			line( houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, 16);
		}

		#ifdef DEBUG
			houghP.copyTo(display_lines(rows,cols));
			imshow("Hough Lines", display_lines);
		#endif

/************************* Blob operations ******************************************/

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours( houghP, contours, hierarchy ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

		temp = Mat::zeros(houghP.size(), CV_8UC1);

		//Fill contours
		for (uint i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(255);
			drawContours( temp, contours, i, color, CV_FILLED );
		}

		//Negate image
		temp = 255 - temp;
		Mat edges = getNearestBlob(temp, SIZE_X, SIZE_Y, 250);

/********************* weighted average of the images intensities *************************/

		vector<Mat> planes;
		Mat old_image, new_image;

		cvtColor(display(rows,cols),old_image,CV_BGR2GRAY);
		//old_image = img.channel[0](rows,cols) * (255.0/180.0);
		equalizeHist(old_image,old_image);

		#ifdef DEBUG
			imshow("grayscale", src);
			otsu_road.copyTo(display_otsu(rows,cols));
			imshow("Otsu", display_otsu);
			edges.copyTo(display_edges(rows,cols));
			imshow("Edges", display_edges);
		#endif

		addWeighted(old_image, 1.0/3.0, otsu_road, 1.0/3.0, 0, new_image);
		addWeighted(new_image, 1.0, edges, 1.0/3.0, 0, new_image);

		Mat detected_road = Mat::zeros(display.size(), CV_8UC1);
		new_image.copyTo(detected_road(rows,cols));

		imshow("w road", detected_road);
		threshold( detected_road, detected_road, 125, 255, THRESH_BINARY);
		imshow("dirty road", detected_road);
		detected_road = getNearestBlob(detected_road, SIZE_X, SIZE_Y, 2500);
		imshow("detected road", detected_road);

/*********************** Keyboard *******************************************/

		while(!keep_running && !(rewind_frame || forward_frame))
		{
			waitKey(10);
		}

		if(rewind_frame)
		{
			if(keep_running)
			{
				rewind_frame = false;
				keep_running = false;
			}

			else
			{
				cap.set(CV_CAP_PROP_POS_FRAMES ,cap.get(CV_CAP_PROP_POS_FRAMES) - 5);
				keep_running = true;
			}
		}

		else if(forward_frame)
		{
			if(keep_running)
			{
				forward_frame = false;
				keep_running = false;
			}

			else
			{
				cap.set(CV_CAP_PROP_POS_FRAMES ,cap.get(CV_CAP_PROP_POS_FRAMES) + 5);
				keep_running = true;
			}
		}

		//while(!keep_running){}
		waitKey(50);
	}

    waitKey(100);

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	destroyAllWindows();

	return 0;
}

static void* userInput_thread(void*)
{
	while(true)
	{
		char cmd = getchar();
		getchar();

		if ( (cmd == 'P') || (cmd == ' ') || (cmd == 'p'))
		{
			//! desired user input 'q' received
			keep_running = !keep_running;
		}

		else if( cmd == 'f')
		{
			forward_frame = true;
		}

		else if( cmd == 'r')
		{
			rewind_frame = true;
		}
	}
}
