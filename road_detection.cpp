/*
 * Image.cpp
 *
 *  Created on: 20 feb 2019
 *      Author: Alan Duran
 */

/******* Includes *************************/
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include "otsu.h"
#include "utils.h"

/***** Definitions ****************************/
#define DEBUG
#define FIND_HORIZON	1
#define HORIZON_SIZE	0.7
#define FILL_LINES		0

#define KERNEL_WIDTH	7
#define KERNEL_HEIGHT	7
#define SIGMA_X 		0
#define SIGMA_Y			0
#define SIZE_X			353
#define SIZE_Y			200

/*
 * iteso.mp4 	--> 10 50
 * base_aerea 	--> 75 200
 * atras_iteso 	--> 10 50
 */
#define CANNY_LOW		10
#define CANNY_HIGH		50
#define CANNY_KERNEL	15

/******** Prototypes *******************************/
void callBackFunc(int event, int x, int y, int flags, void* userdata);

/*****************************************************/
using namespace std;
using namespace cv;

static bool keep_running = true;
static bool rewind_frame = false;
static bool forward_frame = false;

string winName = "GUI v0.1";
Rect pauseButton, forwardButton, rewindButton;
Mat canvas(Size(SIZE_X, SIZE_Y),CV_8UC3,Scalar(0,255,0));

//Read video
VideoCapture cap("data/iteso.mp4");

int main( int argc, char** argv )
{
    /********************* GUI initialization **********************************************/
    pauseButton = Rect(0,0,SIZE_X, 50);
    forwardButton = Rect(0,75,SIZE_X, 50);
    rewindButton = Rect(0,150,SIZE_X, 50);

	// Draw the buttons
    Mat(pauseButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(pauseButton));
    Mat(forwardButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(forwardButton));
    Mat(rewindButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(rewindButton));

    putText(canvas, "Pausa", Point(pauseButton.width*0.35, pauseButton.height*0.7),
    		FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 8);
    putText(canvas, "Avanzar", Point(forwardButton.width*0.30, forwardButton.height*0.7 + forwardButton.y),
        		FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 8);
    putText(canvas, "Regresar", Point(rewindButton.width*0.30, rewindButton.height*0.7 + rewindButton.y),
            		FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 8);

    // Setup callback function
	namedWindow(winName);
	setMouseCallback(winName, callBackFunc);

    imshow(winName,canvas);

    /********************** Load image and pre-processing ******************************/

	if(!cap.isOpened()){
	    cout << "Error opening video stream or file" << endl;
	    return -1;
	}

	while(true)
	{
		Mat src, dst, temp, display, srcBGR;
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
		cvtColor(display,srcBGR,CV_BGR2GRAY);

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
			Rect horizon_area(src.cols * 0.43, 0, src.cols * 0.14, src.rows * HORIZON_SIZE);
			temp = img.channel[0](horizon_area).clone();
			GaussianBlur( temp, temp, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);
			imshow("sss",temp);
			horizon = get_horizon(temp);
			horizon = src.rows - horizon - src.rows * (1 - HORIZON_SIZE);
		}

		else
		{
			horizon = src.rows * 0.35;
		}

		temp = display.clone();
		line( temp, Point(0,horizon), Point(src.cols,horizon),Scalar( 0, 0, 255 ), 2, 1);
		imshow("original_image", temp);

/******************** Road detection with Otsu **************************************/

		Mat otsu_road;

		//Cut the image from horizon to the bottom
		rows.start = horizon;
		rows.end = src.rows;

		img.img = dst(rows,cols).clone();
		img.hist = getHistogram(img.img);
		otsu_road = get_roadImage(&img);

/******** Improvements with Canny detector and Hough Line transform *****************/

		Mat houghP;

		houghP = srcBGR(rows,cols).clone();
		GaussianBlur( houghP, houghP, Size(CANNY_KERNEL, CANNY_KERNEL), 0, 0);

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

		HoughLinesP(houghP, linesP, 10, CV_PI/180, 80, 20, 30); // runs the actual detection 10, CV_PI/180, 80, 20, 30

		float slopes[linesP.size()];
		int slopeCount[linesP.size()];
		int slopeIndex = 0;

		for( size_t i = 0; i < linesP.size(); i++ )
		{
			Vec4i l = linesP[i];

			float currSlope = (l[3] - l[1]) / ( l[2] - l[0] + 0.001);

			if(slopeIndex == 0)
			{
				slopes[0] = currSlope;
				slopeCount[0] = 1;
				slopeIndex++;
			}

			else
			{
				int index = 0;

				for(index = 0; index <= slopeIndex; index++)
				{
					if(currSlope < (slopes[index] + 5) && currSlope > (slopes[index] - 5))
					{
						slopeCount[index]++;
						break;
					}
				}

				if(index > slopeIndex)
				{
					slopeIndex++;
					slopes[slopeIndex] = currSlope;
					slopeCount[slopeIndex] = 1;
				}
			}
		}


		for( size_t i = 0; i < linesP.size(); i++ )
		{
			Vec4i l = linesP[i];
			line( houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, 16);

			if(l[2] > houghP.cols * 0.65 && FILL_LINES)
			{
				line( houghP, Point(l[0], l[1]), Point(houghP.cols, l[1]), Scalar(255), 1, 16);
				line( houghP, Point(l[2], l[3]), Point(houghP.cols, l[3]), Scalar(255), 1, 16);
				line( houghP, Point(houghP.cols, l[1]), Point(houghP.cols, l[3]), Scalar(255), 1, 16);
			}

			else if(l[2] < houghP.cols * 0.35 && FILL_LINES)
			{
				line( houghP, Point(l[0], l[1]), Point(0, l[1]), Scalar(255), 1, 16);
				line( houghP, Point(l[2], l[3]), Point(0, l[3]), Scalar(255), 1, 16);
				line( houghP, Point(0, l[1]), Point(0, l[3]), Scalar(255), 1, 16);
			}

			/*
			if(!keep_running)
			{
			imshow("liddnn",houghP);
			waitKey(0);
			}
			*/
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
			/*
			if(!keep_running)
			{
			imshow("couttt",temp);
			waitKey(0);
			}
			*/
		}

		//Negate image
		temp = 255 - temp;
		Mat edges = getNearestBlob(temp, src.cols, SIZE_Y, 250);

/********************* weighted average of the images intensities *************************/

		vector<Mat> planes;
		Mat old_image, new_image;

		old_image = srcBGR.clone();

		equalizeHist(old_image,old_image);

		#ifdef DEBUG
			imshow("grayscale", old_image);
			otsu_road.copyTo(display_otsu(rows,cols));
			imshow("Otsu", display_otsu);
			edges.copyTo(display_edges(rows,cols));
			imshow("Edges", display_edges);
		#endif

		addWeighted(old_image(rows,cols), 1.0/3.0, otsu_road, 1.0/3.0, 0, new_image);
		addWeighted(new_image, 1.0, edges, 1.0/3.0, 0, new_image);

		Mat detected_road = Mat::zeros(display.size(), CV_8UC1);
		new_image.copyTo(detected_road(rows,cols));

		imshow("w road", detected_road);
		threshold( detected_road, detected_road, 125, 255, THRESH_BINARY);
		detected_road = getNearestBlob(detected_road, src.cols, SIZE_Y, 2500);
		imshow("detected road", detected_road);

/*********************** GUI events *******************************************/

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

		waitKey(30);
	}

    waitKey(100);

	// When everything done, release the video capture object
	cap.release();
	// Closes all the frames
	destroyAllWindows();

	return 0;
}

void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        if (pauseButton.contains(Point(x, y)))
        {
        	keep_running = !keep_running;
        }

        else if (forwardButton.contains(Point(x, y)))
        {
			forward_frame = true;
        }

        else if (rewindButton.contains(Point(x, y)))
        {
			rewind_frame = true;
        }
    }

    imshow(winName, canvas);
    waitKey(1);
}
