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
#define DEBUG				2 // 0 -> no debug, 1 -> only output, 2 -> full debug
#define WRITE_VIDEO			1
#define GUI					0
#define FIND_HORIZON		1
#define HORIZON_SIZE		0.6 //0.60
#define FILL_LINES			1

#define CAMERA				1
#define CAPTURE_WIDTH		3280
#define CAPTURE_HEIGHT		2464
#define FRAMERATE			10
#define FLIP_METHOD			0

#define KERNEL_WIDTH		7
#define KERNEL_HEIGHT		7
#define SIGMA_X 			0
#define SIGMA_Y				0
#define SIZE_Y				200

#define CANNY_LOW			5
#define CANNY_HIGH			50

/*
 * iteso.mp4 				--> 15
 * iteso2.mp4 				--> 15
 * base_aerea_atardecer 	--> 7
 * base_aerea_dia 			--> 11
 * atras_iteso 				--> 11
 * carretera 				--> 11
 */
#define CANNY_KERNEL			15

/************************ Prototypes ***********************************************************/
void menuCallBackFunc(int event, int x, int y, int flags, void* userdata);
void frameCallBackFunc(int event, int x, int y, int flags, void* userdata);

/********************* Memory reservations *****************************************************/
using namespace std;
using namespace cv;

static bool keep_running = true;
static bool rewind_frame = false;
static bool forward_frame = false;

string filename = "iteso.mp4";
string winName = "GUI v0.1";
string outWinName = "Out";

#if CAMERA
	Size frame_size(CAPTURE_WIDTH*((float)SIZE_Y / CAPTURE_HEIGHT), SIZE_Y);
	int framerate = FRAMERATE;

	string pipeline = gstreamer_pipeline(CAPTURE_WIDTH,
				CAPTURE_HEIGHT,
				CAPTURE_WIDTH,
				CAPTURE_HEIGHT,
				framerate,
				FLIP_METHOD);

	 VideoCapture cap(pipeline, 1800); //CAP_GSTREAMER
#else
	//Read video
	VideoCapture cap("data/" + filename);
	Size frame_size(cap.get(CV_CAP_PROP_FRAME_WIDTH)*(SIZE_Y /cap.get(CV_CAP_PROP_FRAME_HEIGHT)), SIZE_Y);
	int framerate = cap.get(CV_CAP_PROP_FPS);
#endif

Rect pauseButton, forwardButton, rewindButton;
Mat canvas(Size(350, 200), CV_8UC3,Scalar(0,255,0));
Mat out_frame(Size(3*frame_size.width, 3*SIZE_Y + 90), CV_8UC3,Scalar(0,0,0));

#if WRITE_VIDEO
	VideoWriter video(filename.substr(0,filename.size() - 4) + ".avi",CV_FOURCC('M','J','P','G'), framerate,
			Size(3*frame_size.width, 3*SIZE_Y + 90), 1);
#endif

/********************************************************************************************/

int main( int argc, char** argv )

{
    /********************* GUI initialization **********************************************/
	#if DEBUG >= 2
		putText(out_frame, "Original             Homogeneous grayscale               Otsu", Point(135,20),
				FONT_HERSHEY_DUPLEX, 0.75, Scalar(255,255,255), 1,32);
		putText(out_frame, "Canny                    Hough Lines                 Hough Road", Point(140, SIZE_Y + 50),
				FONT_HERSHEY_DUPLEX, 0.75, Scalar(255,255,255), 1, 32);
		putText(out_frame, "Equalized grayscale              Yellow area                 Detected road", Point(55, SIZE_Y*2 + 80),
				FONT_HERSHEY_DUPLEX, 0.75, Scalar(255,255,255), 1, 32);

		namedWindow(outWinName);
	#endif

	#if GUI
		// Setup callback function
		pauseButton = Rect(0,0, 350, 50);
		forwardButton = Rect(0,75, 350, 50);
		rewindButton = Rect(0,150, 350, 50);

		// Draw the buttons
		Mat(pauseButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(pauseButton));
		Mat(forwardButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(forwardButton));
		Mat(rewindButton.size(),CV_8UC3,Scalar(200,200,200)).copyTo(canvas(rewindButton));

		putText(canvas, "Pausa", Point(pauseButton.width*0.35, pauseButton.height*0.7),
				FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 16);
		putText(canvas, "Avanzar", Point(forwardButton.width*0.30, forwardButton.height*0.7 + forwardButton.y),
				FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 16);
		putText(canvas, "Regresar", Point(rewindButton.width*0.30, rewindButton.height*0.7 + rewindButton.y),
				FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,0), 1, 16);

		namedWindow(winName);
		setMouseCallback(winName, menuCallBackFunc);
		imshow(winName,canvas);
	#endif

    /********************** Load image and pre-processing ******************************/
	if(!cap.isOpened()){
	    cout << "Error opening video stream or camera" << endl;
	    return -1;
	}


	while(true)
	{
		clock_t start, end;

		start = clock();

		Mat src, dst, temp, display, srcBGR, srcHSV;
		img_type img;

		// Capture frame-by-frame
		#if CAMERA
			cap.read(src);
		#else
			cap >> src;
		#endif

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

		#if DEBUG >= 2 | WRITE_VIDEO
			Mat display_otsu = Mat::zeros(display.size(), CV_8UC1);
			Mat display_canny = Mat::zeros(display.size(), CV_8UC1);
			Mat display_lines = Mat::zeros(display.size(), CV_8UC1);
			Mat display_edges = Mat::zeros(display.size(), CV_8UC1);
		#endif

		//Change color space and remove shadows
		src = removeShadows(src,&img);

		//Smoothing image with Gaussian filter
		GaussianBlur(src, dst, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);

		//Change color space and store channels
		cvtColor(display, srcHSV, CV_BGR2HSV);
		GaussianBlur(srcHSV, srcHSV, Size(KERNEL_WIDTH, KERNEL_HEIGHT), SIGMA_X, SIGMA_Y);

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
			horizon = get_horizon(temp);
			horizon = src.rows - horizon - src.rows * (1 - HORIZON_SIZE);
		}

		else
		{
			horizon = src.rows * 0.35;
		}

		Mat display_original;
		display_original = display.clone();
		line( display_original, Point(0,horizon), Point(src.cols,horizon),Scalar( 0, 0, 255 ), 2, 1);

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

		#if DEBUG >= 2 | WRITE_VIDEO
			houghP.copyTo(display_canny(rows,cols));
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

		HoughLinesP(houghP, linesP, 10, CV_PI/180, 30, 1, 10); // runs the actual detection 10, CV_PI/180, 80, 20, 30

		for( size_t i = 0; i < linesP.size(); i++ )
		{
			Vec4i l = linesP[i];
			line( houghP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, 16);

			double size = (l[0] - l[2])*(l[0] - l[2]) + (l[1] - l[3])*(l[1] - l[3]);
			double slope = (l[3] - l[1]) / ( l[2] - l[0] + 0.0001);

			//printf("size %lf, slope %lf\n", size, slope);

			/*
			 * Check the size, position and slope of the line to define if it's part of the road
			 * and if so, form a contour with it and the sides of the image
			 */
			if((l[0] > houghP.cols * 0.75 || l[2] > houghP.cols * 0.75)
					&& slope > 0.2 && size > 500 && FILL_LINES)
			{
				line( houghP, Point(l[0], l[1]), Point(houghP.cols, l[1]), Scalar(255), 1, 16);
				line( houghP, Point(l[2], l[3]), Point(houghP.cols, l[3]), Scalar(255), 1, 16);
				line( houghP, Point(houghP.cols, l[1]), Point(houghP.cols, l[3]), Scalar(255), 1, 16);
			}

			else if((l[0] < houghP.cols * 0.25 || l[2] < houghP.cols * 0.25)
				&& slope < -0.2 &&  size > 500 && FILL_LINES)
			{
				line( houghP, Point(l[0], l[1]), Point(0, l[1]), Scalar(255), 1, 16);
				line( houghP, Point(l[2], l[3]), Point(0, l[3]), Scalar(255), 1, 16);
				line( houghP, Point(0, l[1]), Point(1, l[3]), Scalar(255), 1, 16);
			}
		}

		#if DEBUG >= 2 | WRITE_VIDEO
			houghP.copyTo(display_lines(rows,cols));
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
		Mat edges = getNearestBlob(temp, src.cols, SIZE_Y, 1500);

/********************* weighted average of the images intensities *************************/

		vector<Mat> planes;
		Mat old_image, new_image;

		old_image = srcBGR.clone();

		equalizeHist(old_image,old_image);

		addWeighted(old_image(rows,cols), 1.0/3.0, otsu_road, 1.0/3.0, 0, new_image);
		addWeighted(new_image, 1.0, edges, 1.0/3.0, 0, new_image);

		Mat w_road = Mat::zeros(display.size(), CV_8UC1);
		Mat thres_road = Mat::zeros(display.size(), CV_8UC1);
		new_image.copyTo(w_road(rows,cols));

		threshold( w_road, thres_road, 125, 255, THRESH_BINARY);
		thres_road = getNearestBlob(thres_road, src.cols, SIZE_Y, 1500);

/************************** Color segmentation ********************************************/

		Mat hsv_out, detected_road;

		thres_road.copyTo(detected_road);
		inRange(srcHSV,Scalar(20, 50, 0), Scalar(30, 255, 255),hsv_out);
		hsv_out = 255 - hsv_out;
		detected_road = detected_road & hsv_out;
		detected_road = getNearestBlob(detected_road, src.cols, SIZE_Y, 500);

/***************** Print and save output **************************************************/
		#if DEBUG >= 2 | WRITE_VIDEO
			otsu_road.copyTo(display_otsu(rows,cols));
			edges.copyTo(display_edges(rows,cols));

			vector<Mat> out_img;

			out_img.push_back(display_original);
			out_img.push_back(src);
			out_img.push_back(display_otsu);

			out_img.push_back(display_canny);
			out_img.push_back(display_lines);
			out_img.push_back(display_edges);

			out_img.push_back(old_image);
			out_img.push_back(hsv_out);
			out_img.push_back(detected_road);

			for(unsigned int i = 0; i < 3; i++)
			{
				for(unsigned int j = 0; j < 3; j++)
				{
					Mat frame = out_img[3*i + j];
					rows = Range(display.rows * i + (30 * (i+1)),display.rows * (i+ 1) + (30 * (i+1)));
					cols = Range(display.cols * j, display.cols * (j + 1));

					if(frame.type() == CV_8UC1)
					{
						cvtColor(frame,frame, CV_GRAY2BGR);
					}

					frame.copyTo(out_frame(rows,cols));
				}

			}

			imshow(outWinName,out_frame);

			#if WRITE_VIDEO
				video.write(out_frame);
			#endif

		#elif DEBUG == 1
			imshow("Original Image",display_original);
			imshow("Detected road",detected_road);
		#endif

		end = clock();

		// Calculating total time taken by the program.
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		printf("Time taken by program is : %f sec", time_taken);

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

		waitKey(10);
	}

    waitKey(100);

	// When everything done, release the video capture object
	cap.release();

	#if WRITE_VIDEO
	video.release();
	#endif

	// Closes all the frames
	destroyAllWindows();

	return 0;
}

void menuCallBackFunc(int event, int x, int y, int flags, void* userdata)
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

    waitKey(1);
}

void frameCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        keep_running = !keep_running;
    }

    waitKey(1);
}
