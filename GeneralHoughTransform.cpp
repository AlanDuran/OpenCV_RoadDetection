/*
 * GeneralHoughTransform.cpp
 *
 *  Created on: 14 févr. 2014
 *      Author: jguillon
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream> // For debugging

#include "GeneralHoughTransform.hpp"

#define DEBUG

using namespace cv;
using namespace std;

const double PI = 4.0*atan(1.0);

GeneralHoughTransform::GeneralHoughTransform(const Mat& templateBGR, const Mat & templateImage, int type) {
	/* Parameters to set */
	m_cannyThreshold1 = 10;
	m_cannyThreshold2 = 100;
	m_deltaScaleRatio = .1;
	m_minScaleRatio = 1.0;
	m_maxScaleRatio = 1.0;
	m_deltaRotationAngle = PI;
	m_minRotationAngle = 0;
	m_maxRotationAngle = 0;
	m_maxValue = (type) ? 179:255;
	/* Computed attributes */
	m_nRotations = (m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle + 1;
	m_nSlices = (2.0*PI) / m_deltaRotationAngle;
	m_nScales = (m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio + 1;

	setTemplate(templateBGR,templateImage);
}

void GeneralHoughTransform::setTemplate(const Mat& templateBGR, const Mat & templateImage) {
	templateBGR.copyTo(m_templateImage);
 	findOrigin();
	GaussianBlur(templateImage, m_template, Size(5,5), 0,0);
	Canny(m_template, m_template, m_cannyThreshold1, m_cannyThreshold2);
	createRTable();
}

void GeneralHoughTransform::findOrigin() {
	m_origin = Vec2f(m_templateImage.cols/2,m_templateImage.rows/2); // By default, the origin is at the center
	for(int j=0 ; j<m_templateImage.rows ; j++) {
		Vec3b* data= (Vec3b*)(m_templateImage.data + m_templateImage.step.p[0]*j);
		for(int i=0 ; i<m_templateImage.cols ; i++)
			if(data[i]==Vec3b(255,0,0)) { // If it's a red pixel...
				m_origin = Vec2f(i,j); // ...then it's the template's origin
			}
	}
}

void GeneralHoughTransform::createRTable() {
	int iSlice;
	double phi;

	Mat direction = gradientDirection(m_template);

	#ifdef DEBUG
		imshow("debug - template", m_template);
		imshow("debug - positive directions", direction);
	#endif

	m_RTable.clear();
	m_RTable.resize(m_nSlices);
	for(int y=0 ; y<m_template.rows ; y++) {
		uchar *templateRow = m_template.ptr<uchar>(y);
		double *directionRow = direction.ptr<double>(y);
		for(int x=0 ; x<m_template.cols ; x++) {
			if(templateRow[x] >= m_maxValue) {
				phi = directionRow[x]; // gradient direction in radians in [-PI;PI]
				iSlice = rad2SliceIndex(phi,m_nSlices);
				m_RTable[iSlice].push_back(Vec2f(m_origin[0]-x, m_origin[1]-y));
			}
		}
	}
}

vector< vector<Vec2f> > GeneralHoughTransform::scaleRTable(const vector< vector<Vec2f> >& RTable, double ratio) {
	vector< vector<Vec2f> > RTableScaled(RTable.size());
	for(vector< vector<Vec2f> >::size_type iSlice=0 ; iSlice<RTable.size() ; iSlice++) {
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size() ; ir++) {
			RTableScaled[iSlice].push_back(Vec2f(ratio*RTable[iSlice][ir][0], ratio*RTable[iSlice][ir][1]));
		}
	}
	return RTableScaled;
}

vector< vector<Vec2f> > GeneralHoughTransform::rotateRTable(const vector< vector<Vec2f> >& RTable, double angle) {
	vector< vector<Vec2f> > RTableRotated(RTable.size());
	double c = cos(angle);
	double s = sin(angle);
	int iSliceRotated;
	for(vector< vector<Vec2f> >::size_type iSlice = 0 ; iSlice<RTable.size() ; iSlice++) {
		iSliceRotated = rad2SliceIndex(iSlice*m_deltaRotationAngle + angle, m_nSlices);
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size(); ir++) {
			RTableRotated[iSliceRotated].push_back(Vec2f(c*RTable[iSlice][ir][0] - s*RTable[iSlice][ir][1], s*RTable[iSlice][ir][0] + c*RTable[iSlice][ir][1]));
		}
	}
	return RTableRotated;
}

void GeneralHoughTransform::showRTable(vector< vector<Vec2f> > RTable) {
	int N(0);
	cout << "--------" << endl;
	for(vector< vector<Vec2f> >::size_type i=0 ; i<RTable.size() ; i++) {
		for(vector<Vec2f>::size_type j=0 ; j<RTable[i].size() ; j++) {
			cout << RTable[i][j];
			N++;
		}
		cout << endl;
	}
	cout << N << " elements" << endl;
}

void GeneralHoughTransform::accumulate(const Mat& image) {
	/* Image preprocessing */
	Mat edges(image.size(), CV_8UC1);
	GaussianBlur(image, edges, Size(5,5), 0,0);
	Canny(edges, edges, m_cannyThreshold1, m_cannyThreshold2);
	Mat direction = gradientDirection(edges);

	#ifdef DEBUG
		imshow("debug - src edges", edges);
		imshow("debug - src edges gradient direction", direction);
		waitKey(1000);
	#endif

	/* Accum size setting */
	int X = image.cols;
	int Y = image.rows;
	int S = ceil((m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio) + 1; // Scale Slices Number
	int R = ceil((m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle) + 1; // Rotation Slices Number

	/* Usefull variables declaration */
	vector< vector< Mat > > accum(R,vector<Mat>(S, Mat::zeros(Size(X,Y),CV_64F)));
	Mat totalAccum = Mat::zeros(Size(X,Y),CV_32S);
	int iSlice(0), iScaleSlice(0), iRotationSlice(0), ix(0), iy(0);
	double max(0.0), phi(0.0);
	vector< vector<Vec2f> > RTableRotated(m_RTable.size()), RTableScaled(m_RTable.size());
	Mat showAccum(Size(X,Y),CV_8UC1);
	vector<GHTPoint> points;
	GHTPoint point;

	/* Main loop */
	for(double angle=m_minRotationAngle ; angle<=m_maxRotationAngle+0.0001 ; angle+=m_deltaRotationAngle) { // For each rotation (0.0001 double comparison)
		iRotationSlice = round((angle-m_minRotationAngle)/m_deltaRotationAngle);
		cout << "Rotation Angle\t: " << angle/PI*180 << "°" << endl;
		RTableRotated = rotateRTable(m_RTable,angle);
		for(double ratio=m_minScaleRatio ; ratio<=m_maxScaleRatio+0.0001 ; ratio+=m_deltaScaleRatio) { // For each scaling (0.0001 double comparison)
 			iScaleSlice = round((ratio-m_minScaleRatio)/m_deltaScaleRatio);
			cout << "|- Scale Ratio\t: " << ratio*100 << "%" << endl;
			RTableScaled = scaleRTable(RTableRotated,ratio);
			accum[iRotationSlice][iScaleSlice] = Mat::zeros(Size(X,Y),CV_64F);
			max = 0;
			for(int y=0 ; y<image.rows ; y++) {
				for(int x=0 ; x<image.cols ; x++) {
					phi = direction.at<double>(y,x);
					if(phi != 0.0) {
						iSlice = rad2SliceIndex(phi,m_nSlices);
						for(vector<Vec2f>::size_type ir=0 ; ir<RTableScaled[iSlice].size() ; ir++) { // For each r related to this angle-slice
							ix = x + round(RTableScaled[iSlice][ir][0]);	// We compute x+r, the supposed template origin position
							iy = y + round(RTableScaled[iSlice][ir][1]);
							if(ix>=0 && ix<image.cols && iy>=0 && iy<image.rows) { // If it's between the image boundaries
								totalAccum.at<int>(iy,ix)++;
								if(++accum[iRotationSlice][iScaleSlice].at<double>(iy,ix) > max) { // Icrement the accum
									max = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix);
									point.phi = angle;
									point.s = ratio;
									point.y.y = iy;
									point.y.x = ix;
									point.hits = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix);
								}
								/* To see the step-by-step accumulation uncomment these lines */
								// normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, 255, NORM_MINMAX, CV_8UC1);
								// imshow("debug - subaccum", showAccum);	waitKey(1);
							}
						}
					}
				}
			}
			/* Pushing back the best point for each transformation (uncomment line 159 : "max = 0") */
			points.push_back(point);
			/* Transformation accumulation visualisation */
			normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, m_maxValue, NORM_MINMAX, CV_8UC1); // To see each transformation accumulation (uncomment line 159 : "max = 0")
			// normalize(totalAccum, showAccum, 0, 255, NORM_MINMAX, CV_8UC1); // To see the cumulated accumulation (comment line 159 : "max = 0")
			imshow("debug - accum", showAccum);	waitKey(1);
			// blur(accum[iRotationSlice][iScaleSlice], accum[iRotationSlice][iScaleSlice], Size(3,3)); // To harmonize the local maxima
		}
	}
	/* Pushing back the best point for cumulated transformations (comment line 159 : "max = 0") */
	// points.push_back(point);

	/* Drawing templates on best points */
	for(vector<GHTPoint>::size_type i=0 ; i<points.size() ; i++) {
		Mat out(image.size(),image.type(),Scalar(0,0,0));
		drawTemplate(out, points[i]);
		imshow("debug - output", out);
		waitKey(0);
	}
}

void GeneralHoughTransform::drawTemplate(Mat& image, GHTPoint params) {
	cout << params.y << " avec un rapport de grandeur de " << params.s << " et une rotation de " << params.phi/PI*180 << "° et avec " << params.hits << " !" << endl;
	double c = cos(params.phi);
	double s = sin(params.phi);
	int x(0), y(0), relx(0), rely(0);
	for(vector< vector<Vec2f> >::size_type iSlice = 0 ; iSlice<m_RTable.size() ; iSlice++)
		for(vector<Vec2f>::size_type ir=0 ; ir<m_RTable[iSlice].size() ; ir++) {
			relx = params.s * (c*m_RTable[iSlice][ir][0] - s*m_RTable[iSlice][ir][1]); // X-Coordinate of the template's point after transformation (relative to the origin)
			rely = params.s * (s*m_RTable[iSlice][ir][0] + c*m_RTable[iSlice][ir][1]); // Y-Coordinate of the template's point after transformation (relative to the origin)
			x = image.cols - params.y.x - relx; // X-Coordinate of the template's point in the image
			y = image.rows - params.y.y - rely; // Y-Coordinate of the template's point in the image
			if(x>=0 && x<image.cols && y>=0 && y<image.rows)
				image.at<Vec3b>(y,x) = Vec3b(255,255,255); // Put the pixel in green
		}
}

int GeneralHoughTransform::rad2SliceIndex(double angle, int nSlices) {
	double a = (angle > 0) ? (fmodf(angle,2*PI)) : (fmodf(angle+2*PI,2*PI));
	return floor( a / (2*PI/nSlices + 0.00000001) );
}

float GeneralHoughTransform::gradientDirection(const Mat& src, int x, int y) {
	int gx,gy;
	if(x==0)				gx = src.at<uchar>(y,x+1) - src.at<uchar>(y,x);
	else if(x==src.cols-1)	gx = src.at<uchar>(y,x) - src.at<uchar>(y,x-1);
	else					gx = src.at<uchar>(y,x+1) - src.at<uchar>(y,x-1);
	if(y==0)				gy = src.at<uchar>(y+1,x) - src.at<uchar>(y,x);
	else if(y==src.rows-1)	gy = src.at<uchar>(y,x) - src.at<uchar>(y-1,x);
	if(y==0)				gy = src.at<uchar>(y+1,x) - src.at<uchar>(y-1,x);
	return atan2(gx,gy);
}

Mat GeneralHoughTransform::gradientDirection(const Mat& src) {
	Mat dst(src.size(),CV_64F);
//	Mat gradX = gradientX(src);
//	Mat gradY = gradientY(src);
	Mat gradX(src.size(), CV_64F);
	Sobel(src, gradX, CV_64F, 1,0,5);
	Mat gradY(src.size(), CV_64F);
	Sobel(src, gradY, CV_64F, 0,1,5);
	double t;
	for(int y=0 ; y<gradX.rows ; y++)
		for(int x=0 ; x<gradX.cols ; x++) {
			t = atan2(gradY.at<double>(y,x), gradX.at<double>(y,x));
			dst.at<double>(y,x) = (t == 180) ? 0 : t;
		}
	return dst;
}

void GeneralHoughTransform::invertIntensities(const Mat& src, Mat& dst) {
	for(int i=0 ; i<src.rows ; i++)
		for(int j=0 ; j<src.cols ; j++)
			dst.at<uchar>(i,j) = m_maxValue - src.at<uchar>(i,j);
}

float GeneralHoughTransform::fastsqrt(float val) {
	int tmp = *(int *)&val;
	tmp -= 1<<23;
	tmp = tmp >> 1;
	tmp += 1<<29;
	return *(float *)&tmp;
}
