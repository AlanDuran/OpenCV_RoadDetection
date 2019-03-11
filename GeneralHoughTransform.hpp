/*
 * GeneralHoughTransform.hpp
 *
 *  Created on: 14 févr. 2014
 *      Author: jguillon
 */

#ifndef GENERALHOUGHTRANSFORM_HPP_
#define GENERALHOUGHTRANSFORM_HPP_

struct GHTPoint {
	double phi;
	double s;
	cv::Point y;
	double hits;
};

class GeneralHoughTransform {

protected:
	std::vector< std::vector<cv::Vec2f> > m_RTable;
	cv::Vec2f m_origin;
	cv::Mat m_templateImage;
	cv::Mat m_grayTemplateImage;
	cv::Mat m_template;
	int m_nScales;
	int m_nRotations;
	int m_nSlices;
	int m_cannyThreshold1;
	int m_cannyThreshold2;
	int m_maxValue;
	double m_minPositivesDistance;
	double m_deltaRotationAngle;
	double m_minRotationAngle;
	double m_maxRotationAngle;
	double m_deltaScaleRatio;
	double m_minScaleRatio;
	double m_maxScaleRatio;

private:
	void createRTable();
	void findOrigin();
	std::vector< std::vector<cv::Vec2f> > scaleRTable(const std::vector< std::vector<cv::Vec2f> >& RTable, double ratio);
	std::vector< std::vector<cv::Vec2f> > rotateRTable(const std::vector< std::vector<cv::Vec2f> >& RTable, double angle);
	void showRTable(std::vector< std::vector<cv::Vec2f> > RTable);

public:
	GeneralHoughTransform(const cv::Mat& templateBGR, const cv::Mat & templateImage, int type);
	void accumulate(const cv::Mat& image);
	void drawTemplate(cv::Mat& image, GHTPoint params);
	void setTemplate(const cv::Mat& templateBGR, const cv::Mat & templateImage);
	cv::Mat gradientDirection(const cv::Mat& src);
	void invertIntensities(const cv::Mat& src, cv::Mat& dst);
	float gradientDirection(const cv::Mat& src, int x, int y);
	float fastsqrt(float val);
	int rad2SliceIndex(double angle, int nSlices);

};

#endif /* GENERALHOUGHTRANSFORM_HPP_ */
