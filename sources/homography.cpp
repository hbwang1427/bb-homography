#include "homography.h"
#include "homography_estimator.h"

homography::homography(void)
{
	initialize();
}

homography::homography(const float h11, const float h12, const float h13,
						   const float h21, const float h22, const float h23,
						   const float h31, const float h32, const float h33)
{
	initialize();

	Set(0, 0, h11); Set(0, 1, h12); Set(0, 2, h13);
	Set(1, 0, h21); Set(1, 1, h22); Set(1, 2, h23);
	Set(2, 0, h31); Set(2, 1, h32); Set(2, 2, h33);
}

homography::homography(const double h11, const double h12, const double h13,
						   const double h21, const double h22, const double h23,
						   const double h31, const double h32, const double h33)
{
	initialize();

	Set(0, 0, h11); Set(0, 1, h12); Set(0, 2, h13);
	Set(1, 0, h21); Set(1, 1, h22); Set(1, 2, h23);
	Set(2, 0, h31); Set(2, 1, h32); Set(2, 2, h33);

}

void homography::initialize(void)
{
	homo = Mat::zeros(3,3,CV_32FC1);		
}

homography::~homography()
{	

}

void homography::transform_point(const int u, const int v, int & up, int & vp)
{
	float inv_k = 1.f / (Get(2, 0) * u + Get(2, 1) * v + Get(2, 2));
	up = int(inv_k * (Get(0, 0) * u + Get(0, 1) * v + Get(0, 2)) + 0.5);
	vp = int(inv_k * (Get(1, 0) * u + Get(1, 1) * v + Get(1, 2)) + 0.5);
}

void homography::transform_point(float u, float v, float & up, float & vp)
{
	float inv_k = 1.f / (Get(2, 0) * u + Get(2, 1) * v + Get(2, 2));
	up = float(inv_k * (Get(0, 0) * u + Get(0, 1) * v + Get(0, 2)) + 0.5);
	vp = float(inv_k * (Get(1, 0) * u + Get(1, 1) * v + Get(1, 2)) + 0.5);
}

void homography::transform_point(double u, double v, double & up, double & vp)
{
	double inv_k = 1. / (Get(2, 0) * u + Get(2, 1) * v + Get(2, 2));
	up = double(inv_k * (Get(0, 0) * u + Get(0, 1) * v + Get(0, 2)) + 0.5);
	vp = double(inv_k * (Get(1, 0) * u + Get(1, 1) * v + Get(1, 2)) + 0.5);
}

float homography::Get(const int i, const int j)
{
	return homo.at<float>(i,j);
}

void  homography::Set(const int i, const int j, const float val)
{
	homo.at<float>(i,j) = val;
}

void  homography::Set(const int i, const int j, const double val)
{
	homo.at<float>(i,j) = (float)val;
}