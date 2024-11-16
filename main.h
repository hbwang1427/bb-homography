#ifndef _MAIN_H
#define _MAIN_H

#include "opencv2/objdetect/objdetect.hpp"   
#include "opencv2/features2d/features2d.hpp"   
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/calib3d/calib3d.hpp"   
#include "opencv2/imgproc/imgproc_c.h"   
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/highgui/highgui.hpp>

#include <string>  
#include <vector>  
#include <iostream>  
#include <fstream>
#include <map>
#include <math.h>
#include <direct.h> // for creating files

#include "homography.h"
#include "homography_estimator.h"

using namespace cv;  
using namespace std;   

struct for_row_column{
	int model_row;
	int model_col;	
	unsigned int score;
};

#define MAX_MODEL_POINTS 50
#define LEAST_CORRESPONDENCE 4
#define LEAST_NUMBER_INLIERS 10
const float REPROJ_THRESHOLD = 3.0;

// function declaration
unsigned int hamdist(unsigned char* a, unsigned char* b, size_t size);

bool operator<(const for_row_column a, const for_row_column b);

void Normalize_Rows_Cols(Mat &matrixA);

float Abs_DifferenceValue_TwoMatrix(Mat &A, Mat &B);

void AssignMat(Mat &A, Mat &B);

void Normalize_With_Sinkhorn(Mat &matrixA);

float l2NormValue(Mat &A, Mat &B);

void Find_Matching_greedy_discretization(Mat &first_order_matrix, vector<DMatch> &matches, 
										 vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
										 Mat &descriptorsA, Mat &descriptorsB,const int frame_num);

bool Probabilistic_Graph_Matching_Method(vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
										  const vector<DMatch>matches,vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB, 
										  Mat &descriptorsA, Mat &descriptorsB,vector<DMatch>&final_matches,
										  const int frame_num,Mat &homo, int &final_inliers_numbers);

void Obtain_KNN_From_Potential_Points(const int K, const vector<DMatch> matches,vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB, 
									  Mat &descriptorsA, Mat &descriptorsB, vector<for_row_column> &result,
									  vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,vector<int>&record_times);

void NearestNeighbor(const int K,const int cur,const vector<DMatch> matches, Mat &descriptorsA, Mat &descriptorsB, vector<for_row_column> &result,
					 vector<pair<int,int>>&row_real_index, vector<pair<int,int>>&column_real_index, vector<int>&cal_times);

void Normalize_Rows_Of_Matrix(Mat &matrixA);

void Graph_With_Binary_Features(const string &model_image,const string &video_file,
								const string &featureName,const bool Graph_Flag);

void ORB_Graph(const string &model_image,const string &video_file,const bool Graph_Flag,const string &featureName);

void BRIEF_Graph(const string &model_image,const string &video_file,const bool Graph_Flag,const string &featureName);

void BRISK_Graph(const string &model_image,const string &video_file,const bool Graph_Flag,const string &featureName);

void FREAK_Graph(const string &model_image,const string &video_file,const bool Graph_Flag,const string &featureName);

bool Original_ORB(Mat &descriptorsA, Mat &descriptorsB,vector<KeyPoint>&keypointsA, 
				  vector<KeyPoint>&keypointsB, vector<DMatch>&final_matches,
				  Mat &homo, int &final_inliers_numbers);

bool Original_BRIEF(Mat&descriptorsA, Mat &descriptorsB,vector<KeyPoint>&keypointsA, 
					vector<KeyPoint>&keypointsB,vector<DMatch>&final_matches,Mat &homo, int &final_inliers_numbers);

bool Original_BRISK(Mat&descriptorsA, Mat &descriptorsB,
					vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
					vector<DMatch>&final_matches,Mat &homo, int &final_inliers_numbers);

bool Original_FREAK(Mat&descriptorsA, Mat &descriptorsB,vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
					vector<DMatch>&final_matches,Mat &homo, int &final_inliers_numbers);

void Soft_Threshold(Mat &matrixA,float lambda);

void Obtain_Reprojected_Error(vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
							  vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
							  vector<for_row_column>&result, Mat &homo, Mat &reproj_error);

void main_run(const string &model_image, const string &video_file,const bool Graph_Flag,const int binary_method);


bool My_PROSAC(Mat &descriptorsA, Mat &descriptorsB,
			   vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
			   Mat &homo,vector<DMatch>&final_matches, int &final_inliers_numbers);

bool Check_Valid_Match(const vector<DMatch> final_matches);


#endif