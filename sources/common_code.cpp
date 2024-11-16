#include "main.h"

unsigned int hamdist(unsigned char* a, unsigned char* b, size_t size)   
{   
	HammingLUT lut;    
	unsigned int result;  
	result = lut((a), (b), size);   // obtain the hamming distance
	return result;   
}    

// This function is used for sort function
bool operator<(const for_row_column a, const for_row_column b) 
{
	if(a.model_row == b.model_row)  
		return a.score < b.score;
	else
		return a.score < b.score;
}

void Normalize_Rows_Cols(Mat &matrixA)
{	
	int rows = matrixA.rows;
	int cols = matrixA.cols;
	float temp = 0;
	float *pix;
	int step0 = matrixA.step[0],step1 = matrixA.step[1];

	// rows normalization
	for (int i=0;i<rows;i++)
	{
		temp = 0;
		for (int j=0;j<cols;j++)
		{
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			temp = temp + *pix;			
		}		
		for (int j=0;j<cols;j++)
		{			
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			*pix = (*pix)/(temp+0.000000001);						
		}
	}
	// cols normalization
	for (int j=0;j<cols;j++)
	{
		temp = 0;
		for (int i=0;i<rows;i++)
		{
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			temp = temp + *pix;
		}
		for (int i=0;i<rows;i++)
		{
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			*pix = (*pix)/(temp+0.000000001);			
		}
	}

}
float Abs_DifferenceValue_TwoMatrix(Mat &A, Mat &B)
{
	float result = 0;
	int rows = A.rows;
	int cols = A.cols;	

	float *Aptr,*Bptr;	
	int Astep0 = A.step[0],Astep1 = A.step[1];
	int Bstep0 = B.step[0],Bstep1 = B.step[1];
	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			Aptr = (float*)(A.data + i*Astep0 + j*Astep1);
			Bptr = (float*)(B.data + i*Bstep0 + j*Bstep1);
			//cout<<*Aptr<<" "<<*Bptr;
			result = result + abs(*Aptr - *Bptr);			
		}
	}
	return result;
}

void AssignMat(Mat &A, Mat &B)
{
	int rows = A.rows;
	int cols = A.cols;	

	float *Aptr,*Bptr;	
	int Astep0 = A.step[0],Astep1 = A.step[1];
	int Bstep0 = B.step[0],Bstep1 = B.step[1];
	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			Aptr = (float*)(A.data + i*Astep0 + j*Astep1);
			Bptr = (float*)(B.data + i*Bstep0 + j*Bstep1);			
			*Bptr = *Aptr;			
		}
	}
}

void Normalize_With_Sinkhorn(Mat &matrixA)
{
	int iter_num = 0;
	int max_Iterative_Time = 10;
	float fEpsilon = 1;// important
	float result = 10;
	int rows = matrixA.rows;
	int cols = matrixA.cols;

	Mat B = Mat::zeros(rows,cols,CV_32FC1);		
	while((result> fEpsilon)&&(iter_num < max_Iterative_Time))               
	{
		iter_num++;                   
		//B = matrixA;// I do not know why it is wrong!	
		AssignMat(matrixA,B);
		Normalize_Rows_Cols(matrixA);
		result = Abs_DifferenceValue_TwoMatrix(matrixA, B);
		//cout<<result<<endl;
	}
	//cout<<" iter_num = "<<iter_num<<endl;
}



float l2NormValue(Mat &A, Mat &B)
{
	int rows = A.rows;
	int cols = A.cols;
	float result = 0.0;

	float *Aptr,*Bptr;
	int Astep0 = A.step[0],Astep1 = A.step[1];
	int Bstep0 = B.step[0],Bstep1 = B.step[1];
	for (int i=0;i<rows;i++)
	{
		for (int j=0;j<cols;j++)
		{
			Aptr = (float*)(A.data + i*Astep0 + j*Astep1);
			Bptr = (float*)(B.data + i*Bstep0 + j*Bstep1);
			result = result + (*Aptr - *Bptr)*(*Aptr - *Bptr);
		}
	}
	result = sqrt(result);	
	return result;
}

void Find_Matching_greedy_discretization(Mat &first_order_matrix, vector<DMatch> &matches, 
										 vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
										 Mat &descriptorsA, Mat &descriptorsB,const int frame_num)
{		
	DMatch temp_match;
	int real_rows = first_order_matrix.rows;
	int real_cols = first_order_matrix.cols;

	Point max_location;
	Point min_location;

	double min_value = 10;
	double max_value = 10;// must be double

	int first_step0 = first_order_matrix.step[0],first_step1 = first_order_matrix.step[1];
	float *first_pix;

	/*ofstream outfile1("./first_order_matrix1.txt",ios::trunc);	
	Restore_Matrix_To_File(first_order_matrix,outfile1);
	outfile1.close();*/

	while(max_value!=0)
	{		
		minMaxLoc(first_order_matrix,&min_value,&max_value,&min_location,&max_location);
		int cols = max_location.x;
		int rows = max_location.y;


		if (abs(max_value-0)<1e-6)
		{
			/*ofstream outfile2("./first_order_matrix2.txt",ios::trunc);
			Restore_Matrix_To_File(first_order_matrix,outfile2);
			outfile2.close();*/	

			break;
		}
		else
		{
			// rebuilding the matching results
			temp_match.queryIdx = row_real_index[rows].second;
			temp_match.trainIdx = column_real_index[cols].second;

			unsigned char* query_feat = descriptorsA.ptr(temp_match.queryIdx);
			unsigned char* train_feat = descriptorsB.ptr(temp_match.trainIdx);

			temp_match.distance = hamdist(query_feat,train_feat,descriptorsA.cols);	// re-obtaining the distance value		
			matches.push_back(temp_match);						
		}
		for(int j = 0;j<real_cols;j++)
		{						
			first_pix = (float*)(first_order_matrix.data + rows*first_step0 + j*first_step1);	
			*first_pix = 0.0;
		}
		for(int i=0;i<real_rows;i++)
		{
			first_pix = (float*)(first_order_matrix.data + i*first_step0 + cols*first_step1);	
			*first_pix = 0.0;
		}
	}	
}

void Soft_Threshold(Mat &matrixA,float lambda)
{
	// Soft Thresholding Method for LSE with l1 regular
	// lambda is an important parameter	
	int rows = matrixA.rows;	
	float temp = 0;
	float *pix;
	int step0 = matrixA.step[0],step1 = matrixA.step[1];

	for (int i=0;i<rows;i++)
	{				
		pix = (float *)(matrixA.data + i*step0 + 0*step1);
		if (*pix>(0.5*lambda))
		{
			*pix = *pix - 0.5*lambda;
		}
		else if (*pix<(-0.5*lambda))
		{
			*pix = *pix + 0.5*lambda;
		}
		else
		{
			*pix = 0;
		}										
	}
}

void Obtain_Reprojected_Error(vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
							  vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
							  vector<for_row_column>&result, Mat &homo, Mat &reproj_error)
{	
	// homo is a 3*3 matrix
	int rows = reproj_error.rows;  //rows = result.size();

	float *homo_pix;
	int homo_step0 = homo.step[0],homo_step1 = homo.step[1];

	double v_homo[9];	
	v_homo[0] = homo.at<float>(0,0);
	v_homo[1] = homo.at<float>(0,1);
	v_homo[2] = homo.at<float>(0,2);
	v_homo[3] = homo.at<float>(1,0);
	v_homo[4] = homo.at<float>(1,1);
	v_homo[5] = homo.at<float>(1,2);
	v_homo[6] = homo.at<float>(2,0);
	v_homo[7] = homo.at<float>(2,1);
	v_homo[8] = homo.at<float>(2,2);				

	int ua,va,ub,vb;
	float rub,rvb;

	float *pix;
	int step0 = reproj_error.step[0],step1 = reproj_error.step[1];	
	for (int i=0;i<result.size();i++)
	{		
		ub = keypointsB[column_real_index[result[i].model_col].second].pt.x;
		vb = keypointsB[column_real_index[result[i].model_col].second].pt.y;

		ua = keypointsA[row_real_index[result[i].model_row].second].pt.x;
		va = keypointsA[row_real_index[result[i].model_row].second].pt.y;


		float inv_k = 1.f / (v_homo[6] * ua + v_homo[7] * va + v_homo[8]);
		rub = float(inv_k * (v_homo[0] * ua + v_homo[1] * va + v_homo[2]) + 0.5);
		rvb = float(inv_k * (v_homo[3] * ua + v_homo[4] * va + v_homo[5]) + 0.5);

		pix = (float *)(reproj_error.data + i*step0 + 0*step1);	
		*pix = sqrt((ub - rub)*(ub - rub) + (vb - rvb)*(vb - rvb));
	}	
}

// the neighbor points number for each model point may be different

void Obtain_KNN_From_Potential_Points(const int K, const vector<DMatch> matches,vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB, 
									  Mat &descriptorsA, Mat &descriptorsB, vector<for_row_column> &result,
									  vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,vector<int>&record_times)
{
	int tmp = matches.size();	// the first match size		
	for_row_column tmp_result;
	// put the original match result
	for (int i=0;i<tmp;i++)
	{
		tmp_result.model_row = i;// important
		tmp_result.model_col = i;// important
		tmp_result.score = matches[i].distance;
		result.push_back(tmp_result);
	}	
	int allKeypintsNumber = descriptorsB.rows;
	vector<bool>flag_matchPoints(allKeypintsNumber,false);
	for(int i=0;i<tmp;i++)
	{
		flag_matchPoints[matches[i].trainIdx] = true;//important
	}	
	for (int i=0;i<allKeypintsNumber;i++)
	{		
		// find nearest neighbor for potential point
		if(flag_matchPoints[i]!= true)
		{
			NearestNeighbor(K, i, matches, descriptorsA, descriptorsB, result, row_real_index, column_real_index, record_times);			
			flag_matchPoints[i] = true;
		}
	}
}


void NearestNeighbor(const int K,const int cur,const vector<DMatch> matches, Mat &descriptorsA, Mat &descriptorsB, vector<for_row_column> &result,
					 vector<pair<int,int>>&row_real_index, vector<pair<int,int>>&column_real_index, vector<int>&cal_times)
{
	unsigned char* train_feat = descriptorsB.ptr(cur);	

	int size_bin = descriptorsB.cols;
	unsigned int minValue = 0x7fffffff;
	unsigned int temp;
	int record;
	for_row_column tmp_result;	
	for(int i=0;i<matches.size();i++)
	{
		unsigned char* query_feat = descriptorsA.ptr(row_real_index[i].second);	// real key-point location
		temp = hamdist(query_feat,train_feat,size_bin);		
		if(temp<minValue)
		{
			minValue = temp;
			record = i;
		}			
	}	
	// If we should add more neighbors
	int len;
	if(cal_times[record]<K)
	{
		cal_times[record] = cal_times[record] + 1;// calculate the neighbor number
		len = column_real_index.size();	
		tmp_result.model_row = record;	
		tmp_result.model_col = len;		
		tmp_result.score = minValue;			
		column_real_index.push_back(pair<int,int>(len,cur)); // add the new match 
		result.push_back(tmp_result);	
	}
}

void Normalize_Rows_Of_Matrix(Mat &matrixA)
{
	int rows = matrixA.rows;
	int cols = matrixA.cols;
	float temp = 0;
	float *pix;
	int step0 = matrixA.step[0],step1 = matrixA.step[1];

	// rows normalization
	for (int i=0;i<rows;i++)
	{
		temp = 0;
		for (int j=0;j<cols;j++)
		{
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			temp = temp + *pix;			
		}		
		for (int j=0;j<cols;j++)
		{			
			pix = (float *)(matrixA.data + i*step0 + j*step1);
			*pix = (*pix)/(temp+0.000000001);						
		}
	}
}

bool Check_Valid_Match(const vector<DMatch> final_matches)
{	
	// two special cases
	if (final_matches.size()<=0)
	{
		return false;
	}
	vector<int>sta_result;
	vector<int>::iterator location;
	for (int ii=0;ii<final_matches.size();ii++)
	{
		location = find(sta_result.begin(),sta_result.end(),final_matches[ii].trainIdx);
		if (location==sta_result.end())
		{
			sta_result.push_back(final_matches[ii].trainIdx);
		}
	}
	// we think there are at least 4 different points pairs to estimate an accurate homography, otherwise it is absolute wrong!.
	if (sta_result.size()<LEAST_CORRESPONDENCE)
	{				
		return false; 
	}	
	return true;
}