#include "main.h"


bool Probabilistic_Graph_Matching_Method(vector<pair<int,int>>&row_real_index,vector<pair<int,int>>&column_real_index,
										  const vector<DMatch>matches,vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB, 
										  Mat &descriptorsA, Mat &descriptorsB,vector<DMatch>&final_matches,
										  const int frame_num, Mat &homo, int &final_inliers_numbers)
{
	// This method constructs a KNN graph from the potential keypoints
	bool flag_homography = false;	
	if(!Check_Valid_Match(matches))
	{
		final_inliers_numbers = 0;
		return flag_homography; // failure, exit the loop 
	}

	// restore the real index
	int K = 5; // neighbor number
	int oriMatchSize = matches.size();
	for(int i=0;i<oriMatchSize;i++)
	{
		row_real_index.push_back(pair<int,int>(i,matches[i].queryIdx));
		column_real_index.push_back(pair<int,int>(i,matches[i].trainIdx));
	}

	// construct second-order matrix				
	vector<for_row_column> result; // next time, you should use it like this.
	vector<int>record_times(oriMatchSize,1);

	Obtain_KNN_From_Potential_Points(K, matches, keypointsA, keypointsB, descriptorsA, descriptorsB,
		result,row_real_index,column_real_index,record_times);	

	// construct first-order matrix
	int real_rows = row_real_index.size();
	int real_cols = column_real_index.size();	
	Mat first_order_matrix = Mat::zeros(real_rows,real_cols,CV_32FC1);// initialization
	
	int first_step0 = first_order_matrix.step[0],first_step1 = first_order_matrix.step[1];
	float *first_pix;
	
	float sigma_Value = 1.75; 
	float elem_value,elem_value1,elem_value2;
	int tmp11,tmp12,tmp21,tmp22;

	int second_rows = result.size();
	int second_cols = result.size();

		
	Mat second_order_matrix = Mat::zeros(second_rows,second_cols,CV_32FC1); // initialization

	int second_step0 = second_order_matrix.step[0],second_step1 = second_order_matrix.step[1];		
	float *second_pix;
	for (int i=0;i<second_rows;i++)
	{
		tmp11 = row_real_index[result[i].model_row].second;
		tmp12 = column_real_index[result[i].model_col].second;			
		for (int j=0;j<second_cols;j++)
		{	
			if(i<j)
			{
				tmp21 = row_real_index[result[j].model_row].second;
				tmp22 = column_real_index[result[j].model_col].second;
				elem_value1 = sqrt((keypointsA[tmp11].pt.x - keypointsA[tmp21].pt.x)*(keypointsA[tmp11].pt.x - keypointsA[tmp21].pt.x) 
					+ (keypointsA[tmp11].pt.y - keypointsA[tmp21].pt.y)*(keypointsA[tmp11].pt.y - keypointsA[tmp21].pt.y)); 
				elem_value2 = sqrt((keypointsB[tmp12].pt.x - keypointsB[tmp22].pt.x)*(keypointsB[tmp12].pt.x - keypointsB[tmp22].pt.x) 
					+ (keypointsB[tmp12].pt.y - keypointsB[tmp22].pt.y)*(keypointsB[tmp12].pt.y - keypointsB[tmp22].pt.y)); 
				elem_value = (elem_value1-elem_value2)*(elem_value1-elem_value2);
				elem_value = exp(-1/(sigma_Value*sigma_Value)*elem_value);				
				second_pix = (float *)(second_order_matrix.data + i*second_step0 + j*second_step1);
				*second_pix = elem_value;				
			}			
			else if (i>j)
			{
				second_pix = (float *)(second_order_matrix.data + j*second_step0 + i*second_step1);
				float temp_tmp = *second_pix;
				second_pix = (float *)(second_order_matrix.data + i*second_step0 + j*second_step1);
				*second_pix = temp_tmp;
			}
			else
			{
				second_pix = (float *)(second_order_matrix.data + i*second_step0 + j*second_step1);
				*second_pix = 1.0;// operator for setting value				
			}				
		}
	}

	const int max_iter = 8;
	Mat p_matrix = Mat::ones(second_cols,1,CV_32FC1);	
	Mat q_matrix = Mat::zeros(second_cols,1,CV_32FC1);
	Mat p_normalize_matrix = Mat::zeros(second_cols,1,CV_32FC1); // initialization

	float sigma = 0.001;
	float iter_result = 0.0;

	int p_step0 = p_matrix.step[0],p_step1 = p_matrix.step[1];
	int p_normalize_step0 = p_normalize_matrix.step[0],p_normalize_step1 = p_normalize_matrix.step[1];	
	int q_step0 = q_matrix.step[0],q_step1 = q_matrix.step[1];

	float *p_pix;					
	float *q_pix;		
	float *p_normalize_pix;
		
	for(int i=0;i<second_cols;i++)
	{				
		p_pix = (float*)(p_matrix.data + p_step0*i + 0*p_step1);
		*p_pix = *p_pix * (1.0/record_times[result[i].model_row]);
	}

	const int total_Refinement_Times = 10;
	float lambda = 0.0001;  

	// total algorithm
	for (int kk=0;kk<total_Refinement_Times;kk++)
	{
		// algorithm
		for (int ii=0;ii<max_iter;ii++)
		{
			q_matrix = second_order_matrix*p_matrix;// matrix multiply					
			// restore the first_order_matrix			

			for (int i=0;i<second_cols;i++)
			{				
				first_pix = (float*)(first_order_matrix.data + result[i].model_row*first_step0 + result[i].model_col*first_step1);				
				q_pix = (float*)(q_matrix.data + i*q_step0 + 0*q_step1);
				*first_pix = *q_pix;				
			}			

			//Normalize_Rows_Of_Matrix(first_order_matrix); 
			Normalize_With_Sinkhorn(first_order_matrix);		

			for(int i=0;i<second_cols;i++)
			{
				first_pix = (float*)(first_order_matrix.data + result[i].model_row*first_step0 + result[i].model_col*first_step1);				
				p_normalize_pix = (float*)(p_normalize_matrix.data + p_normalize_step0*i + 0*p_normalize_step1);
				*p_normalize_pix = *first_pix;				
			}

			// // add sparse constraints
			// // soft threshold

			Soft_Threshold(p_normalize_matrix,lambda);// sparse solution.

			int k=0;
			for(int i=0;i<second_rows;i++)
			{
				for (int j=0;j<second_cols;j++)
				{
					second_pix = (float *)(second_order_matrix.data + i*second_step0 + j*second_step1);
					p_normalize_pix = (float*)(p_normalize_matrix.data + p_normalize_step0*k + 0*p_normalize_step1);
					p_pix = (float*)(p_matrix.data + p_step0*k + 0*p_step1);
					*second_pix = (*second_pix)*(*p_normalize_pix/(*p_pix+0.0000001));					
				}
				k++;
			}
			iter_result = l2NormValue(p_normalize_matrix,p_matrix);				

			if (iter_result<(second_cols*sigma))
			{				
				break;
			}
			else
			{			
				AssignMat(p_normalize_matrix,p_matrix); // assign for next iteration
			}
		}

		// re-obtain the final first order matrix
		for (int i=0;i<second_cols;i++)
		{				
			first_pix = (float*)(first_order_matrix.data + result[i].model_row*first_step0 + result[i].model_col*first_step1);				
			p_normalize_pix = (float*)(p_normalize_matrix.data + p_normalize_step0*i + 0*p_normalize_step1);
			*first_pix = *p_normalize_pix;				
		}	

		if (!final_matches.empty())
		{
			final_matches.clear(); // very important
		}
		Find_Matching_greedy_discretization(first_order_matrix,final_matches,row_real_index,column_real_index,descriptorsA,descriptorsB,frame_num);	

		// // for special case (If the binary features fail detect the image, e.g., brief or brisk on the mouse pad video)				
		if(!Check_Valid_Match(final_matches))
		{			
			final_inliers_numbers = 0;			
			return flag_homography; // failure, exit the loop 
		}

		flag_homography = My_PROSAC(descriptorsA, descriptorsB,keypointsA, keypointsB,homo,final_matches,final_inliers_numbers);	

		if (homo.empty())
		{
			final_inliers_numbers = 0;
			return flag_homography;
		}
					
		if (flag_homography)
		{					
			return flag_homography;
		}	
		Mat reproj_mat = Mat::zeros(p_normalize_matrix.rows,p_normalize_matrix.cols,CV_32FC1);
		Obtain_Reprojected_Error(row_real_index,column_real_index,keypointsA, keypointsB, result, homo, reproj_mat);	

		float *r_pix;
		int r_step0 = reproj_mat.step[0],r_step1 = reproj_mat.step[1];	
		for (int i=0;i<p_normalize_matrix.rows;i++)
		{
			r_pix = (float*)(reproj_mat.data + i*r_step0 + 0*r_step1);				
			p_normalize_pix = (float*)(p_normalize_matrix.data + p_normalize_step0*i + 0*p_normalize_step1);			
			*p_normalize_pix = *p_normalize_pix/(*r_pix+0.00000001);// combine the reprojected error into our algorithm.
			
		}
		AssignMat(p_normalize_matrix,p_matrix); // assign for next iteration

		for(int i=0;i<p_matrix.rows;i++)
		{
			first_pix = (float*)(first_order_matrix.data + result[i].model_row*first_step0 + result[i].model_col*first_step1);				
			p_pix = (float*)(p_matrix.data + p_step0*i + 0*p_step1);
			*first_pix = *p_pix;				
		}
		Normalize_Rows_Of_Matrix(first_order_matrix);   //normalization
		for(int i=0;i<p_matrix.rows;i++)
		{
			first_pix = (float*)(first_order_matrix.data + result[i].model_row*first_step0 + result[i].model_col*first_step1);				
			p_pix = (float*)(p_matrix.data + p_step0*i + 0*p_step1);
			*p_pix = *first_pix;				
		}

	}
	return flag_homography;
}


