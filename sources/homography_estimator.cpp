#include <algorithm>
#include <iostream>
using namespace std;

#include "homography_estimator.h"

homography_estimator::homography_estimator(void)
{
	AA = Mat::zeros(8, 9, CV_32FC1); 
	W  = Mat::zeros(9, 1, CV_32FC1);
	Ut = Mat::zeros(8, 8, CV_32FC1);
	Vt = Mat::zeros(9, 9, CV_32FC1);

	T1	= Mat::zeros(3, 3, CV_32FC1);
	T2inv = Mat::zeros(3, 3, CV_32FC1);
	tmp   = Mat::zeros(3, 3, CV_32FC1);

	AA2 = Mat::zeros(4, 3, CV_32FC1);
	B2  = Mat::zeros(4, 1, CV_32FC1);
	X2  = Mat::zeros(3, 1, CV_32FC1);

	u_v_up_vp = 0;
	normalized_u_v_up_vp = 0;
	scores = 0;  
	sorted_ids = 0;  
	inliers = 0;
	verbose_level = 2;   
}

homography_estimator::~homography_estimator(void)
{		
	delete [] u_v_up_vp;
	delete [] normalized_u_v_up_vp;	
	delete [] inliers;	
	delete [] scores;
	delete [] sorted_ids;		

}

void homography_estimator::set_bottom_right_coefficient_to_one(homography * H)
{	
	float inv_H33 = 1.f/float(HE_Get(H->homo, 2, 2));
	for (int i=0;i<3;i++)
	{
		for (int j=0;j<3;j++)
		{
			if ((i+j)!=4)
			{
				H->homo.at<float>(i,j) = H->homo.at<float>(i,j)*inv_H33;
			}			
		}
	}	
	HE_Set(H->homo, 2, 2, 1.);		

}


bool homography_estimator::estimate(homography * H,
									const float u1, const float v1, const float up1, const float vp1,
									const float u2, const float v2, const float up2, const float vp2,
									const float u3, const float v3, const float up3, const float vp3,
									const float u4, const float v4, const float up4, const float vp4)
{
	const float a[] = { u1, v1 };
	const float b[] = { u2, v2 };
	const float c[] = { u3, v3 };
	const float d[] = { u4, v4 };
	const float x[] = { up1, vp1 };
	const float y[] = { up2, vp2 };
	const float z[] = { up3, vp3 };
	const float w[] = { up4, vp4 };
	float result[3][3];

	for (int i=0;i<3;i++)
	{
		for (int j=0;j<3;j++)
		{
			result[i][j] = H->homo.at<float>(i,j);
		}
	}
	homography_from_4corresp(a,b,c,d,x,y,z,w, result);
	for (int i=0;i<3;i++)
	{
		for (int j=0;j<3;j++)
		{
			H->homo.at<float>(i,j) = result[i][j];
		}
	}

	return true;
}

void homography_estimator::reset_correspondences(int maximum_number_of_correspondences)
{	
	u_v_up_vp = new float[4 * maximum_number_of_correspondences];
	normalized_u_v_up_vp = new float[4 * maximum_number_of_correspondences];
	scores = new float[maximum_number_of_correspondences];
	sorted_ids = new int[maximum_number_of_correspondences];

	number_of_correspondences = 0;
} 

void homography_estimator::add_correspondence(float u, float v, float up, float vp)
{
	u_v_up_vp[4 * number_of_correspondences    ] = u;
	u_v_up_vp[4 * number_of_correspondences + 1] = v;
	u_v_up_vp[4 * number_of_correspondences + 2] = up;
	u_v_up_vp[4 * number_of_correspondences + 3] = vp;

	number_of_correspondences++;
}

void homography_estimator::add_correspondence(float u, float v, float up, float vp, float score)
{
	u_v_up_vp[4 * number_of_correspondences    ] = u;
	u_v_up_vp[4 * number_of_correspondences + 1] = v;
	u_v_up_vp[4 * number_of_correspondences + 2] = up;
	u_v_up_vp[4 * number_of_correspondences + 3] = vp;
	scores[number_of_correspondences] = score;         
	number_of_correspondences++;
}

float homography_estimator::HE_Get(Mat &target,const int i, const int j)
{
	return target.at<float>(i,j);
}

void homography_estimator::HE_Set(Mat &target,const int i, const int j, const float val)
{
	target.at<float>(i,j) = val;
}

void homography_estimator::HE_Set(Mat &target,const int i, const int j, const double val)
{
	target.at<float>(i,j) = (float)val;
}

void homography_estimator::normalize(void)
{
	float u_sum = 0., v_sum = 0., up_sum = 0., vp_sum = 0;

	for(int i = 0; i < number_of_correspondences; i++) {
		u_sum  += u_v_up_vp[4 * i    ];
		v_sum  += u_v_up_vp[4 * i + 1];
		up_sum += u_v_up_vp[4 * i + 2];
		vp_sum += u_v_up_vp[4 * i + 3];
	}

	float u_mean  = u_sum  / number_of_correspondences;
	float v_mean  = v_sum  / number_of_correspondences;
	float up_mean = up_sum / number_of_correspondences;
	float vp_mean = vp_sum / number_of_correspondences;

	// translate mean to origin, compute sum of distances from origin
	float dist_sum = 0, distp_sum = 0;
	for(int i = 0; i < number_of_correspondences; i++) {
		normalized_u_v_up_vp[4 * i    ] = u_v_up_vp[4 * i    ] - u_mean;
		normalized_u_v_up_vp[4 * i + 1] = u_v_up_vp[4 * i + 1] - v_mean;

		dist_sum += sqrt(normalized_u_v_up_vp[4 * i    ] * normalized_u_v_up_vp[4 * i  ] +
			normalized_u_v_up_vp[4 * i + 1] * normalized_u_v_up_vp[4 * i + 1]);

		normalized_u_v_up_vp[4 * i + 2] = u_v_up_vp[4 * i + 2] - up_mean;
		normalized_u_v_up_vp[4 * i + 3] = u_v_up_vp[4 * i + 3] - vp_mean;

		distp_sum += sqrt(normalized_u_v_up_vp[4 * i + 2] * normalized_u_v_up_vp[4 * i + 2] +
			normalized_u_v_up_vp[4 * i + 3] * normalized_u_v_up_vp[4 * i + 3]);
	}

	// compute normalizing scale factor ( average distance from origin = sqrt(2) )
	scale  = (sqrt(2.f) * number_of_correspondences) / dist_sum;
	scalep = (sqrt(2.f) * number_of_correspondences) / distp_sum;

	// apply scaling
	for(int i = 0; i < number_of_correspondences; i++) {
		normalized_u_v_up_vp[4 * i    ] *= scale;
		normalized_u_v_up_vp[4 * i + 1] *= scale;
		normalized_u_v_up_vp[4 * i + 2] *= scalep;
		normalized_u_v_up_vp[4 * i + 3] *= scalep;
	}
	
	HE_Set(T1, 0, 0, scale);
	HE_Set(T1, 0, 1, 0.);
	HE_Set(T1, 0, 2, -scale * u_mean);
	HE_Set(T1, 1, 0, 0.);
	HE_Set(T1, 1, 1, scale);
	HE_Set(T1, 1, 2, -scale * v_mean);
	HE_Set(T1, 2, 0, 0.);
	HE_Set(T1, 2, 1, 0.);
	HE_Set(T1, 2, 2, 1.);

	HE_Set(T2inv, 0, 0, 1. / scalep);
	HE_Set(T2inv, 0, 1, 0.);
	HE_Set(T2inv, 0, 2, up_mean);
	HE_Set(T2inv, 1, 0, 0.);
	HE_Set(T2inv, 1, 1, 1. / scalep);
	HE_Set(T2inv, 1, 2, vp_mean);
	HE_Set(T2inv, 2, 0, 0.);
	HE_Set(T2inv, 2, 1, 0.);
	HE_Set(T2inv, 2, 2, 1.);
}

void homography_estimator::denormalize(homography * H)
{	
	tmp = T2inv*H->homo;
	H->homo = tmp*T1;	
}


bool homography_estimator::nice_homography(homography * H)
{
	double det = HE_Get(H->homo, 0, 0) * HE_Get(H->homo, 1, 1) - HE_Get(H->homo, 1, 0) * HE_Get(H->homo, 0, 1);
	if (det < 0) return false;
	double N1 = sqrt(HE_Get(H->homo, 0, 0) * HE_Get(H->homo, 0, 0) + HE_Get(H->homo, 1, 0) * HE_Get(H->homo, 1, 0));
	if (N1 > 4) return false;
	if (N1 < 0.1) return false;
	double N2 = sqrt(HE_Get(H->homo, 0, 1) * HE_Get(H->homo, 0, 1) + HE_Get(H->homo, 1, 1) * HE_Get(H->homo, 1, 1));
	if (N2 > 4) return false;
	if (N2 < 0.1) return false;
	double N3 = sqrt(HE_Get(H->homo, 2, 0) * HE_Get(H->homo, 2, 0) + HE_Get(H->homo, 2, 1) * HE_Get(H->homo, 2, 1));
	if (N3 > 0.002) return false;
	return true;
}

int homography_estimator::ransac(homography * H, const float threshold, const int maximum_number_of_iterations,
								 const float P, bool prosac_sampling)
{
		if (number_of_correspondences < 4) {
		if (verbose_level >= 1) {
			cerr << "> [homography_estimator::ransac]:" << endl;
			cerr << ">    Can't estimate homography with less than 4 correspondences." << endl;
		}
        
		return 0;
	}
   
	normalize();

	int N = maximum_number_of_iterations;

	number_of_inliers = 0;

	inliers = new bool[number_of_correspondences];
	

	bool * current_inliers = 0;	
	current_inliers = new bool[number_of_correspondences];


	int sample_count = 0;
	int prosac_correspondences = 10;

	if(prosac_sampling) {
		sort_correspondences();
	}

	while (N > sample_count && number_of_inliers<50)
	{ 
		int n1, n2, n3, n4;
		if(prosac_sampling)
		{
			get_4_prosac_indices(prosac_correspondences, n1, n2, n3, n4);
			
			if(prosac_correspondences < number_of_correspondences) 
			{
				++prosac_correspondences;
			}
		}
		else 
		{
			get_4_random_indices(number_of_correspondences, n1, n2, n3, n4);
		}

		bool ok = estimate(H,
			normalized_u_v_up_vp[4 * n1], normalized_u_v_up_vp[4 * n1 + 1], normalized_u_v_up_vp[4 * n1 + 2], normalized_u_v_up_vp[4 * n1 + 3],
			normalized_u_v_up_vp[4 * n2], normalized_u_v_up_vp[4 * n2 + 1], normalized_u_v_up_vp[4 * n2 + 2], normalized_u_v_up_vp[4 * n2 + 3],
			normalized_u_v_up_vp[4 * n3], normalized_u_v_up_vp[4 * n3 + 1], normalized_u_v_up_vp[4 * n3 + 2], normalized_u_v_up_vp[4 * n3 + 3],
			normalized_u_v_up_vp[4 * n4], normalized_u_v_up_vp[4 * n4 + 1], normalized_u_v_up_vp[4 * n4 + 2], normalized_u_v_up_vp[4 * n4 + 3]);

		if (!ok) return false;
       
		denormalize(H);
		set_bottom_right_coefficient_to_one(H);
		       
		if (nice_homography(H)) 
		{
			int current_number_of_inliers = compute_inliers(H, current_inliers, threshold);
           
			if (current_number_of_inliers > number_of_inliers) {
				/*if (verbose_level >= 2) 
				{  
					cout << "> Iteration " << sample_count << ": " << current_number_of_inliers << " inliers. New N = " << N << endl;
				}*/
				
				double eps = 1. - double(current_number_of_inliers) / number_of_correspondences;
				
				int newN = (int)(log(1-P)/log(1-pow((1.-eps), 4))); 
				if (newN < N) N = newN;

				number_of_inliers = current_number_of_inliers;
				
				for (int i = 0; i < number_of_correspondences; i++) 
				{
					inliers[i] = current_inliers[i];
				}
			}
		}
		sample_count++;
	}

	//cout<<"sample_count = " <<sample_count<<" "<<"number_of_inliers = "<<number_of_inliers<<endl;

	
	int old_number_of_inliers = number_of_inliers;
	do {

		bool okok = estimate_from_inliers(H);	
		if (!okok) 
		{			
			return false;
		}

		old_number_of_inliers = number_of_inliers;
		number_of_inliers = compute_inliers(H, inliers, threshold);
		/*if (verbose_level >= 2) 
		{
			cout << "> Refining: " << number_of_inliers << " inliers." << endl;
		}*/
	} while (number_of_inliers > old_number_of_inliers);

	
	
   number_of_inliers = compute_inliers(H, inliers, threshold);
				
  /* if (verbose_level >= 1) 
   {
	   cout<<"Total_number_of_match_points = "<<number_of_correspondences<<endl;
	   cout <<" homography_estimator::ransac( )  "<< number_of_inliers << "inliers found:" << endl;
   }  */    
   delete [] current_inliers;
   return number_of_inliers;
		
}

void homography_estimator::get_4_random_indices(int n_max, int & n1, int & n2, int & n3, int & n4)
{
	
	n1 = rand() % n_max;
	do n2 = rand() % n_max; while(n2 == n1);
	do n3 = rand() % n_max; while(n3 == n1 || n3 == n2);
	do n4 = rand() % n_max; while(n4 == n1 || n4 == n2 || n4 == n3);
}

void homography_estimator::get_4_prosac_indices(int n_max, int & n1, int & n2, int & n3, int & n4)
{	
	n1 = rand() % n_max;
	do n2 = rand() % n_max; while(n2 == n1);
	do n3 = rand() % n_max; while(n3 == n1 || n3 == n2);
	do n4 = rand() % n_max; while(n4 == n1 || n4 == n2 || n4 == n3);
   
	n1 = sorted_ids[n1];
	n2 = sorted_ids[n2];
	n3 = sorted_ids[n3];
	n4 = sorted_ids[n4];
}

int homography_estimator::compute_inliers(homography * H, bool * inliers, float threshold)
{
	int n = 0;
    
	for(int i = 0; i < number_of_correspondences; i++) 
	{
		float eup, evp;
		H->transform_point(u_v_up_vp[4 * i], u_v_up_vp[4 * i + 1], eup, evp);   
		
		inliers[i] = ((u_v_up_vp[4 * i + 2] - eup) * (u_v_up_vp[4 * i + 2] - eup) +
			(u_v_up_vp[4 * i + 3] - evp) * (u_v_up_vp[4 * i + 3] - evp)) < threshold * threshold;

		if (inliers[i])
		{			
			n++;
		}
	}	
	return n;
}

bool homography_estimator::estimate_from_inliers(homography * H)
{
	if (number_of_inliers < 4) return false;

	Mat _AA_ = Mat::zeros(2 * number_of_inliers, 9, CV_32FC1);

	int n = 0;
	for(int i = 0; i < number_of_correspondences; i++)
		if (inliers[i]) 
		{
			HE_Set(_AA_, n, 0, 0.); 
			HE_Set(_AA_, n, 1, 0.); 
			HE_Set(_AA_, n, 2, 0.);
			HE_Set(_AA_, n, 3, -normalized_u_v_up_vp[4 * i]); 
			HE_Set(_AA_, n, 4, -normalized_u_v_up_vp[4 * i + 1]); 
			HE_Set(_AA_, n, 5, -1.); 
			HE_Set(_AA_, n, 6,  normalized_u_v_up_vp[4 * i + 3] * normalized_u_v_up_vp[4 * i]); 
			HE_Set(_AA_, n, 7,  normalized_u_v_up_vp[4 * i + 3] * normalized_u_v_up_vp[4 * i + 1]); 
			HE_Set(_AA_, n, 8,  normalized_u_v_up_vp[4 * i + 3]);
			HE_Set(_AA_, n + 1, 0, normalized_u_v_up_vp[4 * i]); 
			HE_Set(_AA_, n + 1, 1, normalized_u_v_up_vp[4 * i + 1]); 
			HE_Set(_AA_, n + 1, 2, 1.); 
			HE_Set(_AA_, n + 1, 3,  0.);
			HE_Set(_AA_, n + 1, 4,  0.); 
			HE_Set(_AA_, n + 1, 5, 0.); 
			HE_Set(_AA_, n + 1, 6, -normalized_u_v_up_vp[4 * i + 2] * normalized_u_v_up_vp[4 * i]); 
			HE_Set(_AA_, n + 1, 7, -normalized_u_v_up_vp[4 * i + 2] * normalized_u_v_up_vp[4 * i + 1]); 
			HE_Set(_AA_, n + 1, 8, -normalized_u_v_up_vp[4 * i + 2]);
			n += 2;
		}
		SVD::compute(_AA_, W, Ut, Vt, CV_SVD_MODIFY_A | CV_SVD_V_T);

		HE_Set(H->homo, 0, 0, HE_Get(Vt, 8, 0));
		HE_Set(H->homo, 0, 1, HE_Get(Vt, 8, 1));
		HE_Set(H->homo, 0, 2, HE_Get(Vt, 8, 2));

		HE_Set(H->homo, 1, 0, HE_Get(Vt, 8, 3));
		HE_Set(H->homo, 1, 1, HE_Get(Vt, 8, 4));
		HE_Set(H->homo, 1, 2, HE_Get(Vt, 8, 5));

		HE_Set(H->homo, 2, 0, HE_Get(Vt, 8, 6));
		HE_Set(H->homo, 2, 1, HE_Get(Vt, 8, 7));
		HE_Set(H->homo, 2, 2, HE_Get(Vt, 8, 8));
		

		denormalize(H);
		set_bottom_right_coefficient_to_one(H);

		return nice_homography(H);
}

class compare_by_score
{
public:
	compare_by_score(float* s) : scores(s) {}

	bool operator()(int id0, int id1) {  return scores[id0] > scores[id1]; }
	float* scores;
};

void homography_estimator::sort_correspondences()
{
	for(int i = 0; i < number_of_correspondences; i++)
	{
		sorted_ids[i] = i;
	}
	//descending 
	sort(sorted_ids, sorted_ids + number_of_correspondences, compare_by_score(scores)); 
}
