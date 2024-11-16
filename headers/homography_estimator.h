#ifndef homography_estimator_h
#define homography_estimator_h

#include "homography.h"
#include "main.h"


using namespace cv;
using namespace std;

class homography{
public:

	homography(void);

	homography(const float h11, const float h12, const float h13,
		const float h21, const float h22, const float h23,
		const float h31, const float h32, const float h33);

	homography(const double h11, const double h12, const double h13,
		const double h21, const double h22, const double h23,
		const double h31, const double h32, const double h33);

	~homography();

	void transform_point(const int u, const int v, int & up, int & vp);
	void transform_point(const float u, const float v, float & up, float & vp);
	void transform_point(const double u, const double v, double & up, double & vp);

	Mat homo;

	float Get(const int i, const int j);
	void  Set(const int i, const int j, const float val);
	void  Set(const int i, const int j, const double val);

	// private:
	void initialize(void);

};


class homography_estimator{
public:
	homography_estimator(void);
	~homography_estimator(void);
 
	bool estimate(homography *H,
		const float u1, const float v1, const float up1, const float vp1,
		const float u2, const float v2, const float up2, const float vp2,
		const float u3, const float v3, const float up3, const float vp3,
		const float u4, const float v4, const float up4, const float vp4);

	void reset_correspondences(int maximum_number_of_correspondences);
	void add_correspondence(float u, float v, float up, float vp);
	void add_correspondence(float u, float v, float up, float vp, float score);
	
	int ransac(homography *H, const float threshold, const int maximum_number_of_iterations,
		const float P, bool prosac_sampling);

	bool * inliers;
	int number_of_inliers;
	int verbose_level;

	//private:
	void normalize(void);
	float scale, scalep;
	void denormalize(homography *H);
	void get_4_random_indices(int n_max, int & n1, int & n2, int & n3, int & n4);
	void get_4_prosac_indices(int n_max, int & n1, int & n2, int & n3, int & n4);
	int compute_inliers(homography *A, bool * inliers, float threshold);
	bool estimate_from_inliers(homography *A);
	bool nice_homography(homography *H);
	void sort_correspondences();
	void set_bottom_right_coefficient_to_one(homography *H); 

	Mat AA, W, Ut, Vt;                  
	Mat T1, T2inv, tmp;
	Mat AA2, B2, X2;
	float * u_v_up_vp, * normalized_u_v_up_vp, * scores;
	int   * sorted_ids;
	int number_of_correspondences;

	float HE_Get(Mat &target,const int i, const int j);
	void  HE_Set(Mat &target,const int i, const int j, const float val);
	void  HE_Set(Mat &target,const int i, const int j, const double val);
	 	    	
};

#endif
