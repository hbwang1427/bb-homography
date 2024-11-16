#include "main.h"

void ORB_Graph(const string &model_image,const string &video_file,const bool Graph_Flag,const string &featureName)
{
	string nameString = video_file.substr(0,video_file.length()-4);// remove the last ".avi"
	const char* tmp_file = featureName.c_str();
	mkdir(tmp_file);			
	string out_video1,out_video2;	
	string win_match_name, win_homo_name;
	if(Graph_Flag)
	{		
		out_video1.append("./");
		out_video1.append(featureName);
		out_video1.append("/");
		out_video1.append(nameString);
		out_video1.append("_");
		out_video1.append(featureName);
		out_video1.append("_result_graph_");				
		out_video1.append(".avi");

		out_video2.append("./");
		out_video2.append(featureName);
		out_video2.append("/");
		out_video2.append(nameString);
		out_video2.append("_");
		out_video2.append(featureName);
		out_video2.append("_match_graph_");		
		out_video2.append(".avi");
		win_match_name.append(featureName);
		win_match_name.append("_match_graph");		
		win_homo_name.append(featureName);
		win_homo_name.append("_homography_graph");					
	}
	else
	{
		out_video1.append("./");
		out_video1.append(featureName);
		out_video1.append("/");
		out_video1.append(nameString);
		out_video1.append("_");
		out_video1.append(featureName);
		out_video1.append("_original_result.avi");

		out_video2.append("./");
		out_video2.append(featureName);
		out_video2.append("/");
		out_video2.append(nameString);
		out_video2.append("_");
		out_video2.append(featureName);
		out_video2.append("_original_match.avi");

		win_match_name.append(featureName);
		win_match_name.append("_original_match");

		win_homo_name.append(featureName);
		win_homo_name.append("_original_homography");						
	}
		  	
	Mat image1 = imread( model_image,CV_LOAD_IMAGE_GRAYSCALE); 

	vector<KeyPoint> keypointsA, keypointsB;
	Mat descriptorsA, descriptorsB; 

	//ORB orb1,orb2;
	ORB orb1(50),orb2(300);	
	orb1(image1,Mat(),keypointsA,descriptorsA,false);  // important

	vector<DMatch> matches;                            // DMatch(int queryIdx, int trainIdx, float distance)
	vector<DMatch> final_matches; 
	BruteForceMatcher<HammingLUT>matcher;              //BruteForceMatcher support<Hamming> <L1<float>> <L2<float>>  binary feature usually uses hamming distance	

	VideoCapture inputVideo(video_file);
	if (!inputVideo.isOpened())
	{
		cout<<"-- Open Error"<<endl;
		return ;
	}
	VideoWriter outputVideo1,outputVideo2; // output the result;
	int widthValue = (int)inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), heightValue = (int)inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size s = Size(widthValue,heightValue); // get the original video width and height
	int widthModel = image1.cols, heightModel = image1.rows;	
	Size s1 = Size(widthModel+widthValue,max(heightModel,heightValue)); // the size must be same with the result image,oh my god!

	double fps = inputVideo.get(CV_CAP_PROP_FPS);
	//outputVideo1.open(out_video1,-1,15,s,true);
	outputVideo1.open(out_video1,CV_FOURCC('X','V','I','D'),15,s,true);
	outputVideo2.open(out_video2,CV_FOURCC('X','V','I','D'),15,s1,true);

	if (!outputVideo1.isOpened()||!outputVideo2.isOpened())
	{
		cout<<"-- Open Error"<<endl;
		return ;
	}	

	Mat frame,image2;
	Mat img_matches;  
	int count = 0;

	vector<pair<int,int>>row_real_index;    // the rows of graph corresponds to the real key-point index for image1
	vector<pair<int,int>>column_real_index; // the cols of graph corresponds to the real key-point index for image2
	
	bool flag_homography = false;
	Mat homo = Mat::zeros(3,3,CV_32FC1);  // define a homography;	
	int final_inliers_numbers = 0;
	while(true)
	{									      		
		keypointsB.clear();    // important, for next frame		
		matches.clear();       // important, for next frame
		final_matches.clear(); // important, for next frame
		row_real_index.clear();// important, for next frame
		column_real_index.clear();// important, for next frame
		count++;
		
		inputVideo>>frame;

		if (frame.empty())
		{
			cout<<"Have already visited all frames"<<endl;
			break;		
		}
		cvtColor(frame,image2,CV_BGR2GRAY); // convert to gray image						

		orb2(image2,Mat(),keypointsB,descriptorsB,false);  		
		matcher.match(descriptorsA,descriptorsB,matches);  		

		// we use the same maximum model points for all binary features, i.e., orb,brief,brisk,freak		
		for (int i=0;i<matches.size()&&i<MAX_MODEL_POINTS;i++)
		{
			final_matches.push_back(matches[i]);
		}
		if (!matches.empty())
		{
			matches.clear();
		}		
		for (int i=0;i<final_matches.size();i++)
		{
			matches.push_back(final_matches[i]);
		}		
		

		if (Graph_Flag==true)
		{			
			
			flag_homography = Original_ORB(descriptorsA,descriptorsB,
				keypointsA,keypointsB,final_matches,homo,final_inliers_numbers);	
					
			if (!flag_homography)
			{					
				if (!final_matches.empty())
				{
					final_matches.clear();
				}	
				flag_homography = Probabilistic_Graph_Matching_Method(row_real_index,column_real_index,matches,keypointsA, keypointsB,
					descriptorsA, descriptorsB,final_matches,count,homo,final_inliers_numbers);
			}
							
		}
		else
		{		
			flag_homography = Original_ORB(descriptorsA,descriptorsB,
				keypointsA,keypointsB,final_matches,homo,final_inliers_numbers);	
			
		}
		

		// detection fails
		if (homo.empty())
		{			
			continue;
		}				
				
		drawMatches(image1, keypointsA, image2, keypointsB, final_matches, img_matches);	
		imshow(win_match_name,img_matches); 	

		vector<Point2f>src_cornor(4);  
		vector<Point2f>dst_cornor(4);  
		src_cornor[0]=cvPoint(0,0);  
		src_cornor[1]=cvPoint(image1.cols,0);  
		src_cornor[2]=cvPoint(image1.cols,image1.rows);  
		src_cornor[3]=cvPoint(0,image1.rows);  
		perspectiveTransform(src_cornor,dst_cornor,homo); 

		line(frame,dst_cornor[0],dst_cornor[1],Scalar(0,0,255),5);// here it show the blue color in the window, because the image is in BGR format instead of RGB. sgliu 
		line(frame,dst_cornor[1],dst_cornor[2],Scalar(0,0,255),5); 
		line(frame,dst_cornor[2],dst_cornor[3],Scalar(0,0,255),5);
		line(frame,dst_cornor[3],dst_cornor[0],Scalar(0,0,255),5);

		imshow(win_homo_name,frame);  				
		int q = waitKey(20);

		outputVideo1<<frame;
		outputVideo2<<img_matches;
		
		if((char)q == 'q')
		{
			break;
		}
		
	}		
}

bool Original_ORB(Mat &descriptorsA, Mat &descriptorsB,vector<KeyPoint>&keypointsA, 
				  vector<KeyPoint>&keypointsB, vector<DMatch>&final_matches,
				  Mat &homo, int &final_inliers_numbers)
{
	bool flag_homo = false;
	// for special case (If the binary features fail detect the image, e.g., brief or brisk on the mouse pad video)
	if(!Check_Valid_Match(final_matches))
	{		
		return flag_homo;
	}	
	
	flag_homo = My_PROSAC(descriptorsA, descriptorsB, keypointsA, keypointsB, homo,final_matches, final_inliers_numbers);		
	return flag_homo;
}

bool My_PROSAC(Mat &descriptorsA, Mat &descriptorsB,
			   vector<KeyPoint>&keypointsA, vector<KeyPoint>&keypointsB,
			   Mat &homo,vector<DMatch>&final_matches, int &final_inliers_numbers)
{
	bool flag_homography_ransac = false;
	homography_estimator *m_H_estimator = new homography_estimator();
	m_H_estimator->reset_correspondences(final_matches.size());
	float value;
	size_t sizeBin = descriptorsA.cols;
	for (int i=0;i<final_matches.size();i++)
	{		
		unsigned char* query_feat = descriptorsA.ptr(final_matches[i].queryIdx);	
		unsigned char* train_feat = descriptorsB.ptr(final_matches[i].trainIdx);
		value = hamdist(query_feat,train_feat,sizeBin);

		m_H_estimator->add_correspondence(keypointsA[final_matches[i].queryIdx].pt.x,
			keypointsA[final_matches[i].queryIdx].pt.y,keypointsB[final_matches[i].trainIdx].pt.x,
			keypointsB[final_matches[i].trainIdx].pt.y,1.f/value);
	}
	homography m_H;	
	final_inliers_numbers = m_H_estimator->ransac(&m_H, REPROJ_THRESHOLD, 100, 0.99, true);
	flag_homography_ransac = final_inliers_numbers>=LEAST_NUMBER_INLIERS;	
	AssignMat(m_H.homo,homo);

	if (m_H_estimator)
	{
		delete m_H_estimator;
	}

	return flag_homography_ransac;

}

