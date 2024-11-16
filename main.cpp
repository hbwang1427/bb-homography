#include "main.h"

int main()  
{  			
	string model_image,video_file;  
	model_image = "demo.png";
	video_file = "demo.avi";
	
	bool Graph_Flag = false; // whether combine the graph algorithm	
	Graph_Flag = true;
	int binary_method = 1;				
	main_run(model_image, video_file, Graph_Flag, binary_method);
										
	 
	return 0;
}  

void main_run(const string &model_image, const string &video_file,const bool Graph_Flag,const int binary_method)
{

	string feature_name;
	if (binary_method == 1)
	{
		feature_name = "ORB";
	}
	else if (binary_method == 2)
	{
		feature_name = "BRIEF";
	}
	else if(binary_method == 3)
	{
		feature_name = "BRISK";
	}
	else
	{
		feature_name = "FREAK";
	}

	Graph_With_Binary_Features(model_image,video_file,feature_name,Graph_Flag);
}

void Graph_With_Binary_Features(const string &model_image,const string &video_file,
								const string &featureName,const bool Graph_Flag)
{

	if (featureName=="ORB")
	{
		ORB_Graph(model_image,video_file,Graph_Flag,featureName);
	}
	else if (featureName=="BRIEF")
	{
		BRIEF_Graph(model_image,video_file,Graph_Flag,featureName);
	}
	else if(featureName=="BRISK")
	{
		BRISK_Graph(model_image,video_file,Graph_Flag,featureName);
	}
	else
	{
		FREAK_Graph(model_image,video_file,Graph_Flag,featureName);
	}

}




