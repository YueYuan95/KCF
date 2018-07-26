#include<iostream>
#include<fstream>
#include <io.h>  
#include<string>
#include<vector>
#include<opencv2\opencv.hpp>

#include "util.h"

using namespace std;
using namespace cv;

vector<string> getImageNames(string path){
	
	//File Vector
	vector<string> files;
	//文件句柄  
    long   hFile = 0;
    //文件信息，声明一个存储文件信息的结构体  
    struct _finddata_t fileinfo;
    string p;//字符串，存放路径
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
    {
        do
        { 
            if ((fileinfo.attrib &  _A_ARCH))
            {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose函数结束查找
        _findclose(hFile);
    }

	return files;
}

vector<Rect>  getLabel(string labelpath){

	vector<Rect> label;
	ifstream fin(labelpath,ios::in);
	if (!fin){
		cout<<"Cannot open input file."<<endl;
	}
	string rect;
	while(getline(fin,rect,'\n')){

		size_t size = rect.size();
		size_t pos = rect.find(",");

		//get x
		string x = rect.substr(0,pos);
		rect = rect.substr(pos+1,size);
		pos = rect.find(",");

		//get y
		string y = rect.substr(0,pos);
		rect = rect.substr(pos+1,size);
		pos = rect.find(",");

		//get width
		string w = rect.substr(0,pos);
		rect = rect.substr(pos+1,size);
		pos = rect.find(",");

		//get height
		string h = rect.substr(0,pos);
		rect = rect.substr(pos+1,size);
		pos = rect.find(",");

		label.push_back(Rect(stoi(x),stoi(y),stoi(w),stoi(h)));

	}
	return label;

}



int main(){

	string rootPath = "E:\\otb100\\Basketball\\";  
   
    ////获取该路径下的所有图片文件名  
    vector<string> files = getImageNames(rootPath+"img");

	////获取目标真实框位置
	vector<Rect> label = getLabel(rootPath+"groundtruth_rect.txt");

	Size target_sz(label[0].width,label[0].height);
	Point pos(int(label[0].x+target_sz.width/2),int(label[0].y+target_sz.height/2));

	Size window_sz(int(target_sz.width * (1 + padding)),int(target_sz.height * (1 + padding)));

	float output_sigma = sqrt(window_sz.width * window_sz.height) * output_sigma_factor /cell_size;

	Mat gaussian_label = gaussian_shaped_label(output_sigma,window_sz/cell_size);

	dft(gaussian_label,gaussian_label, DFT_COMPLEX_OUTPUT);

	Mat hann(gaussian_label.size().width,gaussian_label.size().height,CV_32FC1);

	cout<<"GaussianLabel"<<gaussian_label.size()<<endl;
	
	createHanningWindow(hann, gaussian_label.size(), CV_32FC1);

	cout<<"CosWindow Size"<<hann.size()<<endl;

	Mat image;
	Mat alphf;
	Mat response;
	vector<Mat> xf;

	Point maxLoc;
	vector<Rect> result;
	
	float w = label[0].width, h = label[0].height;
	for(int i=0;i<files.size();i++){

		if (i==0){

			image = imread(files[i]);

			if (image.channels() > 1){
				cvtColor(image,image,CV_BGR2GRAY);
			}
			Mat patch = get_subwindow(image,pos,window_sz);

			vector<Mat> Feature = get_feature(patch,hann);
			vector<Mat> Feature_f(Feature.size());
			for(int i=0; i < Feature.size();i++){
				dft(Feature[i],Feature_f[i],DFT_COMPLEX_OUTPUT);
			}

			Mat kf = GaussianCorrelation(Feature_f,Feature_f);
			
			alphf = ComplexDiv(gaussian_label,kf+Scalar(lamda, 0));
			vector<Mat> x_temp(Feature.size());
			for(int i=0;i<Feature.size();i++){
				x_temp[i].push_back(Feature_f[i]);
			}
			xf = x_temp;
			result.push_back(label[0]);
			cout<<"Inital Finished...."<<endl;

		}else{

			cout<<"Frame:"<<i<<endl;
			Mat rgb_image = imread(files[i]);
			if (rgb_image.channels() > 1){
				cvtColor(rgb_image,image,CV_BGR2GRAY);
			}
			Mat patch = get_subwindow(image,pos,window_sz);

			vector<Mat> Feature = get_feature(patch,hann);
			vector<Mat> Feature_f(Feature.size());
			for(int i=0;i<Feature.size();i++){
				dft(Feature[i], Feature_f[i],DFT_COMPLEX_OUTPUT);
			}
			Mat kf = GaussianCorrelation(Feature_f,xf);

			idft(ComplexMul(alphf,kf),response,cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

			cout<<response<<endl;

			Point peak;
			Rect result_temp;
			minMaxLoc(response,NULL,NULL,NULL,&peak);
			cout<<peak<<endl;

			if ((peak.x) > (response.cols/2)) peak.x = peak.x - response.cols;
			if ((peak.y) > (response.rows/2)) peak.y = peak.y - response.rows;

			pos.x += cell_size * (peak.x -1);
			pos.y += cell_size * (peak.y -1);

			result_temp.x = pos.x;
			result_temp.y = pos.y;
			result_temp.width = w;
			result_temp.height = h;
			result.push_back(result_temp);

			//rectangle(rgb_image,result_temp,Scalar(255,255,0),2,1,0);
			//imshow("Image",image);
			imshow("Response",response);
			waitKey(-1);

			patch = get_subwindow(image,pos,window_sz);
			Feature = get_feature(patch,hann);
			for(int i=0;i<Feature.size();i++){
				dft(Feature[i], Feature_f[i],DFT_COMPLEX_OUTPUT);
			}

			kf = GaussianCorrelation(Feature_f,Feature_f);

			Mat alphf_temp = ComplexDiv(gaussian_label, kf);
			vector<Mat> xf_temp = Feature_f;

			alphf = (1-interp_factor) * alphf + interp_factor * alphf_temp;
			for(int i=0; i < xf_temp.size();i++){
				xf[i] = (1-interp_factor) * xf[i] + interp_factor * xf_temp[i];
			}

		}
	}


}

