#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include "fhog.hpp"

using namespace std;
using namespace cv;

const float padding = 1.5;
const float output_sigma_factor = 0.1;
const int cell_size = 4;
const float kernel_sigma = 0.5;
const float lamda = 1e-4;
const float interp_factor = 0.075;

Mat gaussian_shaped_label(float sigma, Size size){

	Mat gaussian_label(size.height,size.width,CV_32FC1);			//gaussian_label
	Mat x(size.height,size.width,CV_32FC1);							//The distance between i and 
	Mat y(size.height,size.width,CV_32FC1);

	Mat x_row(1,size.width,CV_32FC1);
	for(int i = 0; i < size.width; i++){
		x_row.at<float>(0,i) = i - size.width/2;
	}
	for(int i = 0; i < size.height;i++){
		x_row.row(0).copyTo(x.row(i));
	}

	Mat y_col(size.height,1,CV_32FC1);
	for(int i = 0; i < size.height; i++){
		y_col.at<float>(i,0) = i - size.height/2;
	}
	for(int i = 0; i < size.width;i++){
		y_col.col(0).copyTo(y.col(i));
	}

	gaussian_label = x.mul(x) + y.mul(y);
	gaussian_label = (gaussian_label * -0.5 )/(sigma*sigma);
	exp(gaussian_label,gaussian_label);

	

	int center_x = gaussian_label.cols / 2;
	int center_y = gaussian_label.rows / 2;

	/*if (gaussian_label.at<float>(center_y-1,center_x-1) == 1){center_y = center_y -1;center_x = center_x -1;}
	if (gaussian_label.at<float>(center_y-1,center_x) == 1){center_y = center_y -1;}
	if (gaussian_label.at<float>(center_y-1,center_x+1) == 1){center_y = center_y -1;center_x = center_x + 1;}
	if (gaussian_label.at<float>(center_y+1,center_x-1) == 1){center_y = center_y +1;center_x = center_x -1;}
	if (gaussian_label.at<float>(center_y+1,center_x) == 1){center_y = center_y +1;}
	if (gaussian_label.at<float>(center_y+1,center_x+1) == 1){center_y = center_y +1;center_x = center_x +1;}
	*/

	/*交换左右两边*/
	Mat temp_lr(size.height,size.width,CV_32FC1);
	Mat l(gaussian_label,Rect(0,0,center_x ,size.height));
	Mat r(gaussian_label,Rect(center_x,0,size.width-center_x,size.height));
	r.copyTo(temp_lr(Rect(0,0,r.cols,r.rows)));
	l.copyTo(temp_lr(Rect(r.cols,0,l.cols,l.rows)));

	gaussian_label = temp_lr;

	/*交换上下两边*/
	Mat temp_ud(size.height,size.width,CV_32FC1);
	Mat u(gaussian_label,Rect(0,0,size.width,center_y));
	Mat d(gaussian_label,Rect(0,center_y,size.width,size.height-center_y));
	d.copyTo(temp_ud(Rect(0,0,d.cols,d.rows)));
	u.copyTo(temp_ud(Rect(0,d.rows,u.cols,u.rows)));

	gaussian_label = temp_ud;
	
	/*确认Label（0，0）的值是1*/
	if(gaussian_label.at<float>(0,0) != 1.0){
		cout<<"Error:Incorrect Label!"<<endl;
	}

	return gaussian_label;

}

Mat get_subwindow(Mat im, Point point, Size sz){

	int crop_x = min(im.cols,max(0,point.x-sz.width/2));
	int crop_y = min(im.rows,max(0,point.y-sz.height/2));

	int crop_width = min(im.cols,(point.x+sz.width/2));
	int crop_height = min(im.rows,(point.y+sz.height/2));

	Mat roi = im(Rect(crop_x,crop_y,crop_width-crop_x,crop_height-crop_y));

	int add_border_left = point.x>(sz.width/2)?0:(point.x - sz.width/2);
	int add_border_right = (point.x+sz.width/2)>im.cols?(point.x+sz.width/2-im.cols):0;
	int add_border_up = point.y>(sz.height/2)?0:(point.y - sz.height/2);
	int add_border_down = (point.y+sz.height/2)>im.rows?(point.y+sz.height/2-im.rows):0;

	copyMakeBorder(roi,roi,add_border_up,add_border_down,add_border_left,add_border_right,BORDER_REPLICATE);


	return roi;

}

vector<cv::Mat> get_feature(Mat patch,Mat cos_window){

	FHoG f_hog_;
	vector<Mat> x_vector;

	patch.convertTo(patch,CV_32FC1, 1.0/255);

	x_vector = f_hog_.extract(patch);

	for(int i=0;i<x_vector.size();i++){
		x_vector[i] = x_vector[i].mul(cos_window);
	}
	
	return x_vector;

} 


Mat GaussianCorrelation(vector<Mat> xf, vector<Mat> yf){

	int N = xf[0].cols * xf[0].rows;

	double xx = 0 ,yy = 0;

	Mat xy(xf[0].rows,xf[0].cols,CV_32FC1),xyf,xy_temp;
	for (int i = 0;i<xf.size();i++){

		xx += norm(xf[i]) * norm(xf[i]) / N;
		yy += norm(yf[i]) * norm(yf[i]) / N;

		mulSpectrums(xf[i], yf[i], xyf, 0, true);

		idft(xyf, xy_temp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		xy += xy_temp;
	}

	float numel_xf = N * xf.size();
	Mat k,kf;
	exp(((-1 / (kernel_sigma * kernel_sigma)) * max(0,(xx + yy - 2 * xy)/numel_xf)),k);
	k.convertTo(k,CV_32FC1);
	dft(k,kf,DFT_COMPLEX_OUTPUT);

	return kf;

}

Mat ComplexDiv(const cv::Mat &x1, const cv::Mat &x2) {

  vector<cv::Mat> planes1;
  split(x1, planes1);
  vector<cv::Mat> planes2;
  split(x2, planes2);
  vector<cv::Mat> complex(2);
  Mat cc = planes2[0].mul(planes2[0]);
  Mat dd = planes2[1].mul(planes2[1]);


  complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
  complex[1] = (-planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0])) / (cc + dd);
  cv::Mat result;
  cv::merge(complex, result);
  return result;

}

Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2) {

  vector<cv::Mat> planes1;
  split(x1, planes1);
  vector<cv::Mat> planes2;
  split(x2, planes2);
  vector<cv::Mat>complex(2);

  complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
  complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
  Mat result;
  merge(complex, result);
  return result;

}