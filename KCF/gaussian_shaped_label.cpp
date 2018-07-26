/*#include <io.h>  
#include <iostream>  
#include <vector>  
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void getFiles(string path, vector<string>& files)
{
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
                files.push_back(fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose函数结束查找
        _findclose(hFile);
    }
}


int main(){
    char * filePath = "C:\\Users\\Yuan\\Desktop\\目标跟踪";//自己设置目录  
    vector<string> files;

    ////获取该路径下的所有文件  
    getFiles(filePath, files);

    char str[30];
    int size = files.size();
    for (int i = 0; i < size; i++)
    {
        cout << files[i].c_str() << endl;
    }
	getchar();
}*/