/*#include <io.h>  
#include <iostream>  
#include <vector>  
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

void getFiles(string path, vector<string>& files)
{
    //�ļ����  
    long   hFile = 0;
    //�ļ���Ϣ������һ���洢�ļ���Ϣ�Ľṹ��  
    struct _finddata_t fileinfo;
    string p;//�ַ��������·��
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//�����ҳɹ��������
    {
        do
        { 
            if ((fileinfo.attrib &  _A_ARCH))
            {
                files.push_back(fileinfo.name);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose������������
        _findclose(hFile);
    }
}


int main(){
    char * filePath = "C:\\Users\\Yuan\\Desktop\\Ŀ�����";//�Լ�����Ŀ¼  
    vector<string> files;

    ////��ȡ��·���µ������ļ�  
    getFiles(filePath, files);

    char str[30];
    int size = files.size();
    for (int i = 0; i < size; i++)
    {
        cout << files[i].c_str() << endl;
    }
	getchar();
}*/