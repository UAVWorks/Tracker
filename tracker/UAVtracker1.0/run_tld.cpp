#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include <iostream>
#include <sstream>
#include "TLD.h"
#include <stdio.h>
#include<time.h>
#include<iomanip>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool tk = true;	//卡尔曼开关
//bool tk = false;
//bool tma = false;	//方向预测开关
bool tma = true;
bool rep = false;
bool fromfile=false;
string video;

int main(int argc, char * argv[]){//int argc, char * argv[]     "-p" "parameters.yml" "-s" 0 "-r"
  VideoCapture capture;
 //capture.open(0);
  FileStorage fs;
  ifstream fv;
  fs.open("parameters.yml", FileStorage::READ);
  fv.open("database.txt", ios::in);
  if (!fv.is_open())
  {
	  cout << "打开文件错误！" << endl; exit(1);
  }
  string line;
  vector<vector<string>> storage;
  vector<int>time;
  vector<float>trackrate;
  while (getline(fv, line))
  {
	  stringstream ss(line);
	  string str;
	  vector<string> st;
	  while (getline(ss, str,','))
	  {
		  st.push_back(str);
	  }
	  storage.push_back(st);
  }
  for (int i = 0; i < storage.size();i++)
  {
	  
	  string st1 = storage[i][0];
	  capture.open(storage[i][0]);

	  if (!capture.isOpened())
	  {
		  cout << "capture device failed to open!" << endl;
		  return 1;
	  }
	  fromfile = true;
	  box.x = atoi(storage[i][1].c_str());
	  box.y = atoi(storage[i][2].c_str());
	  box.width = atoi(storage[i][3].c_str());
	  box.height = atoi(storage[i][4].c_str());
	  gotBB = true;

	 // cvNamedWindow("TLD", CV_WINDOW_AUTOSIZE);
	 // cvSetMouseCallback("TLD", mouseHandler, NULL);
	  //TLD framework
	  TLD tld;
	  //Read parameters file
	  tld.read(fs.getFirstTopLevelNode());
	  Mat frame;
	  Mat last_gray;
	  Mat first;
	  if (fromfile){
		  capture >> frame;
		  cvtColor(frame, last_gray, CV_RGB2GRAY);
		  frame.copyTo(first);
	  }
	  else{
		  capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
		  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	  }

	  ///Initialization
	  //GETBOUNDINGBOX:
	  while (!gotBB)
	  {
		  while (!gotBB)
		  {
			  if (!fromfile){
				  capture >> frame;
			  }
			  else
				  first.copyTo(frame);

			  cvtColor(frame, last_gray, CV_RGB2GRAY);
			  drawBox(frame, box);
			  imshow("TLD", frame);
			  if (cvWaitKey(33) == 'q')
				  return 0;
		  }
		  if (min(box.width, box.height) < (int)fs.getFirstTopLevelNode()["min_win"])
		  {
			  cout << "Bounding box too small, try again." << endl;
			  gotBB = false;
		  }
	  }
	  /*
	  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
	  cout << "Bounding box too small, try again." << endl;
	  gotBB = false;
	  goto GETBOUNDINGBOX;
	  }
	  */
	  //Remove callback

	  //cvSetMouseCallback("TLD", NULL, NULL);
	  // printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
	  //Output file
	  FILE  *bb_file = fopen("missingframe.txt", "w");
	  //TLD initialization
	  tld.init(first, box, bb_file);

	  ///Run- 
	  Mat current_gray;
	  BoundingBox pbox;

	  bool status = true;//status lastboxfound
	  int frames = 1;
	  int detections = 1;
	  int t = 0;
	  Mat current;
	  while (capture.read(frame))
	  {

		  /* if (!capture.read(frame))
		  break;*/
		  clock_t t1 = clock();
		  //get frame
		  cvtColor(frame, current_gray, CV_RGB2GRAY);
		  //Process Frame
		  clock_t t3 = clock();
		  tld.processFrame(frame, last_gray, current_gray, pbox, status, tl, tk, bb_file);//跟踪，检测，综合，学习四个模块
		  clock_t t4 = clock();
		  //printf("运行%dms\n", t4 - t3);
		  //Draw Points
		  if (status){//跟踪成功，这里指的是我们最后通过综合模块输出了一个框，称为输出成功

			  frame.copyTo(current);
			  drawBox(frame, pbox);
			  //detections++;
		  }
		  //Display
		  imshow("TLD", frame);
		  //swap points and images
		  swap(last_gray, current_gray);
		  frames++;
		  if (cvWaitKey(2) == 'q')
			  break;
		  clock_t t2 = clock();
		  //cout << "Time is " << t2 - t1 << "ms"<<endl;
		  if (tld.Rtrack)
		  {
			  t += (t2 - t1);
			  detections++;
		  }
		  time.push_back(t / detections);
		  trackrate.push_back((float)detections / (float)frames);
	  }
	  
	  fclose(bb_file);
	  capture.release();
	/*  system("pause");*/
  }
  cout << left;
  cout << endl;
  cout << "************************************************" << endl;
  cout << setw(24) << "video name" << setw(20) << "fps" << setw(20) << "tracking accuracy(%)" << endl;
  cout << setprecision(3) << fixed;
  for (int i = 0; i < storage.size(); i++)
  {
	  cout << setw(24) << storage[i][0] << setw(20) << 1000 / time[i] << setw(20);
	  cout<< trackrate[i] * 100  << endl;
  }
  cout << "************************************************" << endl;
  
  return 0;
}
