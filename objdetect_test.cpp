#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>


using namespace std;
using namespace cv;
using namespace dnn;


int main(){
//file paths
string path="/home/retro_nokia/objectDetectionInC/";
string config=path+"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
string weight=path+"frozen_inference_graph.pb";
string labelFiles=path+"coco-labels-paper.txt";


//load class labels
vector<string> class_name;
ifstream ifs(labelFiles.c_str());
string line;
while(getline(ifs,line)){
class_name.push_back(line);//makes a list od objects in the text file
//cout<<line<<endl;}



//load model
Net net=readNetFromTensorflow(weight,config);
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);


//open camera
VideoCapture cap(1);//using camera 1
 if(!cap.isOpened){
    cout<<"Problem opening the camera"<<endl;
    return -1;
}
Mat frame;
while (true)
{
    cap>>frame;
    if(frame.empty()){
        break;
    }
    Mat blob =blobFromImage(frame,1.0/127.5,Size(320,320),Scaler(127.5,127.5,127.5),true,false);
    net.setInput(blob);
    Mat output=net.forward();


//extract result
 Mat detectionMat(output.size[2],output.size[3],CV_32F,output.ptr<float>());
 

}

 
}