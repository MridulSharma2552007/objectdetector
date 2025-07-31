//libraries
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

//namespace std
using namespace std;
using namespace cv;
using namespace dnn;


int main(){
//initialization of file paths
string file_path="/home/retro_nokia/objectDetectionInC/";
string config=file_path+"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
string weights=file_path+"frozen_inference_graph.pb";
string labelsFile=file_path+"coco-labels-paper.txt";

// Load class labels
vector<string>class_names;
string line;
ifstream ifs(labelsFile.c_str());
while(getline(ifs,line)){
    class_names.push_back(line);
}
//load model
Net net =readNetFromTensorflow(weights,config);
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

// Open camera
VideoCapture cap(1);
    if(!cap.isOpened()){
    cout<<"Camera failed to open."<<endl;
    }
        else {
            cout<<"Camera opened successfully."<<endl;
        }


Mat frame;
while(true){
    cap>>frame;
    if(frame.empty())break;

    Mat blob=blobFromImage(frame,1.0/127.5,Size(320,320),Scalar(127.5,127.5,127.5),true,false);
    net.setInput(blob);
    Mat output=net.forward();
    
    



}
}