#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main() {
    // File paths
    string path = "/home/retro_nokia/objectDetectionInC/";
    string config = path + "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    string weights = path + "frozen_inference_graph.pb";
    string labelsFile = path + "coco-labels-paper.txt";

    // Load class labels
    vector<string> class_names;
    // class_names.push_back("background");  // SSD MobileNet expects 'background' at index 0
    ifstream ifs(labelsFile.c_str());
    string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }

    // Load the model
    Net net = readNetFromTensorflow(weights, config);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);



    // Open camera
    VideoCapture cap(1);  // Try /dev/video1
    if (!cap.isOpened()) {
        cout << "Camera 1 failed. Trying camera 0..." << endl;
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open any camera." << endl;
            return -1;
        }
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Create input blob
        Mat blob = blobFromImage(frame, 1.0 / 127.5, Size(320, 320), Scalar(127.5, 127.5, 127.5), true, false);
        net.setInput(blob);
        Mat output = net.forward();

        // Extract detection results
        Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.5) {  // Threshold increased to avoid false detections
                int class_id = static_cast<int>(detectionMat.at<float>(i, 1));
                if (class_id >= class_names.size()) continue; // Prevent out-of-range errors

                int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                // Clamp to frame boundaries
                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                right = max(0, min(right, frame.cols - 1));
                bottom = max(0, min(bottom, frame.rows - 1));

                // Draw box
                rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

                // Draw label
                string label = format("%s: %.1f%%", class_names[class_id].c_str(), confidence * 100);
                int baseLine;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int y = max(top, labelSize.height + 10);
                rectangle(frame, Point(left, y - labelSize.height - 5),
                          Point(left + labelSize.width, y + baseLine - 5), Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(left, y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            }
        }

        imshow("Object Detection", frame);
        if (waitKey(1) == 27) break;  // ESC to quit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
