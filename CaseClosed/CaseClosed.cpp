// Objectron.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

#define MASK_RCNN   "models/mask_rcnn_R_50_FPN_1x.onnx"
#define FCN "models/fcn-resnet50-12.onnx"
#define OBJECTRON "models/eth3d_float32.onnx"
#define U2NET   "models/u2netp_256x256.onnx"

// モデルとカメラ
const char* model = U2NET;
const char* axis = "rtsp://mao:pipo0921@192.168.0.135/axis-media/media.amp";


const int INPUT_WIDTH = 256;
const int INPUT_HEIGHT = 256;
const float SCORE_THRESHOLD = 0.2f;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.01f;

// Text parameters.
const float FONT_SCALE = 0.7f;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);


int main()
{
    dnn::Net net;
    net = dnn::readNet(model);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_OPENCL);

    cv::VideoCapture camera;
    if (camera.open(0))
    {
        Mat frame, blob, convert;
        for (bool loop = true; loop && camera.read(frame);)
        {
            double fx = (double)frame.cols / INPUT_WIDTH;
            double fy = (double)frame.rows / INPUT_HEIGHT;

            dnn::blobFromImage(frame, blob, 1.0 / 255, Size(INPUT_WIDTH, INPUT_HEIGHT));
            net.setInput(blob);
            Mat output = net.forward().reshape(0, INPUT_HEIGHT);

            resize(output, convert, frame.size(), fx, fy);

            // 閾値で
            threshold(convert, output, 0.5, 1, THRESH_TOZERO);

            // 
            output.convertTo(convert, CV_8UC1, 255);


            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time : %.2f ms", t);
            putText(frame, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

            imshow("Frame", frame);
            imshow("Output", output);

            switch (waitKey(10))
            {
            case 'q':
            case 'Q':
                loop = false;
            }
        }
    }
}
