// Objectron.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

//#include <stdio.h>

using namespace std;
using namespace cv;

#define U2NET_256x256   "models/u2netp_256x256.onnx"
#define U2NET_320x320   "models/u2netp_320x320.onnx"
#define U2NET_480x640   "models/u2netp_480x640.onnx"

const int INPUT_WIDTH = 256;
const int INPUT_HEIGHT = 256;

// モデルとカメラ
const char* model = U2NET_256x256;
const char* axis = "rtsp://mao:pipo0921@192.168.0.135/axis-media/media.amp";


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

    // 学習済みモデルを読み込む
    net = dnn::readNet(model);

    // OpenCLでの推論処理を設定
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_OPENCL);

    cv::VideoCapture camera;
    if (camera.open(0))
    {
        Mat frame, blob, convert;

        // カメラから映像を取得
        for (bool loop = true; loop && camera.read(frame);)
        {
            double fx = (double)frame.cols / INPUT_WIDTH;
            double fy = (double)frame.rows / INPUT_HEIGHT;

            // 映像を正規化して
            dnn::blobFromImage(frame, blob, 1.0 / 255, Size(INPUT_WIDTH, INPUT_HEIGHT));

            // ネットワークに設定
            net.setInput(blob);

            // シルエットを推定、ヒートマップ(単精度浮動小数)を取得
            Mat heatmap = net.forward().reshape(0, INPUT_HEIGHT);

            // 元画像と同じ大きさに
            resize(heatmap, convert, frame.size(), fx, fy);

            // 閾値で
            //threshold(convert, output, 0.5, 1, THRESH_TOZERO);

            // 
            //convert.convertTo(heatmap, CV_8UC1, 255);

            Mat Afc3, Bfc3;
            Mat t[] = { convert, convert, convert };
            merge(t, 3, Afc3);

            frame.convertTo(Bfc3, CV_32FC3, 1);

            Mat C = Afc3.mul(Bfc3);

            C.convertTo(heatmap, CV_8UC3); 

            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double time = net.getPerfProfile(layersTimes) / freq;
            string label = format("Inference time : %.2f ms", time);
            putText(frame, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

            imshow("Frame", frame);
            imshow("Output", heatmap);

            switch (waitKey(10))
            {
            case ' ':
                if (imwrite("mask.bmp", heatmap))
                    printf("Snapshot mask\n");
                break;
            case 'q':
            case 'Q':
                loop = false;
            }
        }
    }
}
