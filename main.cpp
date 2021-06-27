#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string netname;
};

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame);
private:
	const float anchors[3][6] = { {4,5,  8,10,  13,16}, {23,29,  43,55,  73,105},{146,217,  231,300,  335,433} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;

	char netname[20];
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<int> landmark);
	void sigmoid(Mat* out, int length);
};

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLO::YOLO(Net_config config)
{
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	strcpy_s(this->netname, config.netname.c_str());

	string modelFile = this->netname;
	modelFile += "-face.onnx";
	this->net = readNet(modelFile);
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, vector<int> landmark)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	for (int i = 0; i < 5; i++)
	{
		circle(frame, Point(landmark[2 * i], landmark[2 * i + 1]), 1, Scalar(0, 255, 0), -1);
	}
}

void YOLO::sigmoid(Mat* out, int length)
{
	float* pdata = (float*)(out->data);
	int i = 0; 
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

void YOLO::detect(Mat& frame)
{
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector< vector<int>> landmarks;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = 16, row_ind = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score,x1,y1, ... ,x5,y5,face_score
	for (n = 0; n < 3; n++)   ///特征图尺度
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float* pdata = (float*)outs[0].data + row_ind * nout;
					float box_score = sigmoid_x(pdata[4]);
					if (box_score > this->objThreshold)
					{
						float face_score = sigmoid_x(pdata[15]);
						//if (face_score > this->confThreshold)
						//{ 
						float cx = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  ///h

						int left = (cx - 0.5*w)*ratiow;
						int top = (cy - 0.5*h)*ratioh;   

						confidences.push_back(face_score);
						boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
						vector<int> landmark(10);
						for (k = 5; k < 15; k+=2)
						{
							const int ind = k - 5;
							landmark[ind] = (int)(pdata[k] * anchor_w + j * this->stride[n])*ratiow;
							landmark[ind + 1] = (int)(pdata[k + 1] * anchor_h + i * this->stride[n])*ratioh;
						}
						landmarks.push_back(landmark);
						//}
					}
					row_ind++;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, landmarks[idx]);
	}
}

int main()
{
	Net_config yolo_nets = {0.3, 0.5, 0.3, "yolov5s"};  ///choice = [yolov5s, yolov5m, yolov5l]
	YOLO yolo_model(yolo_nets);
	string imgpath = "selfie.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);
	
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}