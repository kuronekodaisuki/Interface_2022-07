// OpenGL.cpp : OpenGLとの連携サンプル
//

#include <GL/freeglut.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio.hpp>

#include "OpenCL.h"

#ifdef _DEBUG
#pragma comment(lib, "freeglutd.lib")
#else
#pragma comment(lib, "freeglut.lib")
#endif

#ifndef GL_BGR
#define GL_BGR	GL_BGR_EXT
#define GL_BGRA	GL_BGRA_EXT
#endif

enum FILTER
{
	NONE,
	GAUSSIAN3x3,
	MEDIAN3x3,
	MEDIAN5x5
};


cv::VideoCapture camera;	// カメラ
GLuint width = 640;			// 画像横　1280;
GLuint height = 480;		// 画像縦　720;

cv::Mat image, rgba;		// 画像
cv::Mat mx, my;				// 変形マップ

LARGE_INTEGER freq;			// パフォーマンスカウンタ周波数
double msec;				// 処理時間

#ifdef USE_OPENCL_TEXTURE
OpenCL openCL;
#endif

cl_GLuint texture;
cl_mem	clTextute;
cl_mem	bgra, unorm;
FILTER filter = NONE;


void init(int argc, char* argv[]);
void drawScene();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(GLint w, GLint h);
void calcMap(uint width, uint height, float focal, cv::Mat& mx, cv::Mat& my);

cl_image_format format8UC4 = { CL_RGBA, CL_UNSIGNED_INT8 };	// OpenCLは2のべき乗なのでこちら
cl_image_format format8NC4 = { CL_RGBA, CL_UNORM_INT8 };	// テクスチャーは正規化

int main(int argc, char* argv[])
{
#ifdef USE_OPENCL_TEXTURE
	bool bCLGLShare = openCL.CheckCLGLShareing();
	openCL.SelectDevice();
	clTextute = openCL.CreateGLTexture(texture);

	bgra = openCL.CreateImage(width, height, format8UC4, IMAGE_MODE::READ_WRITE);
	unorm = openCL.CreateImage(width, height, format8NC4, IMAGE_MODE::READ_WRITE);
#endif

	// カメラ開始
	camera.open(0, cv::CAP_MSMF, std::vector<int>{ cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_NONE });
	camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
	camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	glutInitWindowSize(width, height);

	init(argc, argv);

	glutMainLoop();

	// カメラ終了
	camera.release();
	return 0;
}

/// <summary>
/// 画像を取得
/// </summary>
/// <returns></returns>
GLvoid idle()
{
	// start timer
	LARGE_INTEGER begin, end;
	QueryPerformanceCounter(&begin);

	// 画像取得
	camera.read(image);

	cv::cvtColor(image, rgba, CV_RGB2BGRA);

#ifdef USE_OPENCL_TEXTURE
	cl_event event;
	cl_int err = openCL.WriteImage(rgba.data, width, height, 4, bgra, &event);
	openCL.UpdateTexture(clTextute, bgra, unorm);
	glBindTexture(GL_TEXTURE_2D, texture);
#else
	// テクスチャに変換
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGBA,
		width,
		height,
		0,
		GL_BGRA,
		GL_UNSIGNED_BYTE,
		rgba.data);
#endif

	// Update display
	glutPostRedisplay();

	QueryPerformanceCounter(&end);
	msec = 1000 * (double)(end.QuadPart - begin.QuadPart) / freq.QuadPart;
}

/// <summary>
/// 表示
/// </summary>
void drawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);

	// Set Projection Matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, width, height, 0);

	// Switch to Model View Matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Draw a textured quad
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f((float)width, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f((float)width, (float)height);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (float)height);
	glEnd();

	glFlush();
	glutSwapBuffers();
}

/// <summary>
/// glut初期化
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
GLvoid init(int argc, char* argv[])
{
	// glut設定
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutCreateWindow(argv[0]);
	glClearColor(0.0, 0.0, 1.0, 1.0);

#ifdef USE_OPENCL_TEXTURE
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glBindTexture(GL_TEXTURE_2D, 0);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
#else
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
#endif

	// コールバック関数を設定
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutDisplayFunc(drawScene);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	// パフォーマンスカウンタ周波数を取得
	QueryPerformanceFrequency(&freq);
}

/// <summary>
/// リサイズ
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <returns></returns>
GLvoid reshape(GLint w, GLint h)
{
	glViewport(0, 0, w, h);
}

/// <summary>
/// キー入力処理
/// </summary>
/// <param name="key"></param>
/// <param name="x"></param>
/// <param name="y"></param>
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'Q':
	case 'q':
		exit(0);
		break;

	case 'n':
	case 'N':
		filter = NONE;
		break;

	case 'g':
	case 'G':
		filter = GAUSSIAN3x3;
		break;

	case '3':
		filter = MEDIAN3x3;
		break;

	case '5':
		filter = MEDIAN5x5;
		break;

	case 'f':
		printf("%f msec\n", msec);
		break;
	}
}

/// <summary>
/// マウス入力処理
/// </summary>
/// <param name="button"></param>
/// <param name="state"></param>
/// <param name="x"></param>
/// <param name="y"></param>
void mouse(int button, int state, int x, int y)
{

}

/// <summary>
/// 変形マップ算出
/// </summary>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="focal"></param>
/// <param name="mx"></param>
/// <param name="my"></param>
void calcMap(uint width, uint height, float focal, cv::Mat &mx, cv::Mat &my)
{
	cv::Mat cameraMatrix(3, 3, CV_32FC1);
	cv::Mat distortionCoeff;
	cv::Size size(width, height), newSize;
	
	cameraMatrix.at<float>(0) = focal;
	cameraMatrix.at<float>(1) = 0.0f;
	cameraMatrix.at<float>(2) = (float)(size.width / 2);
	cameraMatrix.at<float>(3) = 0.0f;
	cameraMatrix.at<float>(4) = focal;
	cameraMatrix.at<float>(5) = (float)(size.width / 2);
	cameraMatrix.at<float>(6) = 0.0f;
	cameraMatrix.at<float>(7) = 0.0f;
	cameraMatrix.at<float>(8) = 1.0f;

	cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, NULL, size, 1, size);
	cv::initUndistortRectifyMap(cameraMatrix, NULL, cv::Mat(), newCameraMatrix, size, CV_32FC1, mx, my);
}