// OpenGL.cpp : OpenGLとの連携サンプル
//

#include <GL/freeglut.h>
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
#endif

cv::VideoCapture camera;	// カメラ
GLuint width = 640;			// 画像横　1280;
GLuint height = 480;		// 画像縦　720;

cv::Mat image, rgba;		// 画像

LARGE_INTEGER freq;			// パフォーマンスカウンタ周波数
double msec;				// 処理時間

OpenCL openCL;

cl_GLuint texture;
cl_mem	clTextute;
cl_mem	bgra, unorm;

void init(int argc, char* argv[]);
void drawScene();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(GLint w, GLint h);
int verifyCLGLSyncObjectsAvailableCL();

cl_image_format format8UC4 = { CL_RGBA, CL_UNSIGNED_INT8 };	// OpenCLは2のべき乗なのでこちら
cl_image_format format8NC4 = { CL_RGBA, CL_UNORM_INT8 };	// テクスチャーは正規化

int main(int argc, char* argv[])
{
	verifyCLGLSyncObjectsAvailableCL();

	// カメラ開始
	camera.open(0);
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
		GL_RGB,
		width,
		height,
		0,
		GL_BGR,
		GL_UNSIGNED_BYTE,
		image.data);
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

	// コールバック関数を設定
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutDisplayFunc(drawScene);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

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

	clTextute = openCL.CreateGLTexture(texture);

	bgra = openCL.CreateImage(width, height, format8UC4, IMAGE_MODE::READ_WRITE);
	unorm = openCL.CreateImage(width, height, format8NC4, IMAGE_MODE::READ_WRITE);
#else
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
#endif
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

	case 'f':
		printf("%f msec\n", msec);
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

void testStatus(int status, const char* errorMsg)
{
	if (status != 0)
	{
		if (errorMsg == NULL)
		{
			printf("Error\n");
		}
		else
		{
			printf("Error: %s", errorMsg);
		}
		exit(EXIT_FAILURE);
	}
}

int g_clEventFromGLsyncObjectSupported = FALSE;

//the extension string can be available at the platform level or at the device level so we check both
int verifyCLGLSyncObjectsAvailableCL()
{
	int status = 0;
	cl_uint numPlatforms = 0;

	printf("\nChecking to see if sync objects are supported...\n");
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	testStatus(status, "clGetPlatformIDs error\n");
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	if (platforms == NULL)
	{
		printf("Error when allocating space for the platforms\n");
		exit(EXIT_FAILURE);
	}

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	testStatus(status, "clGetPlatformIDs error");

	for (unsigned int i = 0; i < numPlatforms; i++)
	{
		printf("******************************************************************************\n");
		char platformVendor[100];
		memset(platformVendor, '\0', 100);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL);
		testStatus(status, "clGetPlatformInfo error");

		char platformName[100];
		memset(platformName, '\0', 100);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
		testStatus(status, "clGetPlatformInfo error");

		char extension_string[1024];
		memset(extension_string, '\0', 1024);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, sizeof(extension_string), extension_string, NULL);
		//printf("Extensions supported: %s\n", extension_string);

		char* extStringStart = NULL;
		extStringStart = strstr(extension_string, "cl_khr_gl_event");
		if (extStringStart == 0)
		{
			printf("Platform %s does not report support for cl_khr_gl_event,\nStill going to check the devices\n", platformName);
		}
		if (extStringStart != 0)
		{
			printf("Platform %s does support cl_khr_gl_event. \nFind out which device (if any) reports support as well\n", platformName);
			g_clEventFromGLsyncObjectSupported = TRUE;

		}

		//get number of devices in the platform
		//for each platform, query extension string
		cl_uint num_devices = 0;
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		testStatus(status, "Error getting number of devices\n");

		cl_device_id* clDevices = NULL;
		clDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		if (clDevices == NULL)
		{
			printf("Error when allocating\n");
		}
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, clDevices, 0);
		testStatus(status, "clGetDeviceIDs error");

		for (unsigned int iDevNum = 0; iDevNum < num_devices; iDevNum++)
		{
			//query each for their extension string
			//print out the device type just to make sure we got it right
			cl_device_type device_type;
			char vendorName[256];
			memset(vendorName, '\0', 256);
			char devExtString[1024];
			memset(devExtString, '\0', 1024);

			clGetDeviceInfo(clDevices[iDevNum], CL_DEVICE_TYPE, sizeof(cl_device_type), (void*)&device_type, NULL);
			clGetDeviceInfo(clDevices[iDevNum], CL_DEVICE_VENDOR, (sizeof(char) * 256), &vendorName, NULL);
			clGetDeviceInfo(clDevices[iDevNum], CL_DEVICE_EXTENSIONS, (sizeof(char) * 1024), &devExtString, NULL);

			char* extStringStart = NULL;
			extStringStart = strstr(devExtString, "cl_khr_gl_event");

			char devTypeString[256];
			memset(devTypeString, '\0', 256);

			if (device_type == CL_DEVICE_TYPE_CPU)
			{
				strcpy_s(devTypeString, "CPU");
			}
			else if (device_type == CL_DEVICE_TYPE_GPU)
			{
				strcpy_s(devTypeString, "GPU");
			}
			else
			{
				strcpy_s(devTypeString, "Not a CPU or GPU"); //for sample code, not product
			}

			if (extStringStart != 0)
			{
				printf("Device %s in %s platform supports synch objects between CL and GL,\n\tNo need for a glFinish() on this device\n", devTypeString, vendorName);
				g_clEventFromGLsyncObjectSupported = TRUE;
			}
			else
			{
				printf("Device %s in %s platform does not support CL/GL sync, \n\tglFinish() would be required on this device\n", devTypeString, vendorName);
			}
		} //end for(...)
		free(clDevices);

	}

	printf("******************************************************************************\n");

	return status;
}
