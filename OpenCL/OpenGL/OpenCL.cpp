#include <vector>
#include <Windows.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>

#include "OpenCL.h"


// OpenGL
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h> // freeglutはNuGetで取得

// OpenCLライブラリ
#pragma comment(lib, "OpenCL.lib")

#define CHECK_ERRORS(error)

const char* kernelSource =
"__constant sampler_t LINEAR = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;\n"
"__constant sampler_t NEAREST = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
"\n"
"__kernel void gaussian3x3( \n"
"		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4\n"
"		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	uint4 pixel;\n"
"	uint4 p[3][3];\n"
"	p[0][0] = read_imageui(src, NEAREST, (int2)(x - 1, y - 1));\n"
"	p[0][1] = read_imageui(src, NEAREST, (int2)(x - 1, y));\n"
"	p[0][2] = read_imageui(src, NEAREST, (int2)(x - 1, y + 1));\n"
"	p[1][0] = read_imageui(src, NEAREST, (int2)(x, y - 1));\n"
"	p[1][1] = read_imageui(src, NEAREST, (int2)(x, y));\n"
"	p[1][2] = read_imageui(src, NEAREST, (int2)(x, y + 1));\n"
"	p[2][0] = read_imageui(src, NEAREST, (int2)(x + 1, y - 1));\n"
"	p[2][1] = read_imageui(src, NEAREST, (int2)(x + 1, y));\n"
"	p[2][2] = read_imageui(src, NEAREST, (int2)(x + 1, y + 1));\n"
"	pixel.x = ((p[0][0].x + p[0][2].x + p[2][0].x + p[2][2].x)"
"			+ (p[0][1].x + p[1][0].x + p[1][2].x + p[2][1].x) * 2 "
"			+ p[1][1].x * 4) / 16;\n"
"	pixel.y = ((p[0][0].y + p[0][2].y + p[2][0].y + p[2][2].y)"
"			+ (p[0][1].y + p[1][0].y + p[1][2].y + p[2][1].y) * 2 "
"			+ p[1][1].y * 4) / 16;\n"
"	pixel.z = ((p[0][0].z + p[0][2].z + p[2][0].z + p[2][2].z)"
"			+ (p[0][1].z + p[1][0].z + p[1][2].z + p[2][1].z) * 2 "
"			+ p[1][1].z * 4) / 16;\n"
"	pixel.w = 255;\n"
"	write_imageui(dst, (int2)(x, y), pixel);\n"
"}\n"
"\n"
"__kernel void median3x3( \n"
"		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4\n"
"		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	uint i, j, k;"
"	uint4 p[9], q[5], min;\n"
"	p[0] = read_imageui(src, NEAREST, (int2)(x - 1, y - 1));\n"
"	p[1] = read_imageui(src, NEAREST, (int2)(x - 1, y));\n"
"	p[2] = read_imageui(src, NEAREST, (int2)(x - 1, y + 1));\n"
"	p[3] = read_imageui(src, NEAREST, (int2)(x, y - 1));\n"
"	p[4] = read_imageui(src, NEAREST, (int2)(x, y));\n"
"	p[5] = read_imageui(src, NEAREST, (int2)(x, y + 1));\n"
"	p[6] = read_imageui(src, NEAREST, (int2)(x + 1, y - 1));\n"
"	p[7] = read_imageui(src, NEAREST, (int2)(x + 1, y));\n"
"	p[8] = read_imageui(src, NEAREST, (int2)(x + 1, y + 1));\n"
"	for (i = 0; i < 5; i++) {\n"
"		min = (uint4)(255, 0, 0, 0); k = 0;\n"
"		for (j = 0; j < 9; j++) {\n"
"			if (p[j].x < min.x) {\n"
"				min = p[j]; k = j;\n"
"			}\n"
"		}\n"
"		q[i] = min; p[k].x = 255;\n"
"	}\n"
"	write_imageui(dst, (int2)(x, y), q[4]);\n"
"}\n"
"\n"
"__kernel void median5x5( \n"
"		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4\n"
"		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	uint i, j, k;"
"	uint4 p[25], q[13], min;\n"
"	p[0] = read_imageui(src, NEAREST, (int2)(x - 1, y - 1));\n"
"	p[1] = read_imageui(src, NEAREST, (int2)(x - 1, y));\n"
"	p[2] = read_imageui(src, NEAREST, (int2)(x - 1, y + 1));\n"
"	p[3] = read_imageui(src, NEAREST, (int2)(x, y - 1));\n"
"	p[4] = read_imageui(src, NEAREST, (int2)(x, y));\n"
"	p[5] = read_imageui(src, NEAREST, (int2)(x, y + 1));\n"
"	p[6] = read_imageui(src, NEAREST, (int2)(x + 1, y - 1));\n"
"	p[7] = read_imageui(src, NEAREST, (int2)(x + 1, y));\n"
"	p[8] = read_imageui(src, NEAREST, (int2)(x + 1, y + 1));\n"
"	p[9] = read_imageui(src, NEAREST, (int2)(x - 2, y - 2));\n"
"	p[10] = read_imageui(src, NEAREST, (int2)(x - 2, y - 1));\n"
"	p[11] = read_imageui(src, NEAREST, (int2)(x - 2, y));\n"
"	p[12] = read_imageui(src, NEAREST, (int2)(x - 2, y + 1));\n"
"	p[13] = read_imageui(src, NEAREST, (int2)(x - 2, y + 2));\n"
"	p[14] = read_imageui(src, NEAREST, (int2)(x + 2, y - 2));\n"
"	p[15] = read_imageui(src, NEAREST, (int2)(x + 2, y - 1));\n"
"	p[16] = read_imageui(src, NEAREST, (int2)(x + 2, y));\n"
"	p[17] = read_imageui(src, NEAREST, (int2)(x + 2, y + 1));\n"
"	p[18] = read_imageui(src, NEAREST, (int2)(x + 2, y + 2));\n"
"	p[19] = read_imageui(src, NEAREST, (int2)(x - 1, y - 2));\n"
"	p[20] = read_imageui(src, NEAREST, (int2)(x, y - 2));\n"
"	p[21] = read_imageui(src, NEAREST, (int2)(x + 1, y - 2));\n"
"	p[22] = read_imageui(src, NEAREST, (int2)(x - 1, y + 2));\n"
"	p[23] = read_imageui(src, NEAREST, (int2)(x, y + 2));\n"
"	p[24] = read_imageui(src, NEAREST, (int2)(x + 1, y + 2));\n"
"\n"
"	for (i = 0; i < 13; i++) {\n"
"		min = (uint4)(255, 0, 0, 0); k = 0;\n"
"		for (j = 0; j < 25; j++) {\n"
"			if (p[j].x < min.x) {\n"
"				min = p[j]; k = j;\n"
"			}\n"
"		}\n"
"		q[i] = min; p[k].x = 255;\n"
"	}\n"
"	write_imageui(dst, (int2)(x, y), q[12]);\n"
"}\n"
"\n"
;

static const cl_image_format format8UC4 = { CL_RGBA, CL_UNSIGNED_INT8 };

/// <summary>
/// コンストラクタ
/// </summary>
OpenCL::OpenCL(bool USE_GPU): m_image(NULL), m_texture(NULL), m_errorCode(0)
{
	if (USE_GPU)
		m_deviceId = SelectDevice(CL_DEVICE_TYPE_GPU);
	else
		m_deviceId = SelectDevice(CL_DEVICE_TYPE_CPU);

	// コンテキスト作成
	m_context = clCreateContext(NULL, 1, &m_deviceId, NULL, NULL, &m_errorCode);

	// 命令キュー作成
	m_commandQueue = clCreateCommandQueueWithProperties(m_context, m_deviceId, NULL, &m_errorCode);

	// カーネルコードからプログラムを生成
	cl_program program = clCreateProgramWithSource(m_context, 1,
		(const char**)&kernelSource, NULL, &m_errorCode);

	// 
	m_errorCode = clBuildProgram(program, 1, &m_deviceId, "-cl-std=CL2.0", NULL, NULL);

	// カーネルを生成
	m_gaussian3x3 = clCreateKernel(program, "gaussian3x3", &m_errorCode);
	m_median3x3 = clCreateKernel(program, "median3x3", &m_errorCode);
	m_median5x5 = clCreateKernel(program, "median5x5", &m_errorCode);

	clReleaseProgram(program);
}

/// <summary>
/// デストラクタ
/// </summary>
OpenCL::~OpenCL()
{
	clReleaseDevice(m_deviceId);
	clReleaseContext(m_context);
	clReleaseCommandQueue(m_commandQueue);
	clReleaseKernel(m_gaussian3x3);
	clReleaseKernel(m_median3x3);
	clReleaseKernel(m_median5x5);
}

/// <summary>
/// デバイスを選択
/// </summary>
/// <param name="type">デバイス種別</param>
/// <returns>デバイス識別子</returns>
cl_device_id OpenCL::SelectDevice(cl_device_type type)
{
	cl_device_id deviceId = 0;
	cl_uint num_of_platforms = 0;

	// プラットフォーム数を取得
	cl_int err = clGetPlatformIDs(0, 0, &num_of_platforms);

	std::vector<cl_platform_id> platforms(num_of_platforms);

	// プラットフォーム識別子を取得
	err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);

	// 各プラットフォームごとに（通常は一つ）
	for (cl_uint i = 0; i < num_of_platforms; i++)
	{
		cl_uint num_of_devices = 0;

		// OpenCL対応デバイス数を取得
		err = clGetDeviceIDs(platforms[i], type, 0, 0, &num_of_devices);

		cl_device_id* id = new cl_device_id[num_of_devices];

		// デバイス識別子を取得
		err = clGetDeviceIDs(platforms[i], type, num_of_devices, id, 0);

		char profile[80];
		if (S_OK == clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(profile), profile, NULL))
		{
			if (strstr(profile, "FULL_PROFILE") != NULL)
			{
				m_platformId = platforms[i];
				// 各デバイスについて
				for (cl_uint j = 0; j < num_of_devices; j++)
				{
					cl_uint freq, units;
					char deviceName[128];
					char deviceExtensions[3000];

					// デバイス情報を所得
					err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &freq, NULL);
					err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
					err = clGetDeviceInfo(id[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
					err = clGetDeviceInfo(id[j], CL_DEVICE_EXTENSIONS, sizeof(deviceExtensions), deviceExtensions, NULL);
					printf("%s %d Compute units %dMHz\n", deviceName, units, freq);
					deviceId = id[j];
				}

			}
			else
				continue;
		}
	}

	return m_deviceId = deviceId;
}

cl_mem OpenCL::CreateImage(size_t width, size_t height, cl_image_format format, IMAGE_MODE mode, void* pHostPtr)
{
	cl_mem_flags flags = CL_MEM_READ_WRITE;
	switch (mode)
	{
	case IMAGE_MODE::READ_ONLY:
		flags = CL_MEM_READ_ONLY;
		break;
	case IMAGE_MODE::READ_WRITE:
		flags = CL_MEM_READ_WRITE;
		break;
	case IMAGE_MODE::WRITE_ONLY:
		flags = CL_MEM_WRITE_ONLY;
		break;
	}
	cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, NULL };
	return clCreateImage(m_context, flags, &format, &desc, pHostPtr, &m_errorCode);
}

cl_int OpenCL::WriteImage(unsigned char* ptr, unsigned int width, unsigned int height, unsigned int channels, cl_mem image, cl_event* wait, cl_event* finish)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	return clEnqueueWriteImage(m_commandQueue, image, CL_TRUE, origin, region, width * (size_t)channels, 0, ptr, 0, wait, finish);
}

cl_int OpenCL::ReadImage(cl_mem image, unsigned int width, unsigned int height, unsigned int channels, unsigned char* ptr, cl_event* wait, cl_event* finish)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	return clEnqueueReadImage(m_commandQueue, image, CL_TRUE, origin, region, width * (size_t)channels, 0, ptr, 0, wait, finish);
}

cl_int OpenCL::EnqueueGaussian(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish)
{
	size_t size[] = { width, height };

	// TODO: Choice most effective filter
	//__kernel void gaussian3x3( 
	//		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4
	//		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4
	m_errorCode = clSetKernelArg(m_gaussian3x3, 0, sizeof(cl_mem), &input);
	m_errorCode = clSetKernelArg(m_gaussian3x3, 1, sizeof(cl_mem), &output);
	return clEnqueueNDRangeKernel(m_commandQueue, m_gaussian3x3, 2, NULL, size, NULL, 1, wait, finish);
}

cl_int OpenCL::EnqueueMedian3x3(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish)
{
	size_t size[] = { width, height };

	// TODO: Choice most effective filter
	//__kernel void median3x3( 
	//		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4
	//		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4
	m_errorCode = clSetKernelArg(m_median3x3, 0, sizeof(cl_mem), &input);
	m_errorCode = clSetKernelArg(m_median3x3, 1, sizeof(cl_mem), &output);
	return clEnqueueNDRangeKernel(m_commandQueue, m_median3x3, 2, NULL, size, NULL, 1, wait, finish);
}

cl_int OpenCL::EnqueueMedian5x5(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish)
{
	size_t size[] = { width, height };

	// TODO: Choice most effective filter
	//__kernel void median5x5( 
	//		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4
	//		__write_only image2d_t dst)	// CL_UNSIGNED_INT8 x 4
	m_errorCode = clSetKernelArg(m_median5x5, 0, sizeof(cl_mem), &input);
	m_errorCode = clSetKernelArg(m_median5x5, 1, sizeof(cl_mem), &output);
	return clEnqueueNDRangeKernel(m_commandQueue, m_median5x5, 2, NULL, size, NULL, 1, wait, finish);
}


void* OpenCL::AllocSVMMemory(size_t size, cl_svm_mem_flags flags)
{
	return clSVMAlloc(m_context, flags, size, 0);
}

void OpenCL::FreeSvmMemory(void* ptr)
{
	clSVMFree(m_context, ptr);
}

cl_int OpenCL::SVMMap(void* ptr, size_t size, cl_map_flags flags)
{
	return clEnqueueSVMMap(m_commandQueue, CL_TRUE, flags, ptr, size, 1, NULL, NULL);
}

cl_int OpenCL::SVMUnmap(void* ptr)
{
	return clEnqueueSVMUnmap(m_commandQueue, ptr, 1, NULL, NULL);
}
