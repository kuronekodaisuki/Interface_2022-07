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
"__kernel void remapImage(\n"
"__read_only image2d_t src,			// CL_UNSIGNED_INT8 x 4\n"
"__read_only image2d_t mapX,		// CL_FLOAT\n"
"__read_only image2d_t mapY,		// CL_FLOAT\n"
"__write_only image2d_t	dst)		// CL_UNSIGNED_INT8 x 4\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	float X = read_imagef(mapX, (int2)(x, y)).x;\n"
"	float Y = read_imagef(mapY, (int2)(x, y)).x;\n"
"	uint4 pixel = read_imageui(src, LINEAR, (float2)(X, Y));\n"
"	write_imageui(dst, (int2)(x, y), pixel);\n"
"}\n"
"\n"
"__kernel void convertTexture( \n"
"		__read_only image2d_t src,	// CL_UNSIGNED_INT8 x 4\n"
"		__write_only image2d_t dst)	// CL_UNORM_INT8 x 4\n"
"{\n"
"	int x = get_global_id(0);\n"
"	int y = get_global_id(1);\n"
"	uint4 pixel = read_imageui(src, NEAREST, (int2)(x, y));\n"
"	float4 f_pixel = (float4)(pixel.z / 255.0f, pixel.y / 255.0f, pixel.x / 255.0f, pixel.w / 255.0f);\n"
"	write_imagef(dst, (int2)(x, y), f_pixel);\n"
"}\n"
;

static const cl_image_format format8UC4 = { CL_RGBA, CL_UNSIGNED_INT8 };
static const cl_image_format formatMap = { CL_R, CL_FLOAT };
static const cl_image_format formatGL = { CL_RGBA, CL_UNORM_INT8 };

#define GETFUNCTION(platform, x) \
    (x ## _fn)clGetExtensionFunctionAddressForPlatform(platform, #x);

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
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// カーネルを生成
	m_remapImage = clCreateKernel(program, "remapImage", &m_errorCode);
	m_convertTexture = clCreateKernel(program, "convertTexture", &m_errorCode);
	
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
	clReleaseKernel(m_remapImage);
	clReleaseKernel(m_convertTexture);
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
		err = clGetDeviceIDs(
			platforms[i],
			type,
			0,
			0,
			&num_of_devices
		);

		cl_device_id* id = new cl_device_id[num_of_devices];

		// デバイス識別子を取得
		err = clGetDeviceIDs(
			platforms[i],
			type,
			num_of_devices,
			id,
			0
		);

		char platformName[80];
		if (S_OK == clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(platformName), platformName, NULL))
		{
			printf("Platform:%s\n", platformName);
		}

		char extension_string[6000];
		memset(extension_string, ' ', 6000);
		if (S_OK == clGetPlatformInfo(platforms[i],	CL_DEVICE_EXTENSIONS,	sizeof(extension_string), extension_string,	NULL))
		{
			char* extStringStart = strstr(extension_string, "cl_khr_gl_sharing");
			if (extStringStart != 0) 
			{
				printf("Platform supports cl_khr_gl_sharing\n");
				// Reference https://software.intel.com/en-us/articles/sharing-surfaces-between-opencl-and-opengl-43-on-intel-processor-graphics-using-implicit
				HGLRC hGLRC = wglGetCurrentContext(); 
				HDC hDC = wglGetCurrentDC();
				cl_context_properties opengl_props[] = {
					CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i],
					CL_GL_CONTEXT_KHR, (cl_context_properties)hGLRC,
					CL_WGL_HDC_KHR, (cl_context_properties)hDC,
					0
				};
				//clCreateContext(cps, 1, g_clDevices, NULL, NULL, &status);
				size_t devSizeInBytes = 0;
				clGetGLContextInfoKHR_fn clGetGLContextInfoKHR = GETFUNCTION(platforms[i], clGetGLContextInfoKHR);
				m_errorCode = clGetGLContextInfoKHR(opengl_props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, 0, NULL, &devSizeInBytes);
				if (m_errorCode == S_OK)
				{
					const size_t devNum = devSizeInBytes / sizeof(cl_device_id);
					std::vector<cl_device_id> devices(devNum);
					clGetGLContextInfoKHR(opengl_props, CL_DEVICES_FOR_GL_CONTEXT_KHR, devSizeInBytes, &devices[0], NULL);
					for (size_t k = 0; k < devNum; k++)
					{
						cl_device_type t;
						clGetDeviceInfo(devices[k], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
						if (t == CL_DEVICE_TYPE_GPU)
						{
							//platformNum++;
							char devicename[80];
							clGetDeviceInfo(devices[k], CL_DEVICE_NAME, sizeof(devicename), devicename, NULL);
							char buffer[32];
							clGetDeviceInfo(devices[k], CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL);
							printf("  %s %s\n", devicename, buffer);
						}
					}
				}

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

/*
cl_mem OpenCL::CreateImage(size_t width, size_t height, cl_channel_order channels, cl_channel_type type, IMAGE_MODE mode)
{
	cl_mem_flags flags = CL_MEM_READ_WRITE;
	switch (mode)
	{
	case IMAGE_MODE::READ_ONLY:
		flags = CL_MEM_READ_ONLY;
		break;
	case IMAGE_MODE::READ_WRITE:
		flags = CL_MEM_WRITE_ONLY;
		break;
	}
	cl_image_format format = { CL_RGBA, type };
	switch (channels)
	{
	case 1:
		format.image_channel_order = CL_R;
		break;
	case 2:
		format.image_channel_order = CL_RG;
		break;
	case 3:
		format.image_channel_order = CL_RGB;
		break;
	case 4:
		format.image_channel_order = CL_RGBA;
		break;
	}
	format.image_channel_data_type = type;
	cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, NULL };
	cl_mem memory = clCreateImage(m_context, CL_MEM_READ_ONLY, &format, &desc, 0, &m_errorCode);
	if (m_errorCode != 0)
		printf("ERROR:%d", m_errorCode);
	return memory;
}
*/

cl_int OpenCL::WriteImage(unsigned char* ptr, unsigned int width, unsigned int height, unsigned int channels, cl_mem memory, cl_event* event)
{
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	return clEnqueueWriteImage(m_commandQueue, memory, CL_TRUE, origin, region, width * (size_t)channels, 0, ptr, 0, NULL, event);
}

cl_mem OpenCL::CreateGLTexture(cl_GLuint texture)
{
	// Attension: 
	// ****************************************************************
	//	OpenGL Internal format is CL_UNORM_INT8, not CL_UNSIGNED_INT8!
	// ****************************************************************
	return  clCreateFromGLTexture(m_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &m_errorCode);
}

void OpenCL::UpdateTexture(cl_mem texture, cl_mem src, cl_mem dest)
{
	cl_event event;
	size_t region[2];
	clGetImageInfo(src, CL_IMAGE_WIDTH, sizeof(size_t), &region[0], NULL);
	clGetImageInfo(src, CL_IMAGE_HEIGHT, sizeof(size_t), &region[1], NULL);

	//	glFinish(); // depend on mode
	m_errorCode = clEnqueueAcquireGLObjects(m_commandQueue, 1, &texture, 0, NULL, NULL);
	CHECK_ERRORS(m_errorCode);

	// dataType == UNORM_INT8 for D3D11 pixel shader
	m_errorCode = clSetKernelArg(m_convertTexture, 0, sizeof(cl_mem), &src);
	m_errorCode = clSetKernelArg(m_convertTexture, 1, sizeof(cl_mem), &dest);
	m_errorCode = clEnqueueNDRangeKernel(m_commandQueue, m_convertTexture, 2, NULL, region, NULL, 0, NULL, &event);
	CHECK_ERRORS(m_errorCode);

	m_errorCode = clEnqueueReleaseGLObjects(m_commandQueue, 1, &texture, 1, &event, NULL);
	CHECK_ERRORS(m_errorCode);
	m_errorCode = clFinish(m_commandQueue);	// NVIDIA has not cl_khr_gl_event
	CHECK_ERRORS(m_errorCode);
	clReleaseEvent(event);	
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

bool OpenCL::CheckCLGLShareing()
{
	bool bclEventFromGLsyncObjectSupported = false;
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
		status = clGetPlatformInfo(platforms[i], CL_DEVICE_EXTENSIONS, sizeof(extension_string), extension_string, NULL);
		//printf("Extensions supported: %s\n", extension_string);

		char* extStringStart = NULL;
		extStringStart = strstr(extension_string, "cl_khr_gl_event");
		if (extStringStart == 0)
		{
			printf("Platform %s does not report support for cl_khr_gl_event,\n", platformName);
		}
		if (extStringStart != 0)
		{
			printf("Platform %s does support cl_khr_gl_event. \nFind out which device (if any) reports support as well\n", platformName);
			bclEventFromGLsyncObjectSupported = TRUE;

		}

		//get number of devices in the platform
		//for each platform, query extension string
		cl_uint num_devices = 0;
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
		//testStatus(status, "Error getting number of devices\n");

		cl_device_id* clDevices = NULL;
		clDevices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
		if (clDevices == NULL)
		{
			printf("Error when allocating\n");
		}
		status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, clDevices, 0);
		//testStatus(status, "clGetDeviceIDs error");

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
				bclEventFromGLsyncObjectSupported = TRUE;
			}
			else
			{
				printf("Device %s in %s platform does not support CL/GL sync, \n\tglFinish() would be required on this device\n", devTypeString, vendorName);
			}
		} //end for(...)
		free(clDevices);

	}

	printf("******************************************************************************\n");

}