// DeviceInformation.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <vector>
#include <string.h>

// OpenCLのヘッダーファイル
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/cl_ext.h>

// OpenCLライブラリ
#pragma comment(lib, "OpenCL.lib")

int main()
{
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
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU,	// CPUとGPUを取得
			0,
			0,
			&num_of_devices
		);

		cl_device_id* id = new cl_device_id[num_of_devices];

		// デバイス識別子を取得
		err = clGetDeviceIDs(
			platforms[i],
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU,	// CPUとGPUを取得
			num_of_devices,
			id,
			0
		);

		// 各デバイスについて
		for (cl_uint j = 0; j < num_of_devices; j++)
		{
			size_t length;
			cl_uint freq, units;
			char deviceName[128];
			char deviceExtensions[5000];

			// デバイス情報を所得
			err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &freq, &length);
			err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, &length);
			err = clGetDeviceInfo(id[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, &length);
			printf("%s %d Compute units %dMHz\n", deviceName, units, freq);
			if (0 == clGetDeviceInfo(id[j], CL_DEVICE_EXTENSIONS, sizeof(deviceExtensions), deviceExtensions, &length))
			{
				printf("%s\n", deviceExtensions);
				for (char* context = NULL, *token = strtok_s(deviceExtensions, " ", &context); token != NULL; token = strtok_s(NULL, " ", &context))
				{
					printf("\t%s\n", token);
				}
			}
		}
	}
}
