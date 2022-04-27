// Kernel.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <Windows.h>
#include <omp.h>
#include <vector>



// OpenCLのヘッダーファイル
#include <CL/cl.h>
#include <CL/cl_platform.h>

// OpenCLライブラリ
#pragma comment(lib, "OpenCL.lib")


cl_device_id SelectDevice(cl_device_type type);

cl_kernel BuildKernel(cl_context context, const char* kernelSource, const char* function);


// カーネルコード（配列どうしの加算）
const char* kernelSource =
"__kernel void vectorAdd(__global double* a,\n"
"	__global double* b,\n"
"	__global double* c,\n"
"	const unsigned int n)\n"
"{\n"
"	// スレッド番号取得\n"
"	int id = get_global_id(0);\n"
"\n"
"	// 各要素の加算\n"
"	if (id < n)\n"
"		c[id] = a[id] + b[id];\n"
"}\n"
"__kernel void vectorMul(__global double* a,\n"
"	__global double* b,\n"
"	__global double* c,\n"
"	const unsigned int n)\n"
"{\n"
"	// スレッド番号取得\n"
"	int id = get_global_id(0);\n"
"\n"
"	// 各要素の加算\n"
"	if (id < n)\n"
"		c[id] = a[id] * b[id];\n"
"}\n";


#define VECTOR_SIZE	50000000
//double A[VECTOR_SIZE], B[VECTOR_SIZE], C[VECTOR_SIZE];

int main()
{
	cl_int err;
	// GPUデバイスを取得
	cl_device_id deviceId = SelectDevice(CL_DEVICE_TYPE_GPU);

	// コンテキスト作成
	cl_context context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &err);

	// 命令キュー作成
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, deviceId, NULL, &err);

	// カーネルを生成
	cl_kernel vectorAdd = BuildKernel(context, kernelSource, "vectorAdd");

	// Shared Virtual Memoryを確保
	double* A = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(double) * VECTOR_SIZE, 0);
	double* B = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(double) * VECTOR_SIZE, 0);
	double* C = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(double) * VECTOR_SIZE, 0);

	// この領域へホスト側でアクセスする
	cl_event events[3];
	clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, A, sizeof(double) * VECTOR_SIZE, 1, NULL, &events[0]);
	clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, B, sizeof(double) * VECTOR_SIZE, 1, NULL, &events[1]);
	clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, C, sizeof(double) * VECTOR_SIZE, 1, NULL, &events[2]);

	// 入力データを設定
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		A[i] = i;
		B[i] = (double)VECTOR_SIZE - i;
	}

	LARGE_INTEGER freq, start, stop;
	QueryPerformanceFrequency(&freq);

	// CPUで
	QueryPerformanceCounter(&start);
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		C[i] = A[i] + B[i];
	}
	QueryPerformanceCounter(&stop);
	printf("CPU %fmsec\n", 1000 * (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart);

	// OpenMPで
	QueryPerformanceCounter(&start);
#pragma omp parallel for
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		C[i] = A[i] + B[i];
	}
	QueryPerformanceCounter(&stop);
	printf("OpenMP %fmsec\n", 1000 * (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart);

	// ホスト側のアクセス終了
	clEnqueueSVMUnmap(queue, A, 1, NULL, &events[0]);
	clEnqueueSVMUnmap(queue, B, 1, NULL, &events[1]);
	clEnqueueSVMUnmap(queue, C, 1, NULL, &events[2]);


	unsigned int vectorSize = VECTOR_SIZE;
	size_t globalSize = VECTOR_SIZE, localSize = 8;

	/*
	//size_t bytes = VECTOR_SIZE * sizeof(double);

	// デバイス上のメモリ領域を確保
	cl_mem a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	cl_mem c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

	// 確保したメモリ領域に入力データをコピー
	err = clEnqueueWriteBuffer(queue, a, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, b, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

	// カーネルのパラメータを設定
	err = clSetKernelArg(vectorAdd, 0, sizeof(cl_mem), &a);
	err |= clSetKernelArg(vectorAdd, 1, sizeof(cl_mem), &b);
	err |= clSetKernelArg(vectorAdd, 2, sizeof(cl_mem), &c);
	err |= clSetKernelArg(vectorAdd, 3, sizeof(unsigned int), &vectorSize);
	*/

	// カーネルのパラメータを設定
	err = clSetKernelArgSVMPointer(vectorAdd, 0, A);
	err |= clSetKernelArgSVMPointer(vectorAdd, 1, B);
	err |= clSetKernelArgSVMPointer(vectorAdd, 2, C);
	err |= clSetKernelArg(vectorAdd, 3, sizeof(unsigned int), &vectorSize);

	// GPUで実行
	QueryPerformanceCounter(&start);

	// カーネルを命令キューに追加して実行
	err = clEnqueueNDRangeKernel(queue, vectorAdd, 1, NULL, &globalSize, &localSize,
		0, NULL, NULL);

	// 命令キューの終了を待つ
	err = clFinish(queue);

	QueryPerformanceCounter(&stop);

	/*
	// デバイス上にある計算結果を取得
	clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, C, sizeof(double) * VECTOR_SIZE, 1, NULL, &events[2]);
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		printf("%f\n", C[i]);
	}
	clEnqueueSVMUnmap(queue, C, 1, NULL, &events[2]);
	*/

	//err = clEnqueueReadBuffer(queue, c, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

	printf("GPU %fmsec\n", 1000 * (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart);


	// 後始末
	clSVMFree(context, A);
	clSVMFree(context, B);
	clSVMFree(context, C);
	//clReleaseMemObject(a);
	//clReleaseMemObject(b);
	//clReleaseMemObject(c);
	clReleaseKernel(vectorAdd);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

/// <summary>
/// デバイスを選択
/// </summary>
/// <param name="type">デバイス種別</param>
/// <returns>デバイス識別子</returns>
cl_device_id SelectDevice(cl_device_type type)
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

		// 各デバイスについて
		for (cl_uint j = 0; j < num_of_devices; j++)
		{
			cl_uint freq, units;
			char deviceName[128];
			cl_device_svm_capabilities svmCapability;

			// デバイス情報を所得
			err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &freq, NULL);
			err = clGetDeviceInfo(id[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
			err = clGetDeviceInfo(id[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
			err = clGetDeviceInfo(id[j], CL_DEVICE_SVM_CAPABILITIES, sizeof(svmCapability), &svmCapability, NULL);
			if (svmCapability & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
				printf("%s %d Compute units %dMHz SVM supported\n", deviceName, units, freq);
			else
				printf("%s %d Compute units %dMHz\n", deviceName, units, freq);
			deviceId = id[j];
		}
	}

	return deviceId;
}

/// <summary>
/// カーネルを生成
/// </summary>
/// <param name="context">コンテキスト</param>
/// <param name="kernelSource">カーネルコード</param>
/// <param name="function">関数名</param>
/// <returns>カーネル</returns>
cl_kernel BuildKernel(cl_context context, const char* kernelSource, const char* function)
{
	cl_int err;
	// カーネルコードからプログラムを生成
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char**)&kernelSource, NULL, &err);

	// 
	err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);

	// カーネルを生成
	cl_kernel kernel = clCreateKernel(program, function, &err);
	
	clReleaseProgram(program);

	return kernel;
}