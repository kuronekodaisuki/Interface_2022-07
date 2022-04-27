#pragma once

// OpenCL�̃w�b�_�[�t�@�C��
#include <CL/cl.h>

enum IMAGE_MODE
{
	READ_ONLY,
	WRITE_ONLY,
	READ_WRITE
};


class OpenCL
{
public:
	cl_device_id SelectDevice(cl_device_type type = CL_DEVICE_TYPE_GPU);
	OpenCL(bool USE_GPU = true);
	~OpenCL();

	/// <summary>
	/// �摜�o�b�t�@�𐶐�
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="format"></param>
	/// <param name="mode"></param>
	/// <returns></returns>
	cl_mem CreateImage(size_t width, size_t height, cl_image_format format, IMAGE_MODE mode, void* pHostPtr = NULL);

	/// <summary>
	/// �摜�f�[�^����������
	/// </summary>
	/// <param name="image"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="channels"></param>
	/// <param name="memory"></param>
	/// <param name="event"></param>
	/// <returns></returns>
	cl_int WriteImage(unsigned char* image, unsigned int width, unsigned int height, unsigned int channels, cl_mem memory, cl_event* wait, cl_event* finish);

	/// <summary>
	/// �摜�f�[�^���擾
	/// </summary>
	/// <param name="memory"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="channels"></param>
	/// <param name="image"></param>
	/// <param name="event"></param>
	/// <param name="finish"></param>
	/// <returns></returns>
	cl_int ReadImage(cl_mem memory, unsigned int width, unsigned int height, unsigned int channels, unsigned char* image, cl_event* event, cl_event* finish);

	/// <summary>
	/// �K�E�V�A���t�B���^�����s
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="input"></param>
	/// <param name="output"></param>
	/// <param name="wait"></param>
	/// <param name="finish"></param>
	/// <returns></returns>
	cl_int EnqueueGaussian(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish);

	/// <summary>
	/// ���f�B�A���t�B���^�����s�i3x3)
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="input"></param>
	/// <param name="output"></param>
	/// <param name="wait"></param>
	/// <param name="finish"></param>
	/// <returns></returns>
	cl_int EnqueueMedian3x3(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish);

	/// <summary>
	/// ���f�B�A���t�B���^�����s(5x5)
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="input"></param>
	/// <param name="output"></param>
	/// <param name="wait"></param>
	/// <param name="finish"></param>
	/// <returns></returns>
	cl_int EnqueueMedian5x5(unsigned int width, unsigned int height, cl_mem input, cl_mem output, cl_event* wait, cl_event* finish);

	/// <summary>
	/// SVM�������m��
	/// </summary>
	/// <param name="size"></param>
	/// <param name="flags"></param>
	/// <returns></returns>
	void* AllocSVMMemory(size_t size, cl_svm_mem_flags flags = CL_MEM_READ_WRITE);

	/// <summary>
	/// SVM���������
	/// </summary>
	/// <param name="ptr"></param>
	void FreeSvmMemory(void* ptr);

	/// <summary>
	/// ���������}�b�v�i�z�X�g���ŃA�N�Z�X�j
	/// </summary>
	/// <param name="ptr"></param>
	/// <param name="size"></param>
	/// <param name="flags"></param>
	/// <returns></returns>
	cl_int SVMMap(void* ptr, size_t size, cl_map_flags flags = CL_MAP_READ | CL_MAP_WRITE);

	/// <summary>
	/// ���������A���}�b�v�i�Ȍ�f�o�C�X���ŃA�N�Z�X�j
	/// </summary>
	/// <param name="ptr"></param>
	/// <returns></returns>
	cl_int SVMUnmap(void* ptr);

private:
	cl_platform_id		m_platformId;
	cl_device_id		m_deviceId;
	cl_context			m_context;
	cl_command_queue	m_commandQueue;
	cl_int				m_errorCode;

	cl_kernel			m_gaussian3x3;
	cl_kernel			m_median3x3;
	cl_kernel			m_median5x5;

	cl_mem				m_image;
	cl_mem				m_texture;
};

