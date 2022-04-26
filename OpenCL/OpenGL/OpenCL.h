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

	static bool CheckCLGLShareing();

	/// <summary>
	/// �e�N�X�`���𐶐�
	/// </summary>
	/// <param name="texture"></param>
	/// <returns></returns>
	cl_mem CreateGLTexture(cl_GLuint texture);

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
	cl_int WriteImage(unsigned char* image, unsigned int width, unsigned int height, unsigned int channels, cl_mem memory, cl_event* event);

	/// <summary>
	/// �e�N�X�`���X�V
	/// </summary>
	/// <param name="texture"></param>
	/// <param name="src"></param>
	/// <param name="dest"></param>
	void UpdateTexture(cl_mem texture, cl_mem src, cl_mem dest);

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
	cl_kernel			m_remapImage;
	cl_kernel			m_convertTexture;
	cl_int				m_errorCode;

	cl_mem				m_image;
	cl_mem				m_texture;
};

