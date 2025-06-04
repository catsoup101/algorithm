#include <iostream>
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "Shader.h"
#include <math.h>

const unsigned int SCR_WIDTH = 800;	 // 屏幕高度
const unsigned int SCR_HEIGHT = 600; // 屏幕宽度
float vertices[] = {
	-0.5f, -0.5f, 0.0f, 1.0f, 0, 0,
	0.5f, -0.5f, 0.0f, 0, 1.0f, 0,
	0.0f, 0.5f, 0.0f, 0, 0, 1.0f,
	/*0.5f, -0.5f, 0.0f,
	0.0f, 0.5f, 0.0f,
	*/
	0.8f, 0.8f, 0.0f, 1.0f, 0.5f, 1.0f};
unsigned indices[] = {
	0, 1, 2,
	2, 1, 3};
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) // 检测指定窗口中某个按键的状态，这里是esc键
		glfwSetWindowShouldClose(window, true);			   // 则将窗口设置为true，即关闭
}
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
	glViewport(0, 0, width, height); // void glViewport(GLint x,GLint y,GLsizei width,GLsizei height)用于指定渲染目标的绘制区域
}
int main()
{
	glfwInit();													   // 初始化函数库
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);				   // 使用的版本,MAJOR指主版本
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);				   // MINOR指的副版本
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 核心流水线
	/*	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);*/	   // mac平台

	GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL); // 设置窗口
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate(); // 终止程序
		return -1;
	}
	glfwMakeContextCurrent(window);									   // 设置这个窗口为上下文，操作将在这个窗口运行
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // 设置帧缓冲回调函数framebuffer_size_callback，以便在窗口的调用帧缓冲
	/*glewExperimental = true;*/									   // 开启实验性功能
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}						// 判断是否成功加载OpenGL函数指针。glfwGetProcAddress用于获取OpenGL类型为GLADloadproc的函数指针
	glEnable(GL_CULL_FACE); // 开启面剔除功能
	glCullFace(GL_BACK);	// 设置遮挡方向

	Shader myShader = Shader("src/vertexSource.txt", "src/fragmentSource.txt");
	// glDrawBuffers
	unsigned int VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO); // 将其作为上下文的vao对象
	unsigned int VBO;
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);										   // 将其绑定在VAO中
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // 传入顶点数据
	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// unsigned int vertexShader;
	// vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// glCompileShader(vertexShader);
	// unsigned int fragmentShader;
	// fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// glCompileShader(fragmentShader);
	// unsigned int shaderProgram;
	// shaderProgram = glCreateProgram();
	// glAttachShader(shaderProgram, vertexShader);
	// glAttachShader(shaderProgram, fragmentShader);
	// glLinkProgram(shaderProgram);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	while (!glfwWindowShouldClose(window)) // 查看窗口是否应该关闭，即用户是否点击了关闭按钮或者触发了关闭事件。
	{
		processInput(window);				  // 处理用户输入
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // 清除颜色并设置窗口颜色
		glClear(GL_COLOR_BUFFER_BIT);		  // 清空颜色缓冲区

		// glBindVertexArray(VAO);
		// glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO);
		myShader.use();

		// glDrawArrays(GL_TRIANGLES, 0, 3);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window); // 交换前后缓冲区
		glfwPollEvents();		 // 处理窗口事件
	}
	glfwTerminate(); // 终止和清理GLFW库的资源，以避免内存泄漏。
	return 0;
}
