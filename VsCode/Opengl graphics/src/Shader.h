#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

using namespace std;
class Shader
{
public:
    Shader(const char *vertexPath, const char *fragmentPath);
    string vertexString;
    string fragmentString;
    const char *vertexSource;
    const char *fragmentSource;
    unsigned int ID;

    void use();

private:
    void checkCompileErroes(unsigned int ID, std::string type);
};

Shader::Shader(const char *vertexPath, const char *fragmentPath)
{
    fstream vertexFile;
    fstream fragmentFile;
    stringstream vertexSStream;
    stringstream fragmentPathSStream;

    vertexSource = "";
    fragmentSource = "";
    vertexFile.flush();
    fragmentFile.flush();

    vertexFile.open(vertexPath);
    fragmentFile.open(fragmentPath);

    vertexFile.exceptions(ifstream::failbit | ifstream::badbit);
    fragmentFile.exceptions(ifstream::failbit | ifstream::badbit);
    if (!vertexFile.is_open() || !fragmentFile.is_open())
    {
        throw std::runtime_error("file open faild!!");
    }

    vertexSStream << vertexFile.rdbuf();
    fragmentPathSStream << fragmentFile.rdbuf();
    vertexString = vertexSStream.str();
    fragmentString = fragmentPathSStream.str();
    vertexSource = vertexString.c_str();
    fragmentSource = fragmentString.c_str();

    unsigned int vertex, fragment;
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexSource, NULL);
    glCompileShader(vertex);
    checkCompileErroes(vertex, "VERTEX");

    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentSource, NULL);
    glCompileShader(fragment);
    checkCompileErroes(fragment, "FRAGMENT");

    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErroes(ID, "PROGRAM");
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    vertexFile.close();
    fragmentFile.close();
}
void Shader::use()
{
    glUseProgram(ID);
}
void Shader::checkCompileErroes(unsigned int ID, std::string type)
{
    // 自定义错误日志函数
    int success;
    char infoLog[512];

    if (type != "PROGRAM")
    {
        glGetShaderiv(ID, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(ID, 512, NULL, infoLog);
            cout << "shader compile error:" << infoLog << endl;
        }
    }
    else
    {
        glGetShaderiv(ID, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(ID, 512, NULL, infoLog);
            cout << "shader link error:" << infoLog << endl;
        }
    }
}