#include <iostream>
#include <opencv2/opencv.hpp>
#include "OBJ_Loader.h"
#include <algorithm>
#include <math.h>
#include <array>
#include <fstream>
#include <stdexcept>
#include <Eigen/Eigen>
constexpr double MY_PI = 3.1415926;
using namespace Eigen;
using namespace std;

namespace shader
{
    struct fragment_shader_payload
    {
        fragment_shader_payload()
        {
            texture = nullptr;
        }

        fragment_shader_payload(const Eigen::Vector3f &col, const Eigen::Vector3f &nor, const Eigen::Vector2f &tc, Texture *tex) : color(col), normal(nor), tex_coords(tc), texture(tex) {}

        Eigen::Vector3f view_pos;   // 当前相机坐标下的点
        Eigen::Vector3f color;      // 当前点的颜色
        Eigen::Vector3f normal;     // 当前点的法向量
        Eigen::Vector2f tex_coords; // 当前点的纹理uv值
        Texture *texture;           // 纹理
    };
    struct vertex_shader_payload
    {
        Eigen::Vector3f position;
    };
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
        0, 2.5, 0, 0,
        0, 0, 2.5, 0,
        0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity(); // 生成单位矩阵

    const float fovy = eye_fov * M_PI / 180; // 角度转弧度
    const float tan_half_fovy = std::tan(fovy / 2.0f);

    projection << 1.0f / (aspect_ratio * tan_half_fovy), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tan_half_fovy, 0.0f, 0.0f,
        0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), (2.0f * zFar * zNear) / (zNear - zFar),
        0.0f, 0.0f, -1.0f, 0.0f;

    return projection;
}

Eigen::Vector3f vertex_shader(const shader::vertex_shader_payload &payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const shader::fragment_shader_payload &payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const shader::fragment_shader_payload &payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        Eigen::Vector3f kd = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
        // TODO: Get the texture value at the texture coordinates of the current fragment
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights)
    {
        Eigen::Vector3f pos_vec = light.position - point;
        Eigen::Vector3f pos_vec = light.position - point;
        Eigen::Vector3f light_dir = pos_vec.normalized();
        Eigen::Vector3f camera_vec = (eye_pos - point).normalized();
        Eigen::Vector3f reflect_vec = (2 * normal * normal.dot(light_dir) - light_dir).normalized();

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);                                                                             // 环境光计算
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::max(0.0f, normal.dot(light_dir));                 // 漫反射光计算
        Eigen::Vector3f half_vector = (camera_vec + light_dir).normalized();                                                                        // 半程向量
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::pow(max(0.0f, reflect_vec.dot(half_vector)), p); // 镜面反射光计算

        result_color += (ambient + diffuse + specular); // 结果累加

        result_color = result_color + (ambient + diffuse + specular); // 结果累加
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const shader::fragment_shader_payload &payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color.normalized();    // 当前点的颜色
    Eigen::Vector3f point = payload.view_pos.normalized(); // 当前点的视角
    Eigen::Vector3f normal = payload.normal.normalized();  // 当前点的法线

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto &light : lights)
    {
        Eigen::Vector3f pos_vec = light.position - point;
        Eigen::Vector3f light_dir = pos_vec.normalized();
        Eigen::Vector3f camera_vec = (eye_pos - point).normalized();
        Eigen::Vector3f reflect_vec = (2 * normal * normal.dot(light_dir) - light_dir).normalized();

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);                                                                             // 环境光计算
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::max(0.0f, normal.dot(light_dir));                 // 漫反射光计算
        Eigen::Vector3f half_vector = (camera_vec + light_dir).normalized();                                                                        // 半程向量
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::pow(max(0.0f, reflect_vec.dot(half_vector)), p); // 镜面反射光计算

        result_color += (ambient + diffuse + specular); // 结果累加
        // std::transform(ka.begin(), ka.end(), ambient.begin(), [](auto &light)
        //                { for (int i; i < 3;i++) return light[i]*ka[i]; });
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.
    }

    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const shader::fragment_shader_payload &payload)
{

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;
    float kh = 0.2, kn = 0.1;

    float x = point.x(), y = point.y(), z = point.z();
    // Eigen::Vector3f t = Eigen::Vector3f(normal.y(), -normal.x(), 0).normalized(); // 计算切线向量t，这里使用了一个简化的方法来找到与n垂直的向量
    Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    t.normalize();
    Eigen::Vector3f n = normal.normalized(); // 标准化的法线向量
    Eigen::Vector3f b = n.cross(t);

    Eigen::Matrix3f TBN;
    TBN << t.x(), t.y(), t.z(),
        b.x(), b.y(), b.z(),
        n.x(), n.y(), n.z();

    float du, dv;
    float u = payload.tex_coords.x();  // 当前纹理u坐标
    float v = payload.tex_coords.y();  // 当前纹理v坐标
    float w = payload.texture->width;  // 纹理的宽度;
    float h = payload.texture->height; // 纹理的高度;

    du = payload.texture->getColor(u + 1.0f / w, v).norm() - payload.texture->getColor(u, v).norm();
    dv = payload.texture->getColor(u, v + 1.0f / h).norm() - payload.texture->getColor(u, v).norm();

    float dU = kh * kn * du;
    float dV = kh * kn * dv;
    
    point += (kn * normal * payload.texture->getColor(u, v).norm()); // 直接改变了顶点坐标的高度

    Eigen::Vector3f ln(-dU, -dV, 1.0f); // 梯度
    Eigen::Vector3f new_normal = (TBN * ln).normalized();

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = new_normal;

    return result_color * 255.f;
    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights)
    {

        Eigen::Vector3f pos_vec = light.position - point;
        Eigen::Vector3f light_dir = pos_vec.normalized();
        Eigen::Vector3f camera_vec = (eye_pos - point).normalized();
        Eigen::Vector3f reflect_vec = (2 * normal * normal.dot(light_dir) - light_dir).normalized();

        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);                                                                            // 环境光计算
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::max(0.0f, normal.dot(light_dir));                // 漫反射光计算
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / pos_vec.squaredNorm()) * std::pow(max(0.0f, reflect_vec.dot(camera_vec)), p); // 镜面反射光计算

        result_color = result_color + (ambient + diffuse + specular); // 结果累加
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const shader::fragment_shader_payload &payload)
{

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;
    float kh = 0.2, kn = 0.1;

    // 修改部分
    float x = point.x(), y = point.y(), z = point.z();
    // Eigen::Vector3f t = Eigen::Vector3f(normal.y(), -normal.x(), 0).normalized(); // 计算切线向量t，这里使用了一个简化的方法来找到与n垂直的向量
    Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    t.normalize();
    Eigen::Vector3f n = normal.normalized(); // 标准化的法线向量
    Eigen::Vector3f b = n.cross(t);

    Eigen::Matrix3f TBN;
    TBN << t.x(), t.y(), t.z(),
        b.x(), b.y(), b.z(),
        n.x(), n.y(), n.z();

    float du, dv;
    float u = payload.tex_coords.x();  // 当前纹理u坐标
    float v = payload.tex_coords.y();  // 当前纹理v坐标
    float w = payload.texture->width;  // 纹理的宽度;
    float h = payload.texture->height; // 纹理的高度;

    du = payload.texture->getColor(u + 1.0f / w, v).norm() - payload.texture->getColor(u, v).norm();
    dv = payload.texture->getColor(u, v + 1.0f / h).norm() - payload.texture->getColor(u, v).norm();

    float dU = kh * kn * du;
    float dV = kh * kn * dv;
    Eigen::Vector3f ln(-dU, -dV, 1.0f); // 梯度
    Eigen::Vector3f new_normal = (TBN * ln).normalized();

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = new_normal;

    return result_color * 255.f;
}

class Texture
{
    // cv库中，原点在左上角，多数采用BGR颜色空间
private:
    cv::Mat image_data;

public:
    int width, height;

    Texture(const std::string &name)
    {
        image_data = cv::imread(name);                           // 通过路径获取纹理，包含像素数据，图像的大小、通道数以及数据类型等信息。
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR); // 从颜色空间RGB转换为BGR
        width = image_data.cols;                                 // 记录纹理的宽度
        height = image_data.rows;                                // 记录纹理的高度
    }
    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;                              // 将U映射到cv库中的纹理坐标
        auto v_img = (1 - v) * height;                       // 将v反转映射到cv库中的纹理坐标
        auto color = image_data.at<cv::Vec3b>(v_img, u_img); // 得到纹理坐标的颜色值
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }
};

class Triangle
{
public:
    Vector4f v[3];

    Vector3f color[3];
    Vector2f tex_coords[3];
    Vector3f normal[3];

    Texture *tex = nullptr;
    Triangle();

    Eigen::Vector4f a() const { return v[0]; }
    Eigen::Vector4f b() const { return v[1]; }
    Eigen::Vector4f c() const { return v[2]; }

    void setVertex(int ind, Vector4f ver)
    {
        v[ind] = ver;
    }
    void setNormal(int ind, Vector3f n)
    {
        normal[ind] = n;
    }
    void setColor(int ind, float r, float g, float b)
    {
        if ((r < 0.0) || (r > 255.) ||
            (g < 0.0) || (g > 255.) ||
            (b < 0.0) || (b > 255.))
        {
            fprintf(stderr, "ERROR! Invalid color values");
            fflush(stderr);
            exit(-1);
        }

        color[ind] = Vector3f((float)r / 255., (float)g / 255., (float)b / 255.);
        return;
    }

    void setNormals(const std::array<Vector3f, 3> &normals)
    {
        normal[0] = normals[0];
        normal[1] = normals[1];
        normal[2] = normals[2];
    }
    void setColors(const std::array<Vector3f, 3> &colors)
    {
        auto first_color = colors[0];
        setColor(0, colors[0][0], colors[0][1], colors[0][2]);
        setColor(1, colors[1][0], colors[1][1], colors[1][2]);
        setColor(2, colors[2][0], colors[2][1], colors[2][2]);
    }
    void setTexCoord(int ind, Vector2f uv)
    {
        tex_coords[ind] = uv;
    }
    std::array<Vector4f, 3> toVector4() const
    {
        std::array<Eigen::Vector4f, 3> res;
        std::transform(begin(v), end(v), res.begin(), [](auto &vec)
                       { return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1.0f); });
        return res;
    }
};

namespace rst
{
    enum class Buffers
    {
        Color = 1,
        Depth = 2
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
    {
        return Vector4f(v3.x(), v3.y(), v3.z(), w);
    }

    enum class Primitive
    {
        Line,
        Triangle
    };

    struct pos_buf_id
    {
        int pos_id = 0;
    };

    struct ind_buf_id
    {
        int ind_id = 0;
    };

    struct col_buf_id
    {
        int col_id = 0;
    };

    class rasterizer
    {
    private:
        int width, height;
        int next_id = 0;
        int normal_id = -1;

        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf;
        std::map<int, std::vector<Eigen::Vector3f>> col_buf;
        std::map<int, std::vector<Eigen::Vector3f>> nor_buf;

        std::optional<Texture> texture;

        std::function<Eigen::Vector3f(shader::fragment_shader_payload)> fragment_shader;
        std::function<Eigen::Vector3f(shader::vertex_shader_payload)> vertex_shader;

        std::vector<Eigen::Vector3f> frame_buf;
        std::vector<float> depth_buf;

        int get_index(int x, int y);
        int get_next_id() { return next_id++; }

        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);

    public:
        rasterizer(int w, int h);
        pos_buf_id load_positions(const std::vector<Eigen::Vector3f> &positions);
        ind_buf_id load_indices(const std::vector<Eigen::Vector3i> &indices);
        col_buf_id load_colors(const std::vector<Eigen::Vector3f> &colors);
        col_buf_id load_normals(const std::vector<Eigen::Vector3f> &normals);

        static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v)
        {
            float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
            float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
            float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
            return {c1, c2, c3};
        }
        static bool insideTriangle(float x, float y, const Vector4f *_v)
        {
            Vector2f v02 = _v[2].head(2) - _v[0].head(2),
                     v10 = _v[0].head(2) - _v[1].head(2),
                     v21 = _v[1].head(2) - _v[2].head(2);

            Vector2f p(x, y);

            Vector2f v0p = p - _v[0].head(2),
                     v1p = p - _v[1].head(2),
                     v2p = p - _v[2].head(2);

            float a1 = v0p.x() * v02.y() - v0p.y() * v02.x(),
                  a2 = v1p.x() * v10.y() - v1p.y() * v10.x(),
                  a3 = v2p.x() * v21.y() - v2p.y() * v21.x();

            return (a1 > 0 && a2 > 0 && a3 > 0) || (a1 < 0 && a2 < 0 && a3 < 0);
        }
        static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1, const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight)
        {
            return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
        }
        static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1, const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight)
        {
            auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
            auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

            u /= weight;
            v /= weight;

            return Eigen::Vector2f(u, v);
        }

        void clear(Buffers buff);

        void set_model(const Eigen::Matrix4f &m);
        void set_view(const Eigen::Matrix4f &v);
        void set_projection(const Eigen::Matrix4f &p);

        void set_texture(Texture tex) { texture = tex; }
        void set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
        {
            // old index: auto ind = point.y() + point.x() * width;
            int ind = (height - point.y()) * width + point.x();
            frame_buf[ind] = color;
        }

        void set_vertex_shader(std::function<Eigen::Vector3f(shader::vertex_shader_payload)> vert_shader)
        {
            vertex_shader = vert_shader;
        }

        void set_fragment_shader(std::function<Eigen::Vector3f(shader::fragment_shader_payload)> frag_shader)
        {
            fragment_shader = frag_shader;
        }

        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type);
        void draw(std::vector<Triangle *> &TriangleList) // TriangleList表示一个三角形集合，里面以指针riangle *存放了一个模型的每个面信息
        {

            float f1 = (50 - 0.1) / 2.0;
            float f2 = (50 + 0.1) / 2.0;

            Eigen::Matrix4f mvp = projection * view * model;
            for (const auto &t : TriangleList) // 遍历三角形集合，其中每个三角形用Triangle *存放数据
            {
                Triangle newtri = *t; // 解引用，得到一个三角形的复制newtri

                std::array<Eigen::Vector4f, 3> mm{
                    (view * model * t->v[0]),
                    (view * model * t->v[1]),
                    (view * model * t->v[2])}; // 得到视图矩阵下的坐标

                std::array<Eigen::Vector3f, 3> viewspace_pos;

                std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v)
                               { return v.template head<3>(); });

                Eigen::Vector4f v[] = {
                    mvp * t->v[0],
                    mvp * t->v[1],
                    mvp * t->v[2]}; // 得到裁剪空间下的坐标

                // Homogeneous division
                for (auto &vec : v) // 透视除法
                {
                    vec.x() /= vec.w();
                    vec.y() /= vec.w();
                    vec.z() /= vec.w();
                }

                Eigen::Matrix4f inv_trans = (view * model).inverse().transpose(); // 法线向量变换
                Eigen::Vector4f n[] = {
                    inv_trans * to_vec4(t->normal[0], 0.0f),
                    inv_trans * to_vec4(t->normal[1], 0.0f),
                    inv_trans * to_vec4(t->normal[2], 0.0f)};

                // Viewport transformation
                for (auto &vert : v)
                {
                    vert.x() = 0.5 * width * (vert.x() + 1.0);
                    vert.y() = 0.5 * height * (vert.y() + 1.0);
                    vert.z() = vert.z() * f1 + f2; // 将深度值缩放到标准规体
                }

                for (int i = 0; i < 3; ++i) // 设置顶点
                {
                    // screen space coordinates
                    newtri.setVertex(i, v[i]);
                }

                for (int i = 0; i < 3; ++i) // 设置法线向量
                {
                    // view space normal
                    newtri.setNormal(i, n[i].head<3>());
                }

                newtri.setColor(0, 148, 121.0, 92.0);
                newtri.setColor(1, 148, 121.0, 92.0);
                newtri.setColor(2, 148, 121.0, 92.0);

                // Also pass view space vertice position
                rasterize_triangle(newtri, viewspace_pos);
            }
        }
        void rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &world_pos)
        {
            auto v = t.toVector4(); // /只改变顶点坐标，将其转换成vetor4
            int min_x, max_x, min_y, max_y;
            Eigen::Vector3f interpolated_normal;
            Eigen::Vector3f interpolated_color;
            Eigen::Vector2f interpolated_texcoords;
            Eigen::Vector3f interpolated_shadingcoords;

            min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
            max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
            min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
            max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

            for (int x = min_x; x <= max_x; x++)
            {
                for (int y = min_y; y <= max_y; y++)
                {
                    float Depth = FLT_MAX;
                    if (insideTriangle(x + 0.5, y + 0.5, t.v))
                    {
                        auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);
                        float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w(); // 深度插值
                        z_interpolated *= w_reciprocal;
                        Depth = std::min(Depth, z_interpolated);

                        if (depth_buf[get_index(x, y)] > Depth)
                        {
                            auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1).normalized();   // 法向量插值
                            auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1);                    // 颜色插值
                            auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1); // 纹理坐标插值
                            auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, world_pos[0], world_pos[1], world_pos[2], 1);      // 相机坐标插值

                            shader::fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                            payload.view_pos = interpolated_shadingcoords; // 得到某点的相机坐标
                            auto pixel_color = fragment_shader(payload);   // 调用片段shader
                            depth_buf[get_index(x, y)] = Depth;            // 更新深度
                            set_pixel(Eigen::Vector2i(x, y), pixel_color);
                        }
                    }
                }
            }
        }

        std::vector<Eigen::Vector3f> &frame_buffer() { return frame_buf; }
    };
}

int main(int argc, const char **argv)
{
    std::vector<Triangle *> TriangleList;
    float angle = 140.0;
    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";

    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj"); // 模型
    for (auto mesh : Loader.LoadedMeshes)
    {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) // 第i个三角形
        {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++)
            {
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y, mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y, mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));
    std::function<Eigen::Vector3f(shader::fragment_shader_payload)> active_shader = phong_fragment_shader;

    Eigen::Vector3f eye_pos = {0, 0, 10};
    int key = 0;
    int frame_count = 0;

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        // r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a')
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }
    }

    return 0;
}