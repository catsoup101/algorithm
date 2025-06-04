#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>
#include <array>
#include <stdexcept>
#include <Eigen/Eigen>
constexpr double MY_PI = 3.1415926;
using namespace Eigen;
using namespace std;

namespace triangle
{
    class Triangle
    {
    public:
        Vector3f v[3];          /*三角形的原始坐标，v0,v1,v2在逆时针顺序*/
        Vector3f color[3];      // 每个顶点的颜色
        Vector2f tex_coords[3]; // 纹理 u,v
        Vector3f normal[3];     // 每个顶点的法向量
        Triangle()
        {
            v[0] << 0, 0, 0;
            v[1] << 0, 0, 0;
            v[2] << 0, 0, 0;

            color[0] << 0.0, 0.0, 0.0;
            color[1] << 0.0, 0.0, 0.0;
            color[2] << 0.0, 0.0, 0.0;

            tex_coords[0] << 0.0, 0.0;
            tex_coords[1] << 0.0, 0.0;
            tex_coords[2] << 0.0, 0.0;
        }

        Eigen::Vector3f a() const { return v[0]; }
        Eigen::Vector3f b() const { return v[1]; }
        Eigen::Vector3f c() const { return v[2]; }

        void setVertex(int ind, Vector3f ver) /*设置第i个顶点坐标*/
        {
            v[ind] = ver;
        }

        void setNormal(int ind, Vector3f n) /*设置第i个顶点的法向量*/
        {
            normal[ind] = n;
        }

        void setColor(int ind, float r, float g, float b) /*设置第 i 个顶点颜色*/
        {
            if ((r < 0.0) || (r > 255.) || (g < 0.0) || (g > 255.) || (b < 0.0) ||
                (b > 255.))
            {
                throw std::runtime_error("Invalid color values");
            }

            color[ind] = Vector3f((float)r / 255., (float)g / 255., (float)b / 255.);
            return;
        }

        void setTexCoord(int ind, float s, float t) /*设置第 i 个顶点纹理坐标*/
        {
            tex_coords[ind] = Vector2f(s, t);
        }

        std::array<Vector4f, 3> toVector4() const /*转化为vector4的向量坐标*/
        {
            std::array<Vector4f, 3> res;
            std::transform(std::begin(v), std::end(v), res.begin(), [](auto &vec)
                           { return Vector4f(vec.x(), vec.y(), vec.z(), 1.f); });
            return res;
        }
    };
}

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

    class rasterizer
    {
    public:
        rasterizer(int w, int h); // 构造函数初始化宽高

        pos_buf_id load_positions(const std::vector<Eigen::Vector3f> &positions) // 实现设置一个三角形位置的哈希表
        {
            auto id = get_next_id();
            pos_buf.emplace(id, positions);

            return {id};
        }

        ind_buf_id load_indices(const std::vector<Eigen::Vector3i> &indices) // 实现设置一个索引缓冲区
        {
            auto id = get_next_id();      // 设置一个索引缓冲区
            ind_buf.emplace(id, indices); // 插入一个ID和索引数据Indices

            return {id};
        }

        /*实现MVP矩阵*/
        void set_model(const Eigen::Matrix4f &m) // 模型变换矩阵
        {
            model = m;
        }
        void set_view(const Eigen::Matrix4f &v) // 视图矩阵
        {
            view = v;
        }
        void set_projection(const Eigen::Matrix4f &p) // 投影矩阵
        {
            projection = p;
        }

        auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) // 转换成齐次坐标系
        {
            return Vector4f(v3.x(), v3.y(), v3.z(), w);
        }

        void set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color) // 设置指定点的颜色值
        {
            // 设置屏幕上指定点的颜色值
            //  old index: auto ind = point.y() + point.x() * width;
            if (point.x() < 0 || point.x() >= width ||
                point.y() < 0 || point.y() >= height)
                return;
            auto ind = (height - point.y()) * width + point.x();
            frame_buf[ind] = color;
        }

        void clear(Buffers buff)
        {
            if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
            {
                std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
            }
            if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
            {
                std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
            }
        }

        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, Primitive type) // 将存储在pos_buffer和ind_buffer的图形绘制到屏幕
        {
            if (type != rst::Primitive::Triangle)
            {
                throw std::runtime_error("Drawing primitives other than triangle is not implemented yet!");
            }
            auto &buf = pos_buf[pos_buffer.pos_id]; // 通过位置ID取得pos_buf哈希中的位置顶点数据
            auto &ind = ind_buf[ind_buffer.ind_id]; // 通过索引ID取得ind_buf哈希中的索引数据

            float f1 = (100 - 0.1) / 2.0;
            /*定义视口*/
            float f2 = (100 + 0.1) / 2.0;

            Eigen::Matrix4f mvp = projection * view * model;
            for (auto &i : ind)
            {
                triangle::Triangle t;

                Eigen::Vector4f v[] = {
                    mvp * to_vec4(buf[i[0]], 1.0f),
                    mvp * to_vec4(buf[i[1]], 1.0f),
                    mvp * to_vec4(buf[i[2]], 1.0f)
                    /*经过mvp矩阵，此时转换到标准规体中*/
                };

                for (auto &vec : v) /*归一化处理*/
                {

                    vec /= vec.w();
                }

                for (auto &vert : v) /*将三角形的顶点从标准规体坐标系映射到屏幕坐标系上*/
                {
                    vert.x() = 0.5 * width * (vert.x() + 1.0);
                    vert.y() = 0.5 * height * (vert.y() + 1.0);
                    vert.z() = vert.z() * f1 + f2;
                }

                for (int i = 0; i < 3; ++i) /*将视口变换后的顶点坐标赋值给三角形对象t的顶点坐标属性*/
                {
                    t.setVertex(i, v[i].head<3>());
                    t.setVertex(i, v[i].head<3>());
                    t.setVertex(i, v[i].head<3>());
                }

                t.setColor(0, 255.0, 0.0, 0.0);
                t.setColor(1, 0.0, 255.0, 0.0); /*设置三个顶点的颜色值*/
                t.setColor(2, 0.0, 0.0, 255.0);

                rasterize_wireframe(t);
            }
        }

        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end) // bresegham画线算法
        {
            auto x1 = begin.x();
            auto y1 = begin.y();
            auto x2 = end.x();
            auto y2 = end.y();

            Eigen::Vector3f line_color = {255, 255, 255};

            int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

            dx = x2 - x1;
            dy = y2 - y1;
            dx1 = fabs(dx);
            dy1 = fabs(dy);
            px = 2 * dy1 - dx1;
            py = 2 * dx1 - dy1;

            if (dy1 <= dx1) // 比较斜率
            {
                if (dx >= 0) // 判断从哪边开始
                {
                    x = x1;
                    y = y1;
                    xe = x2; // 结束点
                }
                else
                {
                    x = x2;
                    y = y2;
                    xe = x1;
                }
                Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
                set_pixel(point, line_color);
                for (i = 0; x < xe; i++)
                {
                    x = x + 1;
                    if (px < 0)
                    {
                        px = px + 2 * dy1;
                    }
                    else
                    {
                        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                        {
                            y = y + 1;
                        }
                        else
                        {
                            y = y - 1;
                        }
                        px = px + 2 * (dy1 - dx1);
                    }
                    //            delay(0);
                    Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
                    set_pixel(point, line_color);
                }
            }
            else
            {
                if (dy >= 0)
                {
                    x = x1;
                    y = y1;
                    ye = y2;
                }
                else
                {
                    x = x2;
                    y = y2;
                    ye = y1;
                }
                Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
                set_pixel(point, line_color);
                for (i = 0; y < ye; i++)
                {
                    y = y + 1;
                    if (py <= 0)
                    {
                        py = py + 2 * dx1;
                    }
                    else
                    {
                        if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                        {
                            x = x + 1;
                        }
                        else
                        {
                            x = x - 1;
                        }
                        py = py + 2 * (dx1 - dy1);
                    }
                    //            delay(0);
                    Eigen::Vector3f point = Eigen::Vector3f(x, y, 1.0f);
                    set_pixel(point, line_color);
                }
            }
        }

        std::vector<Eigen::Vector3f> &frame_buffer() // 返回当前光栅化器维护的帧缓冲区
        {
            return frame_buf;
        }

        void rasterize_wireframe(const triangle::Triangle &t) // 调用draw_line在屏幕上绘制一个三角形的线框
        {
            /*使用画线算法对三角形进行线框渲染*/
            draw_line(t.c(), t.a());
            draw_line(t.c(), t.b());
            draw_line(t.b(), t.a());
        }

        int get_index(int x, int y)
        {
            /*屏幕上的二维坐标 (x, y) 转换为在一维 frame_buf 数组中的索引值*/
            return (height - y) * width + x;
        }

        int get_next_id() { return next_id++; }

    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        std::map<int, std::vector<Eigen::Vector3f>> pos_buf; // 设置一个三角形位置的哈希表
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf; // 设置一个三角形索引缓冲区的哈希表

        std::vector<Eigen::Vector3f> frame_buf;
        std::vector<float> depth_buf;

        int width, height;
        int next_id = 0;
    };
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    float rad = rotation_angle * MY_PI / 180.0f;
    Eigen::Matrix4f rotation;

    rotation << cos(rad), -sin(rad), 0, 0,
        sin(rad), cos(rad), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    model = rotation * model;
    return model;
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

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    const float fovy = eye_fov * M_PI / 180; // 角度转弧度
    const float tan_half_fovy = std::tan(fovy / 2.0f);

    projection << 1.0f / (aspect_ratio * tan_half_fovy), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tan_half_fovy, 0.0f, 0.0f,
        0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), (2.0f * zFar * zNear) / (zNear - zFar),
        0.0f, 0.0f, -1.0f, 0.0f;
    return projection;
}

int main(int argc, const char **argv)
{
    float angle = 0; // 旋转角度的值

    rst::rasterizer r(700, 700); // 屏幕大小

    Eigen::Vector3f eye_pos = {0, 0, 5}; // 初始化相机的位置

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}}; // 初始化三角形顶点

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}}; // 初始化缓冲区索引

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;         // 键盘的值
    int frame_count = 0; // 记录帧缓冲区

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
