#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>
#include <array>
#include <map>
#include <stdexcept>
#include <Eigen/Eigen>
constexpr double MY_PI = 3.1415926;
using namespace Eigen;
using namespace std;

class Triangle
{
    // 三角形结构体，变量提供顶点v。颜色color，纹理颜色tex_coords，法向量normal
    // 函数提供以下：
    // setVertex        //设置第顶点向量
    // setColor         //设置第顶点颜色
    // getColor         //每一个三角形的顶点颜色
    // setTexCoord      //设置顶点纹理坐标
    // toVector4       //Vector3转化toVector4
public:
    Vector3f v[3];
    Vector3f color[3];
    Vector2f tex_coords[3];
    Vector3f normal[3];

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

    void setVertex(int ind, Vector3f ver)
    {
        v[ind] = ver;
    }
    void setNormal(int ind, Vector3f n)
    {
        normal[ind] = n;
    }
    void setTexCoord(int ind, float s, float t)
    {
        tex_coords[ind] = Vector2f(s, t);
    }
    void setColor(int ind, float r, float g, float b)
    {
        if (r < 0.0 || r > 255.0 || g < 0.0 || g > 255.0 || b < 0.0 || b > 255.0)
        {
            // 输出错误信息，并退出程序
            std::cerr << "ERROR! Invalid color values" << std::endl;
            exit(-1);
        }

        color[ind] = Vector3f(r / 255.0, g / 255.0, b / 255.0); // 将颜色值存储在color数组的指定索引处，转换为0到1之间的值
        return;
    }

    Vector3f getColor() const
    {
        return color[0] * 255; // 将归一化的颜色值重新映射回0到255的范围
    }
    std::array<Vector4f, 3> toVector4() const // 将顶点转换成四分量
    {
        std::array<Eigen::Vector4f, 3> res;
        std::transform(begin(v), end(v), res.begin(), [](auto &vec)
                       { return Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1.0f); });
        return res;
    }
};

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

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
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

namespace rst
{
    enum class Buffers // 颜色或者深度
    {
        Color = 1,
        Depth = 2
    };

    enum class Primitive // 画线或者画三角形
    {
        Line,
        Triangle
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

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
        int width, height; // 屏幕宽高
        int next_id = 0;   // 缓冲区iD

        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        map<int, vector<Eigen::Vector3f>> pos_buf; // 顶点缓冲区
        map<int, vector<Eigen::Vector3i>> ind_buf; // 索引缓冲区
        map<int, vector<Eigen::Vector3f>> col_buf; // 颜色缓冲区

        vector<float> depth_buf;           // 深度缓冲区
        vector<Eigen::Vector3f> frame_buf; // 帧缓冲区

    public:
        rasterizer(int w, int h)
        {
            frame_buf.resize(w * h);
            depth_buf.resize(w * h);
        }

        auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
        {
            return Vector4f(v3.x(), v3.y(), v3.z(), w);
        }
        int get_next_id()
        {
            return next_id++;
        }
        int get_index(int x, int y) // 获取当前屏幕像素坐标
        {

            return (height - 1 - y) * width + x; // 翻转成直角坐标系 (height - 1 - y) + 当前x值
        }

        pos_buf_id load_positions(const vector<Eigen::Vector3f> &positions)
        {
            int id = get_next_id();
            pos_buf.emplace(id, positions);

            return {id}; // 将缓冲区iD分配顶点缓冲区索引缓冲区以及颜色缓冲区，保持唯一iD
        }
        ind_buf_id load_indices(const vector<Eigen::Vector3i> &indices)
        {
            int id = get_next_id();
            ind_buf.emplace(id, indices);

            return {id};
        }
        col_buf_id load_colors(const vector<Eigen::Vector3f> &cols)
        {
            int id = get_next_id();
            col_buf.emplace(id, cols);

            return {id};
        }

        void set_model(const Eigen::Matrix4f &m)
        {
            model = m;
        }
        void set_view(const Eigen::Matrix4f &v)
        {
            view = v;
        }
        void set_projection(const Eigen::Matrix4f &p)
        {
            projection = p;
        }

        void set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color) // 设置像素颜色
        {
            auto ind = (height - 1 - point.y()) * width + point.x(); // 用height - 1 - point.y()翻转保证为直角坐标系，然后加上point.x()得到当前坐标
            frame_buf[ind] = color;                                  // 用当前屏幕坐标为下标，存入一个颜色值
        }

        vector<Eigen::Vector3f> &frame_buffer() // 帧缓冲区
        {
            return frame_buf;
        }

        void clear(Buffers buff) // 清除或重置颜色缓冲区或深度缓冲区的内容
        {
            if ((buff & rst::Buffers::Color) == rst::Buffers::Color) // 清除颜色缓冲区

            {
                std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0}); // 初始化为Vector3f{0, 0, 0}
            }
            if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) // 清除深度缓冲区
            {
                std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity()); // 初始化为浮点无限大
            }
        }
        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end) // 点和点之间的画线算法
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
        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) // 三角形着色算法
        {
            auto &buf = pos_buf[pos_buffer.pos_id]; // 这是一个哈希表，用缓冲区[id]返回一个vector<Eigen::Vector3f>的浮点数向量，这行中里面存的是位置
            auto &ind = ind_buf[ind_buffer.ind_id]; // 同理，里面存的是索引
            auto &col = col_buf[col_buffer.col_id]; // 同理，里面存的是颜色

            float f1 = (50 - 0.1) / 2.0;
            float f2 = (50 + 0.1) / 2.0;

            Eigen::Matrix4f mvp = projection * view * model; // mvp矩阵
            for (auto &i : ind)                              // ind总共有两个即 (0，1，2)以及 (3，4，5)
            {
                Triangle t;
                Eigen::Vector4f v[] = {
                    mvp * to_vec4(buf[i[0]], 1.0f), // 通过索引获取位置并且添加 w 分量
                    mvp * to_vec4(buf[i[1]], 1.0f),
                    mvp * to_vec4(buf[i[2]], 1.0f)};

                for (auto &vec : v) // 透视除法
                {
                    vec /= vec.w();
                }

                for (auto &vert : v)
                {
                    vert.x() = 0.5 * width * (vert.x() + 1.0);  // x坐标即水平方向的视口变换
                    vert.y() = 0.5 * height * (vert.y() + 1.0); // y坐标即垂直方向的视口变换
                    vert.z() = vert.z() * f1 + f2;              // z坐标即深度的视口变换
                }

                for (int i = 0; i < 3; ++i)
                {
                    t.setVertex(i, v[i].head<3>());
                }

                auto col_x = col[i[0]]; // 通过索引获取颜色
                auto col_y = col[i[1]];
                auto col_z = col[i[2]];

                t.setColor(0, col_x[0], col_x[1], col_x[2]); // 设置顶点颜色，须注意的是逆时针方向时面向观察者即屏幕方向
                t.setColor(1, col_y[0], col_y[1], col_y[2]);
                t.setColor(2, col_z[0], col_z[1], col_z[2]);

                rasterize_triangle(t);
            }
        }
        static bool insideTriangle(float x, float y, const Vector3f *_v)
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
        static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v)
        {
            // float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
            // float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
            // float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
            //
            // float c1 = ((y - v[0].y()) * (v[2].x() - v[0].x())) - ((x - v[0].x()) * (v[2].y() - v[0].y())) / (v[1].y() - v[0].y()) * (v[2].x() - v[0].x()) - (v[1].x() - v[0].x() * (v[2].y() - v[0].y()));
            // float c2 = ((y - v[0].y()) * (v[1].x() - v[0].x())) - ((x - v[0].x()) * (v[1].y() - v[0].y())) / (v[2].y() - v[0].y()) * (v[1].x() - v[0].x()) - (v[2].x() - v[0].x() * (v[1].y() - v[0].y()));
            // float c3 = 1 - c1 - c2;
            // 计算三角形的面积
            auto triangleArea = [](float x1, float y1, float x2, float y2, float x3, float y3) -> float
            {
                return std::abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0f);
            };

            float areaTotal = triangleArea(v[0].x(), v[0].y(), v[1].x(), v[1].y(), v[2].x(), v[2].y());
            float area1 = triangleArea(x, y, v[1].x(), v[1].y(), v[2].x(), v[2].y());
            float area2 = triangleArea(v[0].x(), v[0].y(), x, y, v[2].x(), v[2].y());
            float area3 = triangleArea(v[0].x(), v[0].y(), v[1].x(), v[1].y(), x, y);

            float c1 = area1 / areaTotal;
            float c2 = area2 / areaTotal;
            float c3 = area3 / areaTotal;

            return {c1, c2, c3};
        }

        void rasterize_triangle(const Triangle &t)
        {
            auto v = t.toVector4(); // 顶点坐标
            int min_x, max_x, min_y, max_y;
            min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
            max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
            min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
            max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));
            std::vector<Eigen::Vector2f> super_sample_step{
                {-0.25, 0.25},
                {0.25, 0.25},
                {-0.25, -0.25},
                {0.25, -0.25}};

            for (int x = min_x; x <= max_x; x++)
            {
                for (int y = min_y; y <= max_y; y++)
                {
                    int count = 0;
                    float Depth = FLT_MAX;
                    float z_interpolated = FLT_MAX;

                    for (int i = 0; i < 4; i++)
                    {
                        if (insideTriangle(x + super_sample_step[i][0], y + super_sample_step[i][1], t.v))
                        {

                            auto [alpha, beta, gamma] = computeBarycentric2D(x + super_sample_step[i][0], y + super_sample_step[i][1], t.v);
                            float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;
                            Depth = std::min(Depth, z_interpolated);
                            count++; // 记录多少个采样点落在三角形中
                        }
                    }

                    if (count > 0 && depth_buf[get_index(x, y)] > Depth) // get_index(x, y)用于获取像素坐标，depth_buf[] 获取当前像素的深度值
                    {
                        Vector3f current_color = t.getColor() * count / 4; // 超采样后的颜色比例
                        depth_buf[get_index(x, y)] = Depth;

                        Vector3f point = {(float)x, (float)y, Depth};
                        set_pixel(point, current_color);
                    }
                }
            }
            // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
        }
    };
}

int main(int argc, const char **argv)
{
    float angle = 0;
    std::string filename = "output.png";

    rst::rasterizer r(700, 700);
    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{
        {2, 0, -2},     // a
        {0, 2, -2},     // b
        {-2, 0, -2},    // c
        {3.5, -1, -5},  // A
        {2.5, 1.5, -5}, // B
        {-1, 0.5, -5}}; // C
    std::vector<Eigen::Vector3i> ind{
        {0, 1, 2},
        {3, 4, 5}};
    std::vector<Eigen::Vector3f> cols{
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth); // 清除缓冲区

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data()); // 700x700像素，CV_32FC3指的是32位浮点数3通道（C3）图像。使用frame_buffer()中数据
        image.convertTo(image, CV_8UC3, 1.0f);                      // image表示图像，转换为8位无符号字符格式（CV_8UC3），1.0f意味着原值按比例1:1缩放
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);              // 第一个和第二个参数是输入图像和输出图像，从RGB格式转换为BGR格式
        cv::imshow("image", image);                                 // 创建一个窗口（标题为image）并显示其中的图像
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }
}