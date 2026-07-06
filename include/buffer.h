#ifndef BUFFER_H
#define BUFFER_H

#include <Eigen/Dense>
#include <vector>

struct point {
    int x;
    int y;
};

struct raw_tri {
    Eigen::Vector3d p1;
    Eigen::Vector3d p2;
    Eigen::Vector3d p3;
};

struct tri_ref {
    int mesh_id;
    int tri_index;
    bool operator==(const tri_ref &other) const {
        return mesh_id == other.mesh_id && tri_index == other.tri_index;
    }
};

template <typename T> struct bound_box {
    T min_x, max_x = 0;
    T min_y, max_y = 0;
};

// colors hold values [0, 1]
class color {
  public:
    Eigen::Array3d val;
    color() : val(0.0, 0.0, 0.0) {}
    color(double r, double g, double b) : val(r, g, b) {}
    double r() const { return val[0]; }
    double g() const { return val[1]; }
    double b() const { return val[2]; }
    void clamp() { val = val.min(1.0).max(0.0); }
};

template <typename T> class buffer {
  public:
    T get(const int i, const int j) const {
        return data[i * (length * sqrt_samples) + j];
    }
    void set(const int i, const int j, const T &value) {
        data[i * (length * sqrt_samples) + j] = value;
    }
    buffer(const int length, const int width, const int sqrt_samples = 1)
        : length(length), width(width), sqrt_samples(sqrt_samples) {
        data = std::vector<T>(length * sqrt_samples * width * sqrt_samples);
    };
    int get_length() const { return length * sqrt_samples; }
    int get_width() const { return width * sqrt_samples; }
    int get_length_p() const { return length; }
    int get_width_p() const { return width; }
    int get_sqrt_samples() const { return sqrt_samples; }
    auto get_start() const { return data.begin(); }
    // assumes custom types have default constructor
    void clear() { std::fill(data.begin(), data.end(), T{}); }

  private:
    int length;            // in pixels
    int width;             // in pixels
    int sqrt_samples;      // in sub_pixels
    std::vector<T> data{}; // intialize buffers as empty
};

class z_buffer : public buffer<double> {
  public:
    z_buffer(const int length, const int width, const int sqrt_samples)
        : buffer(length, width, sqrt_samples) {};

    void set_sample(const int i, const int j, const int sam_i, const int sam_j,
                    const double val) {
        int absolute_i = (i * get_sqrt_samples()) + sam_i;
        int absolute_j = (j * get_sqrt_samples()) + sam_j;
        set(absolute_i, absolute_j, val);
    }
};

class seen_buffer : public buffer<tri_ref> {
  public:
    seen_buffer(const int length, const int width, const int sqrt_samples)
        : buffer(length, width, sqrt_samples) {};
    void set_sample(const int i, const int j, const int sam_i, const int sam_j,
                    const tri_ref val) {
        int absolute_i = (i * get_sqrt_samples()) + sam_i;
        int absolute_j = (j * get_sqrt_samples()) + sam_j;
        set(absolute_i, absolute_j, val);
    }
};

// store a color at each sub_pixel
class color_buffer : public buffer<color> {
  public:
    color_buffer(const int length, const int width, const int sqrt_samples)
        : buffer(length * sqrt_samples, width * sqrt_samples) {}
    bool set_color(const int pix_x, const int pix_y, const int s_pix_x,
                   const int s_pix_y, const color &color);
    color get_color(const int pix_x, const int pix_y, const int s_pix_x,
                    const int s_pix_y) const;
};

// simple buffer, has no sub-pixels
class image_buffer : public buffer<int> {
  public:
    image_buffer(const int length, const int width, const int num_channels)
        : buffer(length * num_channels, width) {};
    bool set_color(const int i, const int j, Eigen::Vector3d color);
    int draw_png(std::string filename, int width, int height, int channels,
                 void *data, int stride);
    Eigen::Vector3d get_color(const int i, const int j) const;
};

class vertex_buffer {
  public:
    vertex_buffer() = default;
    void add(const Eigen::Vector3d &v) { data.push_back(v); }
    Eigen::Vector3d get(const int i) const { return data[i]; }
    int size() const { return data.size(); }
    void clear() { data.clear(); }

  private:
    std::vector<Eigen::Vector3d> data;
};

inline double clamp(double x, double min, double max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}
#endif
