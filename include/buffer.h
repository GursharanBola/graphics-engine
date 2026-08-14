#ifndef BUFFER_H
#define BUFFER_H

#include <Eigen/Dense>
#include <vector>

class engine;

struct triangle {
    triangle(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
             const Eigen::Vector3d &p3)
        : point1(p1), point2(p2), point3(p3) {}

    Eigen::Vector3d point1;
    Eigen::Vector3d point2;
    Eigen::Vector3d point3;
};

struct tri_ref {
    int mesh_id = -1;
    int tri_index = -1;
    bool operator==(const tri_ref &other) const {
        return mesh_id == other.mesh_id && tri_index == other.tri_index;
    }
};

template <typename T> struct bound_box {
    T min_x = 0, max_x = 0;
    T min_y = 0, max_y = 0;
};

struct cached_tri {
    int tri_index;
    triangle p_tri;
    bound_box<int> b_box;
    bool operator<(const cached_tri &other) const {
        return tri_index < other.tri_index;
    }
};

// colors hold values [0, 1]
class color {
  public:
    Eigen::Vector3d val;
    color() : val(0.0, 0.0, 0.0) {}
    color(double r, double g, double b) : val(r, g, b) {}
    double r() const { return val[0]; }
    double g() const { return val[1]; }
    double b() const { return val[2]; }
    void clamp() { val = val.cwiseMin(1.0).cwiseMax(0.0); }
};

template <typename T> class buffer {
  public:
    friend engine;
    virtual ~buffer() = default;
    buffer(buffer &&) noexcept = default;
    buffer &operator=(buffer &&) noexcept = default;
    T get(const int i, const int j) const {
        return data[j * (length * sqrt_samples) + i];
    }
    void set(const int i, const int j, const T &value) {
        data[j * (length * sqrt_samples) + i] = value;
    }
    buffer(const int length, const int width, const int sqrt_samples = 1)
        : length(length), width(width), sqrt_samples(sqrt_samples),
          data(length * sqrt_samples * width * sqrt_samples) {}

    int get_length() const { return length * sqrt_samples; }
    int get_width() const { return width * sqrt_samples; }
    int get_length_p() const { return length; }
    int get_width_p() const { return width; }
    int get_sqrt_samples() const { return sqrt_samples; }
    auto get_start() const { return data.begin(); }
    // determine what user wants to clear default value to
    void clear(T fill_value = T{}) {
        std::fill(data.begin(), data.end(), fill_value);
    }

  private:
    int length;          // in pixels
    int width;           // in pixels
    int sqrt_samples;    // in sub_pixels
    std::vector<T> data; // actual buffer data
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

// tells us what is visible on the program
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
        : buffer(length, width, sqrt_samples) {}
    bool set_color(const int pix_x, const int pix_y, const int s_pix_x,
                   const int s_pix_y, const color &color);
    const color get_color(const int pix_x, const int pix_y, const int s_pix_x,
                          const int s_pix_y) const;
};

// simple buffer, has no sub-pixels
class image_buffer : public buffer<uint8_t> {
  public:
    image_buffer(const int length, const int width, const int num_channels)
        : buffer(length * num_channels, width), channels(num_channels) {};

    bool set_color(const int i, const int j, Eigen::Vector3d color);
    int draw_png(std::string filename, int length, int width, int channels,
                 void *data, int stride);
    Eigen::Vector3d get_color(const int i, const int j) const;

  private:
    int channels;
};

template <typename T> class tiled_buffer {
  public:
    ~tiled_buffer() = default;
    tiled_buffer(tiled_buffer &&) noexcept = default;
    tiled_buffer &operator=(tiled_buffer &&) noexcept = default;

    tiled_buffer(const int length, const int width, const int sqrt_tile_size_p,
                 const int sqrt_samples)
        : length(length), width(width), sqrt_samples(sqrt_samples),
          sqrt_tile_size_p(sqrt_tile_size_p) {

        sqrt_tile_size_s = sqrt_tile_size_p * sqrt_samples;
        tile_size_p = sqrt_tile_size_p * sqrt_tile_size_p;
        tile_size_s = sqrt_tile_size_s * sqrt_tile_size_s;

        num_tiles_x = (length + sqrt_tile_size_p - 1) / sqrt_tile_size_p;
        num_tiles_y = (width + sqrt_tile_size_p - 1) / sqrt_tile_size_p;

        this->data.resize(num_tiles_x * num_tiles_y * tile_size_s);
    };

    const T get_elem(const int i, const int j) const {
        const int tile_x = i / sqrt_tile_size_s;
        const int tile_y = j / sqrt_tile_size_s;
        const int s_pix_x = i % sqrt_tile_size_s;
        const int s_pix_y = j % sqrt_tile_size_s;

        const int tile_index = num_tiles_x * tile_y + tile_x;
        const int tile_start = tile_index * tile_size_s;
        const int index = tile_start + (s_pix_y * sqrt_tile_size_s) + s_pix_x;

        return data[index];
    }

    void set_elem(const int i, const int j, const T &val) {
        const int tile_x = i / sqrt_tile_size_s;
        const int tile_y = j / sqrt_tile_size_s;
        const int s_pix_x = i % sqrt_tile_size_s;
        const int s_pix_y = j % sqrt_tile_size_s;

        const int tile_index = num_tiles_x * tile_y + tile_x;
        const int tile_start = tile_index * tile_size_s;
        const int index = tile_start + (s_pix_y * sqrt_tile_size_s) + s_pix_x;
        data[index] = val;
    }

    int get_length() const { return length * sqrt_samples; }
    int get_width() const { return width * sqrt_samples; }
    int get_length_p() const { return length; }
    int get_width_p() const { return width; }
    int get_sqrt_samples() const { return sqrt_samples; }
    auto get_start() const { return data.begin(); }
    // determine what user wants to clear default value to
    void clear(T fill_value = T{}) {
        std::fill(data.begin(), data.end(), fill_value);
    }

  private:
    int sqrt_tile_size_p; // in pixels
    int sqrt_tile_size_s; // in sub pixels
    int tile_size_p;      // in pixels (area)
    int tile_size_s;      // in sub pixels (area)
    int num_tiles_x;      // how long in tiles
    int num_tiles_y;      // how tall in tiles

    int length;          // in pixels
    int width;           // in pixels
    int sqrt_samples;    // in sub_pixels
    std::vector<T> data; // actual buffer data
};

#endif
