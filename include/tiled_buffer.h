#ifndef TILED_BUFFER_H
#define TILED_BUFFER_H

#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
#include <vector>

class engine;

// TODO: handle there being a row major order tiling and column major order
// tiling

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
    friend engine;
    Eigen::Vector3d val;
    color() : val(0.0, 0.0, 0.0) {}
    color(double r, double g, double b) : val(r, g, b) {}
    double r() const { return val[0]; }
    double g() const { return val[1]; }
    double b() const { return val[2]; }
    void clamp() { val = val.cwiseMin(1.0).cwiseMax(0.0); }

    color &operator+=(const color &other) {
        val += other.val;
        return *this;
    }

    color operator/(double scalar) const {
        color res;
        res.val = this->val / scalar;
        return res;
    }
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

    // if user is interested they can parse row major order buffers and use flip
    // to make buffer column major order considering they size buffer correctly
    std::tuple<int, int> flip(const int i, const int j) { return {j, i}; }

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

    std::vector<T> make_row_major_order() {
        const int width_s = get_length();
        const int height_s = get_width();
        std::vector<T> res(width_s * height_s);

        int src_index = 0;
        for (int tile_y = 0; tile_y < num_tiles_y; ++tile_y) {
            for (int tile_x = 0; tile_x < num_tiles_x; ++tile_x) {
                for (int s_pix_y = 0; s_pix_y < sqrt_tile_size_s; ++s_pix_y) {
                    const int dest_y = tile_y * sqrt_tile_size_s + s_pix_y;
                    for (int s_pix_x = 0; s_pix_x < sqrt_tile_size_s;
                         ++s_pix_x) {
                        const int dest_x = tile_x * sqrt_tile_size_s + s_pix_x;

                        if (dest_x < width_s && dest_y < height_s) {
                            const size_t dest_index =
                                static_cast<size_t>(dest_y) * width_s + dest_x;
                            res[dest_index] = data[src_index];
                        }
                        src_index++;
                    }
                }
            }
        }
        return res;
    }

    std::vector<T> return_avg() {
        const int samples_per_pixel = sqrt_samples * sqrt_samples;
        std::vector<T> res(length * width);
        std::array<T, tile_size_p> pix_sums;
        for (int y = 0; y < width; ++y) {
            for (int x = 0; x < length; ++x) {
                T sum = T{};
                for (int sy = 0; sy < sqrt_samples; ++sy) {
                    for (int sx = 0; sx < sqrt_samples; ++sx) {
                        const int sub_x = x * sqrt_samples + sx;
                        const int sub_y = y * sqrt_samples + sy;
                        sum += get_elem(sub_x, sub_y);
                    }
                }
                const int res_index = y * length + x;
                res[res_index] = sum / static_cast<double>(samples_per_pixel);
            }
        }
        return res;
    }

    int get_length() const { return length * sqrt_samples; }
    int get_width() const { return width * sqrt_samples; }
    int get_length_p() const { return length; }
    int get_width_p() const { return width; }
    int get_sqrt_samples() const { return sqrt_samples; }
    int get_sqrt_tile_size_p() { return sqrt_tile_size_p; }
    int get_sqrt_tile_size_s() { return sqrt_tile_size_s; }
    int get_tile_size_p() { return tile_size_p; }
    int get_tile_size_s() { return tile_size_s; }
    int get_num_tiles_x() { return num_tiles_x; }
    int get_num_tiles_y() { return num_tiles_y; }
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

class tiled_depth_buffer : public tiled_buffer<double> {
  public:
    tiled_depth_buffer(const int length, const int width,
                       const int sqrt_tile_size_p, const int sqrt_samples)
        : tiled_buffer(length, width, sqrt_tile_size_p, sqrt_samples) {}
};

class tiled_visibility_buffer : public tiled_buffer<tri_ref> {
  public:
    tiled_visibility_buffer(const int length, const int width,
                            const int sqrt_tile_size_p, const int sqrt_samples)
        : tiled_buffer(length, width, sqrt_tile_size_p, sqrt_samples) {}
};

class tiled_color_buffer : public tiled_buffer<color> {
  public:
    tiled_color_buffer(const int length, const int width,
                       const int sqrt_tile_size_p, const int sqrt_samples)
        : tiled_buffer(length, width, sqrt_tile_size_p, sqrt_samples) {}
};

class tiled_render_buffer : public tiled_buffer<uint8_t> {
  public:
    tiled_render_buffer(const int length, const int width, const int channels,
                        const int sqrt_tile_size_p)
        : tiled_buffer(length * channels, width, sqrt_tile_size_p, 1),
          channels(channels) {}

  private:
    int channels;
};

#endif
