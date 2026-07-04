#include "buffer.h"

// change the basis onto camera basis
namespace engine_helper {
Eigen::Vector3d project_point(const Eigen::Vector3d p1,
                              const Eigen::Vector3d cam_u,
                              const Eigen::Vector3d cam_v,
                              const Eigen::Vector3d cam_w,
                              const Eigen::Vector3d origin);

raw_tri proj_tri(const raw_tri &tri, const Eigen::Vector3d cam_u,
                 const Eigen::Vector3d cam_v, const Eigen::Vector3d cam_w,
                 const Eigen::Vector3d origin);

// bound_box() runs on world coordinates on the plane of interest
bound_box<double> w_box(const Eigen::Vector3d p1, const Eigen::Vector3d p2,
                        const Eigen::Vector3d p3, const double aspect_ratio);

// create a bound box in terms of pixels on an image_buffer
bound_box<int> create_box(const Eigen::Vector3d p1, const Eigen::Vector3d p2,
                          const Eigen::Vector3d p3, const double aspect_ratio,
                          const int img_length, const int img_width);

// Pineda's edge function
double edge_func(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                 const Eigen::Vector3d &p);

// works regardless of winding order
Eigen::Vector3d get_bary(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                         const Eigen::Vector3d &p3,
                         const Eigen::Vector3d &test_pt);

// push the tile back onto the buffer, offset_x,y is in pixels
template <typename T>
void pull(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y);

// push the tile back onto the buffer, offset_x,y is in pixels
template <typename T>
void push(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y);

// single thread job with tiles given b_box
template <typename Func, typename BuffType, typename ArgType>
void with_tiles(Func &&job, bound_box<int> &b_box, BuffType &&buffs,
                ArgType &&args);

// single thread job on buffers given b_box
template <typename Func, typename BuffType, typename ArgType>
void with_buff(Func &&job, const bound_box<int> &b_box, BuffType &&buffs,
               ArgType &&args);

// args needed for ras_tri
template <typename T> struct ra_tri_args {
    const int paren_len;
    const int paren_wid;
    const raw_tri &p_tri;
    const T &val;
};

// buffers needed for rast_tri
template <typename T> class ra_tri_buffs {
  public:
    buffer<T> &buff;
    buffer<double> &z_buff;
    ra_tri_buffs(buffer<T> &b, buffer<double> &z) : buff(b), z_buff(z) {}
    ra_tri_buffs make_tiles(const int sqrt_tile) const {
        const int sqrt_samples = this->buff.get_sqrt_samples();
        const int side_len = TILE_SIZE * sqrt_samples;
        buffer<T> tile{side_len, side_len};
        buffer<double> z_tile{side_len, side_len};
        return ra_tri_buffs{tile, z_tile};
    }
    void pull_to_tile(ra_tri_buffs &main_b, const int offset_x,
                      const int offset_y) {
        pull(main_b.buff, buff, offset_x, offset_y);
        pull(main_b.z_buff, z_buff, offset_x, offset_y);
    }
    void push_to_buff(ra_tri_buffs &main_b, const int offset_x,
                      const int offset_y) {
        push(buff, main_b.buff, offset_x, offset_y);
        push(z_buff, main_b.z_buff, offset_x, offset_y);
    }

  private:
    static constexpr int TILE_SIZE = 4;
};

// how to call ras_tri in engine.cpp
class rast_tri_fn {
  public:
    template <typename BuffType, typename ArgType>
    void operator()(const bound_box<int> &b, BuffType &bf, ArgType &ar) const {
        rast_tri(b, bf, ar, 0, 0);
    }
};

template <typename T>
void rast_tri(const bound_box<int> &b_box, ra_tri_buffs<T> &buffs,
              ra_tri_args<T> &args, const int off_x = 0, const int off_y = 0);
} // namespace engine_helper
