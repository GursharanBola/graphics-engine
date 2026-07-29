#include "buffer.h"
#include <tuple>

// project a point and correct for depth
namespace engine_helper {
Eigen::Vector3d
project_point(const Eigen::Vector3d &p1, const Eigen::Vector3d &cam_u,
              const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
              const Eigen::Vector3d &origin, const double focal_len);

// project a triangle and correct for depth
triangle proj_tri(const triangle &tri, const Eigen::Vector3d &cam_u,
                  const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
                  const Eigen::Vector3d &origin, const double focal_len);

// create a bound box in terms of pixels on an image_buffer
bound_box<int> create_box(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                          const Eigen::Vector3d &p3, const double aspect_ratio,
                          const int img_length, const int img_width);

// gets [face, local_u, local_v]
std::tuple<int, double, double> get_face(const Eigen::Vector3d &cntrd_surf_pt,
                                         const double max_xy);

// takes the average of each pixel and ouputs the image
void take_avg(const color_buffer &c_buff, image_buffer &img);

// does val ** pow quickly
double f_pow(const double val, const unsigned int pow);

// Pineda's edge function
double edge_func(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                 const Eigen::Vector3d &p);

// get bary fits into the pipeline in the following way:
// triangle objects store point1, point2, point3 in ccw order the
// program then projects the triangle onto the image plane by
// (point1, point2, point3)
//     V        V      V
// (proj_p1, proj_p2, proj_p3)
//
// get_bary() expects the points to be CCW, if the points are CW,
// then the area is negative and will return Eigen::Vector3d{-1,-1,-1}
Eigen::Vector3d get_bary(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                         const Eigen::Vector3d &p3,
                         const Eigen::Vector3d &test_pt);

// run a job directly on threads, is thread safe, b_box can be a thread box
template <typename Func, typename buff_T, typename arg_T>
void with_buff(Func &&job, const bound_box<int> &b_box, buff_T &&buffs,
               arg_T &&args);

// args needed for ras_tri_fn
template <typename T> struct ra_tri_args {
    const int paren_len; // in sub_pixels
    const int paren_wid; // in sub_pixels
    const triangle &p_tri;
    const T &val;
};

// buffers needed for rast_tri_fn
template <typename T> class ra_tri_buffs {
  public:
    buffer<T> &buff;
    buffer<double> &z_buff;
    ra_tri_buffs(buffer<T> &b, buffer<double> &z) : buff(b), z_buff(z) {}
};

// how to call rast_tri
class rast_tri_fn {
  public:
    template <typename buff_T, typename arg_T>
    void operator()(const bound_box<int> &b, buff_T &&bf, arg_T &&ar) const {
        rast_tri(b, std::forward<buff_T>(bf), std::forward<arg_T>(ar));
    }
};

template <typename T>
void rast_tri(const bound_box<int> &b_box, ra_tri_buffs<T> &buffs,
              ra_tri_args<T> &args);
} // namespace engine_helper
