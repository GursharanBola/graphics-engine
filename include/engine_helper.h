#include "buffer.h"
#include <memory>
#include <vector>

// efficient cache map, astores tri_refs densely to avoid cache misses
// also optimizes iteration order, using post_process()

// counts_in is the max number of triangles in each mesh
struct cached_tri {
    int tri_index;
    raw_tri p_tri;
    bound_box<int> b_box;
    bool operator<(const cached_tri &other) const {
        return tri_index < other.tri_index;
    }
};

template <typename T> struct e_cache_map {
    std::vector<T> data;
    std::vector<int> initial;
    std::vector<int> offsets;
    e_cache_map(const std::vector<int> &counts_in) {
        int num_meshes = counts_in.size();
        offsets.resize(num_meshes);
        initial.resize(num_meshes);
        int current_total = 0;
        for (int i = 0; i < num_meshes; ++i) {
            initial[i] = current_total;
            offsets[i] = current_total;
            current_total += counts_in[i];
        }
        data.resize(current_total);
    }
    void add_tri(const int m_id, const T &item) {
        data[offsets[m_id]] = item;
        offsets[m_id]++;
    }
    void post_process() {
        int num_meshes = initial.size();
        for (int m_id = 0; m_id < num_meshes; ++m_id) {
            int start = initial[m_id];
            int end = offsets[m_id];
            std::sort(data.begin() + start, data.begin() + end);
        }
    }
    void clear() { offsets = initial; }
};

// project a point and correct for depth
namespace engine_helper {
Eigen::Vector3d
project_point(const Eigen::Vector3d &p1, const Eigen::Vector3d &cam_u,
              const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
              const Eigen::Vector3d &origin, const double focal_len);

// project a triangle and correct for depth
raw_tri proj_tri(const raw_tri &tri, const Eigen::Vector3d &cam_u,
                 const Eigen::Vector3d &cam_v, const Eigen::Vector3d &cam_w,
                 const Eigen::Vector3d &origin, const double focal_len);

// bound_box() runs on world coordinates on the plane of interest
bound_box<double> w_box(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                        const Eigen::Vector3d &p3, const double aspect_ratio);

// create a bound box in terms of pixels on an image_buffer
bound_box<int> create_box(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                          const Eigen::Vector3d &p3, const double aspect_ratio,
                          const int img_length, const int img_width);

// Pineda's edge function
double edge_func(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                 const Eigen::Vector3d &p);

// get bary fits into the pipeline in the following way:
// triangle objects store p1.v_id, p2.v_id, p_3.v_id in ccw order the
// program then projects the triangle onto the image plane in the following
// way (p1.v_id, p2.v_id, p_3.v_id)
//     V        V         V
// (proj_p1, proj_p2, proj_p3)
//
// get_bary() expects the pointsp1, p2 and p3 to be CCW, if the point is CW,
// then the area is negative and will return Eigen::Vector3d{-1,-1,-1}
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
void with_buff(Func &&job, const bound_box<int> &b_box, BuffType &buffs,
               ArgType &args);

// args needed for ras_tri
template <typename T> struct ra_tri_args {
    const int paren_len; // in sub_pixels
    const int paren_wid; // in sub_pixels
    const raw_tri &p_tri;
    const T &val;
};

// buffers needed for rast_tri
template <typename T> class ra_tri_buffs {
  public:
    buffer<T> *buff;
    buffer<double> *z_buff;
    ra_tri_buffs(buffer<T> *b, buffer<double> *z) : buff(b), z_buff(z) {}
    void pull_to_tile(ra_tri_buffs &main_b, const int offset_x,
                      const int offset_y) {
        pull(*(main_b.buff), *buff, offset_x, offset_y);
        pull(*(main_b.z_buff), *z_buff, offset_x, offset_y);
    }
    void push_to_buff(ra_tri_buffs &main_b, const int offset_x,
                      const int offset_y) {
        push(*buff, *(main_b.buff), offset_x, offset_y);
        push(*z_buff, *(main_b.z_buff), offset_x, offset_y);
    }
};

// function will generate buffers and allow us to manage the ptrs to data
template <typename T> struct ra_tri_tile_manager {
    std::unique_ptr<buffer<T>> owned_buff;
    std::unique_ptr<buffer<double>> owned_z_buff;
    ra_tri_buffs<T> view;

    ra_tri_tile_manager(int sqrt_tile, int sqrt_samples)
        : owned_buff(
              std::make_unique<buffer<T>>(sqrt_tile, sqrt_tile, sqrt_samples)),
          owned_z_buff(std::make_unique<buffer<double>>(sqrt_tile, sqrt_tile,
                                                        sqrt_samples)),
          view(owned_buff.get(), owned_z_buff.get()) {}
};

// helper function to help create managers which can be used in the program
// sqrt_tile is in pixels
template <typename T>
ra_tri_tile_manager<T> make_tiles_for(ra_tri_buffs<T> &main_b, int sqrt_tile) {
    int sqrt_samples = main_b.buff->get_sqrt_samples();
    return ra_tri_tile_manager<T>(sqrt_tile, sqrt_samples);
}

// how to call ras_tri in engine.cpp
class rast_tri_fn {
  public:
    template <typename BuffType, typename ArgType>
    void operator()(const bound_box<int> &b, BuffType &bf, ArgType &ar,
                    const int off_x, const int off_y) const {
        rast_tri(b, bf, ar, off_x, off_y);
    }
};

template <typename T>
void rast_tri(const bound_box<int> &b_box, ra_tri_buffs<T> &buffs,
              ra_tri_args<T> &args, const int off_x = 0, const int off_y = 0);
} // namespace engine_helper
