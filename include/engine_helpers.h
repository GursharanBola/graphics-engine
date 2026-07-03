#include "buffer.h"
#include <cmath>
#include <stdexcept>
#include <utility>

// TODO: Follow the one definition rule
namespace engine_helpers {
Eigen::Vector3d project_point(const Eigen::Vector3d p1,
                              const Eigen::Vector3d cam_u,
                              const Eigen::Vector3d cam_v,
                              const Eigen::Vector3d cam_w,
                              const Eigen::Vector3d origin) {
    Eigen::Vector3d translated = p1 - origin;
    double x_cam = translated.dot(cam_u);
    double y_cam = translated.dot(cam_v);
    double z_cam = -translated.dot(cam_w);
    return Eigen::Vector3d(x_cam, y_cam, z_cam);
}

raw_tri proj_tri(const raw_tri &tri, const Eigen::Vector3d cam_u,
                 const Eigen::Vector3d cam_v, const Eigen::Vector3d cam_w,
                 const Eigen::Vector3d origin) {
    Eigen::Vector3d p1 = tri.p1;
    Eigen::Vector3d p2 = tri.p2;
    Eigen::Vector3d p3 = tri.p3;
    Eigen::Vector3d proj_1 = project_point(p1, cam_u, cam_v, cam_w, origin);
    Eigen::Vector3d proj_2 = project_point(p2, cam_u, cam_v, cam_w, origin);
    Eigen::Vector3d proj_3 = project_point(p3, cam_u, cam_v, cam_w, origin);
    return raw_tri{proj_1, proj_2, proj_3};
}

// bound_box() runs on world coordinates on the plane of interest
bound_box<double> w_box(const Eigen::Vector3d p1, const Eigen::Vector3d p2,
                        const Eigen::Vector3d p3, const double aspect_ratio) {
    double hor_min = std::min({p1[0], p2[0], p3[0]});
    double ver_min = std::min({p1[1], p2[1], p3[1]});
    double hor_max = std::max({p1[0], p2[0], p3[0]});
    double ver_max = std::max({p1[1], p2[1], p3[1]});
    // image planes are bound by the box: [-aspect_ratio, aspect_ratio, -1, 1]
    if (hor_min < -aspect_ratio) {
        hor_min = -aspect_ratio;
    }
    if (hor_max > aspect_ratio) {
        hor_max = aspect_ratio;
    }
    if (ver_min < -1) {
        ver_min = -1;
    }
    if (ver_max > 1) {
        ver_max = 1;
    }
    bound_box<double> bbox{hor_min, hor_max, ver_min, ver_max};
    return bbox;
}

// create a bound box in terms of pixels on an image_buffer
bound_box<int> create_box(const Eigen::Vector3d p1, const Eigen::Vector3d p2,
                          const Eigen::Vector3d p3, const double aspect_ratio,
                          const int img_length, const int img_width) {
    bound_box<double> w_bbox = w_box(p1, p2, p3, aspect_ratio);
    int left_pixel = std::ceil((w_bbox.min_x / aspect_ratio + 1) * img_length);
    int right_pixel = std::ceil((w_bbox.max_x / aspect_ratio + 1) * img_length);
    int top_pixel = std::ceil(1 - 0.5 * (w_bbox.min_y + 1) * img_width);
    int bot_pixel = std::ceil(1 - 0.5 * (w_bbox.max_y + 1) * img_width);
    return bound_box<int>{left_pixel, right_pixel, top_pixel, bot_pixel};
}

// Pineda's edge function
bool is_in_tri(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
               const Eigen::Vector3d &p3, const Eigen::Vector3d &test) {
    double dx1 = p2.x() - p1.x();
    double dy1 = p2.y() - p1.y();
    double tx1 = test.x() - p1.x();
    double ty1 = test.y() - p1.y();
    double cross1 = dx1 * ty1 - dy1 * tx1;
    int val = (cross1 > 0.0) ? 1 : -1;
    double dx2 = p3.x() - p2.x();
    double dy2 = p3.y() - p2.y();
    double tx2 = test.x() - p2.x();
    double ty2 = test.y() - p2.y();
    if (val * (dx2 * ty2 - dy2 * tx2) < 0.0)
        return false;
    double dx3 = p1.x() - p3.x();
    double dy3 = p1.y() - p3.y();
    double tx3 = test.x() - p3.x();
    double ty3 = test.y() - p3.y();
    return val * (dx3 * ty3 - dy3 * tx3) >= 0.0;
}

// push the tile back onto the buffer, offset_x,y is in pixels
template <typename T>
void pull(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y) {
    const int sqrt_tile = dest.get_length_p();
    const int tile_w = dest.get_width_p();
    if (sqrt_tile != tile_w) {
        std::runtime_error("src is a tile, it must be square");
    }
    const int sqrt_samples = dest.get_sqrt_samples();
    const int buff_sqrt_samples = src.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        std::runtime_error("src and dest must have same num_samples");
    }
    const int buff_len_p = src.get_length_p();
    const int buff_wid_p = src.get_width_p();
    if (offset_x > buff_len_p || offset_y > buff_len_p) {
        std::runtime_error("offset out of range");
    }
    if (buff_len_p < sqrt_tile || buff_wid_p < sqrt_tile) {
        std::runtime_error("src must be smaller than buffer");
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    const int rem_x = buff_len_p - offset_x;
    const int rem_y = buff_wid_p - offset_y;
    if (rem_x < sqrt_tile || rem_y < sqrt_tile) {
        for (int i = 0; i < rem_x * sqrt_samples; i++) {
            for (int j = 0; j < rem_y * sqrt_samples; j++) {
                dest.set(i, j, src.get(s_offset_x + i, s_offset_y + j));
            }
        }
    }
    for (int i = 0; i < sqrt_tile * sqrt_samples; i++) {
        for (int j = 0; j < sqrt_tile * sqrt_samples; j++) {
            dest.set(i, j, src.get(s_offset_x + i, s_offset_y + j));
        }
    }
}

// push the tile back onto the buffer, offset_x,y is in pixels
template <typename T>
void push(buffer<T> &src, buffer<T> &dest, const int offset_x,
          const int offset_y) {
    const int sqrt_tile = src.get_length_p();
    const int tile_w = src.get_width_p();
    if (sqrt_tile != tile_w) {
        std::runtime_error("src is a tile, it must be square");
    }
    const int sqrt_samples = src.get_sqrt_samples();
    const int buff_sqrt_samples = dest.get_sqrt_samples();
    if (sqrt_samples != buff_sqrt_samples) {
        std::runtime_error("src and dest must have same num_samples");
    }
    const int buff_len_p = dest.get_length_p();
    const int buff_wid_p = dest.get_width_p();
    if (offset_x > buff_len_p || offset_y > buff_len_p) {
        std::runtime_error("offset out of range");
    }
    if (buff_len_p < sqrt_tile || buff_wid_p < sqrt_tile) {
        std::runtime_error("src must be smaller than buffer");
    }
    const int s_offset_x = offset_x * sqrt_samples;
    const int s_offset_y = offset_y * sqrt_samples;
    const int rem_x = buff_len_p - offset_x;
    const int rem_y = buff_wid_p - offset_y;
    if (rem_x < sqrt_tile || rem_y < sqrt_tile) {
        for (int i = 0; i < rem_x * sqrt_samples; i++) {
            for (int j = 0; j < rem_y * sqrt_samples; j++) {
                dest.set(s_offset_x + i, s_offset_y + j, src.get(i, j));
            }
        }
    }
    for (int i = 0; i < sqrt_tile * sqrt_samples; i++) {
        for (int j = 0; j < sqrt_tile * sqrt_samples; j++) {
            dest.set(s_offset_x + i, s_offset_y + j, src.get(i, j));
        }
    }
}

// single thread job given the correct b_box
template <typename Func, typename BuffType, typename ArgType>
void with_tiles(Func &&job, bound_box<int> &b_box, BuffType &&buffs,
                ArgType &&args) {
    constexpr int sqrt_tile = 4;
    int sqrt_samples = buffs.get_sqrt_samples();
    int box_len = b_box.max_x - b_box.min_x;
    int box_wid = b_box.max_y - b_box.min_y;
    auto tiles = buffs.make_tiles(sqrt_tile);
    int x_tiles = (box_len + sqrt_tile - 1) / sqrt_tile;
    int y_tiles = (box_wid + sqrt_tile - 1) / sqrt_tile;
    for (int i = 0; i < x_tiles; ++i) {
        int offset_x = b_box.min_x + (i * sqrt_tile);
        for (int j = 0; j < y_tiles; ++j) {
            int offset_y = b_box.min_y + (j * sqrt_tile);
            tiles.pull_to_tile(buffs, offset_x, offset_y);
            job(b_box, std::forward<BuffType>(tiles),
                std::forward<ArgType>(args));
            tiles.push_to_buff(buffs, offset_x, offset_y);
        }
    }
}
// TODO: write the with_buffs function
template <typename T> struct ra_tri_args {
    const int paren_len;
    const int paren_wid;
    const raw_tri &p_tri;
    const T &val;
    const double alpha = 1.0 / 3.0;
    const double beta = 1.0 / 3.0;
    const double gamma = 1.0 / 3.0;
};

template <typename T> struct ra_tri_buffs {
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

template <typename T>
void rast_tri(const bound_box<int> &b_box, ra_tri_buffs<T> &buffs,
              ra_tri_args<T> &args, const int off_x, const int off_y) {
    const int paren_len = args.paren_len;
    const int paren_wid = args.paren_wid;
    const double a_ratio = (double)paren_len / paren_wid;
    const int sqrt_samples = buffs.buff.get_sqrt_samples();
    const int left = b_box.min_x * sqrt_samples;
    const int right = b_box.max_x * sqrt_samples;
    const int top = b_box.min_y * sqrt_samples;
    const int bot = b_box.max_y * sqrt_samples;
    const int alpha = args.alpha;
    const int beta = args.beta;
    const int gamma = args.gamma;
    const int s_pix_to_world_x = paren_len * 2 * a_ratio - a_ratio;
    const int s_pix_to_world_y = paren_wid * 2 - 1;
    const raw_tri &p_tri = args.raw_tri;
    const double inv_p1_z = 1.0 / p_tri.p1[2];
    const double inv_p2_z = 1.0 / p_tri.p2[2];
    const double inv_p3_z = 1.0 / p_tri.p3[2];
    for (int k = left; k < right; ++k) {
        for (int l = top; l < bot; ++l) {
            double world_x = (double)k / s_pix_to_world_x;
            double world_y = (double)l / s_pix_to_world_y;
            Eigen::Vector3d test{world_x, world_y, 0};
            bool is_in = is_in_tri(p_tri.p1, p_tri.p2, p_tri.p3, test);
            if (!is_in) {
                continue;
            }
            double z_rep =
                alpha * inv_p1_z + beta * inv_p2_z + gamma * inv_p3_z;
            double z_sub = (z_rep > 0) ? (1.0 / z_rep)
                                       : std::numeric_limits<double>::max();
            int x = k - off_x;
            int y = l - off_y;
            if (args.z_buff.get(x, y) > z_sub) {
                args.z_buff.set(x, y, z_sub);
                args.buff.set(x, y, args.val);
            }
        }
    }
}
} // namespace engine_helpers
