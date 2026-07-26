#include "engine_helper.h"

Eigen::Vector3d engine_helper::project_point(const Eigen::Vector3d &p1,
                                             const Eigen::Vector3d &cam_u,
                                             const Eigen::Vector3d &cam_v,
                                             const Eigen::Vector3d &cam_w,
                                             const Eigen::Vector3d &origin,
                                             const double focal_len) {
    Eigen::Vector3d translated = p1 - origin;
    double x_cam = translated.dot(cam_u);
    double y_cam = translated.dot(cam_v);
    double z_cam = -translated.dot(cam_w);
    if (std::abs(z_cam) < 1e-6) {
        z_cam = 1e-6;
    }
    double ratio = focal_len / z_cam;
    return Eigen::Vector3d(x_cam * ratio, y_cam * ratio, z_cam);
}

triangle engine_helper::proj_tri(const triangle &tri,
                                 const Eigen::Vector3d &cam_u,
                                 const Eigen::Vector3d &cam_v,
                                 const Eigen::Vector3d &cam_w,
                                 const Eigen::Vector3d &origin,
                                 const double focal_len) {
    return triangle{
        project_point(tri.point1, cam_u, cam_v, cam_w, origin, focal_len),
        project_point(tri.point2, cam_u, cam_v, cam_w, origin, focal_len),
        project_point(tri.point3, cam_u, cam_v, cam_w, origin, focal_len)};
}

bound_box<int>
engine_helper::create_box(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                          const Eigen::Vector3d &p3, const double aspect_ratio,
                          const int img_length, const int img_width) {
    const double inv_aspect = 1.0 / aspect_ratio;
    const double half_len = img_length * 0.5;
    const double half_wid = img_width * 0.5;
    const double x_scale = half_len * inv_aspect;
    const double y_scale = -half_wid;
    const double sx1 = p1[0] * x_scale + half_len;
    const double sy1 = p1[1] * y_scale + half_wid;
    const double sx2 = p2[0] * x_scale + half_len;
    const double sy2 = p2[1] * y_scale + half_wid;
    const double sx3 = p3[0] * x_scale + half_len;
    const double sy3 = p3[1] * y_scale + half_wid;
    const double min_x = std::min(sx1, std::min(sx2, sx3));
    const double max_x = std::max(sx1, std::max(sx2, sx3));
    const double min_y = std::min(sy1, std::min(sy2, sy3));
    const double max_y = std::max(sy1, std::max(sy2, sy3));
    int left = static_cast<int>(std::floor(min_x));
    int right = static_cast<int>(std::ceil(max_x));
    int top = static_cast<int>(std::floor(min_y));
    int bottom = static_cast<int>(std::ceil(max_y));
    return bound_box<int>{
        std::clamp(left, 0, img_length), std::clamp(right, 0, img_length),
        std::clamp(top, 0, img_width), std::clamp(bottom, 0, img_width)};
}

void engine_helper::take_avg(const color_buffer &color_buff,
                             image_buffer &img) {
    const int sqrt_samples = color_buff.get_sqrt_samples();
    const int inv_samples = 1 / (sqrt_samples * sqrt_samples);
    const int img_length = color_buff.get_length_p();
    const int img_height = color_buff.get_width_p();
    std::vector<Eigen::Vector3d> run_tot(img_length);
    int img_height_s = img_height * sqrt_samples;
    int img_length_s = img_length * sqrt_samples;
    int pixel_y = 0;
    int sub_y = 0;
    for (int j = 0; j < img_height_s; ++j) {
        int pixel_x = 0;
        int sub_x = 0;
        for (int i = 0; i < img_length_s; ++i) {
            run_tot[pixel_x] += color_buff.get(i, j).val;
            sub_x++;
            if (sub_x == sqrt_samples) {
                pixel_x++;
                sub_x = 0;
            }
        }
        sub_y++;
        if (sub_y == sqrt_samples) {
            for (int x = 0; x < run_tot.size(); ++x) {
                Eigen::Vector3d avg_col =
                    (run_tot[x] * inv_samples).array().min(1.0).max(0.0);
                img.set_color(x, pixel_y, avg_col);
            }
            std::fill(run_tot.begin(), run_tot.end(), Eigen::Vector3d::Zero());
            pixel_y++;
            sub_y = 0;
        }
    }
}

double engine_helper::f_pow(double val, unsigned int pow) {
    double res = 1.0;
    double base = val;
    while (pow > 0) {
        if (pow & 1) {
            res *= base;
        }
        base *= base;
        pow >>= 1;
    }
    return res;
}

double engine_helper::edge_func(const Eigen::Vector3d &a,
                                const Eigen::Vector3d &b,
                                const Eigen::Vector3d &p) {
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0]);
}

Eigen::Vector3d engine_helper::get_bary(const Eigen::Vector3d &p1,
                                        const Eigen::Vector3d &p2,
                                        const Eigen::Vector3d &p3,
                                        const Eigen::Vector3d &test_pt) {
    double area = edge_func(p1, p2, p3);
    // backface cull
    if (area < 0) {
        return {-1.0, -1.0, -1.0};
    }
    if (std::abs(area) < 1e-9) {
        return {-1.0, -1.0, -1.0};
    }
    double w1 = edge_func(p2, p3, test_pt);
    double w2 = edge_func(p3, p1, test_pt);
    double w3 = edge_func(p1, p2, test_pt);
    double inv_area = 1.0 / area;
    double b1 = w1 * inv_area;
    double b2 = w2 * inv_area;
    double b3 = w3 * inv_area;
    if (b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0) {
        return {b1, b2, b3};
    }
    return {-1.0, -1.0, -1.0};
}

template <typename Func, typename buff_T, typename arg_T>
void with_buff(Func &&job, const bound_box<int> &b_box, buff_T &&buffs,
               arg_T &&args) {
    std::forward<Func>(job)(b_box, std::forward<buff_T>(buffs),
                            std::forward<arg_T>(args));
}

template <typename T>
void engine_helper::rast_tri(const bound_box<int> &b_box,
                             ra_tri_buffs<T> &buffs, ra_tri_args<T> &args) {
    const int paren_len = args.paren_len; // in sub_pixels
    const int paren_wid = args.paren_wid; // in sub_pixels
    const int buf_w = buffs.buff.get_length_p();
    const int buf_h = buffs.buff.get_width_p();
    const int sqrt_samples = buffs.buff.get_sqrt_samples();
    const double a_ratio = (double)paren_len / paren_wid;
    const int left = std::max(b_box.min_x * sqrt_samples, 0);
    const int right =
        std::min(b_box.max_x * sqrt_samples, buf_w * sqrt_samples);
    const int top = std::max(b_box.min_y * sqrt_samples, 0);
    const int bot = std::min(b_box.max_y * sqrt_samples, buf_h * sqrt_samples);

    const double s_pix_to_world_x = 2.0 * a_ratio / paren_len;
    const double s_pix_to_world_y = 2.0 / paren_wid;
    const triangle &p_tri = args.p_tri;
    const double p1_z = p_tri.point1[2];
    const double p2_z = p_tri.point2[2];
    const double p3_z = p_tri.point3[2];
    const double near_plane = 0.1;
    if (p1_z < near_plane || p2_z < near_plane || p3_z < near_plane) {
        return;
    }
    const double inv_p1_z = 1.0 / p1_z;
    const double inv_p2_z = 1.0 / p2_z;
    const double inv_p3_z = 1.0 / p3_z;

    const int stride = buf_w * sqrt_samples;
    for (int l = top; l < bot; ++l) {
        double world_y = 1.0 - (l + 0.5) + s_pix_to_world_y;
        double world_x = (left + 0.5) * s_pix_to_world_x - a_ratio;
        int mem_idx = l * stride + left;
        for (int k = left; k < right;
             ++k, world_x += s_pix_to_world_x, ++mem_idx) {
            Eigen::Vector3d test{world_x, world_y, 0};
            Eigen::Vector3d bary =
                get_bary(p_tri.point1, p_tri.point2, p_tri.point3, test);
            if (bary[0] < 0.0 || bary[1] < 0.0 || bary[2] < 0.0) {
                continue;
            }
            double z_rep =
                bary[0] * inv_p1_z + bary[1] * inv_p2_z + bary[2] * inv_p3_z;
            double z_sub = 1.0 / z_rep;
            if (buffs.z_buff[mem_idx] > z_sub) {
                buffs.z_buff[mem_idx] = z_sub;
                buffs.buff[mem_idx] = args.val;
            }
        }
    }
}
