#include "engine.h"
#include "buffer.h"
#include "engine_helper.h"
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

void engine::fill_z_s(const projector &projector,
                      const std::vector<std::unique_ptr<mesh>> &meshes,
                      const vertex_buffer &v_buff, z_buffer &z_buff,
                      seen_buffer &s_buff) const {
    const Eigen::Vector3d cam_u = projector.get_u();
    const Eigen::Vector3d cam_v = projector.get_v();
    const Eigen::Vector3d cam_w = projector.get_w();
    const Eigen::Vector3d cam_o = projector.get_o();
    const int length = z_buff.get_length();
    const int width = z_buff.get_width();
    const int length_p = z_buff.get_width_p();
    const int width_p = z_buff.get_width_p();
    const int sqrt_samples = z_buff.get_sqrt_samples();
    const double a_ratio = static_cast<double>(length) / width;
    if (length != s_buff.get_length() || width != s_buff.get_width() ||
        s_buff.get_sqrt_samples() != sqrt_samples) {
        throw std::runtime_error(
            "s_buff must have same width, length, and sqrt_samples");
    }
    constexpr int NUM_THREADS = 4;
    engine_helper::ra_tri_buffs<tri_ref> buffs{s_buff, z_buff};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    const int t_height = (width + NUM_THREADS - 1) / NUM_THREADS;
    for (int top = 0; top < width; top += t_height) {
        int next_row = top + t_height;
        int bot = next_row > width_p ? width_p : next_row;
        bound_box thread_box = bound_box<int>{0, length_p, top, bot};
        threads.emplace_back([&, thread_box]() {
            for (const auto &mesh : meshes) {
                int tri_index = 0;
                for (const triangle &tri : mesh->list_of_triangles) {
                    raw_tri new_tri =
                        raw_tri{v_buff.get(tri.point1), v_buff.get(tri.point2),
                                v_buff.get(tri.point3)};
                    raw_tri p_tri = engine_helper::proj_tri(
                        new_tri, cam_u, cam_v, cam_w, cam_o);
                    bound_box<int> p_b_box = engine_helper::create_box(
                        p_tri.p1, p_tri.p2, p_tri.p3, a_ratio, length, width);
                    int left = std::max(p_b_box.min_x, thread_box.min_x);
                    int right = std::min(p_b_box.max_x, thread_box.max_x);
                    int top = std::max(p_b_box.min_y, thread_box.min_y);
                    int bot = std::min(p_b_box.max_y, thread_box.max_y);
                    if (left > right || bot > top) {
                        continue;
                    }
                    bound_box t_b_box = bound_box<int>{left, right, top, bot};
                    tri_ref current_val = tri_ref{mesh->get_id(), tri_index};
                    engine_helper::ra_tri_args<tri_ref> args{
                        length, width, p_tri, current_val};
                    engine_helper::with_buff(engine_helper::rast_tri_fn{},
                                             t_b_box, buffs, args);
                    tri_index++;
                }
            }
        });
    }
}
