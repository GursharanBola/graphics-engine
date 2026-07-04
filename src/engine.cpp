#include "engine.h"
#include "buffer.h"
#include "engine_helper.h"
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

void engine::fill_v_s(const projector &projector,
                      const std::vector<std::unique_ptr<mesh>> &meshes,
                      const vertex_buffer &v_buff, z_buffer &z_buff,
                      seen_buffer &s_buff) const {
    constexpr int num_threads = 4;
    std::vector<std::thread> threads(num_threads);
    const Eigen::Vector3d cam_u = projector.get_u();
    const Eigen::Vector3d cam_v = projector.get_v();
    const Eigen::Vector3d cam_w = projector.get_w();
    const Eigen::Vector3d cam_o = projector.get_o();
    const int length = z_buff.get_length();
    const int width = z_buff.get_width();
    const int sqrt_samples = z_buff.get_sqrt_samples();
    const double a_ratio = static_cast<double>(length) / width;
    if (length != s_buff.get_length() || width != s_buff.get_width() ||
        s_buff.get_sqrt_samples() != sqrt_samples) {
        throw std::runtime_error(
            "s_buff must have same width, length, and sqrt_samples");
    }
    engine_helper::ra_tri_buffs<tri_ref> buffs{s_buff, z_buff};
    for (const auto &mesh : meshes) {
        int tri_index = 0;
        for (const triangle &tri : mesh->list_of_triangles) {
            raw_tri new_tri =
                raw_tri{v_buff.get(tri.point1), v_buff.get(tri.point2),
                        v_buff.get(tri.point3)};
            raw_tri p_tri =
                engine_helper::proj_tri(new_tri, cam_u, cam_v, cam_w, cam_o);
            bound_box<int> p_b_box = engine_helper::create_box(
                p_tri.p1, p_tri.p2, p_tri.p3, a_ratio, length, width);
            tri_ref current_val = tri_ref{mesh->get_id(), tri_index};
            engine_helper::ra_tri_args<tri_ref> args{length, width, p_tri,
                                                     current_val};
            const int b_box_h = p_b_box.max_y - p_b_box.min_y;
            const int left = p_b_box.min_x;
            const int right = p_b_box.max_x;
            const int bot = p_b_box.max_y;
            const int t_num_rows = (b_box_h + num_threads - 1) / num_threads;
            for (int top = 0; top < b_box_h; top += t_num_rows) {
                const int next_row = top + t_num_rows;
                const int right = next_row > bot ? bot : next_row;
                bound_box<int> t_p_b_box{left, right, top, right};
                std::thread job([t_p_b_box, &buffs, &args]() {
                    engine_helper::with_buff(engine_helper::rast_tri_fn{},
                                             t_p_b_box, buffs, args);
                });
                threads.push_back(job);
            }
            for (auto &t : threads) {
                t.join();
            }
            tri_index++;
        }
    }
}
