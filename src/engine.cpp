#include "engine.h"
#include "buffer.h"
#include <memory>
#include <stdexcept>
#include <vector>

// TODO: redo this function with refactored code
void engine::fill_v_s(const projector &projector,
                      const std::vector<std::unique_ptr<mesh>> &meshes,
                      const vertex_buffer &v_buff, z_buffer &z_buff,
                      seen_buffer &s_buff) const {

    Eigen::Vector3d cam_u = projector.get_u();
    Eigen::Vector3d cam_v = projector.get_v();
    Eigen::Vector3d cam_w = projector.get_w();
    Eigen::Vector3d cam_o = projector.get_o();

    int length = z_buff.get_length();
    int width = z_buff.get_width();
    int sqrt_samples = z_buff.get_sqrt_samples();
    double a_ratio = static_cast<double>(length) / width;

    if (length != s_buff.get_length() || width != s_buff.get_width() ||
        s_buff.get_sqrt_samples() != sqrt_samples) {
        std::runtime_error(
            "s_buff must have same width, length, and sqrt_samples");
    }
    for (const auto &mesh : meshes) {
        int tri_index = 0;
        for (triangle tri : mesh->list_of_triangles) {

            std::array<Eigen::Vector3d, 3> p_tri =
                proj_tri(tri, cam_u, cam_v, cam_w, cam_o, v_buff);

            // create a bound_box at the pixel level
            bound_box<int> p_b_box = create_box(p_tri[0], p_tri[1], p_tri[2],
                                                a_ratio, length, width);

            // create a bound_box at the sub_pixel level
            bound_box s_b_box = bound_box<int>{
                p_b_box.min_x * sqrt_samples, p_b_box.max_x * sqrt_samples,
                p_b_box.min_y * sqrt_samples, p_b_box.max_y * sqrt_samples};

            int box_length = s_b_box.max_x - s_b_box.min_x;
            int box_width = s_b_box.max_y - s_b_box.min_y;

            for (int i = 0; i < box_length; ++i) {
                for (int j = 0; j < box_width; ++j) {
                    tri_ref tri = tri_ref{mesh->get_id(), tri_index};

                    on_buff(rast_s_pix_fn{}
                }
            }
            tri_index++;
        }
    }
};
