#include "mesh.h"
#include <Eigen/Dense>

void quad::build(ds::e_cache_map<triangle> &list_of_tri) {
    const Eigen::Vector3d bot_le = origin;
    const Eigen::Vector3d top_le = origin + u;
    const Eigen::Vector3d bot_ri = origin + v;
    const Eigen::Vector3d top_ri = bot_ri + u;
    triangle &tri1 = list_of_tri.claim_next_slot(mesh_id);
    triangle &tri2 = list_of_tri.claim_next_slot(mesh_id);
    tri1 = triangle{top_le, bot_le, bot_ri};
    tri2 = triangle{top_le, bot_ri, top_ri};

    double half_width = std::abs(u.x()) + std::abs(v.x());
    double half_height = std::abs(u.y()) + std::abs(v.y());
    double max_xy = std::max(half_width, half_height);
    const double neg_x = origin.x() - max_xy;
    const double pos_x = origin.x() + max_xy;
    const double neg_y = origin.y() - max_xy;
    const double pos_y = origin.y() + max_xy;
    const double neg_z = origin.z() - max_xy;
    const double pos_z = origin.z() + max_xy;
    b_cube = {Eigen::Vector3d{pos_x, neg_y, pos_z},
              Eigen::Vector3d{neg_x, pos_y, neg_z}};
}

Eigen::Vector3d quad::find_normal(const Eigen::Vector3d &point) const {
    return norm;
}
