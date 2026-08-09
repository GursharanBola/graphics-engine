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
    Eigen::Vector3d box_min =
        bot_le.cwiseMin(top_le).cwiseMin(bot_ri).cwiseMin(top_ri);
    Eigen::Vector3d box_max =
        bot_le.cwiseMax(top_le).cwiseMax(bot_ri).cwiseMax(top_ri);
    const double eps = 1e-4;
    for (int i = 0; i < 3; ++i) {
        if (box_max[i] - box_min[i] < eps) {
            box_max[i] += eps;
            box_min[i] -= eps;
        }
    }
    b_cube = {box_min, box_max};
}

Eigen::Vector3d quad::find_normal(const Eigen::Vector3d &point) const {
    return norm;
}
