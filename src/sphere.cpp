#include "mesh.h"
#include <Eigen/Dense>

void sphere::build(ds::e_cache_map<triangle> &list_of_tri) {
    Eigen::Vector3d center = get_origin();
    double delta_phi = EIGEN_PI / num_samples;
    double delta_theta = (2.0 * EIGEN_PI) / num_samples;
    Eigen::Vector3d top(0.0, radius, 0.0);
    double ring_y = radius * std::cos(delta_phi);
    double ring_rad_y = radius * std::sin(delta_phi);
    Eigen::Vector3d bot_prev(ring_rad_y * 1.0, ring_y, ring_rad_y * 0.0);
    for (int i = 1; i <= num_samples; ++i) {
        double c_theta = delta_theta * i;
        Eigen::Vector3d bot_curr(ring_rad_y * std::cos(c_theta), ring_y,
                                 ring_rad_y * std::sin(c_theta));
        triangle &tri = list_of_tri.claim_next_slot(mesh_id);
        tri = triangle{top + center, bot_prev + center, bot_curr + center};
        bot_prev = bot_curr;
    }
    for (int i = 1; i < num_samples - 1; ++i) {
        const double top_y = radius * std::cos(i * delta_phi);
        const double top_rad = radius * std::sin(i * delta_phi);
        const double bot_y = radius * std::cos((i + 1) * delta_phi);
        const double bot_rad = radius * std::sin((i + 1) * delta_phi);
        Eigen::Vector3d top_le = center + Eigen::Vector3d{top_rad, top_y, 0.0};
        Eigen::Vector3d bot_le = center + Eigen::Vector3d{bot_rad, bot_y, 0.0};
        for (int j = 1; j <= num_samples; ++j) {
            double c_theta_cos = std::cos(j * delta_theta);
            double c_theta_sin = std::sin(j * delta_theta);
            Eigen::Vector3d top_ri =
                center + Eigen::Vector3d{top_rad * c_theta_cos, top_y,
                                         top_rad * c_theta_sin};
            Eigen::Vector3d bot_ri =
                center + Eigen::Vector3d{bot_rad * c_theta_cos, bot_y,
                                         bot_rad * c_theta_sin};
            triangle &tri1 = list_of_tri.claim_next_slot(mesh_id);
            triangle &tri2 = list_of_tri.claim_next_slot(mesh_id);
            tri1 = triangle{top_le, bot_le, bot_ri};
            tri2 = triangle{top_le, bot_ri, top_ri};
            top_le = top_ri;
            bot_le = bot_ri;
        }
    }
    Eigen::Vector3d bot(0.0, -radius, 0.0);
    double l_ring_y = radius * std::cos(delta_phi * (num_samples - 1));
    double l_ring_rad_y = radius * std::sin(delta_phi * (num_samples - 1));
    Eigen::Vector3d top_prev(l_ring_rad_y * 1.0, l_ring_y, l_ring_rad_y * 0.0);
    for (int i = 1; i <= num_samples; ++i) {
        double c_theta = delta_theta * i;
        Eigen::Vector3d bot_curr(l_ring_rad_y * std::cos(c_theta), l_ring_y,
                                 l_ring_rad_y * std::sin(c_theta));
        triangle &tri = list_of_tri.claim_next_slot(mesh_id);
        tri = triangle{top_prev + center, bot_curr + center, bot + center};
        bot_prev = bot_curr;
    }
}

Eigen::Vector3d sphere::find_normal(const Eigen::Vector3d &point) const {
    return (point - origin).normalized();
}
