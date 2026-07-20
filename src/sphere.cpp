#include "buffer.h"
#include "mesh.h"
#include <Eigen/Dense>

void sphere::build(vertex_buffer &v_buffer) {
    int i_index = v_buffer.size();
    Eigen::Vector3d center = get_origin();
    Eigen::Vector3d start_p(0, radius, 0);
    double delta_phi = EIGEN_PI / num_samples;
    double delta_theta = (2 * EIGEN_PI) / num_samples;
    v_buffer.add(start_p + center);
    for (int i = 1; i < num_samples; i++) {
        double c_phi = i * delta_phi;
        for (int j = 0; j < num_samples; j++) {
            double c_theta = j * delta_theta;
            Eigen::Vector3d vertex =
                center +
                radius * Eigen::Vector3d{std::sin(c_phi) * std::cos(c_theta),
                                         std::cos(c_phi),
                                         std::sin(c_phi) * std::sin(c_theta)};
            v_buffer.add(vertex);
        }
    }
    v_buffer.add(-start_p + center);
    // add the top triangles
    for (int k = 0; k < num_samples; k++) {
        triangle t;
        t.point1 = i_index;
        t.point2 = i_index + 1 + k;
        t.point3 = i_index + 1 + ((k + 1) % num_samples);
        list_of_triangles.push_back(t);
    }
    // middle layers of the sphere
    for (int k = 1; k < num_samples - 1; k++) {
        int prev_row_start = i_index + 1 + (k - 1) * num_samples;
        int curr_row_start = i_index + 1 + k * num_samples;
        for (int l = 0; l < num_samples; l++) {
            int next_l = (l + 1) % num_samples;
            int p1 = prev_row_start + l;
            int p2 = prev_row_start + next_l;
            int p3 = curr_row_start + l;
            int p4 = curr_row_start + next_l;
            triangle t1, t2;
            t1.point1 = p1;
            t1.point2 = p3;
            t1.point3 = p4;
            t2.point1 = p1;
            t2.point2 = p4;
            t2.point3 = p2;
            list_of_triangles.push_back(t1);
            list_of_triangles.push_back(t2);
        }
    }
    // add bottom layers of the sphere
    int bottom_pole = v_buffer.size() - 1;
    int last_row_start = i_index + 1 + (num_samples - 2) * num_samples;
    for (int k = 0; k < num_samples; k++) {
        triangle t;
        t.point1 = bottom_pole;
        t.point2 = last_row_start + ((k + 1) % num_samples);
        t.point3 = last_row_start + k;
        list_of_triangles.push_back(t);
    }
}

Eigen::Vector3d sphere::find_normal(const Eigen::Vector3d &point) const {
    return (point - center).normalized();
}
