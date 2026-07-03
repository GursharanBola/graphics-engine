#include "mesh.h"
#include <Eigen/Dense>

void quad::build(vertex_buffer &v_buffer) {
    int i_index = v_buffer.size();
    const Eigen::Vector3d v1 = origin;
    const Eigen::Vector3d v2 = origin + u;
    const Eigen::Vector3d v3 = origin + v;
    const Eigen::Vector3d v4 = v3 + u;
    v_buffer.add(v1);
    v_buffer.add(v2);
    v_buffer.add(v3);
    v_buffer.add(v4);
    triangle t1{i_index, i_index + 3, i_index + 1};
    triangle t2{i_index, i_index + 2, i_index + 3};
    list_of_triangles.push_back(t1);
    list_of_triangles.push_back(t2);
}
Eigen::Vector3d quad::find_normal(const Eigen::Vector3d point) const {
    return u.cross(v).normalized();
}
