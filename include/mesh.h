#ifndef MESH_H
#define MESH_H

#include "buffer.h"
#include "material.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

typedef struct triangle {
    // note all of these are v_id.
    int point1;
    int point2;
    int point3;
} triangle;

// mesh interface, NOTE that the winding type is CCW for triangles
// there are reduant checks for winding order however...
class mesh {
  public:
    std::vector<triangle> list_of_triangles;

    mesh(const int mesh_id, const color mesh_color,
         const std::shared_ptr<material> mat = nullptr,
         const int num_samples = 1)
        : mat(mat), mesh_id(mesh_id), num_samples(num_samples),
          mesh_color(mesh_color) {}

    virtual ~mesh() = default;
    virtual void build(vertex_buffer &v_buffer) = 0;
    virtual Eigen::Vector3d find_normal(const Eigen::Vector3d &point) const = 0;

    int get_id() const { return mesh_id; }
    int get_samples() const { return num_samples; }
    color get_color() const { return mesh_color; }

  protected:
    std::shared_ptr<material> mat;
    int mesh_id;
    int num_samples;
    color mesh_color;
    friend class engine;
};

// much like RTIOW, main test_object
class sphere : public mesh {
  public:
    sphere(const int mesh_id, const Eigen::Vector3d &center,
           const double radius, const color &mesh_color,
           const std::shared_ptr<material> &mat, const int num_samples)
        : mesh(mesh_id, mesh_color, mat, num_samples), center(center),
          radius(radius) {}

    virtual void build(vertex_buffer &v_buffer) override final;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d &point) const override final;

    Eigen::Vector3d get_center() const { return center; }
    double get_radius() const { return radius; }

  private:
    Eigen::Vector3d center;
    double radius;
};

// a quad useful for backgrounds
class quad : public mesh {
  public:
    quad(const int mesh_id, const Eigen::Vector3d &origin,
         const Eigen::Vector3d &u, const Eigen::Vector3d &v,
         const color &mesh_color, const std::shared_ptr<material> &mat)
        : mesh(mesh_id, mesh_color, mat), u(u), v(v), origin(origin) {}
    virtual void build(vertex_buffer &v_buffer) override final;
    virtual Eigen::Vector3d
    find_normal(const Eigen::Vector3d &point) const override final;

    Eigen::Vector3d get_u() const { return u; }
    Eigen::Vector3d get_v() const { return v; }

  private:
    Eigen::Vector3d u;
    Eigen::Vector3d v;
    Eigen::Vector3d origin;
};

#endif
