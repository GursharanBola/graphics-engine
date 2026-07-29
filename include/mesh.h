#ifndef MESH_H
#define MESH_H

#include "buffer.h"
#include "ds.h"
#include <Eigen/Dense>
#include <variant>

// the program expects all triangles to have a winding order of CCW, this is
// VERY important allows for backface culling

// since bounding boxes are axis aligned the program can use what is called
// the slab method, only the bot left and top right corners are needed
// standard: [min, max]

class mesh {
  public:
    mesh(const int mesh_id,
         const Eigen::Vector3d &origin = Eigen::Vector3d::Zero(),
         const int num_samples = 1)
        : mesh_id(mesh_id), num_samples(num_samples), origin(origin) {}

    Eigen::Vector3d get_origin() const { return origin; }
    int get_id() const { return mesh_id; }
    int get_samples() const { return num_samples; }

  protected:
    int mesh_id;
    int num_samples;
    Eigen::Vector3d origin;
    friend class engine;
};

// much like RTIOW, main test object
class sphere : public mesh {
  public:
    sphere(const int mesh_id, const Eigen::Vector3d &center,
           const double radius, const int num_samples,
           ds::e_cache_map<triangle> &list_of_tri)
        : mesh(mesh_id, center, num_samples), radius(radius) {
        build(list_of_tri);
    }

    void build(ds::e_cache_map<triangle> &list_of_tri);
    Eigen::Vector3d find_normal(const Eigen::Vector3d &point) const;
    double get_radius() const { return radius; }

  private:
    friend class engine;
    std::array<Eigen::Vector3d, 2> b_cube;
    double radius;
};

// quads are useful for making backgrounds, they must be orthoginal simplifies
// the math by a lot
class quad : public mesh {
  public:
    quad(const int mesh_id, const Eigen::Vector3d &origin,
         const Eigen::Vector3d &u, const Eigen::Vector3d &v,
         ds::e_cache_map<triangle> &list_of_tri)
        : mesh(mesh_id, origin), u(u), v(v) {
        if (std::abs(u.dot(v)) > 1e-6) {
            throw std::invalid_argument("Vectors u and v must be orthogonal.");
        }
        build(list_of_tri);
        u_norm = u.norm();
        v_norm = v.norm();
        norm = u.cross(v).normalized();
    }

    Eigen::Vector3d find_normal(const Eigen::Vector3d &point) const;
    void build(ds::e_cache_map<triangle> &list_of_tri);
    Eigen::Vector3d get_u() const { return u; }
    Eigen::Vector3d get_v() const { return v; }

  private:
    friend class engine;
    Eigen::Vector3d u;
    Eigen::Vector3d v;
    Eigen::Vector3d norm;
    std::array<Eigen::Vector3d, 2> b_cube;
    double u_norm;
    double v_norm;
};

using shape = std::variant<sphere, quad>;

inline Eigen::Vector3d find_normal_at(const shape &s,
                                      const Eigen::Vector3d &point) {
    if (std::holds_alternative<sphere>(s)) {
        return std::get<sphere>(s).find_normal(point);
    } else {
        return std::get<quad>(s).find_normal(point);
    }
}

inline Eigen::Vector3d get_origin_of(const shape &s) {
    if (std::holds_alternative<sphere>(s)) {
        return std::get<sphere>(s).get_origin();
    } else {
        return std::get<quad>(s).get_origin();
    }
}

inline int get_id(const shape &s) {
    if (std::holds_alternative<sphere>(s)) {
        return std::get<sphere>(s).get_id();
    } else {
        return std::get<quad>(s).get_id();
    }
}

#endif
