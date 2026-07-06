#ifndef ENGINE_H
#define ENGINE_H

#include "buffer.h"
#include "projector.h"
#include "scene.h"
#include <memory>
#include <vector>
// NOTE: add file parsing and video rendering using interpolation

/**
 * The program for optimal performance will store all lights in their own
 * buffers and then later use these for a shading pass. This is optimal
 * for performance since it optimizes the cache blocking. To avoid for the
 * camera's buffers it will complete a v and s buffer completion, as well as
 * a coloring pass for phong shading. The shading pass is then done using
 * tiling.
 *
 * coloring pass is done using the color_buffer
 *
 * tiling only happens on the shading pass
 */

class engine {
  public:
    engine(scene &scene, camera &camera) : scene(scene), camera(camera) {};
    void fill_z_s(const projector &projector,
                  const std::vector<std::unique_ptr<mesh>> &meshes,
                  const vertex_buffer &v_buff, z_buffer &z_buff,
                  seen_buffer &s_buff) const;
    void shade();
    // TODO: determine additional functions need to be added to the engine
  private:
    scene &scene;
    camera &camera;
};
#endif
