#include "buffer.h"
#include <Eigen/Dense>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * Colors are rendered from [0,1] only internally are they actually represented
 * as 8 bit width channels, colors in this way is clever for later functions.
 * This idea was taken from RTIOW.
 */

int image_buffer::draw_png(std::string filename, int width, int height,
                           int channels, void *data, int stride) {
    int png_truth =
        stbi_write_png(filename.c_str(), width, height, channels, data, stride);
    return png_truth;
}

Eigen::Vector3d image_buffer::get_color(const int i, const int j) const {
    if (i < 0 || i >= (get_length() / 3) || j < 0 || j >= get_width()) {
        return Eigen::Vector3d(0, 0, 0);
    }
    double r = get(i * 3 + 0, j);
    double g = get(i * 3 + 1, j);
    double b = get(i * 3 + 2, j);
    return Eigen::Vector3d(r / 255.0, g / 255.0, b / 255.0);
}

bool image_buffer::set_color(const int i, const int j, Eigen::Vector3d color) {
    if (i < 0 || i >= (get_length() / 3) || j < 0 || j >= get_width()) {
        return false; // We failed
    }
    double r = std::clamp(color(0), 0.0, 1.0);
    double g = std::clamp(color(1), 0.0, 1.0);
    double b = std::clamp(color(2), 0.0, 1.0);
    set(i * 3 + 0, j, static_cast<int>(255.999 * r));
    set(i * 3 + 1, j, static_cast<int>(255.999 * g));
    set(i * 3 + 2, j, static_cast<int>(255.999 * b));
    return true;
}
