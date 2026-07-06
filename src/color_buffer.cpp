#include "buffer.h"

/**
 * Colors are rendered from [0,1] only internally are they actually represented
 * as 8 bit width channels, colors in this way is clever for later functions.
 * This idea was taken from RTIOW.
 */

color color_buffer::get_color(const int pix_x, const int pix_y,
                              const int s_pix_x, const int s_pix_y) const {
    if (pix_y < 0 || pix_y >= get_width_p() || pix_x < 0 ||
        pix_x >= get_length_p()) {
        return color{0, 0, 0};
    }
    int sqrt_samples = this->get_sqrt_samples();
    int loc_x = (pix_x * sqrt_samples) + s_pix_x;
    int loc_y = (pix_y * sqrt_samples) + s_pix_y;
    return this->get(loc_x, loc_y);
}

bool color_buffer::set_color(const int pix_x, const int pix_y,
                             const int s_pix_x, const int s_pix_y,
                             const color &color) {
    if (pix_y < 0 || pix_y >= get_width_p() || pix_x < 0 ||
        pix_x >= get_length_p()) {
        return false;
    }
    int sqrt_samples = this->get_sqrt_samples();
    int loc_x = (pix_x * sqrt_samples) + s_pix_x;
    int loc_y = (pix_y * sqrt_samples) + s_pix_y;
    this->set(loc_x, loc_y, color);
    return true;
}
