

#include <cstdio>
#include <arrayfire.h>
#include <cstdlib>
#include <iostream>


using namespace af;
using namespace std;


static void harris_demo(bool console)
{
    af::Window wnd("Harris Corner Detector");

    // Load image
    af::array img_color;
    img_color = loadImage("src.jpg", true);

    // Convert the image from RGB to gray-scale
    af::array img = colorSpace(img_color, AF_GRAY, AF_RGB);
    img_color /= 255.f;

    // Calculate image gradients
    af::array ix, iy;
    grad(ix, iy, img);

    // Compute second-order derivatives
    af::array ixx = ix * ix;
    af::array ixy = ix * iy;
    af::array iyy = iy * iy;

    af::array gauss_filt = gaussianKernel(5, 5, 1.0, 1.0);

    ixx = convolve(ixx, gauss_filt);
    ixy = convolve(ixy, gauss_filt);
    iyy = convolve(iyy, gauss_filt);

    // Calculate trace
    af::array itr = ixx + iyy;
    // Calculate determinant
    af::array idet = ixx * iyy - ixy * ixy;

    // Calculate Harris response
    af::array response = idet - 0.04f * (itr * itr);

    af::array mask = constant(1,3,3);
    af::array max_resp = dilate(response, mask);

    // Discard responses that are not greater than threshold
    af::array corners = response > 1e5f;
    corners = corners * response;

    corners = (corners == max_resp) * corners;

    // Gets host pointer to response data
    float* h_corners = corners.host<float>();

    unsigned good_corners = 0;

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 3;
    for (int y = draw_len; y < img_color.dims(0) - draw_len; y++) {
        for (int x = draw_len; x < img_color.dims(1) - draw_len; x++) {
            // Only draws crosshair if is a corner
            if (h_corners[x * corners.dims(0) + y] > 1e5f) {
                img_color(y, seq(x-draw_len, x+draw_len), 0) = 0.f;
                img_color(y, seq(x-draw_len, x+draw_len), 1) = 1.f;
                img_color(y, seq(x-draw_len, x+draw_len), 2) = 0.f;

                img_color(seq(y-draw_len, y+draw_len), x, 0) = 0.f;
                img_color(seq(y-draw_len, y+draw_len), x, 1) = 1.f;
                img_color(seq(y-draw_len, y+draw_len), x, 2) = 0.f;

                good_corners++;
            }
        }
    }

    printf("Corners found: %u\n", good_corners);

    if (!console) {
        // Previews color image with green crosshairs
        while(!wnd.close())
            wnd.image(img_color);
    } else {
        // Find corner indexes in the image as 1D indexes
        af::array idx = where(corners);

        // Calculate 2D corner indexes
        af::array corners_x = idx / corners.dims()[0];
        af::array corners_y = idx % corners.dims()[0];

        const int good_corners = corners_x.dims()[0];
        std::cout << "Corners found: " << good_corners << std::endl << std::endl;

        af_print(corners_x);
        af_print(corners_y);
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();
        std::cout << "ArrayFire Harris Corner Detector" << std::endl << std::endl;
        harris_demo(console);

    } catch (af::exception& ae) {
        std::cerr << ae.what() << std::endl;
        throw;
    }

    return 0;
}
