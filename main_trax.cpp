#include <stdlib.h>

#include <trax/opencv.hpp>
#include <memory>
#include "kcf.h"

int main()
{
    trax::Image img;
    trax::Region reg;

    std::unique_ptr<KCF_Tracker> tracker;
    cv::Mat image;
    cv::Rect rectangle;

    trax::Server handle(trax::Configuration(TRAX_IMAGE_PATH | TRAX_IMAGE_MEMORY | TRAX_IMAGE_BUFFER, TRAX_REGION_RECTANGLE), trax_no_log);

    std::cout << handle.configuration().format_region << " " << TRAX_SUPPORTS(handle.configuration().format_region, TRAX_REGION_POLYGON) << std::endl;

    while (true) {
        trax::Properties prop;
        int tr = handle.wait(img, reg, prop);

        if (tr == TRAX_INITIALIZE) {
            //create new tracker
            tracker.reset(new KCF_Tracker());

            rectangle = trax::region_to_rect(reg);
            image = trax::image_to_mat(img);

            // Dynamically configure tracker
            tracker->m_use_scale = prop.get("use_scale", true);
            tracker->m_use_color = prop.get("use_color", true);
            tracker->m_use_subpixel_localization = prop.get("use_subpixel_localization", true);
            tracker->m_use_subgrid_scale = prop.get("use_subgrid_scale", true);
            tracker->m_use_multithreading = prop.get("use_multithreading", true);
            tracker->m_use_cnfeat = prop.get("use_cnfeat", true);
            tracker->m_use_linearkernel = prop.get("use_linearkernel", false);

            tracker->init(image, rectangle);

        } else if (tr == TRAX_FRAME) {

            image = trax::image_to_mat(img);
            if (tracker) {
                tracker->track(image);
                BBox_c bb = tracker->getBBox();
                rectangle = bb.get_rect();
            }
        } else {
            break;
        }

        trax::Region status = trax::rect_to_region(rectangle);
        handle.reply(status, trax::Properties());

    }

    return EXIT_SUCCESS;
}
