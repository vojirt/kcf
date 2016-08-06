#include <stdlib.h>

#include "kcf.h"

#define VOT_RECTANGLE
#include "trax.h"
#include "vot_trax.h"

int main()
{
    VOT vot; // Initialize the communcation

    VOTRegion region = vot.region(); // Get region and first frame
    string path = vot.frame();

    cv::Mat image = cv::imread(path);

    KCF_Tracker tracker;

    cv::Rect init_rect;
    region.get(init_rect);
    tracker.init(image, init_rect);

    BBox_c bb;
    while (true) {
        path = vot.frame(); // Get the next frame
        if (path.empty()) break; // Are we done?

        image = cv::imread(path);
        tracker.track(image);
        bb = tracker.getBBox();

        region.set(bb.get_rect());
        vot.report(region); // Report the position of the tracker
    }

    return EXIT_SUCCESS;
}