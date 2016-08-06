/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4; tab-width: 4 -*- */
/*
 * This header file contains C functions that can be used to quickly integrate
 * VOT challenge support into your C or C++ tracker.
 *
 * Copyright (c) 2015, VOT Committee
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 

 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution. 

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies, 
 * either expressed or implied, of the FreeBSD Project.
 */

#ifndef _VOT_TOOLKIT_H
#define _VOT_TOOLKIT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#define VOT_READ_BUFFER 2024

// Newer compilers support interactive checks for headers, otherwise we have to enable TraX support manually
#ifdef __has_include
#  if __has_include("trax.h")
#    include <trax.h>
#    define VOT_TRAX
#  endif
#else
#  ifdef TRAX
#    include <trax.h>
#    define VOT_TRAX
#  endif
#endif

// Alternative getline implementation for Windows compatibility
size_t _getline(char **lineptr, size_t *n, FILE *stream) {

    char *bufptr = NULL;
    char *p = bufptr;
    size_t size;
    int c;

    if (lineptr == NULL) {
    	return -1;
    }

    if (stream == NULL) {
    	return -1;
    }

    if (n == NULL) {
    	return -1;
    }

    bufptr = *lineptr;
    size = *n;

    c = fgetc(stream);
    if (c == EOF) {
    	return -1;
    }
    if (bufptr == NULL) {
    	bufptr = (char *) malloc(128);
    	if (bufptr == NULL) {
    		return -1;
    	}
    	size = 128;
    }
    p = bufptr;
    while(c != EOF) {
    	if ((p - bufptr) > (size - 1)) {
    		size = size + 128;
    		bufptr = (char *) realloc(bufptr, size);
    		if (bufptr == NULL) {
    			return -1;
    		}
    	}
    	*p++ = c;
    	if (c == '\n') {
    		break;
    	}
    	c = fgetc(stream);
    }

    *p++ = '\0';
    *lineptr = bufptr;
    *n = size;

    return p - bufptr - 1;
}

// Define VOT_OPENCV after including OpenCV core header to enable better OpenCV support
#ifdef __OPENCV_CORE_HPP__
#  define VOT_OPENCV
#endif

#ifndef VOT_RECTANGLE
#define VOT_POLYGON
#endif

#ifdef VOT_POLYGON
typedef struct vot_region {
    float* x;
    float* y;
    int count;
} vot_region;

void vot_region_release(vot_region** region) {
    if (!(*region)) return;

    if ((*region)->x) {
        free((*region)->x);
        (*region)->x = NULL;
    }
    if ((*region)->y) {
        free((*region)->y);
        (*region)->y = NULL;
    }

    free(*region);

    *region = NULL;
}

vot_region* vot_region_create(int n) {
    vot_region* region = (vot_region*) malloc(sizeof(vot_region));
    region->x = (float *) malloc(sizeof(float) * n);
    region->y = (float *) malloc(sizeof(float) * n);
    memset(region->x, 0, sizeof(float) * n);
    memset(region->y, 0, sizeof(float) * n);
    region->count = n;
    return region;
}

vot_region* vot_region_copy(const vot_region* region) {
    vot_region* copy = vot_region_create(region->count);
    int i;
    for (i = 0; i < region->count; i++) {
        copy->x[i] = region->x[i];
        copy->y[i] = region->y[i];
    }
    return copy;
}

#else
typedef struct vot_region {
    float x;
    float y;
    float width;
    float height;
} vot_region;

void vot_region_release(vot_region** region) {

    if (!(*region)) return;

    free(*region);

    *region = NULL;

}

vot_region* vot_region_create() {
    vot_region* region = (vot_region*) malloc(sizeof(vot_region));
    region->x = 0;
    region->y = 0;
    region->width = 0;
    region->height = 0;
    return region;
}

vot_region* vot_region_copy(const vot_region* region) {
    vot_region* copy = vot_region_create();
    copy->x = region->x;
    copy->y = region->y;
    copy->width = region->width;
    copy->height = region->height;
    return copy;
}


#endif

vot_region* _parse_region(char* buffer) {

    int i;
    float* numbers = (float*) malloc(sizeof(float) * (strlen(buffer) / 2));
	char* pch = strtok(buffer, ",");

    for (i = 0; ; i++) {
        if (!pch) break;
        numbers[i] = (float) atof(pch); 
		pch = strtok(NULL, ",");
    }

    vot_region* region;

#ifdef VOT_POLYGON
    {
        // Check if region is actually a rectangle and convert it
        if (i == 4) {

            region = vot_region_create(4);

			region->count = 4;

			region->x[0] = numbers[0];
			region->x[1] = numbers[0] + numbers[2];
			region->x[2] = numbers[0] + numbers[2];
			region->x[3] = numbers[0];

			region->y[0] = numbers[1];
			region->y[1] = numbers[1];
			region->y[2] = numbers[1] + numbers[3];
			region->y[3] = numbers[1] + numbers[3];


        } else {
            int count = i / 2;
            assert(count >= 3);

            region = vot_region_create(count);

            for (i = 0; i < count; i++) {
                region->x[i] = numbers[i*2];
                region->y[i] = numbers[i*2+1];
            }

            region->count = count;
        }
    }
#else
    {
        assert(i > 3);

        region = vot_region_create();

        // Check if the input region is actually a polygon and convert it
        if (i > 6) {
            int j;
		    float top = FLT_MAX;
		    float bottom = FLT_MIN;
		    float left = FLT_MAX;
		    float right = FLT_MIN;

		    for (j = 0; j < i / 2; j++) {
			    top = MIN(top, numbers[j * 2 + 1]);
			    bottom = MAX(bottom, numbers[j * 2 + 1]);
			    left = MIN(left, numbers[j * 2]);
			    right = MAX(right, numbers[j * 2]);
		    }

	        region->x = left;
	        region->y = top;
	        region->width = right - left;
	        region->height = bottom - top;

        } else {
            region = vot_region_create();
            region->x = numbers[0];
            region->y = numbers[1];
            region->width = numbers[2];
            region->height = numbers[3];
        }
    }
#endif

    free(numbers);

    return region;
}

#ifdef __cplusplus

#include <string>
#include <fstream>
#include <iostream>

using namespace std;

class VOT;

class VOTRegion {
    friend class VOT;
public:

    ~VOTRegion() {
        vot_region_release(&_region);
    }

    VOTRegion(const vot_region* region) {
        _region = vot_region_copy(region);
    }

#ifdef VOT_POLYGON
    VOTRegion(int count) {
        _region = vot_region_create(count);
    }

    void set(int i, float x, float y) { assert(i >= 0 && i < _region->count); _region->x[i] = x; _region->y[i] = y; }
    float get_x(int i) const { assert(i >= 0 && i < _region->count); return _region->x[i]; } 
    float get_y(int i) const { assert(i >= 0 && i < _region->count); return _region->y[i]; }
    int count() const { return _region->count; }

#else

    VOTRegion() {
        _region = vot_region_create();
    }

    float get_x() const { return _region->x; }
    float get_y() const { return _region->y; }
    float get_width() const { return _region->width; }
    float get_height() const { return _region->height; }

    float set_x(float x) { return _region->x = x; }
    float set_y(float y) { return _region->y = y; }
    float set_width(float width) { return _region->width = width; }
    float set_height(float height) { return _region->height = height; }

#endif

    VOTRegion& operator= (const VOTRegion &source) {

        if (this == &source)
            return *this;

#ifdef VOT_POLYGON

        if (this->_region->count != source.count()) {
            vot_region_release(&(this->_region));
            this->_region = vot_region_create(source.count());
        }

        for (int i = 0; i < source.count(); i++) {
            set(i, source.get_x(i), source.get_y(i));
        }

#else

        set_x(source.get_x());
        set_y(source.get_y());
        set_width(source.get_width());
        set_height(source.get_height());

#endif

        return *this;
    }

#ifdef VOT_OPENCV

    VOTRegion(const cv::Rect& rectangle) {
#ifdef VOT_POLYGON
        _region = vot_region_create(4);
#else
        _region = vot_region_create();
#endif
        set(rectangle);
    }

    void set(const cv::Rect& rectangle) {

#ifdef VOT_POLYGON

        if (_region->count != 4) {
            vot_region_release(&(this->_region));
            _region = vot_region_create(4);
        }

	    set(0, rectangle.x, rectangle.y);
	    set(1, rectangle.x + rectangle.width, rectangle.y);
	    set(2, rectangle.x + rectangle.width, rectangle.y + rectangle.height);
	    set(3, rectangle.x, rectangle.y + rectangle.height);

#else

        set_x(rectangle.x);
        set_y(rectangle.y);
        set_width(rectangle.width);
        set_height(rectangle.height);

#endif

    }

    void get(cv::Rect& rectangle) const {

#ifdef VOT_POLYGON

	    float top = FLT_MAX;
	    float bottom = FLT_MIN;
	    float left = FLT_MAX;
	    float right = FLT_MIN;

	    for (int j = 0; j < _region->count; j++) {
		    top = MIN(top, _region->y[j]);
		    bottom = MAX(bottom, _region->y[j]);
		    left = MIN(left, _region->x[j]);
		    right = MAX(right, _region->x[j]);
	    }

        rectangle.x = left;
        rectangle.y = top;
        rectangle.width = right - left;
        rectangle.height = bottom - top;
#else

        rectangle.x = get_x();
        rectangle.y = get_y();
        rectangle.width = get_width();
        rectangle.height = get_height();

#endif

    }

    void operator= (cv::Rect& rectangle) {
        this->get(rectangle);
    }

#endif

protected:

    vot_region* _region;

};

#ifdef VOT_OPENCV

void operator<< (VOTRegion &source, const cv::Rect& rectangle) {
    source.set(rectangle);
}

void operator>> (const VOTRegion &source, cv::Rect& rectangle) {
    source.get(rectangle);
}

void operator<< (cv::Rect& rectangle, const VOTRegion &source) {
    source.get(rectangle);
}

void operator>> (const cv::Rect& rectangle, VOTRegion &source) {
    source.set(rectangle);
}


#endif

class VOT {
public:
    VOT() {
        _region = vot_initialize(); 
    }

    ~VOT() {
        vot_quit();
    }

    const VOTRegion region() { 
        return VOTRegion(_region);
    }

    void report(const VOTRegion& region) {

        vot_report(region._region);

    }

    const string frame() {

        const char* result = vot_frame();

        if (!result)
            return string();

        return string(result);
    }

    bool end() {
        return vot_end() != 0;
    }


private:

    vot_region* vot_initialize();

    void vot_quit();

    const char* vot_frame();

    void vot_report(vot_region* region);

    int vot_end();

    vot_region* _region;

#endif

    // Current position in the sequence
    int _vot_sequence_position;
    // Size of the sequence
    int _vot_sequence_size;
    // List of image file names
    char** _vot_sequence;
    // List of results
    vot_region** _vot_result;

#ifdef VOT_TRAX

    trax_handle* _trax_handle;
    char _trax_image_buffer[VOT_READ_BUFFER];

#ifdef VOT_POLYGON

vot_region* _trax_to_region(const trax_region* _trax_region) {
    int i;
    int count = trax_region_get_polygon_count(_trax_region);
    vot_region* region = vot_region_create(count);
    for (i = 0; i < count; i++)
        trax_region_get_polygon_point(_trax_region, i, &(region->x[i]), &(region->y[i]));
    return region;
}
trax_region* _region_to_trax(const vot_region* region) {
    int i;
    trax_region* _trax_region = trax_region_create_polygon(region->count);
    assert(trax_region_get_type(_trax_region) == TRAX_REGION_POLYGON);
    for (i = 0; i < region->count; i++)
        trax_region_set_polygon_point(_trax_region, i, region->x[i], region->y[i]);
    return _trax_region;
}
#else

vot_region* _trax_to_region(const trax_region* _trax_region) {
    vot_region* region = vot_region_create();
    assert(trax_region_get_type(_trax_region) == TRAX_REGION_RECTANGLE);
    trax_region_get_rectangle(_trax_region, &(region->x), &(region->y), &(region->width), &(region->height));
    return region;
}
trax_region* _region_to_trax(const vot_region* region) {
    return trax_region_create_rectangle(region->x, region->y, region->width, region->height);
}

#endif
#endif

#ifdef __cplusplus

};

#endif

#ifdef __cplusplus
#  define VOT_PREFIX(FUN) VOT::FUN
#else
#  define VOT_PREFIX(FUN) FUN
#endif


/**
 * Reads the input data and initializes all structures. Returns the initial 
 * position of the object as specified in the input data. This function should
 * be called at the beginning of the program.
 */
vot_region* VOT_PREFIX(vot_initialize)() {

    int j;
    FILE *inputfile;
    FILE *imagesfile;

    _vot_sequence_position = 0;
    _vot_sequence_size = 0;

#ifdef VOT_TRAX
    if (getenv("TRAX")) {
        trax_configuration config;
        trax_image* _trax_image = NULL;
        trax_region* _trax_region = NULL;
        _trax_handle = NULL;
        int response;

        #ifdef VOT_POLYGON
        config.format_region = TRAX_REGION_POLYGON;
        #else
        config.format_region = TRAX_REGION_RECTANGLE;
        #endif
        config.format_image = TRAX_IMAGE_PATH;
        _trax_handle = trax_server_setup(config, NULL);

        response = trax_server_wait(_trax_handle, &_trax_image, &_trax_region, NULL);

        assert(response == TRAX_INITIALIZE);

        strcpy(_trax_image_buffer, trax_image_get_path(_trax_image));

        trax_server_reply(_trax_handle, _trax_region, NULL);

        vot_region* region = _trax_to_region(_trax_region);

        trax_region_release(&_trax_region);
        trax_image_release(&_trax_image);
        
        return region;
    }
#endif

    inputfile = fopen("region.txt", "r");
    imagesfile = fopen("images.txt", "r");

    if (!inputfile) {
        fprintf(stderr, "Initial region file (region.txt) not available. Stopping.\n");
        exit(-1);
    }

    if (!imagesfile) {
        fprintf(stderr, "Image list file (images.txt) not available. Stopping.\n");
        exit(-1);
    }

    int linelen;
    size_t linesiz = sizeof(char) * VOT_READ_BUFFER;
    char* linebuf = (char*) malloc(sizeof(char) * VOT_READ_BUFFER);
    
    _getline(&linebuf, &linesiz, inputfile);
    vot_region* region = _parse_region(linebuf);

    fclose(inputfile);

    j = 32;
    _vot_sequence = (char**) malloc(sizeof(char*) * j);

    while (1) {

        if ((linelen = _getline(&linebuf, &linesiz, imagesfile))<1)
            break;

        if ((linebuf)[linelen - 1] == '\n') {
            (linebuf)[linelen - 1] = '\0';
        }

        if (_vot_sequence_size == j) {
            j += 32;
            _vot_sequence = (char**) realloc(_vot_sequence, sizeof(char*) * j);
        }

        _vot_sequence[_vot_sequence_size] = (char *) malloc(sizeof(char) * (strlen(linebuf) + 1));

        strcpy(_vot_sequence[_vot_sequence_size], linebuf);

        _vot_sequence_size++;
    }

    free(linebuf);

    _vot_result = (vot_region**) malloc(sizeof(vot_region*) * _vot_sequence_size);

    return region;
}

/**
 * Stores results to the result file and frees memory. This function should be 
 * called at the end of the tracking program.
 */
void VOT_PREFIX(vot_quit)() {

    int i;

#ifdef VOT_TRAX
    if (_trax_handle) {
        trax_cleanup(&_trax_handle);
        return;
    }
#endif

    FILE *outputfile = fopen("output.txt", "w");

    for (i = 0; i < _vot_sequence_position; i++) {
#ifdef VOT_POLYGON
        {
            int j;
            fprintf(outputfile, "%f,%f", _vot_result[i]->x[0], _vot_result[i]->y[0]); 
            for (j = 1; j < _vot_result[i]->count; j++)
                fprintf(outputfile, ",%f,%f", _vot_result[i]->x[j], _vot_result[i]->y[j]); 
            fprintf(outputfile, "\n"); 
        }
#else
        fprintf(outputfile, "%f,%f,%f,%f\n", _vot_result[i]->x, _vot_result[i]->y, _vot_result[i]->width, _vot_result[i]->height); 
#endif
        vot_region_release(&(_vot_result[i]));
    }

    fclose(outputfile);

    if (_vot_sequence) {
        for (i = 0; i < _vot_sequence_size; i++)
            free(_vot_sequence[i]);

        free(_vot_sequence);
    }

    if (_vot_result)
        free(_vot_result);

}

/**
 * Returns the file name of the current frame. This function does not advance 
 * the current position.
 */
const char* VOT_PREFIX(vot_frame)() {

#ifdef VOT_TRAX
    if (_trax_handle) {
        int response;
        trax_image* _trax_image = NULL;
        trax_region* _trax_region = NULL;

        if (_vot_sequence_position == 0) {
            _vot_sequence_position++;
            return _trax_image_buffer;
        }

        response = trax_server_wait(_trax_handle, &_trax_image, &_trax_region, NULL);

        if (response != TRAX_FRAME) {
            vot_quit();
            exit(0);
        }

        strcpy(_trax_image_buffer, trax_image_get_path(_trax_image));
        trax_image_release(&_trax_image);

        return _trax_image_buffer;

    }
#endif

    if (_vot_sequence_position >= _vot_sequence_size)
        return NULL;

    return _vot_sequence[_vot_sequence_position];

}

/**
 * Used to report position of the object. This function also advances the
 * current position.
 */
void VOT_PREFIX(vot_report)(vot_region* region) {

#ifdef VOT_TRAX
    if (_trax_handle) {
        trax_region* _trax_region = _region_to_trax(region);
        trax_server_reply(_trax_handle, _trax_region, NULL);
        trax_region_release(&_trax_region);
        return;
    }
#endif

    if (_vot_sequence_position >= _vot_sequence_size)
        return;
        
    _vot_result[_vot_sequence_position] = vot_region_copy(region);
    _vot_sequence_position++;
}

int VOT_PREFIX(vot_end)() {

#ifdef VOT_TRAX
    return 0;
#endif

    if (_vot_sequence_position >= _vot_sequence_size)
        return 1;
        
    return 0;
}

#endif

