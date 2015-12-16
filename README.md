## Tracking with Kernelized Correlation Filters

Code author : Tomas Vojir

________________

This is a C++ reimplementation of algorithm presented in "High-Speed Tracking with Kernelized Correlation Filters" paper.
For more info and implementation in other languages visit the [autor's webpage!](http://home.isr.uc.pt/~henriques/circulant/). 

It is free for research use. If you find it useful or use it in your research, please acknowledge my git repository 
and cite the original paper [1].

The code depends on OpenCV 2.4+ library and is build via cmake toolchain.

_________________
Quick start guide

for linux: open terminal in the directory with the code

$ mkdir build; cd build; cmake .. ; make

This code compiles into binary **kcf_vot**

./kcf_vot 
- using VOT 2014 methodology (http://www.votchallenge.net/)
 - INPUT : expecting two files, images.txt (list of sequence images with absolute path) and 
           region.txt with initial bounding box in the first frame in format "top_left_x, top_left_y, width, height" or 
           four corner points listed clockwise starting from bottom left corner.
 - OUTPUT : output.txt containing the bounding boxes in the format "top_left_x, top_left_y, width, height"

__________
References

[1] João F. Henriques, Rui Caseiro, Pedro Martins, Jorge Batista, “High-Speed Tracking with Kernelized Correlation Filters“, 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015

_____________________________________
Copyright (c) 2014, Tomáš Vojíř

Permission to use, copy, modify, and distribute this software for research
purposes is hereby granted, provided that the above copyright notice and 
this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

__________________
Additional Library

NOTE: The following files are part of Piotr's Toolbox, and were modified for usage with c++

   src/piotr_fhog/gradientMex.cpp
   src/piotr_fhog/sse.hpp
   src/piotr_fhog/wrappers.hpp
  
You are encouraged to get the [full version of this library here.](http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html)

______________________________________________________________________________

Copyright (c) 2012, Piotr Dollar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.

