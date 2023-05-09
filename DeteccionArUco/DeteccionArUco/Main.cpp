#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "Deteccion.cpp"

//#define VIDEO "./videos/video_2.mp4"
//#define FICHERO "./ficheros/camera_calibration_parameters_v2.txt"

#define VIDEO "./videos/video_3.mp4"
#define FICHERO "./ficheros/camera_calibration_parameters_v2.txt"

//#define VIDEO "./videos/video_1.avi"
//#define FICHERO "./ficheros/camera_calibration_parameters.txt"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	cout << "Has introducido " << argc << "argumentos" << endl;
	for (int i = 0; i < argc; i++) {
		cout << argv[i] << endl;
	}

	switch (argc) {
		case 1:
			deteccion(VIDEO, FICHERO);
			break;
		case 3:
			string video = argv[1];
			string fichero = argv[2];
			deteccion(video, fichero);
			break;
	}
	return 0;
}