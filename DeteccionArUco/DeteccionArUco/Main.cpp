#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "Deteccion.cpp"
#define VIDEO "./videos/video_1.avi"
#define FICHERO "./ficheros/camera_calibration_parameters.txt"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	int valor;

	cout << "Has introducido " << argc << "argumentos" << endl;
	for (int i = 0; i < argc; i++) {
		cout << argv[i] << endl;
	}

	deteccion(VIDEO, FICHERO);


	return 1;
}