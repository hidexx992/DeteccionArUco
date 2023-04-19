//Se cargan imagenes desde un video y se muestran por pantalla
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <time.h>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define TECLA_ESCAPE 27 //ASCII
#define VIDEO "./videos/video_1.avi"
int main()
{
	/*Variables*/
	//Frame
	Mat frame;
	//Crear objeto VideoCapture
	VideoCapture capture(VIDEO);
	//Guarda el número total de frames
	// Tecla presionada
	char pressedKey = 0;
	// Se crea un lienzo donde mostrar imagenes
	namedWindow("Video", WINDOW_AUTOSIZE);
	//Carga el video?
	if (!capture.isOpened()) {
		cout << "Error al cargar el video!" << endl;
		return -1;
	}

	//Seguimiento del tiempo transcurrido para calcular FPS
	auto start = chrono::steady_clock::now();
	double fps;
	//fps = capture.get(CAP_PROP_FPS);
	

	while (1) {
		//Se lee el video iamgen a imagen
		capture.read(frame); //capture >> frame;
		//fps = capture.get(CAP_PROP_FPS);
		// Se comprueba que no se ha llegado al final
		if (frame.empty()) {
			cout << "Se ha llegado al final del video" << endl;
			break;
		}	
		// Obtener FPS
		auto end = chrono::steady_clock::now();
		double milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
		start = end;
		fps = 1.0 / (milliseconds/1000.0);
		string fps_text = "FPS: " + to_string(fps);
		putText(frame, fps_text, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
		//Mostrar frame
		imshow("Video", frame);
		//Se espera 20 ms
		pressedKey = waitKey(50);
		//Presiona ESC en el teclado para salir
		if (pressedKey == TECLA_ESCAPE) {
			break;
		}
	}
	//Liberar el objeto videoCapture
	capture.release();
	//Destruir todas las ventanas
	destroyAllWindows();
	return 1;
}