#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#define TECLA_ESCAPE 27 //ASCII
#define TECLA_a 97 
#define TECLA_A 65 
#define CAMARA_0 0
const int n_markers = 10;

using namespace cv;
using namespace std;


void inicializarCubo() {
	cv::Mat axis = (cv::Mat_<float>(4, 3) << 3, 0, 0, 0, 3, 0, 0, 0, -3, 0, 0, 0);
	// this creates a matrix with 4 rows and 3 columns, 
	//containing the coordinates for the points to draw the x, y, and z axis. 
	//The fourth row is initialized with 0,0,0 to make the matrix homogeneous.
}

void resolverPose(vector<cv::Point3f> objectPoints, vector<cv::Point2f> corners1, Mat cameraMatrix, Mat distCoeffs, Mat &rvec, Mat &tvec) {
	solvePnP(objectPoints, corners1, cameraMatrix, distCoeffs, rvec, tvec);
	std::vector<cv::Point2f> projectedPoints;
	cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
	for (unsigned int i = 0; i < projectedPoints.size(); ++i)
	{
		std::cout << "Image point: " << corners1[i] << " Projected to " << projectedPoints[i] << std::endl;
	}
}

cv::Mat drawAxis(cv::Mat img, cv::Mat corners, cv::Mat imgpts) {
	cv::Point3f corner = cv::Point3f(corners.at<float>(0, 0), corners.at<float>(0, 1), corners.at<float>(0, 2));
	cv::line(img, cv::Point(corner.x, corner.y), cv::Point(imgpts.at<float>(0, 0), imgpts.at<float>(0, 1)), cv::Scalar(255, 0, 0), 5);
	cv::line(img, cv::Point(corner.x, corner.y), cv::Point(imgpts.at<float>(1, 0), imgpts.at<float>(1, 1)), cv::Scalar(0, 255, 0), 5);
	cv::line(img, cv::Point(corner.x, corner.y), cv::Point(imgpts.at<float>(2, 0), imgpts.at<float>(2, 1)), cv::Scalar(0, 0, 255), 5);
	return img;
}

cv::Mat draw_cube(cv::Mat img, cv::Mat corners, cv::Mat imgpts) {
	cv::Mat imgpts_int;
	imgpts.convertTo(imgpts_int, CV_32S);
	std::vector<std::vector<cv::Point>> contours = { std::vector<cv::Point>(imgpts_int.rowRange(0, 4)) };
	cv::drawContours(img, contours, -1, cv::Scalar(0, 255, 0), 3);
	for (int i = 0; i < 4; i++) {
		cv::line(img, cv::Point(imgpts_int.at<int>(i, 0), imgpts_int.at<int>(i, 1)),
			cv::Point(imgpts_int.at<int>(i + 4, 0), imgpts_int.at<int>(i + 4, 1)), cv::Scalar(255), 3);
	}
	contours = { std::vector<cv::Point>(imgpts_int.rowRange(4, 8)) };
	cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 3);
	return img;
}

vector<Mat> cargarImagenes() {
	vector<Mat> aruco_images;
	for (int i = 0; i < n_markers; i++) {
		string filename = "./imagenes/MarcadoresAruco/4x4_1000-";
		filename += to_string(i);
		filename += ".png";
		cout << "Leyendo " << filename << endl;
		Mat aruco_marker = imread(filename);
		if (aruco_marker.empty()) {
			cerr << "Error: no se pudo leer la imagen " << filename << endl;
			exit(1);
		}
		aruco_images.push_back(aruco_marker);
	}
	return aruco_images;
}

vector<vector<int>> bordesNegros(vector<vector<int>> result_matrix) //Matrix 6x6
{
	result_matrix[0] = { 0,0,0,0,0,0 };
	result_matrix[5] = { 0,0,0,0,0,0 };
	result_matrix[1][0] = 0;
	result_matrix[2][0] = 0;
	result_matrix[3][0] = 0;
	result_matrix[4][0] = 0;
	result_matrix[1][5] = 0;
	result_matrix[2][5] = 0;
	result_matrix[3][5] = 0;
	result_matrix[4][5] = 0;
	return result_matrix;

}

int determinarID(vector<vector<vector<int>>> aruco_images_matrixs, vector<vector<int>> result_matrix) {
	int id;
	for (int i = 0; i < 40; i++) 
	{
		if (std::equal(result_matrix.begin(), result_matrix.end(), aruco_images_matrixs[i].begin())) // comparar vectores de matrices
		{
			if ((i >= 0) && (i < 4)) id = 0;
			else if ((i >= 4) && (i < 8)) id = 1;
			else if ((i >= 8) && (i < 12)) id = 2;
			else if ((i >= 12) && (i < 16)) id = 3;
			else if ((i >= 16) && (i < 20)) id = 4;
			else if ((i >= 20) && (i < 24)) id = 5;
			else if ((i >= 24) && (i < 28)) id = 6;
			else if ((i >= 28) && (i < 32)) id = 7;
			else if ((i >= 32) && (i < 36)) id = 8;
			else if ((i >= 36) && (i < 40)) id = 9;
			break;
		}
		else {
			id = -1;
		}
	}
	return id;
}

vector<vector<vector<int>>> generarMatricesImagenes(vector<Mat> aruco_images) {
	vector<vector<vector<int>>> aruco_images_matrixs(0);
	
	for(int i = 0 ; i < aruco_images.size(); i++)
	{
		
		resize(aruco_images[i], aruco_images[i], Size(60, 60), 0, 0, INTER_LINEAR);
		cvtColor(aruco_images[i], aruco_images[i], COLOR_BGR2GRAY);
		threshold(aruco_images[i], aruco_images[i], 125, 255, THRESH_BINARY | THRESH_OTSU);
		int cellSize = aruco_images[i].rows / 6;

		vector <vector<int>> aruco_matrix(0); // Inicializar con ceros
		vector <int> aruco_vector(0); // Inicializar con ceros
		
		for (int j = 0; j < 4; j++)
		{
			for (int y = 0; y < 6; y++)
			{
				for (int x = 0; x < 6; x++) //+= inc)
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = aruco_images[i](Rect(cellX, cellY, cellSize, cellSize)); //Celdas 
					int cellValue = cv::sum(cell)[0]; // suma de los valores de píxeles en la celda
					int msbValue = (cellValue >> 7) & 1; // valor del bit más significativo
					aruco_vector.push_back (msbValue);
				}
				aruco_matrix.push_back(aruco_vector);
				aruco_vector.clear();
			}
			aruco_images_matrixs.push_back(aruco_matrix);
			aruco_matrix.clear();
			rotate(aruco_images[i], aruco_images[i], ROTATE_90_CLOCKWISE);
		}
	}
	return aruco_images_matrixs;
}



void getCamMatrix_and_distCoeff(String ruta_fichero, Mat& matriz_camara_m, Mat& coeff_dist_m) {
	//Leer cada línea
	String linea;
	//Matriz camara y coeficientes de distorsión
	double matriz_camara[3][3], coeff_dist[1][5];
	//Abre el archivo
	ifstream archivo(ruta_fichero);
	// Vector para almacenar los valores de la línea actual	
	vector<long double> valores;
	//Verifica que el archivo se haya abierto correctamente
	if (!archivo.is_open()) {
		cout << "No se pudo abrir el archivo" << endl;
		exit(1);
	}

	// Leer línea por línea del archivo
	while (getline(archivo, linea)) {
		// Leer los valores de la línea actual
		istringstream iss(linea);
		double valor;
		while (iss >> valor) {
			valores.push_back(valor);
		}
	}

	//Guardar valores en Matriz Camara
	cout << "Matriz Camara" << endl;
	int k = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			matriz_camara[i][j] = valores[k];
			cout << valores[k] << endl;
			k++;
		}
	}

	//Guardar valores en vector de coeficientes
	k = 0;
	cout << "Coeficientes de distorsion" << endl;
	for (int i = 9; i < valores.size(); i++) {
		coeff_dist[k][0] = valores[i];
		cout << valores[i] << endl;
		k++;
	}
	//Convertir las matrices en objetos Mat
	matriz_camara_m = Mat(3, 3, CV_64FC1, matriz_camara);
	coeff_dist_m = Mat(5, 1, CV_64FC1, coeff_dist);
}

vector<Point> ordenarPuntos(vector<Point> approx){
	vector<Point> points; //Puntos
	for (int j = 0; j < 4; j++)
		points.push_back(Point(approx[j].x, approx[j].y));
	// Ordenar en sentido contrario a las agujas del reloj
	Point v1 = points[1] - points[0];
	Point v2 = points[2] - points[0];
	double o = (v1.x * v2.y) - (v1.y * v2.x);
	if (o < 0.0)
		swap(points[1], points[3]);
	return points;
}

vector<Point2f> ordenarPuntos_f(vector<Point2f> approx) {
	vector<Point2f> points; //Puntos
	for (int j = 0; j < 4; j++)
		points.push_back(Point2f(approx[j].x, approx[j].y));
	// Ordenar en sentido contrario a las agujas del reloj
	Point v1 = points[1] - points[0];
	Point v2 = points[2] - points[0];
	double o = (v1.x * v2.y) - (v1.y * v2.x);
	if (o < 0.0)
		swap(points[1], points[3]);
	return points;
}


void deteccion(String ruta_video, String ruta_fichero) {
	/*Variables*/
	//Crear el objeto de tipo VideoCapture
	VideoCapture captura = VideoCapture(ruta_video);
	//VideoCapture captura = VideoCapture(CAMARA_0);
	
	//Frames
	Mat frame;
	//Tecla presionada
	char pressedKey = 0;
	//Selección dibujo 
	int dibujo = 0;
	/*Leer las imágenes*/
	vector<Mat> aruco_images;
	aruco_images = cargarImagenes();
	/*Generar matrices binarias de las imagenes*/
	vector <vector<vector<int>>> aruco_images_matrixs = generarMatricesImagenes(aruco_images);
	/*Obtener matriz de cámara y coeficientes de */
	Mat cameraMatrix, distCoeffs;
	getCamMatrix_and_distCoeff(ruta_fichero, cameraMatrix, distCoeffs);

	if (!captura.isOpened()) {
		cout << "Error al cargar el video!" << endl;
		exit(1);
	}
	//Seguimiento del tiempo transcurrido para calcular FPS
	auto start = chrono::steady_clock::now();
	double fps;

	//Cuántas veces detecta si hay más 1 contorno
	int error = 0;
	int id_error = 0;
	int correcta_teccion = 0;
	int error_id = 0;


	while (1) {
		//Se lee el video imagen a imagen
		captura.read(frame); //capture >> frame;
		// Se comprueba que no se ha llegado al final
		if (frame.empty()) {
			cout << "Se ha llegado al final del video" << endl;
			break;
		}
		//Conversion a Escala de Grises
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//Filtro Gaussiano
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		namedWindow("Escala de Grises con filtro", WINDOW_AUTOSIZE);
		imshow("Escala de Grises con filtro", gray);
		//Amplitud de la escala
		Mat thresh, equalized;
		equalizeHist(gray, equalized);
		//Umbralización para binarizar la imagen
		adaptiveThreshold(equalized, thresh,
			255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 3);
		//Encontrar contornos
		Mat contour_image = thresh.clone();
		vector<vector<Point>> contours;
		findContours(contour_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//Aproximar el contorno
		vector<Point2f> approx_f; //Las esquinas
		vector<Point> approx;     //Esquinas  
		//Inicializa un vector para almacenar las esquinas
		vector<vector<Point> > corners; //las esquinas de todos los marcadores
		vector<vector<Point2f> > corners_f; //las esquinas de todos los marcadores
		
		//Cuantos contornos hay
		/////////////////////////
		//vector<Point3f> axis;
		//axis.push_back(cv::Point3f(3, 0, 0));
		//axis.push_back(cv::Point3f(0, 3, 0));
		//axis.push_back(cv::Point3f(0, 0, -3));
		/////////////////////////

		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			if (area > 2700) {
				// Perimetro
				double peri = arcLength(contours[i], true); 
				//Aproximacion del contorno
				approxPolyDP(contours[i], approx_f, 0.02 * peri, true);
				approxPolyDP(contours[i], approx, 0.02 * peri, true);
				// Si el polígono tiene cuatro lados y es convexo, entonces es un cuadrado
				if ((approx_f.size() == 4) && (isContourConvex(approx) == true))
				{
					corners.push_back(ordenarPuntos(approx));
					corners_f.push_back(ordenarPuntos_f(approx_f));
				}
			}
		}
		
		

		vector<Point2f> square_points;
		
		//Cuando el tamaño del marcador es 4x4, el tamaño incluyendo el borde negro es 6x6
		//Si el ancho de píxel de una celda se establece en 10 al dividir la imagen en una cuadrícula en un paso posterior
		//La longitud de un lado de la imagen del marcador es 60	
		int marker_image_side_length = 60; 
		square_points.push_back(Point2f(0, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
		square_points.push_back(Point2f(0, marker_image_side_length - 1));
		
		Mat marker_image;
		vector <vector<int>> result_matrix(0); // Inicializar con ceros
		for (int i = 0; i < corners_f.size(); i++)
		{
			vector<Point2f> m = corners_f[i];
			//Obtenga una matriz de transformación de perspectiva para transformar el marcador en un rectángulo.
			Mat PerspectiveTransformMatrix = getPerspectiveTransform(m, square_points);
			//Aplicar transformación de perspectiva. 
			warpPerspective(gray, marker_image, PerspectiveTransformMatrix,
				Size(marker_image_side_length, marker_image_side_length));
			//Aplicar la binarización por el método otsu.
			threshold(marker_image, marker_image, 127, 255, THRESH_BINARY | THRESH_OTSU);
			// Definimos el kernel
			Mat kernel = Mat::ones(6, 6, CV_8UC1);
			// Aplicamos la operación de erosión
			erode(marker_image, marker_image, kernel);
			//El tamaño del marcador es 4, y el tamaño incluyendo el borde negro es 6
			int cellSize = marker_image.rows / 6;
			imshow("result", marker_image);
			vector <int> result_vector(0); // Inicializar con ceros
			for (int y = 0; y < 6; y++)
			{
				for (int x = 0; x < 6; x++) //+= inc)
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = marker_image(Rect(cellX, cellY, cellSize, cellSize)); //Celdas negras
					
					int cellValue = cv::sum(cell)[0]; // suma de los valores de píxeles en la celda
					int msbValue = (cellValue >> 7) & 1; // valor del bit más significativo
					result_vector.push_back(msbValue);					
					imshow("Celda", cell);
					
				}
				result_matrix.push_back(result_vector);
				
				result_vector.clear();
			}

			result_matrix = bordesNegros(result_matrix);
			int id = determinarID(aruco_images_matrixs, result_matrix);
			cout << "ID = " << id  <<endl;
			if (id == -1) {
				break;
			}
			int centerX = 0; int centerY = 0;
			int sumX = 0; int sumY = 0;
			for (int j = 0; j < 4; j++) {
				sumX += corners[i][j].x;
				sumY += corners[i][j].y;
			}
			centerX = sumX / corners[i].size();
			centerY = sumY / corners[i].size();
			Point marker_center = Point(centerX - 30, centerY);
			putText(frame, "ID = " + to_string(id), marker_center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
			drawContours(frame, vector<vector<Point>>{corners[i]}, -1, Scalar(0, 255, 0), 5, LINE_AA);
		}
		
		result_matrix.clear();

		/*Poner texto en imagen*/
		//string letrero = "Objetos: ";// + to_string(total);
		//putText(frame, letrero, Point(10, 150), FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);
		
		// Obtener FPS
		auto end = chrono::steady_clock::now();
		double milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
		start = end;
		fps = 1.0 / (milliseconds / 1000.0);
		string fps_text = "FPS: " + to_string(fps);
		putText(frame, fps_text, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
		//Mostrar frame
		//Se crea un lienzo donde mostrar imagenes
		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", frame);
		//Se espera 20 ms
		pressedKey = waitKey(40);
		
		//Presiona ESC en el teclado para salir
		if (pressedKey == TECLA_ESCAPE) {
			break;
		}
		if ((pressedKey == TECLA_A) || (pressedKey == TECLA_a)) {
			dibujo++;
			if (dibujo == 1)
				//dibujoCubo();
				cout << dibujo << endl;
				if (dibujo == 2) {
					//dibujoPiramide();
					dibujo = 0;

				}
		}

	}
	cout << "Veces que han detectado mas de 1 contorno: " << error << endl;
	cout << "Veces que el id es incorrecto " << error_id << endl;
	cout << "Veces que ha detectado bien la imagen " << correcta_teccion << endl;
	//int count_matrix =0;
	for (int i = 0; i < aruco_images_matrixs.size(); i++) {
		for (int j = 0; j < aruco_images_matrixs[i].size(); j++) {
			for (int k = 0; k < aruco_images_matrixs[i][j].size(); k++) {
				//int element = aruco_images_matrixs[i][j][k];
				//cout << element << " ";
			}
			//cout << endl ;
		}
		//cout << endl << endl ;
		//count_matrix++;
	}
	//cout << count_matrix;
	
	//Liberar el objeto videoCapture
	captura.release();
	//Destruir todas las ventanas
	destroyAllWindows();

}



	
