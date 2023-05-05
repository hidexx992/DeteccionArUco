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

// Ordenar en sentido contrario a las agujas del reloj (Puntos entreros)
vector<Point> ordenarPuntos(vector<Point> approx) {
	vector<Point> points; //Puntos
	for (int j = 0; j < 4; j++)
		points.push_back(Point(approx[j].x, approx[j].y));	
	Point v1 = points[1] - points[0];
	Point v2 = points[2] - points[0];
	double o = (v1.x * v2.y) - (v1.y * v2.x);
	if (o < 0.0)
		swap(points[1], points[3]);
	return points;
}

// Ordenar en sentido contrario a las agujas del reloj (Puntos flotantes)
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

int determinarID(vector<vector<vector<int>>> aruco_images_matrixs, vector<vector<int>> result_matrix, int &rotation) {
	int id;
	rotation = -1;
	for (int i = 0; i < 40; i++) 
	{
		if (std::equal(result_matrix.begin(), result_matrix.end(), aruco_images_matrixs[i].begin())) // comparar vectores de matrices
		{
			if ((i >= 0) && (i < 4)) {
				id = 0;
				rotation = i;
			}
			else if ((i >= 4) && (i < 8)) {
				id = 1;
				rotation = i - 4;
			}
			else if ((i >= 8) && (i < 12)){ 
				id = 2;
				rotation = i - 8;
			}
			else if ((i >= 12) && (i < 16)){ 
				id = 3;
				rotation = i - 12;
			}
			else if ((i >= 16) && (i < 20)){ 
				id = 4;
				rotation = i - 16;
			}
			else if ((i >= 20) && (i < 24)){ 
				id = 5;
				rotation = i - 20;
			}
			else if ((i >= 24) && (i < 28)){ 
				id = 6;
				rotation = i - 24;
			}
			else if ((i >= 28) && (i < 32)){ 
				id = 7;
				rotation = i - 28;
			}
			else if ((i >= 32) && (i < 36)){ 
				id = 8;
				rotation = i - 32;
			}
			else if ((i >= 36) && (i < 40)){ 
				id = 9;
				rotation = i - 36;
			}
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



void getCamMatrix_and_distCoeff(String ruta_fichero, Mat& camera_matrix_m, Mat& coeff_dist_m) {
	//Leer cada línea
	String linea;
	//Matriz camara y coeficientes de distorsión
	double matriz_camara[3][3], coeff_dist[5][1];
	//Abre el archivo
	ifstream archivo(ruta_fichero);
	// Vector para almacenar los valores de la línea actual	
	vector<double> valores;
	
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

	Mat k(3, 3, CV_64FC1);
	Mat dc(5, 1, CV_64FC1);
	//Guardar valores en Matriz Camara
	cout << "Matriz Camara" << endl;
	int idx = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			k.at<double>(i, j) = valores.at(idx);
			//cout << matriz_camara[i][j] << endl;
			idx++;
		}
	}

	//Guardar valores en vector de coeficientes
	idx = 0;
	cout << "Coeficientes de distorsion" << endl;
	for (int i = 9; i < valores.size(); i++) {
		dc.at<double>(idx) = valores.at(i); 
		idx++;
	}
	//Convertir las matrices en objetos Mat
	camera_matrix_m = k.clone();
	coeff_dist_m = dc.clone();
}



vector<Point2f> refinedCorners(Mat descsFrame, Mat descsRef, vector<KeyPoint> kpsFrame, vector<KeyPoint> kpsRef, Mat aruco_img) {
	resize(aruco_img, aruco_img, Size(), 0.5, 0.5, INTER_AREA);
	// Realizar emparejamiento de características
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector< DMatch > matches;
	matcher->match(descsRef, descsFrame, matches, 2);
	

	//-- Paso 4: Filtramos los emparejamientos en funcion de lo bueno que sean 
	// En este caso, valor relativo frente al mejor caso
	double max_dist = 0; 
	double min_dist = 10000;
	for (int i = 0; i < descsRef.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	vector<DMatch> good_matches;
	for (int i = 0; i < descsRef.rows; i++) {
		if (matches[i].distance < 2 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}
	//-- Paso 5: Localizamos el objeto
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches
		obj.push_back(kpsRef[good_matches[i].queryIdx].pt);
		scene.push_back(kpsFrame[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, RANSAC, 3, noArray(), 3000, 0.20);

	// Tomamos las esquinas de la imagen del objeto
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); 
	obj_corners[1] = Point(aruco_img.cols, 0);
	obj_corners[2] = Point(aruco_img.cols, aruco_img.rows); 
	obj_corners[3] = Point(0, aruco_img.rows);
	vector<Point2f> scene_corners(4);
	// Proyectamos susando la homografia
	perspectiveTransform(obj_corners, scene_corners, H);
	return scene_corners;
}
vector<Point> pts2f_to_pts (vector<Point2f> cornersFloat) {
	// Convertir cada esquina de Point2f a Point
	vector<Point> cornersInt;
	for (int i = 0; i < cornersFloat.size(); i++) {
		Point corner(cornersFloat[i].x, cornersFloat[i].y);
		cornersInt.push_back(corner);
	}
	return cornersInt;
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
	/*Obtener matriz de cámara y coeficientes de de distorsión*/
	Mat cameraMatrix, distCoeffs;
	getCamMatrix_and_distCoeff(ruta_fichero, cameraMatrix, distCoeffs);
	
	vector<Point> aruco_corners
	{
		Point(0, 0),
		Point(0, 591),
		Point(591, 591),
		Point(591, 0)
	};
		 
	vector<Point2f> aruco_corners_f
	{
		Point2f(0, 0),
		Point2f(0, 591),
		Point2f(591, 591),
		Point2f(591, 0) };

	vector<Point3f> corners_3d;
	corners_3d.push_back(Point3f(-5, 5, 0));
	corners_3d.push_back(Point3f(5, 5, 0));
	corners_3d.push_back(Point3f(5, -5, 0));
	corners_3d.push_back(Point3f(-5, -5, 0));

	vector<Point3f> axis_cube{ //Dibujo cubo
		Point3f(-5, -5, 8),  // esquina inferior izquierda frontal
		Point3f(-5, 5, 8),   // esquina superior izquierda frontal
		Point3f(5, 5, 8),    // esquina superior derecha frontal
		Point3f(5, -5, 8),   // esquina inferior derecha frontal
		Point3f(-5, -5, 0),  // esquina inferior izquierda trasera
		Point3f(-5, 5, 0),   // esquina superior izquierda trasera
		Point3f(5, 5, 0),    // esquina superior derecha trasera
		Point3f(5, -5, 0)    // esquina inferior derecha trasera
	};

	vector<Point3f> axis_pyramid{
		Point3f(5, 5, 0),
		Point3f(5, -5, 0),
		Point3f(5, -5, 5),
		Point3f(5, 5, 5),
		Point3f(0, 0, 2.5)
	};



	// Set coordinate system
	float markerLength = 0.05;
	cv::Mat objPoints(4, 1, CV_32FC3);
	objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
	objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
	
	//Dibujar los ejes 
	std::vector<cv::Point3f> axis = {
	cv::Point3f(0, 0, 0),      // origen
	cv::Point3f(5, 0, 0),      // eje X
	cv::Point3f(0, 5, 0),      // eje Y
	cv::Point3f(0, 0, 5)       // eje Z
	};
		
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

	bool drawing_cube = false;
	bool drawing_pyramid = false;
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
		//Umbralización para binarizar la imagen
		Mat thresh;
		adaptiveThreshold(gray, thresh,
			255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 3);
		
		//Inicializa un vector para almacenar las esquinas
		vector<vector<Point>> corners; //Enteros
		vector<vector<Point2f>> corners_f; //Flotantes
		//Aproximar el contorno
		vector<Point2f> approx_f; //Enteros
		vector<Point> approx;     //Flotantes
		//Guardar esquinas que corresponden a marcadores ArUco
		vector<vector<Point> > corrected_corners; //Enteros
		vector<vector<Point2f> > corrected_corners_f; //Flotantes
		//Definir el tamaño de la imagen que va a contener el marcador que se va a extraer para determinar su id
		vector<Point2f> square_points;
		int marker_image_side_length = 60;
		square_points.push_back(Point2f(0, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
		square_points.push_back(Point2f(0, marker_image_side_length - 1));		
		//Encontrar contornos
		Mat contour_image = thresh.clone();
		vector<vector<Point>> contours;
		findContours(contour_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//Encontrar las esquinas que nos interesan analizar
		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			if (area > 2300) {
				double peri = arcLength(contours[i], true); 
				//Aproximacion del contorno
				approxPolyDP(contours[i], approx_f, 0.02 * peri, true);
				approxPolyDP(contours[i], approx, 0.02 * peri, true);
				//Si el polígono tiene cuatro lados y es convexo, entonces es un cuadrado
				if ((approx_f.size() == 4) && (isContourConvex(approx) == true))
				{
					corners.push_back(ordenarPuntos(approx));
					corners_f.push_back(ordenarPuntos_f(approx_f));
				}
			}
		}
		//Vector de objetos tipos Mat que contienen los marcadores resultantes
		vector<Mat> result_markers;
		//Vector que guarda los id de los marcadores
		vector<int> result_ids;
		//Vector que guarda las matrices de los marcadores 
		vector<vector<vector<int>>> result_matrixs; 
		//Vector que guarda las rotaciones
		vector<int> rotations;
		for (int i = 0; i < corners.size(); i++)
		{
			Mat result_marker;
			vector<vector<int>>result_matrix;
			vector<Point2f> m = corners_f[i];
			//Calcular la homografía entre el marcador y el cuadrado de referencia
			Mat H = findHomography(corners_f[i], square_points);
			//Aplicar transformación de perspectiva. 
			warpPerspective(gray, result_marker, H,
				Size(marker_image_side_length, marker_image_side_length));
			//Aplicar la binarización por el método otsu.
			threshold(result_marker, result_marker, 127, 255, THRESH_BINARY | THRESH_OTSU);
			//Definimos el kernel
			Mat kernel = Mat::ones(6, 6, CV_8UC1);
			//Aplicamos la operación de erosión
			erode(result_marker, result_marker, kernel);
			rotate(result_marker, result_marker, cv::ROTATE_90_CLOCKWISE);
			//El tamaño del marcador es 4, y el tamaño incluyendo el borde negro es 6
			int cellSize = result_marker.rows / 6;
			vector <int> result_vector;
			int rotation;
			for (int y = 0; y < 6; y++)
			{
				for (int x = 0; x < 6; x++) 
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = result_marker(Rect(cellX, cellY, cellSize, cellSize)); //Celda
					int cellValue = cv::sum(cell)[0];		//suma de los valores de píxeles en la celda
					int msbValue = (cellValue >> 7) & 1;	//valor del bit más significativo
					result_vector.push_back(msbValue);						
				}
				result_matrix.push_back(result_vector);
				result_vector.clear();
			}
			result_matrixs.push_back(result_matrix);
			result_markers.push_back(result_marker);
			result_matrix = bordesNegros(result_matrix);
			int id = determinarID(aruco_images_matrixs, result_matrix, rotation);
			rotations.push_back(rotation);
			cout << "ID = " << id  <<endl;
			imshow("res", result_marker);

			if (id != -1) {
				//trackPuntos_f(corners_f[i], rotation);
				//trackPuntos(corners[i], rotation);
				// Dibujar un círculo rojo de radio 5 píxeles alrededor del punto
				//cout << rotation << endl;
				std::rotate(corners_f[i].begin(), corners_f[i].begin() + 4 - rotation, corners_f[i].end());
				cout << rotation << endl;
				//cv::circle(frame, corners_f[i][0], 10, cv::Scalar(0, 0, 255), -1);
				// Mostrar la imagen resultante
				corrected_corners.push_back(corners[i]);
				corrected_corners_f.push_back(corners_f[i]);
				result_ids.push_back(id);
			}
			result_matrix.clear();
		}

		//Calcular homografía entre esquinas del objeto fijo y objeto girado
		for (int i=0; i < corrected_corners.size(); i++) 
		{
			int centerX = 0; int centerY = 0;
			int sumX = 0; int sumY = 0;
			for (int j = 0; j < 4; j++) {
				sumX += corrected_corners[i][j].x;
				sumY += corrected_corners[i][j].y;
			}
			centerX = sumX / corrected_corners[i].size();
			centerY = sumY / corrected_corners[i].size();
			Point marker_center = Point(centerX - 30, centerY);
			drawContours(frame, vector<vector<Point>>{corrected_corners[i]}, -1, Scalar(0, 255, 0), 5, LINE_AA);
			putText(frame, "ID = " + to_string(result_ids[i]), marker_center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
		}
		result_matrixs.clear();
		result_ids.clear();
		result_markers.clear();
		
		
		if (corrected_corners.size() >= 1){ 
			//std::cout << "Initial cameraMatrix: " << cameraMatrix << std::endl;
			//std::cout << "rvec: " << rvec << std::endl;
			//std::cout << "tvec: " << tvec << std::endl;
			// rvec is 3x1, tvec is 3x1 ?
			std::vector<cv::Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
			//for(int i=0;; )
			solvePnPRansac(corners_3d, corrected_corners_f[0], cameraMatrix, distCoeffs, rvecs.at(0), tvecs.at(0));
			// Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
			vector<Point2f>  imPointsProjected;
			projectPoints(axis, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imPointsProjected);
			//Dibujar ejes
			ordenarPuntos_f(imPointsProjected);
			line(frame, imPointsProjected[0], imPointsProjected[1], cv::Scalar(0, 0, 255), 2);    // eje X en rojo
			line(frame, imPointsProjected[0], imPointsProjected[2], cv::Scalar(0, 255, 0), 2);    // eje Y en verde
			line(frame, imPointsProjected[0], imPointsProjected[3], cv::Scalar(255, 0, 0), 2);    // eje Z en azul
		}
	

		// Obtener FPS
		auto end = chrono::steady_clock::now();
		double milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
		start = end;
		fps = 1.0 / (milliseconds / 1000.0);
		string fps_text = "FPS: " + to_string(fps);
		putText(frame, fps_text, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
		
		//Dibujar cubo
		if (drawing_cube)	
		{
			if (corrected_corners.size() >= 1)
			{
				std::vector<cv::Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
				vector<vector<Point2f>>  imPointsProjected(corrected_corners.size());
				for (int i = 0; i < corrected_corners.size(); i++) 
				{
					solvePnPRansac(corners_3d, corrected_corners_f[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
					//Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
					projectPoints(axis_cube, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imPointsProjected[i]);
					//Dibujar las líneas del cubo en la imagen
					Scalar green(0, 255, 0);
					Scalar red(0, 0, 255);
					Scalar violeta(242, 90, 232);
					cv::line(frame, imPointsProjected[i][0], imPointsProjected[i][1], violeta, 2);
					cv::line(frame, imPointsProjected[i][1], imPointsProjected[i][2], violeta, 2);
					cv::line(frame, imPointsProjected[i][2], imPointsProjected[i][3], violeta, 2);
					cv::line(frame, imPointsProjected[i][3], imPointsProjected[i][0], violeta, 2);
					cv::line(frame, imPointsProjected[i][0], imPointsProjected[i][4], violeta, 2);
					cv::line(frame, imPointsProjected[i][1], imPointsProjected[i][5], violeta, 2);
					cv::line(frame, imPointsProjected[i][2], imPointsProjected[i][6], violeta, 2);
					cv::line(frame, imPointsProjected[i][3], imPointsProjected[i][7], violeta, 2);
					cv::line(frame, imPointsProjected[i][4], imPointsProjected[i][5], violeta, 2);
					cv::line(frame, imPointsProjected[i][5], imPointsProjected[i][6], violeta, 2);
					cv::line(frame, imPointsProjected[i][6], imPointsProjected[i][7], violeta, 2);
					cv::line(frame, imPointsProjected[i][7], imPointsProjected[i][4], violeta, 2);
				}
			}	
		}
		//Dibujar pirámide
		if (drawing_pyramid)
		{
			if (corrected_corners.size() >= 1)
			{
				std::vector<cv::Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
				vector<vector<Point2f>>  imPointsProjected(corrected_corners.size());
				for (int i = 0; i < corrected_corners.size(); i++) 
				{
					solvePnPRansac(corners_3d, corrected_corners_f[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
					//Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
					projectPoints(axis, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imPointsProjected[i]); Scalar green(0, 255, 0);
					ordenarPuntos_f(imPointsProjected[i]);
					int a1x = (imPointsProjected[i][3].x - imPointsProjected[i][0].x);
					int a1y = (imPointsProjected[i][3].y - imPointsProjected[i][0].y);
					Point p1_start = corrected_corners[0][(rotations[0] + 3) % 4];
					Point p2_start = corrected_corners[0][(rotations[0] + 2) % 4];
					Point p3_start = (p1_start + corrected_corners[0][(rotations[0] + 1) % 4]) / 2;
					Point p4_start = (p2_start + corrected_corners[0][(rotations[0] + 0) % 4]) / 2;
					Point p1_end(p1_start.x + a1x, p1_start.y + a1y);
					Point p2_end(p2_start.x + a1x, p2_start.y + a1y);
					Point p3_end(p3_start.x + a1x, p3_start.y + a1y);
					Point p4_end(p4_start.x + a1x, p4_start.y + a1y);
					// base piramido
					Scalar blue(255, 0, 0);
					line(frame, p1_start, p2_start, blue, 2);
					line(frame, p1_start, p1_end, blue, 2);
					line(frame, p2_start, p2_end, blue, 2);
					line(frame, p1_end, p2_end, blue, 2);

					int center_x = (p3_start.x + p4_start.x) / 2;
					int center_y = (p3_start.y + p3_end.y) / 2;

					cv::Point tip(center_x, center_y);
					line(frame, p1_start, tip, blue, 2);
					line(frame, p1_end, tip, blue, 2);
					line(frame, p2_start, tip, blue, 2);
					line(frame, p2_end, tip, blue, 2);
				}
			}
		}

		
		//Presiona ESC en el teclado para salir
		if (pressedKey == TECLA_ESCAPE) {
			break;
		}
		if ((pressedKey == TECLA_A) || (pressedKey == TECLA_a))
		{
			dibujo++;
		}

		switch (dibujo) {
		case 0:
			drawing_cube = false;
			drawing_pyramid = false;
			break;
		case 1:
			drawing_cube = true;
			drawing_pyramid = false;
			break;
		case 2:
			drawing_pyramid = true;
			drawing_cube = false;
			break;
		case 3:
			dibujo = 0;
			break;
		}
		//Se espera 60 ms
		pressedKey = waitKey(20);
		//Mostrar frame
		//Se crea un lienzo donde mostrar imagenes
		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", frame);

	}

	//VideoWriter writer("nombre_del_archivo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size, isColor);
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



	
