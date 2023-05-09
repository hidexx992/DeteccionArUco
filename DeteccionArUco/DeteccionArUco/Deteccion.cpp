#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>

#define TECLA_ESCAPE 27 //ASCII
#define TECLA_a 97 
#define TECLA_A 65 
#define CAMARA_0 0

const int n_markers = 10;
using namespace cv;
using namespace std;

/*Todos los bordes del marcador aruco deben ser negros*/
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

/*Determinar el ID y rotación del marcador*/
int determinarID(vector<vector<vector<int>>> aruco_images_matrixs, vector<vector<int>> result_matrix, int &rotation) {
	int id;
	rotation = -1;
	for (int i = 0; i < 40; i++) 
	{
		if (equal(result_matrix.begin(), result_matrix.end(), aruco_images_matrixs[i].begin())) // Comparar vectores de matrices
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
/*Generar las matrices binarias de los marcadores de referencia*/
vector<vector<vector<int>>> generarMatricesImagenes(vector<Mat> aruco_images) {
	vector<vector<vector<int>>> aruco_images_matrixs(0);
	
	for(int i = 0 ; i < aruco_images.size(); i++)
	{
		resize(aruco_images[i], aruco_images[i], Size(60, 60), 0, 0, INTER_LINEAR);
		cvtColor(aruco_images[i], aruco_images[i], COLOR_BGR2GRAY);
		threshold(aruco_images[i], aruco_images[i], 125, 255, THRESH_BINARY | THRESH_OTSU);
		int cellSize = aruco_images[i].rows / 6;

		vector <vector<int>> aruco_matrix(0);	// Inicializar con ceros
		vector <int> aruco_vector(0);			// Inicializar con ceros
		
		for (int j = 0; j < 4; j++)
		{
			for (int y = 0; y < 6; y++)
			{
				for (int x = 0; x < 6; x++) //+= inc)
				{
					int cellX = x * cellSize;
					int cellY = y * cellSize;
					Mat cell = aruco_images[i](Rect(cellX, cellY, cellSize, cellSize)); //Celdas 
					int cellValue = sum(cell)[0]; // suma de los valores de píxeles en la celda
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

/*Obtener la matriz cámara y los coeficientes de distorsión como objetos tipo Mat*/
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


/*Converetir vector de puntos flotantes a puntos enteros*/
vector<Point> pts2f_to_pts (vector<Point2f> cornersFloat) {
	// Convertir cada esquina de Point2f a Point
	vector<Point> cornersInt;
	for (int i = 0; i < cornersFloat.size(); i++) {
		Point corner(cornersFloat[i].x, cornersFloat[i].y);
		cornersInt.push_back(corner);
	}
	return cornersInt;
}

/*Cargar las imágenes de los 10 marcadores ArUco a detectar*/
vector<Mat> cargarImagenes() {
	vector<Mat> aruco_images;
	for (int i = 0; i < n_markers; i++) {
		string filename = "./imagenes/MarcadoresAruco/4x4_1000-";
		filename += to_string(i);
		filename += ".png";
		//cout << "Leyendo " << filename << endl;
		Mat aruco_marker = imread(filename);
		if (aruco_marker.empty()) {
			cerr << "Error: no se pudo leer la imagen " << filename << endl;
			exit(1);
		}
		aruco_images.push_back(aruco_marker);
	}
	return aruco_images;
}


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
	vector<vector<vector<int>>> aruco_images_matrixs = generarMatricesImagenes(aruco_images);
	/*Obtener matriz de cámara y coeficientes de de distorsión*/
	Mat cameraMatrix, distCoeffs;
	getCamMatrix_and_distCoeff(ruta_fichero, cameraMatrix, distCoeffs);
	cout << "cameraMatrix : " << cameraMatrix << endl;
	cout << "distCoeffs : " << distCoeffs << endl;

	vector<Point3f> corners_3d;
	corners_3d.push_back(Point3f(-5, 5, 0));
	corners_3d.push_back(Point3f(5, 5, 0));
	corners_3d.push_back(Point3f(5, -5, 0));
	corners_3d.push_back(Point3f(-5, -5, 0));

	//Dibujar los ejes 
	vector<Point3f> axis = {
	Point3f(0, 0, 0),      // origen
	Point3f(5, 0, 0),      // eje X
	Point3f(0, 5, 0),      // eje Y
	Point3f(0, 0, 5)       // eje Z
	};

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

	vector<Point3f> axis_pyramid{ //Dibujo pirámide
		Point3f(5, 5, 0),
		Point3f(5, -5, 0),
		Point3f(5, -5, 5),
		Point3f(5, 5, 5),
		Point3f(0, 0, 2.5)
	};
	
	//Carga el video?
	if (!captura.isOpened()) {
		cout << "Error al cargar el video!" << endl;
		exit(1);
	}
	// Obtener el tamaño del video original
	int frame_width = captura.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = captura.get(CAP_PROP_FRAME_HEIGHT);
	//Guardar video
	VideoWriter outputVideo;
	string outputFileName = ruta_video.erase(ruta_video.length() - 4, 4) + "_final.mp4";
	int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
	double fps_video = 30.0;
	Size frameSize(frame_width, frame_height);
	outputVideo.open(outputFileName, codec, fps_video, frameSize);

	//Seguimiento del tiempo transcurrido para calcular FPS
	auto start = chrono::steady_clock::now();
	double fps;
	//Dibujar el cubo
	bool drawing_cube = false;
	//Dibujar la pirámide
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
		//imshow("Gris", gray);
		
		//Filtro Gaussiano
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		//imshow("Gaussiano", gray);

		//Umbralización para binarizar la imagen
		Mat thresh;
		adaptiveThreshold(gray, thresh,
			255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 3);
		//imshow("Thresh", thresh);
		
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
			if (area > 1700) {
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
		contours.clear();

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
			warpPerspective(gray, result_marker, H, Size(marker_image_side_length, marker_image_side_length),
				INTER_NEAREST);
			rotate(result_marker, result_marker, ROTATE_90_CLOCKWISE);
			//Aplicar la binarización por el método otsu.
			threshold(result_marker, result_marker, 127, 255, THRESH_BINARY | THRESH_OTSU);
			//Definimos el kernel
			Mat kernel = Mat::ones(6, 6, CV_8UC1);
			//Aplicamos la operación de erosión
			erode(result_marker, result_marker, kernel);
			//rotate(result_marker, result_marker, ROTATE_90_CLOCKWISE);
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
					int cellValue = sum(cell)[0];		//suma de los valores de píxeles en la celda
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
			//cout << "ID = " << id  <<endl;
			//imshow("res", result_marker);

			if (id != -1) {
				rotate(corners_f[i].begin(), corners_f[i].begin() + 4 - rotation, corners_f[i].end());
				rotations.push_back(rotation);
				//cout << rotation << endl;
				corrected_corners.push_back(corners[i]);
				corrected_corners_f.push_back(corners_f[i]);
				result_ids.push_back(id);
			}
			result_matrix.clear();
		}
		corners_f.clear();
		corners.clear();

		//Escribir el ID
		//Se calcula el centroide de cada marcador sumando las coordenadas x e y 
		//de cada esquina y dividiendo por el número total de esquinas. 
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
			putText(frame, "ID = " + to_string (result_ids[i]), marker_center, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
			Rect tlCorner(corrected_corners[i][rotations[i]].x, corrected_corners[i][rotations[i]].y, 10, 10);
			rectangle(frame, tlCorner, Scalar(0, 0, 255));

		}

		result_matrixs.clear();
		result_ids.clear();
		result_markers.clear();
		
		
		if (corrected_corners.size() >= 1){ 
			vector<Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
			vector<vector<Point2f>>  imPointsProjected(corrected_corners.size());
			for(int i = 0; i < corrected_corners.size(); i++){
				
				solvePnPRansac(corners_3d, corrected_corners_f[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
				// Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
				projectPoints(axis, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imPointsProjected[i]);
				//Dibujar ejes
				ordenarPuntos_f(imPointsProjected[i]);
				//Dibujar ejes
				line(frame, imPointsProjected[i][0], imPointsProjected[i][1], Scalar(0, 0, 255), 2);    // eje X en rojo
				line(frame, imPointsProjected[i][0], imPointsProjected[i][2], Scalar(0, 255, 0), 2);    // eje Y en verde
				line(frame, imPointsProjected[i][0], imPointsProjected[i][3], Scalar(255, 0, 0), 2);    // eje Z en azul
			}
			rvecs.clear();
			tvecs.clear();
			imPointsProjected.clear();
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
				vector<Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
				vector<vector<Point2f>>  cubeProjected(corrected_corners.size());
				Scalar rojo(0, 0, 255);
				for (int i = 0; i < corrected_corners.size(); i++) 
				{
					solvePnPRansac(corners_3d, corrected_corners_f[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
					//Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
					projectPoints(axis_cube, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, cubeProjected[i]);
					ordenarPuntos_f(cubeProjected[i]);
					//Dibujar las líneas del cubo en la imagen
					line(frame, cubeProjected[i][0], cubeProjected[i][1], rojo, 2);
					line(frame, cubeProjected[i][1], cubeProjected[i][2], rojo, 2);
					line(frame, cubeProjected[i][2], cubeProjected[i][3], rojo, 2);
					line(frame, cubeProjected[i][3], cubeProjected[i][0], rojo, 2);
					line(frame, cubeProjected[i][0], cubeProjected[i][4], rojo, 2);
					line(frame, cubeProjected[i][1], cubeProjected[i][5], rojo, 2);
					line(frame, cubeProjected[i][2], cubeProjected[i][6], rojo, 2);
					line(frame, cubeProjected[i][3], cubeProjected[i][7], rojo, 2);
					line(frame, cubeProjected[i][4], cubeProjected[i][5], rojo, 2);
					line(frame, cubeProjected[i][5], cubeProjected[i][6], rojo, 2);
					line(frame, cubeProjected[i][6], cubeProjected[i][7], rojo, 2);
					line(frame, cubeProjected[i][7], cubeProjected[i][4], rojo, 2);
				}
				rvecs.clear();
				tvecs.clear();
				cubeProjected.clear();
			}			
		}

		//Dibujar pirámide
		if (drawing_pyramid)
		{
			if (corrected_corners.size() >= 1)
			{
				
				vector<Vec3d> rvecs(corrected_corners.size()), tvecs(corrected_corners.size());
				vector<vector<Point2f>>  pyramidProjected(corrected_corners.size());
				Scalar green(0, 255, 0);
				for (int i = 0; i < corrected_corners.size(); i++) 
				{
					solvePnPRansac(corners_3d, corrected_corners_f[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
					//Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
					projectPoints(axis_pyramid, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, pyramidProjected[i]); 
					/*
					//int a1x = (imPointsProjected[i][3].x - imPointsProjected[i][0].x);
					//int a1y = (imPointsProjected[i][3].y - imPointsProjected[i][0].y);
					
					Point p1_start = corrected_corners[i][(rotations[i] + 3) % 4];
					Point p2_start = corrected_corners[i][(rotations[i] + 2) % 4];
					Point p3_start = (p1_start + (corrected_corners[i][(rotations[i] + 1) % 4])) / 2;
					Point p4_start = (p2_start + (corrected_corners[i][(rotations[i] + 0) % 4])) / 2;
					Point p1_end(p1_start.x + a1x, p1_start.y + a1y);
					Point p2_end(p2_start.x + a1x, p2_start.y + a1y);
					Point p3_end(p3_start.x + a1x, p3_start.y + a1y);
					Point p4_end(p4_start.x + a1x, p4_start.y + a1y);
					//Base pirámide
					
					line(frame, p1_start, p2_start, blue, 2);
					line(frame, p1_start, p1_end,   blue, 2);
					line(frame, p2_start, p2_end,   blue, 2);
					line(frame, p1_end,   p2_end,   blue, 2);

					int center_x = (p3_start.x + p4_start.x) / 2;
					int center_y = (p3_start.y + p3_end.y) / 2;
					
					Point tip(center_x, center_y);
					line(frame, p1_start, tip, blue, 2);
					line(frame, p1_end, tip, blue, 2);
					line(frame, p2_start, tip, blue, 2);
					line(frame, p2_end, tip, blue, 2);
					*/

					line(frame, pyramidProjected[i][0], pyramidProjected[i][1], green, 2);
					line(frame, pyramidProjected[i][1], pyramidProjected[i][2], green, 2);
					line(frame, pyramidProjected[i][2], pyramidProjected[i][3], green, 2);
					line(frame, pyramidProjected[i][3], pyramidProjected[i][0], green, 2);
					line(frame, pyramidProjected[i][0], pyramidProjected[i][4], green, 2);
					line(frame, pyramidProjected[i][1], pyramidProjected[i][4], green, 2);
					line(frame, pyramidProjected[i][2], pyramidProjected[i][4], green, 2);
					line(frame, pyramidProjected[i][3], pyramidProjected[i][4], green, 2);

				}
				rvecs.clear();
				tvecs.clear();
				pyramidProjected.clear();
			}
		}
		outputVideo.write(frame);
		
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
		//Se espera 10 ms
		pressedKey = waitKey(10);
		//Se crea un lienzo donde mostrar imagenes
		namedWindow("Video", WINDOW_AUTOSIZE);
		//Mostrar frame
		imshow("Video", frame);

	}

	//Mostrar las matrices de las 4 rotaciones de los marcadores ArUco a detectar
	/*
	int count_matrix =0;
	for (int i = 0; i < aruco_images_matrixs.size(); i++) {
		for (int j = 0; j < aruco_images_matrixs[i].size(); j++) {
			for (int k = 0; k < aruco_images_matrixs[i][j].size(); k++) {
				int element = aruco_images_matrixs[i][j][k];
				cout << element << " ";
			}
			cout << endl ;
		}
		cout << endl << endl ;
		count_matrix++;
	}
	cout << "\n" << count_matrix << " matrices.";
	*/
	
	//Liberar el objeto videoCapture
	captura.release();
	//Cerrar archivo
	outputVideo.release();
	//Destruir todas las ventanas
	destroyAllWindows();
}