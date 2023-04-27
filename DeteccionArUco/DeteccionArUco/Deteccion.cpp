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

#include <iostream>
#include <fstream>

using namespace std;



vector<Point> ordenarPuntos(vector<Point> approx) {
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

//No funciona
void trackPuntos(vector<Point>& corner, int rotation) {
	switch (rotation) {
	case(1):
		swap(corner[0], corner[1]); //0 -> 1
		swap(corner[0], corner[2]);	//2 -> 3
		swap(corner[0], corner[3]);	//0 -> 2
		break;

	case(2):
		swap(corner[0], corner[2]); //0 -> 2
		swap(corner[1], corner[3]);	//1 -> 3
		break;

	case(3):
		swap(corner[0], corner[3]);
		swap(corner[1], corner[3]);
		swap(corner[2], corner[3]);
		break;
	}

}

//No funciona
void trackPuntos_f(vector<Point2f> & corner, int rotation) {
	switch (rotation) {
	case(1):
		swap(corner[0], corner[1]); //0 -> 1
		swap(corner[0], corner[2]);	//2 -> 3
		swap(corner[0], corner[3]);	//0 -> 2
		break;

	case(2):
		swap(corner[0], corner[2]); //0 -> 2
		swap(corner[1], corner[3]);	//1 -> 3
		break;

	case(3):
		swap(corner[0], corner[3]);//0->3
		swap(corner[1], corner[3]);
		swap(corner[2], corner[3]);
		break;
	default:
		break;
	}
}




void drawCube(Mat& img, vector<Point2f>& corners, vector<Point2f>& imgpts) {
	imgpts = Mat(imgpts).reshape(2);
	vector<vector<Point>> contour = { vector<Point>{imgpts[0], imgpts[1], imgpts[2], imgpts[3]} };
	drawContours(img, contour, 0, Scalar(0, 255, 0), 3);
	for (int i = 0; i < 4; i++) {
		line(img, imgpts[i], imgpts[(i + 4) % 8], Scalar(255), 3);
	}
	contour = { vector<Point>{imgpts[4], imgpts[5], imgpts[6], imgpts[7]} };
	drawContours(img, contour, 0, Scalar(0, 0, 255), 3);
}

void cargar_kps_descs(vector<vector<KeyPoint>> & keypointsRefs, vector<Mat> & descriptorsRefs) {
	keypointsRefs.resize(10);
	descriptorsRefs.resize(10);
	// Detección de características y obtención de descriptores
	Ptr<Feature2D> detector = AKAZE::create();
	for (int i = 0; i < 10; i++) {
		string filename = "./imagenes/MarcadoresAruco/4x4_1000-";
		filename += to_string(i);
		filename += ".png";
		Mat aruco_marker = imread(filename, IMREAD_GRAYSCALE);
		resize(aruco_marker, aruco_marker,Size(), 0.5, 0.5, INTER_AREA);
		detector->detectAndCompute(aruco_marker, Mat(), keypointsRefs[i], descriptorsRefs[i]);
	}

	
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
void sortCorners(vector<Point2f>& corners) {
	vector<Point2f> top, bottom;
	Point2f center(0, 0);

	// Encuentra el centroide del rectángulo
	for (int i = 0; i < corners.size(); i++) {
		center += corners[i];
	}
	center *= (1. / corners.size());

	// Separa las esquinas superiores e inferiores basándose en su posición relativa al centroide
	for (int i = 0; i < corners.size(); i++) {
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bottom.push_back(corners[i]);
	}

	// Ordena las esquinas superiores e inferiores por coordenadas x
	sort(top.begin(), top.end(), [](Point2f a, Point2f b) { return a.x < b.x; });
	sort(bottom.begin(), bottom.end(), [](Point2f a, Point2f b) { return a.x < b.x; });

	// Ordena las esquinas en sentido horario o antihorario, dependiendo de la ubicación de la esquina superior izquierda
	if (top.size() == 2 && bottom.size() == 2) {
		if (top[1].x < bottom[0].x)
			swap(top[1], bottom[0]);
		corners.clear();
		corners.push_back(top[0]);
		corners.push_back(top[1]);
		corners.push_back(bottom[1]);
		corners.push_back(bottom[0]);
	}
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
	/*Obtener matriz de cámara y coeficientes de */
	Mat cameraMatrix, distCoeffs;
	getCamMatrix_and_distCoeff(ruta_fichero, cameraMatrix, distCoeffs);
	
	//Descriptores y keypoints
	//vector<vector<KeyPoint>> keypointsRefs;
	//vector<Mat> descriptorsRefs;
	//cargar_kps_descs(keypointsRefs, descriptorsRefs);
	

	vector<Point2f> aruco_corners
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
	//corners_3d.push_back(Point3f(0, 10, 0));
	//corners_3d.push_back(Point3f(0, 0, 0));
	//corners_3d.push_back(Point3f(10, 0, 0));
	//corners_3d.push_back(Point3f(10, 10, 0));
	corners_3d.push_back(Point3f(-5, 5, 0));
	corners_3d.push_back(Point3f(5, 5, 0));
	corners_3d.push_back(Point3f(5, -5, 0));
	corners_3d.push_back(Point3f(-5, -5, 0));

	vector<Point3f> axis_cube{ //Dibujo cubo
		
		Point3f(0, 0, 5),
		Point3f(0, 5, 5),
		Point3f(5, 5, 5),
		Point3f(5, 0, 5),
		Point3f(0, 0, 0),
		Point3f(0, 5, 0),
		Point3f(5, 5, 0),
		Point3f(5, 0, 0)
		
		/*
		Point3f(-5,-5, 5),
		Point3f(-5, 5, 5),
		Point3f(5, 5, 5),
		Point3f(5, -5, 5),

		Point3f(-5, -5, 0),
		Point3f(-5, 5, 0),
		Point3f(5, 5, 0),
		Point3f(5, -5, 0)*/
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
		imshow("Thresh", thresh);
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
		//Corregir esquinas
		vector<vector<Point> > corrected_corners; //las esquinas de todos los marcadores
		vector<vector<Point2f> > corrected_corners_f; //las esquinas de todos los marcadores
		//Cuando el tamaño del marcador es 4x4, el tamaño incluyendo el borde negro es 6x6
		//Si el ancho de píxel de una celda se establece en 10 al dividir la imagen en una cuadrícula en un paso posterior
		//La longitud de un lado de la imagen del marcador es 60	
		vector<Point2f> square_points;
		int marker_image_side_length = 60;
		square_points.push_back(Point2f(0, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, 0));
		square_points.push_back(Point2f(marker_image_side_length - 1, marker_image_side_length - 1));
		square_points.push_back(Point2f(0, marker_image_side_length - 1));		
		
		

		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			if ((area > 1700) && (area > 3700)) {
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
					
					//corners.push_back(approx);
					//corners_f.push_back(approx_f));
					

				}
			}
		}
		
	
		//cout << "Esquinas" << corners.size() << endl;
		//cout << "Esquinas flotantes" << corners_f.size() << endl;

		vector<Mat> result_markers(corners.size());
		vector<int> result_ids;
		vector<vector<vector<int>>> result_matrixs; 
		for (int i = 0; i < corners.size(); i++)
		{
			Mat result_marker;
			vector<vector<int>>result_matrix;
			vector<Point2f> m = corners_f[i];

			//Obtenga una matriz de transformación de perspectiva para transformar el marcador en un rectángulo.
			Mat PerspectiveTransformMatrix = getPerspectiveTransform(m, square_points);
			
			//Aplicar transformación de perspectiva. 
			warpPerspective(gray, result_marker, PerspectiveTransformMatrix,
				Size(marker_image_side_length, marker_image_side_length));
			
			//Aplicar la binarización por el método otsu.
			threshold(result_marker, result_marker, 127, 255, THRESH_BINARY | THRESH_OTSU);
			// Definimos el kernel
			Mat kernel = Mat::ones(6, 6, CV_8UC1);
			// Aplicamos la operación de erosión
			erode(result_marker, result_marker, kernel);

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
					Mat cell = result_marker(Rect(cellX, cellY, cellSize, cellSize)); //Celdas negras
					
					int cellValue = cv::sum(cell)[0]; // suma de los valores de píxeles en la celda
					int msbValue = (cellValue >> 7) & 1; // valor del bit más significativo
					result_vector.push_back(msbValue);					
					imshow("Celda", cell);
					
				}
				result_matrix.push_back(result_vector);
				result_vector.clear();
			}
			result_matrixs.push_back(result_matrix);
			result_markers.push_back(result_marker);
			result_matrix = bordesNegros(result_matrix);
			int id = determinarID(aruco_images_matrixs, result_matrix, rotation);
			cout << "ID = " << id  <<endl;
			

			if (id != -1) {
				

				//vector<KeyPoint> kps;
				//Mat descs;
				//Ptr<Feature2D> detector = AKAZE::create();
				//Mat img_gray; 
			    //cvtColor(frame, img_gray, IMREAD_GRAYSCALE);
				//detector->detectAndCompute(img_gray, Mat(), kps, descs);
				//cout << descs.size() << endl;
				//cout << descriptorsRefs[id].size() << endl;
				
				//vector<Point2f> ref_corners_f = refinedCorners( descs, descriptorsRefs[id], kps, keypointsRefs[id], aruco_images[id]);
				
				//corrected_corners_f.push_back(ref_corners_f);
				//corrected_corners.push_back(pts2f_to_pts(ref_corners_f));
				//corrected_corners.push_back(pts2f_to_pts(ref_corners_f));
				
				//La homografía es fácilmente con: 
				// remove perspective
					
				// remove perspective
				/*
				Mat result_marker; // marker image after removing perspective
				int resultImgSize = 591;
				Mat resultMarkerCorners(4, 1, CV_32FC2);
				resultMarkerCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
				resultMarkerCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
				resultMarkerCorners.ptr<Point2f>(0)[2] =
					Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
				resultMarkerCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);
				Mat transformation = getPerspectiveTransform(aruco_corners, resultMarkerCorners);
				warpPerspective(frame, result_marker, transformation, Size(resultImgSize, resultImgSize),
					INTER_NEAREST);
				namedWindow("result", WINDOW_AUTOSIZE);
				imshow("result", result_marker);
				*/

				//Mat H = findHomography(aruco_corners_f, corners_f[i]);
				// Usar la función perspectiveTransform() para transformar el punto
				//vector<Point2f> transf_points;
				//perspectiveTransform(aruco_corners_f, transf_points, H);
				// Convertir el Mat a un vector de Point2f
				//std::vector<cv::Point2f> points;
				// Crear un Mat de tamaño 2x2 y tipo CV_32FC1
				
				//circle(frame, points[0], 5, Scalar(242, 159, 90), -1);
				//Mat frame_warp;
				//int frame_warp_size = 60;
				//Mat transformation = getPerspectiveTransform(corners[i], corners[i]);
				//warpPerspective(frame, frame_warp, transformation,  Size(frame_warp_size, frame_warp_size), INTER_NEAREST);
				//imshow("Frame warp", frame_warp);
				



				// Convertir Mat a vector de Point
				//vector<Point> points(mat.rows);
				//memcpy(points.data(), mat.data, mat.rows * mat.cols * sizeof(int));

				
				// Convertir el vector de puntos en un vector de puntos flotantes
				//vector<Point2f> puntos_flotantes;
				//cv::Mat(rotated_corners).convertTo(puntos_flotantes, CV_32F);
				trackPuntos_f(corners_f[i], rotation);
				trackPuntos(corners[i], rotation);
				corrected_corners.push_back(corners[i]);
				corrected_corners_f.push_back(corners_f[i]);
				result_ids.push_back(id);

			}
			result_matrix.clear();
		}
		// Calcular homografía entre esquinas del objeto fijo y objeto girado
	

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
			solvePnPRansac(corners_3d, corrected_corners_f[0], cameraMatrix, distCoeffs, rvecs.at(0), tvecs.at(0));
			// Proyectar los puntos del cubo en 3D a la imagen en 2D utilizando projectPoints
			vector<Point2f>  imPointsProjected;
			projectPoints(axis, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, imPointsProjected);
			

			// Dibujar las líneas del cubo en la imagen
			/////////////////////////////////777
			/*
			cv::Scalar green(0, 255, 0);
			cv::Scalar red(0, 0, 255);
			Scalar violeta(242, 90, 232);
			cv::line(frame, imPointsProjected[0], imPointsProjected[1], green, 2);
			cv::line(frame, imPointsProjected[1], imPointsProjected[2], green, 2);
			cv::line(frame, imPointsProjected[2], imPointsProjected[3], green, 2);
			cv::line(frame, imPointsProjected[3], imPointsProjected[0], green, 2);
			cv::line(frame, imPointsProjected[0], imPointsProjected[4], red, 2);
			cv::line(frame, imPointsProjected[1], imPointsProjected[5], red, 2);
			cv::line(frame, imPointsProjected[2], imPointsProjected[6], red, 2);
			cv::line(frame, imPointsProjected[3], imPointsProjected[7], red, 2);
			cv::line(frame, imPointsProjected[4], imPointsProjected[5], green, 2);
			cv::line(frame, imPointsProjected[5], imPointsProjected[6], green, 2);
			cv::line(frame, imPointsProjected[6], imPointsProjected[7], green, 2);
			cv::line(frame, imPointsProjected[7], imPointsProjected[4], green, 2);
			*/
			//Dibujar ejes
			ordenarPuntos_f(imPointsProjected);
			line(frame , imPointsProjected[0], imPointsProjected[1], cv::Scalar(0, 0, 255), 2);    // eje X en rojo
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




	
