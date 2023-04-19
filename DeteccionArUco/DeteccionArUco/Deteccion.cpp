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
const int n_images = 10;
const int MIN_MATCH_COUNT = 10;

using namespace cv;
using namespace std;






vector<Mat> cargarImagenes() {
	vector<Mat> aruco_images;
	for (int i = 0; i < n_images; i++) {
		string filename = "./imagenes/MarcadoresAruco/4x4_1000_";
		filename += to_string(i);
		filename += ".png";
		cout << "Leyendo " << filename << endl;
		Mat aruco_marker = imread(filename, IMREAD_GRAYSCALE);
		if (aruco_marker.empty()) {
			cerr << "Error: no se pudo leer la imagen " << filename << endl;
			exit(1);
		}
		aruco_images.push_back(aruco_marker);
	}
	return aruco_images;
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
/*
void homography(Mat frame_desc, vector<KeyPoint> frame_kps, Mat frame_image, Mat aruco_desc, 
	vector<KeyPoint> aruco_kps, Mat aruco_image) 
{
	//Emparejar los descriptores 
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> matches;
	matcher->match(aruco_desc, frame_desc, matches);
	double min_dist = 10000;
	double max_dist = 0;


	// Seleccionar las mejores correspondencias (por distancia)
	for (int i = 0; i < aruco_desc.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "Max dist: " << max_dist << endl;
	cout << "Min dist: " << min_dist << endl;
	

	vector<DMatch> good_matches;
	for (int i = 0; i < aruco_desc.rows; i++) {
		if (matches[i].distance < 2 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(aruco_image, aruco_kps, frame_image, frame_kps, good_matches, 
	img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), 
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
	//-- Paso 5: Localizamos el objeto
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (int i = 0; i < good_matches.size(); i++) {
		//-- Get the keypoints from the good matches
		obj.push_back(aruco_kps[good_matches[i].queryIdx].pt);
		scene.push_back(frame_kps[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC, 5.0);
	
	//-- Tomamos las esquinas d ela imagen del objeto
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); 
	obj_corners[1] = Point(aruco_image.cols, 0);
	obj_corners[2] = Point(aruco_image.cols, aruco_image.rows); 
	obj_corners[3] = Point(0, aruco_image.rows);
	vector<Point2f> scene_corners(4);
	// Proyectamos usando la homografia
	perspectiveTransform(obj_corners, scene_corners, H);
	
	//-- Dibujamos las lineas (hay q trasladar Point2f(img_object.cols, 0) a la dcha

	line(img_matches, scene_corners[0] + Point2f(aruco_image.cols, 0), scene_corners[1] + Point2f(aruco_image.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(aruco_image.cols, 0), scene_corners[2] + Point2f(aruco_image.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(aruco_image.cols, 0), scene_corners[3] + Point2f(aruco_image.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(aruco_image.cols, 0), scene_corners[0] + Point2f(aruco_image.cols, 0), Scalar(0, 255, 0), 4);

	//-- Mostramos la deteccion
	namedWindow("Emparejamientos & Deteccion", WINDOW_NORMAL); // Crear ventana
	imshow("Emparejamientos & Deteccion", img_matches);


	resizeWindow("Emparejamientos & Deteccion", 640, 480); // Cambiar tamaño de la ventana
	cout << "good_matches " << good_matches.size() << endl;

	// Vamos a ver cuales son los q han deterninado la homografia
	vector< DMatch > homografy_matches;
	for (int i = 0; i < good_matches.size(); i++) {
		Point2f pto_objeto, pto_imagen, proyeccion_imagen;
		pto_objeto = aruco_kps[good_matches[i].queryIdx].pt;
		pto_imagen = frame_kps[good_matches[i].trainIdx].pt;

		// La homografia multiplica una matriz de 3x3 por un vector 3
		Mat punto = Mat(3, 1, CV_64F);
		punto.at<double>(0) = pto_objeto.x;
		punto.at<double>(1) = pto_objeto.y;
		punto.at<double>(2) = 1.0;
		// Proyectamos
		punto = H * punto;
		punto /= punto.at<double>(2);

		proyeccion_imagen.x = punto.at<double>(0);
		proyeccion_imagen.y = punto.at<double>(1);

		float dist = sqrt(pow(pto_imagen.x - proyeccion_imagen.x, 2) + pow(pto_imagen.y - proyeccion_imagen.y, 2));

		if (dist < 3) homografy_matches.push_back(good_matches[i]);
	}
	cout << "Homografia " << homografy_matches.size() << endl;

	Mat img_homografia;
	drawMatches(aruco_image, aruco_kps, frame_image, frame_kps, homografy_matches, img_homografia, 
		Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("homografia", WINDOW_NORMAL); // Crear ventana
	imshow("homografia", img_homografia);
	
}*/

void convertImgToMat(Mat aruco_img, Mat mat) {
	int cell_size = aruco_img.rows / 6;
	for (int i = 1; i < 5; i++) {
		for (int j = 1; j < 5; j++) {
			int centerX = cell_size * i + cell_size / 2;
			int centerY = cell_size * j + cell_size / 2;
			cout << "(" << centerX << ", " << centerY << ") = " << aruco_img.at<int>(centerX, centerY) << endl;
			if (aruco_img.at<int>(centerX, centerY) > 128) {
				mat.at<int>(i - 1, j - 1) = 1;
			}
			else {
				mat.at<int>(i - 1, j - 1) = 0;
			}
		}
	}
}

void convertImagesToMat(vector<Mat> arucoMats) {
	for (int i = 0; i < 1; i++) {
		String filename = "./imagenes/MarcadoresAruco/4x4_1000-" + to_string(i) + ".png";
		Mat aruco_img = imread(filename, IMREAD_GRAYSCALE);
		Mat mat = Mat(4, 4, CV_32SC1);
		cout << "Image: " << i << endl;
		convertImgToMat(aruco_img, mat);
		cout << mat << endl;
		arucoMats.push_back(mat);
	}
}




void determineId(Mat resultImg) {

}

/*
// Compare two images by getting the L2 error (square-root of sum of squared error).
void getSimilarity(const cv::Mat A, const cv::Mat B, int i) {
	Mat gray_A, gray_B;
	//Gris
	cvtColor(A, gray_A, COLOR_BGR2GRAY);
	cvtColor(A, gray_B, COLOR_BGR2GRAY);
	//Filtro Gaussiano
	GaussianBlur(gray_A, gray_A, Size(3, 3), 0, 0);
	GaussianBlur(gray_B, gray_B, Size(3, 3), 0, 0);
	//Umbralizacion
	Mat thresh_A, thresh_B;
	adaptiveThreshold(gray_A, thresh_A, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 31, 0);
	adaptiveThreshold(gray_B, thresh_B, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 31, 0);
	//Transformaciones morfologicas
	thresh_A = 255 - thresh_A;
	thresh_B = 255 - thresh_B;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat rect_A = thresh_A.clone();
	Mat rect_B = thresh_B.clone();
	morphologyEx(rect_A, rect_A, MORPH_OPEN, kernel);
	morphologyEx(rect_B, rect_B, MORPH_ERODE, kernel);

	if (rect_A.rows > 0 && rect_A.rows == rect_B.rows && rect_A.cols > 0 && rect_A.cols == rect_B.cols) {
		// Calculate the L2 relative error between images.
		double errorL2 = cv::norm(rect_A, rect_B, cv::NORM_L2);
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double similarity = errorL2 / static_cast<double>(rect_A.rows * rect_A.cols);
		cout << "Similitud" << similarity << endl;
		cout << "Similar a imagen " << i << endl;
	}
	else {
		// Images have a different size.
		cout << "No es similar con la imagen " << i << endl;
	}
	
}
*/

void deteccion(String ruta_video, String ruta_fichero) {

	/*Variables*/
	//Crear el objeto de tipo VideoCapture
	VideoCapture captura = VideoCapture(ruta_video);
	//Frames
	Mat frame;
	//Tecla presionada
	char pressedKey = 0;
	//Selección dibujo 
	int dibujo = 0;
	/*Leer las imágenes y etiquetas*/
	vector<Mat> aruco_images;
	vector<int> markerIds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	aruco_images = cargarImagenes();
	/*Obtener matriz de cámara y coeficientes de */
	Mat cameraMatrix, distCoeffs;
	getCamMatrix_and_distCoeff(ruta_fichero, cameraMatrix, distCoeffs);
	/*Descriptores*/
	Ptr<SIFT> detector = SIFT::create();
	vector <Mat> aruco_descriptors;
	vector<vector<KeyPoint>> aruco_keypoints;
	for (int i = 0; i < n_images; i++) {
		Mat desc;
		vector<KeyPoint> kp; 
		detector->detectAndCompute(aruco_images[i], Mat(), kp, desc);
		aruco_descriptors.push_back(desc);
		aruco_keypoints.push_back(kp);
	}
	
	

 
	if (!captura.isOpened()) {
		cout << "Error al cargar el video!" << endl;
		exit(1);
	}
	//Seguimiento del tiempo transcurrido para calcular FPS
	auto start = chrono::steady_clock::now();
	double fps;


	//vector<Mat> arucoMats;
	//convertImagesToMat(arucoMats);
	
	//Cuántas veces detecta si hay más 1 contorno
	int error = 0;

	while (1) {
		//Se lee el video imagen a imagen
		captura.read(frame); //capture >> frame;
		// Se comprueba que no se ha llegado al final
		if (frame.empty()) {
			cout << "Se ha llegado al final del video" << endl;
			break;
		}
		//Descriptores 
		vector<KeyPoint> frameKeypoints;
		Mat frameDescriptors;
		detector->detectAndCompute(frame, Mat(), frameKeypoints, frameDescriptors);
		


		//Conversion a Escala de Grises
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//Filtro Gaussiano
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		namedWindow("Escala de Grises con filtro", WINDOW_AUTOSIZE);
		imshow("Escala de Grises con filtro", gray);
		//Umbralizacion
		Mat thresh;
		adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 17, 0);
		//Transformaciones morfologicas
		thresh = 255 - thresh;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat rect = thresh.clone();
		morphologyEx(rect, rect, MORPH_OPEN, kernel);
		morphologyEx(rect, rect, MORPH_ERODE, kernel);
		imshow("Eliminar ruido - Imagen Binaria", rect);

		/////////////////////////
		std::vector<cv::Point3f> object_points;
		object_points.push_back(cv::Point3f(0, 0, 0));
		object_points.push_back(cv::Point3f(0, 591, 0));
		object_points.push_back(cv::Point3f(591, 0, 0));
		object_points.push_back(cv::Point3f(591, 591, 0));

		vector<Point3f> axis;
		axis.push_back(cv::Point3f(3, 0, 0));
		axis.push_back(cv::Point3f(0, 3, 0));
		axis.push_back(cv::Point3f(0, 0, -3));


		/////////////////////////
		
		//Encontrar contornos
		vector<vector<Point>> contours;
		findContours(rect, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//Inicializa un vector para almacenar las esquinas
		vector<vector<Point>> corners;
		//Cuantos contornos hay
		int total = 0;
		
		for (size_t i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			// cout << "area " << area << endl;
			if (area > 2700) {
				// Aproximación de contorno
				double peri = arcLength(contours[i], true); // Perimetro
				vector<Point> approx;
				vector<Point2f> approx_f;
				approxPolyDP(contours[i], approx_f, 0.02 * peri, true);
				approxPolyDP(contours[i], approx, 0.02 * peri, true);
				// Si el polígono tiene cuatro lados y es convexo, entonces es un cuadrado
				if ((approx.size() == 4) && (isContourConvex(approx) == true))
				{
					Mat resultImg; // marker image after removing perspective
					int resultImgSize = 500;
					Mat resultImgCorners(4, 1, CV_32FC2);
					resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
					resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
					resultImgCorners.ptr<Point2f>(0)[2] =
						Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
					resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);

					// remove perspective
					Mat transformation = getPerspectiveTransform(approx_f, resultImgCorners);
					warpPerspective(frame, resultImg, transformation, Size(resultImgSize, resultImgSize),
						INTER_NEAREST);
					namedWindow("result", WINDOW_AUTOSIZE);
					imshow("result", resultImg);
					drawContours(frame, vector<vector<Point>>{approx}, 0, Scalar(0, 255, 0), 3, LINE_AA);
					// Agrega las esquinas del cuadrado al vector de esquinas
					corners.push_back(approx);
					// Calcula el punto central del cuadrado
					Point center((approx[0].x + approx[2].x) / 2, (approx[0].y + approx[2].y) / 2);
					// Escribe un texto en el punto central del cuadrado
					String text = "ID =" + to_string(corners.size());
					putText(frame, text, center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
					total++; //Número de objetos detectados
					cout << "Contornos encontrados: " << total << endl;
				}
				if (total > 1) {
					error++;
				}
			}
		}
		
	

		cout << "Esquinas" << endl;
		for (vector<Point> corner : corners) {
			cout <<  corner  << endl;
		}
		imshow("THRESHOLD", thresh);
		/*Poner texto en imagen*/
		string letrero = "Objetos: ";// + to_string(total);
		putText(frame, letrero, Point(10, 150), FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2);
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
				if (dibujo == 2) {
					//dibujoPiramide();
					dibujo = 0;
				}
		}
	}
	cout << "Veces que han detectado mas de 1 contorno: " << error << endl;
	//Liberar el objeto videoCapture
	captura.release();
	//Destruir todas las ventanas
	destroyAllWindows();

}
	
	
