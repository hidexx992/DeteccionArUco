#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <iomanip> // para usar std::setprecision
#include <fstream>
using namespace cv;
using namespace std;


//tam_imagen ={1280, 720};
int main()
{
    //Dimensiones del tablero de ajedrez
    int tablero[2] = {6, 9};
    //Vector que guarda los puntos del mundo real
    vector <vector<Point3f>> objpoints = {};
    //Vector para guardar los puntos de las imágenes del tablero
    vector <vector<Point2f>>  imgpoints = {};
    //Puntos del tablero
    vector<Point3f> objp = {};

    for (int i=0 ; i < tablero[1]; i++)
    {
        for (int j = 0; j < tablero[0]; j++)
        {
            objp.push_back(Point3f((float)j, (float)i, 0.0));
        }
    }

    // Extrayendo la ruta de la imagen individual guardado en un directorio determinado
    vector<String>rutaImagenes;
    // Ruta de la carpeta que contiene las imágenes del tablero
    string path = "./imagenes/CameraCalibration/*.jpg";
    glob(path, rutaImagenes);
    Mat imagen, gray;
    //Vector que guarda los puntos de las esquinas del tablero 
    vector<Point2f> corner_pts;
    bool success;
    //Recorremos todas las imágenes del directorio
    for (int i = 0; i < rutaImagenes.size(); i++)
    {
        imagen = imread(rutaImagenes[i]);
        if (!imagen.data) 
        {
            cout << "Error al cargar la imagen: " << rutaImagenes[i] << endl;
            exit(1);
        }
        cvtColor(imagen, gray, COLOR_BGR2GRAY);
        
        //Encontramos las esquinas del tablero
        //Si el número deseado de corners son encontrados en la imagen entonces entonces success = true  
        success = cv::findChessboardCorners(gray, Size(tablero[0],
            tablero[1]), corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        
        /* Si el número deseado de esquinas son detectados, refinamosn las coordenadas del pixel
         * y las mostramos en las imágenes del tablero
         */


        if (success)
        {
            //Criterio con las que vamos a definir esquinas
            TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001); 
            //Refinando las coordenadas del pixel para puntos 2D determinados.
            cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);
            //Mostrando las esquinas en el tablero de ajedrez
            drawChessboardCorners(imagen, Size(tablero[0], tablero[1]), corner_pts, success);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        imshow("Imagen", imagen);
        waitKey(0);
    }
    
    destroyAllWindows();

    Mat cameraMatrix, distCoeffs, R, T;
    /* Realizando la calibración de la cámara pasando los valores de los puntos 3D conocidos 
     * (objpoints) y las correspondientes coordenadas de píxeles de las esqunas detectadas (imgpoints).
     */
    calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
    cout << "cameraMatrix : " << cameraMatrix << endl;
    cout << "distCoeffs : " << distCoeffs << endl;
    cout << "Rotation vector : " << R << endl;
    cout << "Translation vector : " << T << endl;

    // Guardar matriz de cámara y coeficientes de distorsión en archivo de texto 
    vector <double> camera_matrix_f;
    vector <double> dist_coeffs_f;

    if (cameraMatrix.isContinuous())
    {
        camera_matrix_f.assign((double*)cameraMatrix.data, (double*)cameraMatrix.data + cameraMatrix.total());
    }
    else
    {
        for (int i = 0; i < cameraMatrix.rows; ++i)
        {
            camera_matrix_f.insert(camera_matrix_f.end(), cameraMatrix.ptr<double>(i), cameraMatrix.ptr<double>(i) + cameraMatrix.cols);
        }
    }

    if (distCoeffs.isContinuous())
    {
        dist_coeffs_f.assign((double*)distCoeffs.data, (double*)distCoeffs.data + distCoeffs.total());
    }
    else
    {
        for (int i = 0; i < distCoeffs.rows; ++i)
        {
            dist_coeffs_f.insert(dist_coeffs_f.end(), distCoeffs.ptr<double>(i), distCoeffs.ptr<double>(i) + distCoeffs.cols);
        }
    }
    // Crear un archivo de salida
    ofstream archivo("./ficheros/camera_calibration_parameters_v2.txt");
    archivo << std::fixed << std::setprecision(6);
    // Escribir los valores de la matriz en la primera línea del archivo
    for (int i = 0; i < camera_matrix_f.size(); i++)
    {
        if (i == (camera_matrix_f.size() - 1))
            archivo << camera_matrix_f[i];
        else
            archivo << camera_matrix_f[i] << " ";
    }
    archivo << "\n";
    for (int i = 0; i < dist_coeffs_f.size(); i++)
    {
        if (i == (dist_coeffs_f.size() - 1))
            archivo << dist_coeffs_f[i];
        else
            archivo << dist_coeffs_f[i] << " ";
    }
    // Cerrar el archivo
    archivo.close();
    return 0;   
}
