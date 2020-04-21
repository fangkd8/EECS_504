#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap 
#include <octomap/ColorOcTree.h>

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings

using namespace std;
using namespace Eigen;
using namespace octomap;

const int ROW = 375;
const int COL = 1242;

/*
  If you want to run KITTI odometry mapping, please edit 
  data_dir = [directory] to .bin files.
  calib_dir = [path to calib.txt]
  pose_dir = [path to ground truth]
*/

const string data_dir = "/home/fangkd/Desktop/dataset/06/velodyne/";
const string calib_dir = "../calib.txt";
const string pose_dir = "../data/06.txt";
const string save_dir = "../data/mapping.ot";

const bool fullMap = false;
const int partMap = 100;

// You may want to use provided data.
const bool defaultdata = true;

MatrixXf readbinfile(const string dir);

template <size_t Rows, size_t Cols>
Eigen::Matrix<float, Rows, Cols> ReadCalibrationVariable(
  const std::string& prefix, std::istream& file);

vector<Matrix4f> ReadPoses(const string filename);


int main(int argc, char const *argv[]){

  bool viewProjection = false;

  // KITTI intrinsic & extrinsic, using left image.
  std::ifstream calib_file(calib_dir);
  MatrixXf K;
  K.resize(3, 4);
  K = ReadCalibrationVariable<3, 4>("P0:", calib_file);
  Matrix4f T = Matrix4f::Identity();
  T.block(0, 0, 3, 4) = ReadCalibrationVariable<3, 4>("Tr:", calib_file);

  octomap::ColorOcTree tree(0.1);

  cv::Mat map, mD;
  
  vector<Matrix4f> v = ReadPoses(pose_dir);
  int size;
  if (fullMap && !defaultdata){
  	cout << "Mapping with whole dataset." << endl;
  	size = v.size();
  }
  else if (!fullMap && !defaultdata){
  	cout << "Mapping with " << partMap << " point clouds." << endl;
  	size = partMap;
  }
  else{
  	cout << "Mapping with given data" << endl;
  	size = 2;
  }

  for (int k = 0; k < size; k++){
    cout << "Processing " << k+1 << "-th frame." << endl;

    if (viewProjection)
      map = cv::Mat::zeros(ROW, COL, CV_32FC1);      

    char buff1[100];
    snprintf(buff1, sizeof(buff1), "%006d.bin", k);
    string file = data_dir + string(buff1);

    Matrix4f pose = v[k];
    MatrixXf data = readbinfile(file);
    MatrixXf camPts = K * T * data;
    octomap::Pointcloud cloud;

    for (int i = 0; i < camPts.cols(); i++){
      Vector4f point(data(0, i), data(1, i), data(2, i), 1);
      Vector4f pt = pose * T * point;
      if (camPts(2, i) > 0){
        float x = camPts(0, i) / camPts(2, i);
        float y = camPts(1, i) / camPts(2, i);
        if ( (x > 0 && x < COL - 0.5) && (y > 0 && y < ROW - 0.5) ){
          if (viewProjection)
            map.at<float>(round(y), round(x)) = 255;

          octomap::point3d endpoint(pt[0], pt[1], pt[2]);
          octomap::ColorOcTreeNode* n = tree.updateNode(endpoint, true);
          
          // Till now, only set them to the same color.
          n->setColor(0, 0, 255);
          if (defaultdata){
          	n->setColor(255*(k==0), 0, 255*(k==1));
          }
        }
  
      }
    }
    if (viewProjection){
      double minVal, maxVal;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(map, &minVal, &maxVal, &minLoc, &maxLoc);
      // cout << minVal << " " << maxVal << endl;
      map = 255 * (map - minVal) / (maxVal - minVal);
      map.convertTo(map, CV_8UC1);
      cv::imshow("result", map);
      cv::waitKey(0);    
    }
  }

  tree.updateInnerOccupancy();
  cout << "saving octomap ... " << endl;
  // tree.writeBinary("octomap.bt");
  tree.write(save_dir);
  return 0;
}

MatrixXf readbinfile(const string dir){

  ifstream fin(dir.c_str(), ios::binary);
  assert(fin);

  fin.seekg(0, ios::end);
  const size_t num_elements = fin.tellg() / sizeof(float);
  fin.seekg(0, ios::beg);

  vector<float> l_data(num_elements);
  fin.read(reinterpret_cast<char*>(&l_data[0]), num_elements*sizeof(float));

  MatrixXf data = Map<MatrixXf>(l_data.data(), 4, l_data.size()/4);

  return data;
}

template <size_t Rows, size_t Cols>
Eigen::Matrix<float, Rows, Cols> ReadCalibrationVariable(
    const std::string& prefix, std::istream& file) {
  // rewind
  file.seekg(0, std::ios::beg);
  file.clear();

  double buff[20] = {0};

  int itter = 0;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    size_t found = line.find(prefix);
    if (found != std::string::npos) {
      std::cout << prefix << std::endl;
      std::cout << line << std::endl;
      std::stringstream stream(
          line.substr(found + prefix.length(), std::string::npos));
      while (!stream.eof()) {
        stream >> buff[itter++];
      }
      std::cout << std::endl;
      break;
    }
  }

  Eigen::Matrix<float, Rows, Cols> results;
  for (size_t i = 0; i < Rows; i++) {
    for (size_t j = 0; j < Cols; j++) {
      results(i, j) = buff[Cols * i + j];
    }
  }
  return results;
}

vector<Matrix4f> ReadPoses(const string filename){
  vector<Matrix4f> poses;
  ifstream file(filename);

  std::string line;
  while (std::getline(file, line)) {
    double buff[20] = {0};
    stringstream stream(line);
    int itter = 0;
    while (!stream.eof()) {
      stream >> buff[itter++];
    }
    // break; 
    Matrix4f result = Matrix4f::Identity();
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        result(i, j) = buff[4 * i + j];
      }
    }
    poses.push_back(result);
  }

  return poses;
}
