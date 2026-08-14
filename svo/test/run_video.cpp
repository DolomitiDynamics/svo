// Standalone tool to run SVO on an arbitrary video file using an OpenCV-style
// camera calibration YAML file (as produced by cv::calibrateCamera /
// opencv_interactive-calibration), and print the estimated pose per frame.
//
// Usage:
//   run_video <video_file> <calibration.yml> [start_frame] [--set name=value ...]
//
// svo::Config (include/svo/config.h) is a singleton read by every stage of
// the pipeline (DepthFilter, Matcher, Reprojector, PoseOptimizer, ...) via
// static accessors such as Config::gridSize(). Those accessors return
// mutable references, so tuning a parameter is just a matter of assigning to
// them before the FrameHandlerMono is constructed. --set exposes that here
// on the command line instead of requiring a recompile.

#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
#include <opencv2/opencv.hpp>
#include <vikit/pinhole_camera.h>
#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame.h>
#include <svo/map.h>

namespace {

/// Maps the name used on the command line to a setter that parses the string
/// value and assigns it to the corresponding svo::Config field.
const std::unordered_map<std::string, std::function<void(const std::string&)>>&
configSetters()
{
  static const std::unordered_map<std::string, std::function<void(const std::string&)>> setters = {
    {"n_pyr_levels",            [](const std::string& v){ svo::Config::nPyrLevels() = std::stoul(v); }},
    {"core_n_kfs",               [](const std::string& v){ svo::Config::coreNKfs() = std::stoul(v); }},
    {"map_scale",                [](const std::string& v){ svo::Config::mapScale() = std::stod(v); }},
    {"grid_size",                [](const std::string& v){ svo::Config::gridSize() = std::stoul(v); }},
    {"init_min_disparity",       [](const std::string& v){ svo::Config::initMinDisparity() = std::stod(v); }},
    {"init_min_tracked",         [](const std::string& v){ svo::Config::initMinTracked() = std::stoul(v); }},
    {"init_min_inliers",         [](const std::string& v){ svo::Config::initMinInliers() = std::stoul(v); }},
    {"klt_max_level",            [](const std::string& v){ svo::Config::kltMaxLevel() = std::stoul(v); }},
    {"klt_min_level",            [](const std::string& v){ svo::Config::kltMinLevel() = std::stoul(v); }},
    {"reproj_thresh",            [](const std::string& v){ svo::Config::reprojThresh() = std::stod(v); }},
    {"poseoptim_thresh",         [](const std::string& v){ svo::Config::poseOptimThresh() = std::stod(v); }},
    {"poseoptim_num_iter",       [](const std::string& v){ svo::Config::poseOptimNumIter() = std::stoul(v); }},
    {"structureoptim_max_pts",   [](const std::string& v){ svo::Config::structureOptimMaxPts() = std::stoul(v); }},
    {"structureoptim_num_iter",  [](const std::string& v){ svo::Config::structureOptimNumIter() = std::stoul(v); }},
    {"loba_thresh",              [](const std::string& v){ svo::Config::lobaThresh() = std::stod(v); }},
    {"loba_robust_huber_width",  [](const std::string& v){ svo::Config::lobaRobustHuberWidth() = std::stod(v); }},
    {"loba_num_iter",            [](const std::string& v){ svo::Config::lobaNumIter() = std::stoul(v); }},
    {"kfselect_mindist",         [](const std::string& v){ svo::Config::kfSelectMinDist() = std::stod(v); }},
    {"triang_min_corner_score",  [](const std::string& v){ svo::Config::triangMinCornerScore() = std::stod(v); }},
    {"subpix_n_iter",            [](const std::string& v){ svo::Config::subpixNIter() = std::stoul(v); }},
    {"max_n_kfs",                [](const std::string& v){ svo::Config::maxNKfs() = std::stoul(v); }},
    {"img_imu_delay",            [](const std::string& v){ svo::Config::imgImuDelay() = std::stod(v); }},
    {"max_fts",                  [](const std::string& v){ svo::Config::maxFts() = std::stoul(v); }},
    {"quality_min_fts",          [](const std::string& v){ svo::Config::qualityMinFts() = std::stoul(v); }},
    {"quality_max_drop_fts",     [](const std::string& v){ svo::Config::qualityMaxFtsDrop() = std::stoi(v); }},
  };
  return setters;
}

/// Parses "name=value" and applies it via configSetters(). Throws on unknown
/// name or malformed value.
void applyConfigSetting(const std::string& arg)
{
  const auto eq = arg.find('=');
  if(eq == std::string::npos)
    throw std::runtime_error("--set expects name=value, got: " + arg);
  const std::string name = arg.substr(0, eq);
  const std::string value = arg.substr(eq + 1);

  const auto& setters = configSetters();
  const auto it = setters.find(name);
  if(it == setters.end())
    throw std::runtime_error("Unknown config parameter: " + name);
  it->second(value);
  std::cout << "Config: " << name << " = " << value << "\n";
}

vk::PinholeCamera* loadCameraFromYaml(const std::string& path, int& cam_width, int& cam_height)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if(!fs.isOpened())
    throw std::runtime_error("Could not open calibration file: " + path);

  cv::Mat camera_matrix, distortion_coefficients;
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> distortion_coefficients;
  int image_width = static_cast<int>(fs["image_width"]);
  int image_height = static_cast<int>(fs["image_height"]);
  fs.release();

  if(camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3)
    throw std::runtime_error("Calibration file is missing a valid 3x3 camera_matrix");
  if(image_width <= 0 || image_height <= 0)
    throw std::runtime_error("Calibration file is missing valid image_width/image_height");

  double fx = camera_matrix.at<double>(0, 0);
  double fy = camera_matrix.at<double>(1, 1);
  double cx = camera_matrix.at<double>(0, 2);
  double cy = camera_matrix.at<double>(1, 2);

  double d[5] = {0, 0, 0, 0, 0};
  if(!distortion_coefficients.empty())
  {
    int n = std::min(5, distortion_coefficients.rows * distortion_coefficients.cols);
    for(int i = 0; i < n; ++i)
      d[i] = distortion_coefficients.at<double>(i);
  }

  cam_width = image_width;
  cam_height = image_height;

  std::cout << "Loaded camera calibration from " << path << ":\n"
            << "  image size: " << image_width << " x " << image_height << "\n"
            << "  fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << "\n"
            << "  distortion (k1,k2,p1,p2,k3): "
            << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << ", " << d[4] << "\n";

  return new vk::PinholeCamera(image_width, image_height, fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4]);
}

} // namespace

int main(int argc, char** argv)
{
  if(argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <video_file> <calibration.yml> [start_frame] [--set name=value ...]\n";
    return 1;
  }
  const std::string video_path = argv[1];
  const std::string calib_path = argv[2];
  int start_frame = 0;

  for(int i = 3; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if(arg == "--set")
    {
      if(i + 1 >= argc)
      {
        std::cerr << "--set requires a name=value argument\n";
        return 1;
      }
      applyConfigSetting(argv[++i]);
    }
    else
    {
      // Backwards-compatible positional start_frame.
      start_frame = std::atoi(arg.c_str());
    }
  }

  int cam_width = 0, cam_height = 0;
  vk::PinholeCamera* cam = loadCameraFromYaml(calib_path, cam_width, cam_height);

  cv::VideoCapture cap(video_path);
  if(!cap.isOpened())
  {
    std::cerr << "Could not open video file: " << video_path << std::endl;
    delete cam;
    return 1;
  }

  const int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if(fps <= 1e-3)
    fps = 30.0;

  std::cout << "Video: " << video_path << " (" << video_width << "x" << video_height
            << " @ " << fps << " fps)\n";

  if(video_width != cam_width || video_height != cam_height)
  {
    std::cerr << "WARNING: video resolution (" << video_width << "x" << video_height
              << ") does not match calibration resolution (" << cam_width << "x" << cam_height
              << "). Frames will be resized to the calibration resolution, but the intrinsics\n"
              << "         (fx, fy, cx, cy) were only calibrated for the resolution above, so\n"
              << "         results may be inaccurate. Recalibrate at the video resolution for\n"
              << "         reliable results.\n";
  }

  if(start_frame > 0)
    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);

  svo::FrameHandlerMono vo(cam);
  vo.start();

  cv::Mat frame_bgr, frame_gray;
  int img_id = start_frame;
  while(cap.read(frame_bgr))
  {
    if(frame_bgr.size() != cv::Size(cam_width, cam_height))
      cv::resize(frame_bgr, frame_bgr, cv::Size(cam_width, cam_height));

    if(frame_bgr.channels() == 3)
      cv::cvtColor(frame_bgr, frame_gray, cv::COLOR_BGR2GRAY);
    else
      frame_gray = frame_bgr;

    const double timestamp = img_id / fps;
    vo.addImage(frame_gray, timestamp);

    if(vo.lastFrame() != NULL)
    {
      const Sophus::SE3& T_f_w = vo.lastFrame()->T_f_w_;
      const Eigen::Vector3d twf = T_f_w.inverse().translation();
      const Eigen::Quaterniond q(T_f_w.unit_quaternion());
      std::cout << "frame=" << img_id
                << " t=" << timestamp
                << " stage=" << vo.stage()
                << " nobs=" << vo.lastNumObservations()
                << " time_ms=" << vo.lastProcessingTime()*1000
                << " pos=[" << twf.x() << ", " << twf.y() << ", " << twf.z() << "]"
                << " quat=[" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]"
                << std::endl;
    }
    else
    {
      std::cout << "frame=" << img_id << " t=" << timestamp << " (not yet tracking)" << std::endl;
    }

    ++img_id;
  }

  delete cam;
  std::cout << "Done. Processed " << (img_id - start_frame) << " frames." << std::endl;
  return 0;
}
