// Standalone tool to run SVO on an arbitrary video file using an OpenCV-style
// camera calibration YAML file (as produced by cv::calibrateCamera /
// opencv_interactive-calibration), and print the estimated pose per frame.
//
// Usage:
//   run_video <video_file> <calibration.yml> [start_frame] [--set name=value ...]
//             [--reset-after-lost N]
//
// svo::Config (include/svo/config.h) is a singleton read by every stage of
// the pipeline (DepthFilter, Matcher, Reprojector, PoseOptimizer, ...) via
// static accessors such as Config::gridSize(). Those accessors return
// mutable references, so tuning a parameter is just a matter of assigning to
// them before the FrameHandlerMono is constructed. --set exposes that here
// on the command line instead of requiring a recompile.
//
// --reset-after-lost N (default 30, 0 disables): if SVO stays in
// STAGE_RELOCALIZING for N consecutive frames without recovering, the map
// is torn down and reinitialized from scratch on the next frame. Since the
// only output that matters here is per-frame linear/angular velocity (not
// a pose in one consistent world frame), a fresh, arbitrary map origin
// after reinit is harmless.
//
// IMPORTANT: destroying a FrameHandlerMono that went through a successful
// init followed by tracking loss corrupts the heap -- confirmed via a
// debugger backtrace into KltHomographyInit::~KltHomographyInit() (a
// pre-existing bug in SVO's own manual Frame/Feature/Point pointer
// ownership, unrelated to this reset feature: it reproduces even with
// --reset-after-lost 0, on the *natural* end-of-video teardown). Since the
// only thing this tool needs is the already-printed per-frame output, old
// FrameHandlerMono instances are deliberately never destructed -- only
// their DepthFilter background thread is stopped (to avoid leaking a
// running thread per reset) -- and main() exits via std::_Exit() to skip
// the crashing destructor chain entirely.

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vikit/pinhole_camera.h>
#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame.h>
#include <svo/map.h>
#include <svo/depth_filter.h>

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

/// Formats a scalar with fixed width/precision so that per-frame output
/// lines up in columns instead of jittering with the value's magnitude.
/// Values that are effectively zero (e.g. -0.0000 from sign flips) are
/// clamped to a clean 0 first.
std::string fmt(double v, int width = 9, int prec = 4)
{
  if(std::abs(v) < 5e-10)
    v = 0.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(prec) << std::setw(width) << v;
  return oss.str();
}

/// Formats a 3D vector as "[ x,  y,  z]" using fmt() for each component.
std::string fmtVec3(const Eigen::Vector3d& v, int width = 9, int prec = 4)
{
  return "[" + fmt(v.x(), width, prec) + ", " + fmt(v.y(), width, prec) + ", " + fmt(v.z(), width, prec) + "]";
}

/// Formats the current wall-clock time as HH:MM:SS.mmm, i.e. the real time
/// at which this frame was actually processed (as opposed to "t=", which is
/// the frame's timestamp within the video).
std::string wallClockNow()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;

  std::tm tm_buf;
  localtime_r(&now_c, &tm_buf);

  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%H:%M:%S")
      << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
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
    std::cerr << "Usage: " << argv[0] << " <video_file> <calibration.yml> [start_frame]"
              << " [--set name=value ...] [--reset-after-lost N]\n";
    return 1;
  }
  const std::string video_path = argv[1];
  const std::string calib_path = argv[2];
  int start_frame = 0;
  // Number of consecutive STAGE_RELOCALIZING frames after which the VO state
  // (map, keyframes, depth filter) is reset and reinitialized from scratch,
  // since SVO has no built-in give-up-and-restart behavior of its own.
  // 0 disables the automatic reset.
  int reset_after_lost = 30;

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
    else if(arg == "--reset-after-lost")
    {
      if(i + 1 >= argc)
      {
        std::cerr << "--reset-after-lost requires an integer frame count\n";
        return 1;
      }
      reset_after_lost = std::atoi(argv[++i]);
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

  // Scoped so that *vo (and every Frame it holds, all of which store a raw
  // cam_ pointer) is destroyed before "delete cam" below -- otherwise
  // vo's destructor runs after cam is freed (use-after-free/segfault).
  //
  // vo is heap-allocated (rather than a plain stack object) because a
  // forced reset (below) destroys and reconstructs it from scratch, rather
  // than calling FrameHandlerBase::reset() in place. SVO's manual
  // Frame/Feature/Point graph (raw pointers, deleted directly by
  // Map::reset()/MapPointCandidates::reset()) is only ever exercised
  // through a single construct-run-destruct lifecycle upstream; repeatedly
  // reset()-ing one long-lived instance leaves that graph in a state whose
  // *final* destructor corrupts memory (observed as a crash strictly after
  // "Done" -- i.e. once the video ends -- on any run that had reset at
  // least once, with or without in-place reset()/start()). A fresh object
  // per restart sidesteps that entirely: each instance only ever goes
  // through the one lifecycle that's known to work.
  {
  svo::FrameHandlerMono* vo = new svo::FrameHandlerMono(cam);
  vo->start();

  // State kept to differentiate the camera pose (world-frame position +
  // orientation) between consecutive tracked frames into linear and
  // angular velocity.
  bool have_prev_pose = false;
  double prev_t = 0.0;
  Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond prev_q = Eigen::Quaterniond::Identity();

  // Number of consecutive frames spent in STAGE_RELOCALIZING. Reset to 0
  // whenever tracking succeeds (or during normal startup init).
  int relocalizing_streak = 0;

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
    vo->addImage(frame_gray, timestamp);

    // FrameHandlerMono::addImage() unconditionally assigns last_frame_ on
    // every call, success or failure -- so lastFrame() != NULL only means
    // "some frame has been processed", not "this frame was tracked". On a
    // tracking failure SVO deliberately freezes new_frame_->T_f_w_ at the
    // previous pose ("reset to avoid crazy pose jumps") and switches to
    // STAGE_RELOCALIZING, so the pose only reflects a genuine new estimate
    // while stage() == STAGE_DEFAULT_FRAME.
    const bool is_tracking =
        vo->lastFrame() != NULL && vo->stage() == svo::FrameHandlerBase::STAGE_DEFAULT_FRAME;

    if(is_tracking)
    {
      // Camera pose in the world frame (position + orientation), both
      // derived from the same T_w_f = T_f_w_^-1 so that differentiating them
      // yields a consistent linear/angular velocity.
      const Sophus::SE3 T_w_f = vo->lastFrame()->T_f_w_.inverse();
      const Eigen::Vector3d twf = T_w_f.translation();
      Eigen::Quaterniond q = T_w_f.unit_quaternion();

      Eigen::Vector3d vel = Eigen::Vector3d::Zero();
      Eigen::Vector3d angvel = Eigen::Vector3d::Zero();
      double vel_mag = 0.0, angvel_mag = 0.0;

      if(have_prev_pose)
      {
        const double dt = timestamp - prev_t;
        if(dt > 1e-9)
        {
          vel = (twf - prev_pos) / dt;
          vel_mag = vel.norm();

          // Choose the quaternion hemisphere closest to prev_q so the
          // relative rotation takes the shortest path.
          if(q.dot(prev_q) < 0.0)
            q.coeffs() *= -1.0;
          Eigen::Quaterniond q_rel = q * prev_q.inverse();
          q_rel.normalize();
          const Eigen::AngleAxisd aa(q_rel);
          angvel = aa.axis() * (aa.angle() / dt);
          angvel_mag = std::abs(aa.angle() / dt);
        }
      }
      prev_pos = twf;
      prev_q = q;
      prev_t = timestamp;
      have_prev_pose = true;

      std::cout << "frame=" << std::setw(6) << img_id
                << " tracked_at=" << wallClockNow()
                << " t="     << fmt(timestamp, 8, 3)
                << " stage=" << vo->stage()
                << " nobs="  << std::setw(4) << vo->lastNumObservations()
                << " time_ms=" << fmt(vo->lastProcessingTime()*1000, 7, 2)
                << " pos="   << fmtVec3(twf)
                << " quat="  << "[" << fmt(q.x()) << ", " << fmt(q.y()) << ", " << fmt(q.z()) << ", " << fmt(q.w()) << "]"
                << " vel="   << fmtVec3(vel)
                << " |vel|="    << fmt(vel_mag, 8) << " m/s"
                << " angvel="   << fmtVec3(angvel)
                << " |angvel|=" << fmt(angvel_mag, 8) << " rad/s"
                << std::endl;
    }
    else
    {
      have_prev_pose = false;
      const char* reason = (vo->lastFrame() == NULL) ? "not yet tracking" : "lost tracking, relocalizing";
      std::cout << "frame=" << std::setw(6) << img_id
                << " tracked_at=" << wallClockNow()
                << " t=" << fmt(timestamp, 8, 3)
                << " stage=" << vo->stage()
                << " (" << reason << ")" << std::endl;
    }

    // SVO has no built-in give-up-and-restart: once RESULT_FAILURE hits, it
    // just keeps calling relocalizeFrame() against the old map forever (see
    // frame_handler_base.cpp finishFrameProcessingCommon). If that streak
    // goes on too long, tear down vo and construct a fresh one, exactly
    // like the very first call. Since the caller only wants relative
    // velocities (not a pose in a single consistent world frame), a new
    // arbitrary map origin after reinit is harmless -- have_prev_pose is
    // already false here, so velocity output naturally resumes cleanly
    // from the next tracked pair.
    if(vo->stage() == svo::FrameHandlerBase::STAGE_RELOCALIZING)
      ++relocalizing_streak;
    else
      relocalizing_streak = 0;

    if(reset_after_lost > 0 && relocalizing_streak >= reset_after_lost)
    {
      std::cout << "frame=" << std::setw(6) << img_id
                << " tracked_at=" << wallClockNow()
                << " RESET: lost tracking for " << relocalizing_streak
                << " frames, reinitializing map from scratch" << std::endl;
      // Deliberately not deleting vo -- see the note at the top of this
      // file. Only stop its background thread so it doesn't keep running
      // forever, then abandon the rest of the object.
      vo->depthFilter()->stopThread();
      vo = new svo::FrameHandlerMono(cam);
      vo->start();
      relocalizing_streak = 0;
    }

    ++img_id;
  }

  std::cout << "Done. Processed " << (img_id - start_frame) << " frames." << std::endl;
  vo->depthFilter()->stopThread();
  } // end vo scope -- vo is deliberately never deleted, see note above.

  std::cout.flush();
  std::_Exit(0); // Skip C++ destructors (cam, cap, vo, ...) entirely.
}
