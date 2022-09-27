/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

bool point_in_image(const Eigen::Vector2d& p_2d,
                    const std::shared_ptr<AbstractCamera<double>>& cam) {
  bool res;
  res = (p_2d.x() >= 0) && (abs(p_2d.x()) <= cam->width()) && (p_2d.y() >= 0) &&
        (abs(p_2d.y()) <= cam->height());
  return res;
}

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (auto& landmark : landmarks) {
    Eigen::Vector3d landmark_cam = current_pose.inverse() * landmark.second.p;
    if (landmark_cam.z() >= cam_z_threshold) {
      Eigen::Vector2d p_2d = cam->project(landmark_cam);
      if (point_in_image(p_2d, cam)) {
        projected_points.push_back(p_2d);
        projected_track_ids.push_back(landmark.first);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  using TDescriptor = std::bitset<256>;

  for (size_t i = 0; i < kdl.corners.size(); i++) {
    TDescriptor keypoint_des = kdl.corner_descriptors[i];
    TrackId best_index = -1;
    int best_dist = std::numeric_limits<int>::max(),
        best_dist_2 = std::numeric_limits<int>::max();

    for (size_t j = 0; j < projected_points.size(); j++) {
      if ((kdl.corners[i] - projected_points[j]).norm() <= match_max_dist_2d) {
        int best_dist_obs = std::numeric_limits<int>::max();
        for (auto& obs : landmarks.at(projected_track_ids.at(j)).obs) {
          FrameCamId fcid = obs.first;
          FeatureId fid = obs.second;
          TDescriptor obs_des =
              feature_corners.at(fcid).corner_descriptors.at(fid);
          auto hamming_dist = int((keypoint_des ^ obs_des).count());

          if (hamming_dist < best_dist_obs) best_dist_obs = hamming_dist;
        }

        if (best_dist_obs <= best_dist) {
          best_dist_2 = best_dist;
          best_dist = best_dist_obs;
          best_index = projected_track_ids.at(j);
        } else if (best_dist_obs < best_dist_2) {
          best_dist_2 = best_dist_obs;
        }
      }
    }

    if (best_dist < feature_match_threshold &&
        best_dist_2 >= best_dist * feature_match_dist_2_best) {
      md.matches.emplace_back(i, best_index);
    }
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.
  opengv::bearingVectors_t bearingVectors;
  opengv::points_t points;

  for (auto& match : md.matches) {
    FeatureId fid = match.first;
    TrackId tid = match.second;
    points.push_back(landmarks.at(tid).p);
    Eigen::Vector2d p_2d = kdl.corners.at(fid);
    Eigen::Vector3d p_3d = cam->unproject(p_2d);
    bearingVectors.push_back(p_3d);
  }

  // create the central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));

  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  double focal_length = 500.0;
  ransac.threshold_ =
      1.0 -
      cos(atan(reprojection_error_pnp_inlier_threshold_pixel / focal_length));
  ransac.computeModel();

  // non-linear optimization (using all correspondences)
  adapter.sett(ransac.model_coefficients_.rightCols(1));
  adapter.setR(ransac.model_coefficients_.leftCols(3));
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  for (auto& inlier : ransac.inliers_) {
    md.inliers.push_back(md.matches.at(inlier));
  }

  md.T_w_c = Sophus::SE3d(nonlinear_transformation.leftCols(3),
                          nonlinear_transformation.rightCols(1));
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.
  std::set<FeatureId> md_stereo_used;

  // md contains feature to landmark matches for the left camera (camera 0)
  // std::pair<FeatureId, TrackId> LandmarkMatchData::inliers
  for (auto& inlier : md.inliers) {
    FeatureId fid = inlier.first;
    TrackId tid = inlier.second;
    landmarks.at(tid).obs.insert(std::make_pair(fcidl, fid));

    /// collection of {featureId_i, featureId_j} pairs of inlier matches
    for (auto& stereo_inlier : md_stereo.inliers) {
      if (stereo_inlier.first == fid) {
        landmarks.at(tid).obs.insert(
            std::make_pair(fcidr, stereo_inlier.second));
        md_stereo_used.insert(stereo_inlier.first);
      }
    }
  }

  for (auto& stereo_inlier : md_stereo.inliers) {
    if (md_stereo_used.count(stereo_inlier.first)) continue;
    opengv::bearingVectors_t bearingVectorsL, bearingVectorsR;
    Eigen::Vector2d p0_2d = kdl.corners.at(stereo_inlier.first);
    Eigen::Vector2d p1_2d = kdr.corners.at(stereo_inlier.second);
    Eigen::Vector3d p0_3d =
        calib_cam.intrinsics[fcidl.cam_id]->unproject(p0_2d);
    Eigen::Vector3d p1_3d =
        calib_cam.intrinsics[fcidr.cam_id]->unproject(p1_2d);
    bearingVectorsL.emplace_back(p0_3d);
    bearingVectorsR.emplace_back(p1_3d);

    opengv::relative_pose::CentralRelativeAdapter adapter(
        bearingVectorsL, bearingVectorsR, t_0_1, R_0_1);
    Eigen::Vector3d point = opengv::triangulation::triangulate(adapter, 0);
    Landmark lm;
    lm.p = md.T_w_c * point;
    lm.obs.emplace(fcidl, stereo_inlier.first);
    lm.obs.emplace(fcidr, stereo_inlier.second);
    landmarks.emplace(next_landmark_id++, lm);
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.

  // should only work when kf_frames larger than max_num_kfs
  while (kf_frames.size() > size_t(max_num_kfs)) {
    // find the oldest frame
    auto itFrameidToRemove =
        std::min_element(kf_frames.begin(), kf_frames.end());
    auto frameidToRemove = *itFrameidToRemove;

    for (auto it_camera = cameras.begin(); it_camera != cameras.end();) {
      it_camera->first.frame_id == frameidToRemove ? cameras.erase(it_camera++)
                                                   : it_camera++;
    }

    for (auto it_landmark = landmarks.begin();
         it_landmark != landmarks.end();) {
      for (auto it_obs = it_landmark->second.obs.begin();
           it_obs != it_landmark->second.obs.end();) {
        it_obs->first.frame_id == frameidToRemove
            ? it_landmark->second.obs.erase(it_obs++)
            : it_obs++;
      }

      // remove 0 obs landmark to old_landmarks
      if (it_landmark->second.obs.empty()) {
        old_landmarks.insert(*it_landmark);
        landmarks.erase(it_landmark++);
      } else
        it_landmark++;
    }
    kf_frames.erase(itFrameidToRemove);
  }
}
}  // namespace visnav
