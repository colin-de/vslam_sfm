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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  Eigen::Vector3d t_0_1_n = t_0_1.normalized();
  Eigen::Matrix3d t_0_1_hat;
  t_0_1_hat << 0, -t_0_1_n(2), t_0_1_n(1), t_0_1_n(2), 0, -t_0_1_n(0),
      -t_0_1_n(1), t_0_1_n(0), 0;
  E = t_0_1_hat * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    double res = 0.0;
    const Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    const Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    res = p0_3d.transpose() * E * p1_3d;
    if (abs(res) <= epipolar_error_threshold) {
      md.inliers.emplace_back(md.matches[j]);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.
  opengv::bearingVectors_t bearingVectors1, bearingVectors2;
  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    const Eigen::Vector3d p0_3d = cam1->unproject(p0_2d);
    const Eigen::Vector3d p1_3d = cam2->unproject(p1_2d);

    bearingVectors1.emplace_back(p0_3d.normalized());
    bearingVectors2.emplace_back(p1_3d.normalized());
  };
  // create the central relative adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                        bearingVectors2);
  // create a RANSAC object
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  // create a CentralRelativePoseSacProblem
  // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  // run ransac
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.max_iterations_ = 1e6;
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation = ransac.model_coefficients_;
  Eigen::Vector3d t = best_transformation.topRightCorner(3, 1);
  Eigen::Matrix3d R = best_transformation.topLeftCorner(3, 3);

  // non-linear optimization (using all correspondences)
  adapter.sett12(t);
  adapter.setR12(R);
  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  Eigen::Vector3d final_t = nonlinear_transformation.topRightCorner(3, 1);
  Eigen::Matrix3d final_R = nonlinear_transformation.topLeftCorner(3, 3);

  // Get the set of inliers that correspond to the best model found so far
  ransac.sac_model_->selectWithinDistance(ransac.model_coefficients_,
                                          ransac.threshold_, ransac.inliers_);

  for (size_t i = 0; i < ransac.inliers_.size(); i++) {
    md.inliers.emplace_back(md.matches[ransac.inliers_[i]].first,
                            md.matches[ransac.inliers_[i]].second);
  }
  if (static_cast<int>(md.inliers.size()) < ransac_min_inliers) {
    md.inliers.clear();
  }

  Sophus::SE3d final_pose(final_R, final_t);
  md.T_i_j = final_pose;
}
}  // namespace visnav
