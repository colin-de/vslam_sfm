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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  UNUSED(xi);
  T theta = sqrt(xi.transpose() * xi);
  Eigen::Vector3d v = xi / theta;
  Eigen::Matrix<T, 3, 3> v_hat, res;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  v_hat << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;

  if (theta != 0) {
    res = cos(theta) * I + (1 - cos(theta)) * v * v.transpose() +
          sin(theta) * v_hat;
  } else {
    res = I;
  }
  // return Sophus::SO3d::exp(xi).matrix(); // using sophus
  return res;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  UNUSED(mat);
  T theta = acos((mat.trace() - 1) / 2);
  Eigen::Matrix<T, 3, 1> res;
  if (theta != 0) {
    Eigen::Matrix<T, 3, 1> vec, axis;
    vec << (mat(2, 1) - mat(1, 2)), (mat(0, 2) - mat(2, 0)),
        (mat(1, 0) - mat(0, 1));
    axis = 1 / (2 * sin(theta)) * vec;
    res = theta * axis;
  } else {
    res = Eigen::Matrix<T, 3, 1>::Zero();
  }
  return res;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  UNUSED(xi);
  Eigen::Matrix<T, 3, 1> rot, trans;
  Eigen::Matrix<T, 3, 3> R, v_hat, J;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Vector3d v;
  Eigen::Matrix<T, 4, 4> res;

  trans << xi(0), xi(1), xi(2);
  rot << xi(3), xi(4), xi(5);
  R = user_implemented_expmap(rot);
  T theta = sqrt(rot.transpose() * rot);
  v = rot / theta;
  v_hat << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;

  if (theta != 0) {
    J = (sin(theta) / theta) * I +
        (1 - (sin(theta) / theta)) * v * v.transpose() +
        (1 - cos(theta)) / theta * v_hat;
  } else {
    J = I;
  }

  res.topLeftCorner(3, 3) = R;
  res.topRightCorner(3, 1) = (J * trans);
  res.bottomLeftCorner(1, 3) = Eigen::Matrix<T, 1, 3>::Zero();
  res.bottomRightCorner(1, 1) << 1;
  return res;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  UNUSED(mat);
  Eigen::Matrix<T, 3, 1> rot, trans, v;
  Eigen::Matrix<T, 3, 3> R = mat.topLeftCorner(3, 3);
  Eigen::Matrix<T, 3, 3> J, J_inv, w_hat, v_hat;
  Eigen::Matrix<T, 6, 1> res;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

  rot = user_implemented_logmap(R);
  w_hat << 0, -rot(2), rot(1), rot(2), 0, -rot(0), -rot(1), rot(0), 0;
  T theta = acos((R.trace() - 1) / 2);
  v = rot / theta;
  v_hat << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  if (theta != 0) {
    J = (sin(theta) / theta) * I +
        (1 - (sin(theta) / theta)) * v * v.transpose() +
        (1 - cos(theta)) / theta * v_hat;
  } else {
    J = I;
  }

  trans = J.inverse() * mat.topRightCorner(3, 1);
  res << trans, rot;
  return res;
}

}  // namespace visnav
