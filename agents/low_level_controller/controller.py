#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math

import numpy as np
import pandas as pd

import carla
from agents.tools.misc import get_speed
from config import cfg


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral=None, args_longitudinal=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following
        semantics:
                             K_P -- Proportional term
                             K_D -- Differential term
                             K_I -- Integral term
        """
        if not args_lateral:
            args_lateral = {'K_P': 0.3, 'K_D': 0.0, 'K_I': 0.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 40.0, 'K_D': 0.1, 'K_I': 4}

        self._vehicle = vehicle
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def reset(self):
        self._lon_controller.reset()
        self._lat_controller.reset()
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = True
        control.manual_gear_shift = False
        self._vehicle.apply_control(control)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle, speed = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def run_step_2_wp(self, target_speed, waypoint1, waypoint2):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target_speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: distance (in meters) to the waypoint
        """
        throttle, speed = self._lon_controller.run_step(target_speed)
        steering = self._lat_controller.run_step_2_wp(waypoint1, waypoint2)
        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        return control


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=10.0, K_D=0.0, K_I=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05
        self._e_buffer = deque(maxlen=10)

    def reset(self):
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_speed):
        """
        Execute one step of longitudinal control to reach a given target speed.

        :param target_speed: target speed in m/s
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        return self._pid_control(target_speed, current_speed), current_speed

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in m/s
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        _e = (target_speed - current_speed)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de / self.dt) + (self._K_I * _ie * self.dt), 0.0, 1.0)


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=0.2, K_D=0.0, K_I=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05
        self._e_buffer = deque(maxlen=10)

        self.prev_prop = np.nan
        self.prev_prev_prop = np.nan
        self.curr_prop = np.nan
        self.deriv_list = []
        self.deriv_len = 5

    def reset(self):
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint [x, y]
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint[0] -
                          v_begin.x, waypoint[1] -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self.dt) + (self._K_I * _ie * self.dt), -1.0, 1.0)

    def run_step_2_wp(self, waypoint1, waypoint2):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control_2_wp(waypoint1, waypoint2, self._vehicle.get_transform())

    def _pid_control_2_wp(self, waypoint1, waypoint2, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint [x, y]
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint2[0] -
                          waypoint1[0], waypoint2[1] -
                          waypoint1[1], 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self.dt) + (self._K_I * _ie * self.dt), -1.0, 1.0)


class PIDCrossTrackController:
    """
    PID control for the trajectory tracking
    Acceptable performance: 'K_P': 0.01, 'K_D': 0.01, 'K_I': 0.15,
    """

    def __init__(self, params):
        """
        params: dictionary of PID coefficients
        """
        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05
        self.params = params
        self.e_buffer = deque(maxlen=30)  # error buffer; error: deviation from center lane -/+ value

    def reset(self):
        self.e_buffer = deque(maxlen=30)

    def run_step(self, cte):
        """
        cte: a weak definition for cross track error. i.e. cross track error = |cte|
        ***************** modify the code to use dt in correct places ***************
        """
        self.e_buffer.append(cte)
        if len(self.e_buffer) >= 2:
            _de = (self.e_buffer[-1] - self.e_buffer[-2]) / self.dt
            _ie = sum(self.e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self.params['K_P'] * cte) + (self.params['K_D'] * _de / self.dt)
                       + (self.params['K_I'] * _ie * self.dt), -0.5, 0.5)


class IntelligentDriverModel:
    """
    Intelligent Driver Model (Cruise Control)
    https://arxiv.org/pdf/1909.11538.pdf
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.a_max = 1     # needs tuning (depending on the vehicle dynamics)
        self.delta = 4
        self.T = 1.6
        self.d0 = 2
        self.b = 1.7
        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

    def run_step(self, vd, vehicle_ahead):
        v = get_speed(self.vehicle)

        if vehicle_ahead is None:
            acc_cmd = self.a_max * (1 - (v / vd)**self.delta)
        else:
            loc1 = [vehicle_ahead.get_location().x, vehicle_ahead.get_location().y, vehicle_ahead.get_location().z]
            loc2 = [self.vehicle.get_location().x, self.vehicle.get_location().y, self.vehicle.get_location().z]
            d = euclidean_distance(loc1, loc2)
            v2 = get_speed(vehicle_ahead)
            dv = v - v2

            d_star = self.d0 + max(0, v*self.T + v*dv/(2*math.sqrt(self.b*self.a_max)))

            acc_cmd = self.a_max * (1 - (v / vd)**self.delta - (d_star/d)**2)

        cmdSpeed = get_speed(self.vehicle) + acc_cmd * self.dt

        # if self.vehicle.attributes['role_name'] == 'hero':
        #     print(v, cmdSpeed, acc_cmd)
        #     print([x * 3.6 for x in [cmdSpeed, acc_cmd, v, vd]])
        return cmdSpeed
