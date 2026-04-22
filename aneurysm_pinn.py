# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import torch
import numpy as np
from sympy import Symbol, sqrt, Max

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.geometry.tessellation import Tessellation


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # read stl files to make geometry
    point_path = to_absolute_path("./stl_files")
    inlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_inlet.stl", airtight=False
    )
    outlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_outlet.stl", airtight=False
    )
    noslip_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_noslip.stl", airtight=False
    )
    integral_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_integral.stl", airtight=False
    )
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    # params
    nu = 7.989e-05
    inlet_vel = 1.5

    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar

    # scale and normalize mesh
    center = (0.01151354, 0.03322, 0.0)
    scale = 12.753

    inlet_mesh = normalize_mesh(inlet_mesh, center, scale)
    outlet_mesh = normalize_mesh(outlet_mesh, center, scale)
    noslip_mesh = normalize_mesh(noslip_mesh, center, scale)
    integral_mesh = normalize_mesh(integral_mesh, center, scale)
    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # geom params
    inlet_normal = (0.0, 1.0, 0.0)
    inlet_area_raw = 5.0179867253e-05
    inlet_area = inlet_area_raw * (scale**2)
    inlet_center = (-0.1469, -0.4996, 0.0)
    inlet_radius = np.sqrt(inlet_area / np.pi)

    outlet_area_raw = 5.0179867253e-05
    outlet_area = outlet_area_raw * (scale**2)

    Q_scaled = 0.006122

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # Integral Continuity 1 - outlet
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"normal_dot_vel": Q_scaled},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    domain.add_constraint(integral_continuity, "integral_continuity_1")

    # Integral Continuity 2 - integral mesh
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": -Q_scaled},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    domain.add_constraint(integral_continuity, "integral_continuity_2")

    # ---- StarCCM+ Validation & L2 Pressure Error ----
    file_path = "./starccm/starccm_pressure.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "X": "x",
            "Y": "y",
            "Z": "z",
            "Pressure": "p",
        }
        starccm_var = csv_to_dict(to_absolute_path(file_path), mapping)

        starccm_invar = {
            key: value for key, value in starccm_var.items() if key in ["x", "y", "z"]
        }
        starccm_invar = normalize_invar(starccm_invar, center, scale, dims=3)

        starccm_outvar = {
            key: value for key, value in starccm_var.items() if key in ["p"]
        }

        starccm_validator = PointwiseValidator(
            nodes=nodes,
            invar=starccm_invar,
            true_outvar=starccm_outvar,
            batch_size=4096,
        )
        domain.add_validator(starccm_validator)

    else:
        warnings.warn(
            f"StarCCM+ file {file_path} not found. Skipping validator. "
            f"Place your CSV at ./starccm/starccm_pressure.csv"
        )

    # pressure monitor at inlet
    pressure_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(16),
        output_names=["p"],
        metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    # make solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
