cmake_minimum_required(VERSION 3.15)
project(zmp_pinocchio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── conda 환경 경로 ──
set(CONDA_ENV "/home/frlab/anaconda3/envs/frmoco")
set(MUJOCO_ROOT "${CONDA_ENV}/lib/python3.12/site-packages/mujoco")

# ── Eigen ──
find_package(Eigen3 REQUIRED)

# ── Pinocchio ──
find_package(pinocchio REQUIRED
    PATHS "${CONDA_ENV}/lib/cmake/pinocchio" NO_DEFAULT_PATH)

# ── Qhull (pinocchio_extra dependency) ──
if(NOT TARGET Qhull::qhullcpp)
    find_library(QHULL_CPP_LIB qhullcpp)
    if(QHULL_CPP_LIB)
        add_library(Qhull::qhullcpp SHARED IMPORTED)
        set_target_properties(Qhull::qhullcpp PROPERTIES
            IMPORTED_LOCATION "${QHULL_CPP_LIB}")
    endif()
endif()

# ── ProxSuite ──
find_package(proxsuite REQUIRED
    PATHS "${CONDA_ENV}/lib/cmake/proxsuite" NO_DEFAULT_PATH)

# ── MuJoCo ──
set(MUJOCO_INCLUDE "${MUJOCO_ROOT}/include")
set(MUJOCO_LIB     "${MUJOCO_ROOT}/libmujoco.so.3.5.0")

# ── GLFW (시스템 직접 지정) ──
set(GLFW_INCLUDE_DIR /usr/include)
set(GLFW_LIBRARY /usr/lib/x86_64-linux-gnu/libglfw.so)

# ── 소스 파일 ──
set(SOURCES
    main.cpp
    src/trajectory_planner/zmp_trajectory.cpp
    src/trajectory_planner/foot_trajectory.cpp
    src/controller/LIPM_LQR.cpp
    src/controller/LIPM_MPC.cpp
    src/dynamics_model/LIPM.cpp
    src/dynamics_model/SRBD.cpp
    src/dynamics_model/com_dynamics.cpp
    src/whole_body_controller/Force_Optimizer.cpp
    src/whole_body_controller/whole_body_torque.cpp
    src/whole_body_controller/whole_body_ik.cpp
    src/whole_body_controller/DBFC_core.cpp
    src/whole_body_controller/tasks/balance_task.cpp
    src/main_controller/g1_walking_controller.cpp
    src/constraints/friction_cone.cpp
    src/constraints/convex_hull.cpp
    src/constraints/cop_limits.cpp
    src/constraints/task_space.cpp
)

add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MUJOCO_INCLUDE}
    ${GLFW_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(main
    pinocchio::pinocchio
    proxsuite::proxsuite
    ${MUJOCO_LIB}
    ${GLFW_LIBRARY}
    dl
)
