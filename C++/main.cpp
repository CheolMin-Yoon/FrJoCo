// C++ API 기반 Mujoco + Pinocchio 공통
#include <iostream>
#include <string>
#include <cstring>

#define COAL_DISABLE_HPP_FCL_WARNINGS 
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Dense>
#include "mujoco/mujoco.h"
#include "GLFW/glfw3.h"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

// G1 Walking Control을 위한 헤더 파일들
#include "config.hpp"
#include "main_controller/g1_walking_controller.hpp"

// ── 경로 ──────────────────────────────────────────────────────
static const char* MJ_XML   = "/home/frlab/mujoco_demo/model/g1/scene_29dof.xml";
static const char* PIN_URDF = "/home/frlab/mujoco_demo/humanoid_study26/C++/RoMoCo/src/g1_stack/model_files/g1_29dof.urdf";

// ── GLFW 전역 ─────────────────────────────────────────────────
static mjModel* g_m = nullptr;
static mjData*  g_d = nullptr;
static mjvCamera cam; static mjvOption opt;
static mjvScene scn;  static mjrContext con;
static bool btn_l = false, btn_r = false;
static double lastx = 0, lasty = 0;

void keyboard(GLFWwindow*, int key, int, int act, int) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(g_m, g_d); mj_forward(g_m, g_d);
    }
}
void mouse_button(GLFWwindow* w, int, int, int) {
    btn_l = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS);
    btn_r = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(w, &lastx, &lasty);
}
void mouse_move(GLFWwindow* w, double xpos, double ypos) {
    if (!btn_l && !btn_r) return;
    double dx = xpos - lastx, dy = ypos - lasty;
    lastx = xpos; lasty = ypos;
    int W, H; glfwGetWindowSize(w, &W, &H);
    bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
    mjtMouse action = btn_r ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                             : (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);
    mjv_moveCamera(g_m, action, dx/H, dy/H, &scn, &cam);
}
void scroll(GLFWwindow*, double, double dy) {
    mjv_moveCamera(g_m, mjMOUSE_ZOOM, 0, -0.05*dy, &scn, &cam);
}

// ── MuJoCo → Pinocchio 변환 ───────────────────────────────────
Eigen::VectorXd mj2pin_q(const double* qpos, int nq_pin) {
    Eigen::VectorXd q(nq_pin);
    q[0]=qpos[0]; q[1]=qpos[1]; q[2]=qpos[2];
    q[3]=qpos[4]; q[4]=qpos[5]; q[5]=qpos[6]; q[6]=qpos[3]; // MJ wxyz → Pin xyzw
    for (int i = 7; i < nq_pin; ++i) q[i] = qpos[i];
    return q;
}

// ── 디버그 시각화 헬퍼 ────────────────────────────────────────
static void drawSphere(mjvScene* scn, const Eigen::Vector3d& pos,
                        float radius, float r, float g, float b, float a = 0.9f)
{
    if (scn->ngeom >= scn->maxgeom) return;
    mjvGeom* geom = &scn->geoms[scn->ngeom++];
    mjv_initGeom(geom, mjGEOM_SPHERE, nullptr, nullptr, nullptr, nullptr);
    geom->size[0] = radius;
    geom->pos[0] = pos.x(); geom->pos[1] = pos.y(); geom->pos[2] = pos.z();
    geom->rgba[0] = r; geom->rgba[1] = g; geom->rgba[2] = b; geom->rgba[3] = a;
}

static void drawArrow(mjvScene* scn, const Eigen::Vector3d& from,
                       const Eigen::Vector3d& force, float width,
                       float r, float g, float b, float a = 0.8f)
{
    if (scn->ngeom >= scn->maxgeom) return;
    constexpr double SCALE = 0.001;
    Eigen::Vector3d to = from + force * SCALE;
    mjtNum f[3] = {from.x(), from.y(), from.z()};
    mjtNum t[3] = {to.x(),   to.y(),   to.z()};
    mjvGeom* geom = &scn->geoms[scn->ngeom++];
    mjv_initGeom(geom, mjGEOM_ARROW, nullptr, nullptr, nullptr, nullptr);
    mjv_connector(geom, mjGEOM_ARROW, width, f, t);
    geom->rgba[0] = r; geom->rgba[1] = g; geom->rgba[2] = b; geom->rgba[3] = a;
}

static void drawLine(mjvScene* scn, const Eigen::Vector3d& p0,
                      const Eigen::Vector3d& p1, float width,
                      float r, float g, float b, float a = 0.7f)
{
    if (scn->ngeom >= scn->maxgeom) return;
    mjtNum f0[3] = {p0.x(), p0.y(), p0.z()};
    mjtNum f1[3] = {p1.x(), p1.y(), p1.z()};
    mjvGeom* geom = &scn->geoms[scn->ngeom++];
    mjv_initGeom(geom, mjGEOM_CAPSULE, nullptr, nullptr, nullptr, nullptr);
    mjv_connector(geom, mjGEOM_CAPSULE, width, f0, f1);
    geom->rgba[0] = r; geom->rgba[1] = g; geom->rgba[2] = b; geom->rgba[3] = a;
}

// ── 메인 ──────────────────────────────────────────────────────
int main()
{
    // MuJoCo 로드
    char err[1000] = {};
    g_m = mj_loadXML(MJ_XML, nullptr, err, sizeof(err));
    if (!g_m) { std::cerr << "MuJoCo load failed: " << err << "\n"; return 1; }
    g_d = mj_makeData(g_m);

    std::cout << "=== MuJoCo Model ===\n";
    std::cout << "  nq=" << g_m->nq << "  nv=" << g_m->nv << "  nu=" << g_m->nu << "\n";

    // knees_bent 자세 설정
    g_d->qpos[0]=0.0; g_d->qpos[1]=0.0; g_d->qpos[2]=0.755;
    g_d->qpos[3]=1.0; g_d->qpos[4]=0.0; g_d->qpos[5]=0.0; g_d->qpos[6]=0.0;
    g_d->qpos[7]=-0.312; g_d->qpos[8]=0.0;  g_d->qpos[9]=0.0;
    g_d->qpos[10]=0.669; g_d->qpos[11]=-0.363; g_d->qpos[12]=0.0;
    g_d->qpos[13]=-0.312; g_d->qpos[14]=0.0; g_d->qpos[15]=0.0;
    g_d->qpos[16]=0.669;  g_d->qpos[17]=-0.363; g_d->qpos[18]=0.0;
    g_d->qpos[19]=0.073;  g_d->qpos[20]=0.0; g_d->qpos[21]=0.0;
    mj_forward(g_m, g_d);

    // Pinocchio 로드
    pinocchio::Model pin_model;
    try {
        pinocchio::urdf::buildModel(PIN_URDF, pinocchio::JointModelFreeFlyer(), pin_model);
    } catch (const std::exception& e) {
        std::cerr << "[Pinocchio] 로드 실패: " << e.what() << "\n"; return 1;
    }
    pinocchio::Data pin_data(pin_model);

    std::cout << "=== Pinocchio Model ===\n";
    std::cout << "  nq=" << pin_model.nq << "  nv=" << pin_model.nv
              << "  na=" << (pin_model.nv - 6) << "\n";
    double total_mass = pinocchio::computeTotalMass(pin_model);
    std::cout << "  총 질량: " << total_mass << " kg\n";

    // 초기 CoM
    Eigen::VectorXd q0 = mj2pin_q(g_d->qpos, pin_model.nq);
    pinocchio::forwardKinematics(pin_model, pin_data, q0);
    pinocchio::updateFramePlacements(pin_model, pin_data);
    pinocchio::centerOfMass(pin_model, pin_data, q0, false);
    Eigen::Vector2d init_com_xy = pin_data.com[0].head<2>();
    std::cout << "  초기 CoM  : " << pin_data.com[0].transpose() << "\n";

    // Pinocchio frame ID (한 번만 조회)
    int rf_frame_id = pin_model.getFrameId("right_ankle_roll_link");
    int lf_frame_id = pin_model.getFrameId("left_ankle_roll_link");

    // FK 기반 초기 발 위치
    Eigen::Vector3d init_rf_pos = pin_data.oMf[rf_frame_id].translation();
    Eigen::Vector3d init_lf_pos = pin_data.oMf[lf_frame_id].translation();
    std::cout << "  초기 RF   : " << init_rf_pos.transpose() << "\n";
    std::cout << "  초기 LF   : " << init_lf_pos.transpose() << "\n";

    // controller 초기화 (Pinocchio model에서 nv, na, mass 자동 추출)
    G1WalkingController walker(pin_model, init_com_xy, pin_data.com[0].z(),
                               init_rf_pos, init_lf_pos);

    std::cout << "=== Layers 초기화 완료 ===\n";
    std::cout << "  omega = " << std::sqrt(GRAVITY / COM_HEIGHT) << " rad/s\n";

    // GLFW 뷰어
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }
    GLFWwindow* window = glfwCreateWindow(1200, 900, "G1 — WBC", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);

    mjv_defaultCamera(&cam); mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);  mjr_defaultContext(&con);
    mjv_makeScene(g_m, &scn, 5000);
    mjr_makeContext(g_m, &con, mjFONTSCALE_150);
    cam.distance = 3.0; cam.elevation = -20; cam.azimuth = 140;

    // 시각화 옵션 플래그
    opt.flags[mjVIS_TRANSPARENT]  = 1;
    opt.flags[mjVIS_COM]          = 1;
    opt.flags[mjVIS_CONTACTFORCE] = 1;
    opt.flags[mjVIS_CONTACTPOINT] = 1;

    // ── 메인 루프 ──
    int step_cnt  = 0;
    int print_cnt = 0;
    double sim_time = 0.0;

    while (!glfwWindowShouldClose(window))
    {
        // ① 시뮬 전진
        mj_step(g_m, g_d);
        sim_time += MJ_TIMESTEP;

        // ② Pinocchio 상태 최신화
        Eigen::VectorXd q_pin = mj2pin_q(g_d->qpos, pin_model.nq);
        Eigen::VectorXd dq    = Eigen::Map<const Eigen::VectorXd>(g_d->qvel, g_m->nv);
        pinocchio::forwardKinematics(pin_model, pin_data, q_pin, dq,
                                     Eigen::VectorXd::Zero(pin_model.nv));
        pinocchio::updateFramePlacements(pin_model, pin_data);
        pinocchio::centerOfMass(pin_model, pin_data,
                                pinocchio::KinematicLevel::VELOCITY, false);

        // ③ MPC 100Hz — standing 모드에서는 비활성
        // if (step_cnt % MPC_DECIMATION == 0)
        //     walker.mpcLoop(pin_data, sim_time);

        // ④ WBC 1kHz — standing balance (τ_ff + τ_fb)
        {
            Eigen::VectorXd tau = walker.standingLoop(
                pin_model, pin_data, q_pin, dq,
                rf_frame_id, lf_frame_id);

            for (int i = 0; i < g_m->nu && i < static_cast<int>(tau.size()); ++i)
                g_d->ctrl[i] = tau(i);
        }

        ++step_cnt;

        // 출력 (500 스텝마다)
        if (++print_cnt % 500 == 0) {
            std::cout << "\n[step=" << print_cnt << "]\n";
            std::cout << "  com_pin : " << pin_data.com[0].transpose() << "\n";
            std::cout << "  x_state : x=" << walker.x_state_(0)
                      << "  dx="  << walker.x_state_(1)
                      << "  ddx=" << walker.x_state_(2) << "\n";
            std::cout << "  y_state : y=" << walker.y_state_(0)
                      << "  dy="  << walker.y_state_(1)
                      << "  ddy=" << walker.y_state_(2) << "\n";
            std::cout << "  [timing] MPC=" << walker.getMpcSolveUs()
                      << "us  ForceOpt=" << walker.getForceSolveUs()
                      << "us  TorqueGen=" << walker.getTorqueGenUs() << "us\n";
        }

        mjrRect vp = {0,0,0,0};
        glfwGetFramebufferSize(window, &vp.width, &vp.height);
        mjv_updateScene(g_m, g_d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

        // ── 커스텀 시각화 ──
        // 로봇 반투명
        for (int i = 0; i < scn.ngeom; i++) {
            if (scn.geoms[i].category == mjCAT_DYNAMIC ||
                scn.geoms[i].category == mjCAT_DECOR)
                scn.geoms[i].rgba[3] = 0.4f;
        }

        // Contact Force 화살표
        for (int i = 0; i < g_d->ncon; ++i) {
            mjContact* ct = &g_d->contact[i];
            Eigen::Vector3d cpos(ct->pos[0], ct->pos[1], ct->pos[2]);
            mjtNum f_contact[6];
            mj_contactForce(g_m, g_d, i, f_contact);
            Eigen::Matrix3d R_ct;
            R_ct << ct->frame[0], ct->frame[1], ct->frame[2],
                    ct->frame[3], ct->frame[4], ct->frame[5],
                    ct->frame[6], ct->frame[7], ct->frame[8];
            Eigen::Vector3d f_global = R_ct.transpose() *
                Eigen::Vector3d(f_contact[0], f_contact[1], f_contact[2]);
            if (f_global.norm() > 1.0)
                drawArrow(&scn, cpos, f_global, 0.01f, 0.0f, 1.0f, 1.0f);
        }

        // 궤적 시각화
        {
            const auto& zmp_traj = walker.getZmpTrajectory();
            int walk_samples = zmp_traj.getWalkSamples();
            const auto& zmp_x = zmp_traj.getZmpRefX();
            const auto& zmp_y = zmp_traj.getZmpRefY();
            const auto& com_ref = walker.getComRefTraj();

            for (int i = 0; i < walk_samples - 5; i += 5) {
                if (scn.ngeom >= scn.maxgeom - 100) break;
                Eigen::Vector3d p0(zmp_x(i), zmp_y(i), 0.008);
                Eigen::Vector3d p1(zmp_x(i+5), zmp_y(i+5), 0.008);
                if ((p1-p0).norm() > 0.5) continue;
                drawLine(&scn, p0, p1, 0.002f, 0.0f, 1.0f, 0.0f, 0.6f);
            }
            for (int i = 0; i < walk_samples - 5; i += 5) {
                if (scn.ngeom >= scn.maxgeom - 100) break;
                Eigen::Vector3d p0(com_ref(i,0), com_ref(i,1), COM_HEIGHT);
                Eigen::Vector3d p1(com_ref(i+5,0), com_ref(i+5,1), COM_HEIGHT);
                if ((p1-p0).norm() > 0.5) continue;
                drawLine(&scn, p0, p1, 0.002f, 1.0f, 1.0f, 1.0f, 0.6f);
            }
        }

        mjr_render(vp, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    mj_deleteData(g_d);  mj_deleteModel(g_m);
    glfwTerminate();
    return 0;
}
