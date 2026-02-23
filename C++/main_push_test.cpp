// C++ API 기반 Mujoco + Pinocchio 공통 (Push Test — mjvPerturb 외력 인가)
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

#include "config.hpp"
#include "main_controller/g1_walking_controller.hpp"

// ── 경로 ──────────────────────────────────────────────────────
static const char* MJ_XML   = "/home/ycm/FrJoCo/model/xml/scene_29dof.xml";
static const char* PIN_URDF = "/home/ycm/FrJoCo/model/urdf/g1_29dof.urdf";

// ── GLFW / MuJoCo 전역 ────────────────────────────────────────
static mjModel*   g_m = nullptr;
static mjData*    g_d = nullptr;
static mjvCamera  cam;
static mjvOption  opt;
static mjvScene   scn;
static mjrContext con;
static mjvPerturb pert;          // ← 외력 인가용

static bool   btn_l = false, btn_r = false;
static double lastx = 0, lasty = 0;

// ── 콜백 ──────────────────────────────────────────────────────
void keyboard(GLFWwindow*, int key, int, int act, int) {
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(g_m, g_d); mj_forward(g_m, g_d);
    }
    // ESC: perturb 해제
    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        pert.active = 0;
        pert.select = 0;
    }
}

void mouse_button(GLFWwindow* w, int button, int act, int) {
    btn_l = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS);
    btn_r = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    double x, y;
    glfwGetCursorPos(w, &x, &y);
    lastx = x; lasty = y;

    // 더블클릭으로 바디 선택 → perturb 활성화
    if (act == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
        static double last_click_time = 0.0;
        double now = glfwGetTime();
        bool double_click = (now - last_click_time) < 0.3;
        last_click_time = now;

        if (double_click) {
            int W, H;
            glfwGetWindowSize(w, &W, &H);
            mjtNum selpnt[3];
            int geomid[1] = {-1}, flexid[1] = {-1}, skinid[1] = {-1};
            int selbody = mjv_select(g_m, g_d, &opt,
                                     (mjtNum)W / H,
                                     (mjtNum)x / W,
                                     (mjtNum)(H - y) / H,
                                     &scn, selpnt, geomid, flexid, skinid);
            if (selbody >= 0) {
                pert.select     = selbody;
                pert.skinselect = skinid[0];
                mju_copy3(pert.refpos, selpnt);
                mju_copy3(pert.refselpos, selpnt);
                mjv_initPerturb(g_m, g_d, &scn, &pert);
            } else {
                pert.select = 0;
                pert.active = 0;
            }
        }
    }

    // 버튼 떼면 perturb force 해제 (pose는 유지)
    if (act == GLFW_RELEASE) {
        pert.active = 0;
    }
}

void mouse_move(GLFWwindow* w, double xpos, double ypos) {
    double dx = xpos - lastx, dy = ypos - lasty;
    lastx = xpos; lasty = ypos;

    if (!btn_l && !btn_r) return;

    int W, H;
    glfwGetWindowSize(w, &W, &H);
    bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
    bool ctrl  = (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS);

    // Ctrl + 드래그 → perturb (바디 선택된 경우)
    if (ctrl && pert.select > 0) {
        mjtMouse action = btn_r ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        pert.active = mjPERT_TRANSLATE;
        mjv_movePerturb(g_m, g_d, action, dx / H, dy / H, &scn, &pert);
        return;
    }

    // 일반 카메라 조작
    mjtMouse action = btn_r ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                             : (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);
    mjv_moveCamera(g_m, action, dx / H, dy / H, &scn, &cam);
}

void scroll(GLFWwindow*, double, double dy) {
    mjv_moveCamera(g_m, mjMOUSE_ZOOM, 0, -0.05 * dy, &scn, &cam);
}

// ── MuJoCo → Pinocchio 변환 ───────────────────────────────────
Eigen::VectorXd mj2pin_q(const double* qpos, int nq_pin) {
    Eigen::VectorXd q(nq_pin);
    q[0]=qpos[0]; q[1]=qpos[1]; q[2]=qpos[2];
    q[3]=qpos[4]; q[4]=qpos[5]; q[5]=qpos[6]; q[6]=qpos[3];
    for (int i = 7; i < nq_pin; ++i) q[i] = qpos[i];
    return q;
}

// ── 시각화 헬퍼 ───────────────────────────────────────────────
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
    char err[1000] = {};
    g_m = mj_loadXML(MJ_XML, nullptr, err, sizeof(err));
    if (!g_m) { std::cerr << "MuJoCo load failed: " << err << "\n"; return 1; }
    g_d = mj_makeData(g_m);

    std::cout << "=== MuJoCo Model ===\n";
    std::cout << "  nq=" << g_m->nq << "  nv=" << g_m->nv << "  nu=" << g_m->nu << "\n";

    g_d->qpos[0]=0.0; g_d->qpos[1]=0.0; g_d->qpos[2]=0.755;
    g_d->qpos[3]=1.0; g_d->qpos[4]=0.0; g_d->qpos[5]=0.0; g_d->qpos[6]=0.0;
    g_d->qpos[7]=-0.312; g_d->qpos[8]=0.0;  g_d->qpos[9]=0.0;
    g_d->qpos[10]=0.669; g_d->qpos[11]=-0.363; g_d->qpos[12]=0.0;
    g_d->qpos[13]=-0.312; g_d->qpos[14]=0.0; g_d->qpos[15]=0.0;
    g_d->qpos[16]=0.669;  g_d->qpos[17]=-0.363; g_d->qpos[18]=0.0;
    g_d->qpos[19]=0.073;  g_d->qpos[20]=0.0; g_d->qpos[21]=0.0;
    mj_forward(g_m, g_d);

    pinocchio::Model pin_model;
    try {
        pinocchio::urdf::buildModel(PIN_URDF, pinocchio::JointModelFreeFlyer(), pin_model);
    } catch (const std::exception& e) {
        std::cerr << "[Pinocchio] 로드 실패: " << e.what() << "\n"; return 1;
    }
    pinocchio::Data pin_data(pin_model);

    Eigen::VectorXd q0 = mj2pin_q(g_d->qpos, pin_model.nq);
    pinocchio::forwardKinematics(pin_model, pin_data, q0);
    pinocchio::updateFramePlacements(pin_model, pin_data);
    pinocchio::centerOfMass(pin_model, pin_data, q0, false);
    Eigen::Vector2d init_com_xy = pin_data.com[0].head<2>();

    int rf_frame_id = pin_model.getFrameId("right_ankle_roll_link");
    int lf_frame_id = pin_model.getFrameId("left_ankle_roll_link");
    Eigen::Vector3d init_rf_pos = pin_data.oMf[rf_frame_id].translation();
    Eigen::Vector3d init_lf_pos = pin_data.oMf[lf_frame_id].translation();

    G1WalkingController walker(pin_model, init_com_xy, pin_data.com[0].z(),
                               init_rf_pos, init_lf_pos);

    std::cout << "=== Push Test 준비 완료 ===\n";
    std::cout << "  더블클릭으로 바디 선택 → Ctrl+드래그로 외력 인가\n";
    std::cout << "  ESC: perturb 해제  |  Backspace: 리셋\n";

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return 1; }
    GLFWwindow* window = glfwCreateWindow(1200, 900, "G1 — Push Test", nullptr, nullptr);
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
    mjv_defaultPerturb(&pert);   // perturb 초기화

    cam.distance = 3.0; cam.elevation = -20; cam.azimuth = 140;
    opt.flags[mjVIS_TRANSPARENT]  = 1;
    opt.flags[mjVIS_COM]          = 1;
    opt.flags[mjVIS_CONTACTFORCE] = 1;
    opt.flags[mjVIS_CONTACTPOINT] = 1;

    int step_cnt = 0, print_cnt = 0;
    double sim_time = 0.0;

    while (!glfwWindowShouldClose(window))
    {
        // ① 외력 인가 후 시뮬 전진
        mjv_applyPerturbForce(g_m, g_d, &pert);
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

        // ③ WBC
        {
            Eigen::VectorXd tau = walker.standingLoop(
                pin_model, pin_data, q_pin, dq,
                rf_frame_id, lf_frame_id);
            for (int i = 0; i < g_m->nu && i < static_cast<int>(tau.size()); ++i)
                g_d->ctrl[i] = tau(i);
        }

        ++step_cnt;
        if (++print_cnt % 500 == 0) {
            std::cout << "[step=" << print_cnt << "]"
                      << "  com: " << pin_data.com[0].transpose()
                      << "  perturb_body=" << pert.select
                      << "  active=" << pert.active << "\n";
        }

        // ④ 렌더링
        mjrRect vp = {0,0,0,0};
        glfwGetFramebufferSize(window, &vp.width, &vp.height);
        mjv_updateScene(g_m, g_d, &opt, &pert, &cam, mjCAT_ALL, &scn);

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

        mjr_render(vp, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    mjv_freeScene(&scn); mjr_freeContext(&con);
    mj_deleteData(g_d);  mj_deleteModel(g_m);
    glfwTerminate();
    return 0;
}
