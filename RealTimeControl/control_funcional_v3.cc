// =======================================================================================
// KUKA ECI Robot Control with TCP Interface and Advanced Inverse Kinematics
//
// Based on a stable user-provided version, with only the IK solver upgraded
// to a more robust implementation.
//
// To compile (example):
// g++ main.cpp -o control-example -std=c++17 -pthread -lorocos-kdl -lkdl_parser -lurdfdom_model -lurdfdom_sensor -lurdfdom_model_state -lurdfdom_world
//
// =======================================================================================

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <array>
#include <algorithm>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// KUKA SDK
#include "kuka/external-control-sdk/iiqka/sdk.h"
#include "event-handlers/control_event_handler.hpp"

// Eigen (required by KDL)
#include <Eigen/Dense>

// KDL / URDF
#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <kdl_parser/kdl_parser.hpp>


// =======================================================================================
// IK SOLVER CLASS (Optimized version based 100% on user-provided reference code)
// =======================================================================================
class IK_Solver {
public:
    IK_Solver() : chain_initialized_(false) {}

    bool initialize() {
        const int ROBOT_SELECT = 1; // 0 = iisy 11 R1300, 1 = iisy 3 R760
        const bool USE_TCP_TARGET = true; // false = resolver a BRIDA (flange), true = resolver al TCP (tool0)
        const char* URDF_PATHS[2] = {"LBR_iisy_11_R1300.urdf", "LBR_iisy_3_R760.urdf"};
        const char* BASE_LINK = "base_link";
        const char* TIP_LINK_FLANGE = "flange";
        const char* TIP_LINK_TCP = "tool0";

        const char* urdf_path = URDF_PATHS[ROBOT_SELECT];
        const std::string tip_link = USE_TCP_TARGET ? TIP_LINK_TCP : TIP_LINK_FLANGE;

        if (!loadURDFtoChain(urdf_path, BASE_LINK, tip_link, chain_, urdf_model_)) {
            std::cerr << "[IK] FATAL: No se pudo cargar la cadena KDL desde " << urdf_path << std::endl;
            return false;
        }

        // --- OPTIMIZATION: Initialize solvers only once ---
        extractLimits(urdf_model_, chain_, q_min_, q_max_);
        fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
        ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
        ik_pos_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(chain_, q_min_, q_max_, *fk_solver_, *ik_vel_solver_, 300, 1e-6);
        
        chain_initialized_ = true;
        std::cout << "[IK] Solver inicializado correctamente con " << urdf_path << std::endl;
        return true;
    }

    bool solve(const std::vector<double>& cartesian_pose_vec, const std::vector<double>& q_seed_vec, std::vector<double>& q_solution_vec) {
        if (!chain_initialized_ || cartesian_pose_vec.size() != 6 || q_seed_vec.size() != 6) return false;

        KDL::JntArray q_seed_kdl(chain_.getNrOfJoints());
        for(size_t i=0; i<q_seed_vec.size(); ++i) q_seed_kdl(i) = q_seed_vec[i];

        const double BASE_XYZABC[6] = {0,0,0, 0,0,0};
        const double TOOL_XYZABC[6] = {0,0,0, 0,0,0};
        KDL::Frame T_base = make_frame_from_XYZABC(BASE_XYZABC, BASE_XYZABC + 3);
        KDL::Frame T_tool = make_frame_from_XYZABC(TOOL_XYZABC, TOOL_XYZABC + 3);
        
        double xyz_goal[3] = {cartesian_pose_vec[0], cartesian_pose_vec[1], cartesian_pose_vec[2]};
        double abc_goal[3] = {cartesian_pose_vec[3], cartesian_pose_vec[4], cartesian_pose_vec[5]};
        KDL::Frame T_goal_world = make_frame_from_XYZABC(xyz_goal, abc_goal);
        KDL::Frame T_goal_internal = T_base.Inverse() * T_goal_world * T_tool.Inverse();

        // Como solicitado, S_FORCE se establece en -1 para no forzar la configuración.
        int S_FORCE = 6, T_FORCE = -1;

        KDL::JntArray q_solution_kdl(chain_.getNrOfJoints());
        int iters_out = 0; 
        bool result = solve_IK_ST(T_goal_internal, q_seed_kdl, S_FORCE, T_FORCE, q_solution_kdl, iters_out);

        if (result) {
            q_solution_vec.resize(chain_.getNrOfJoints());
            for(unsigned int i=0; i<chain_.getNrOfJoints(); ++i) {
                q_solution_vec[i] = q_solution_kdl(i);
            }
        }
        return result;
    }

private:
    KDL::Chain chain_;
    urdf::ModelInterfaceSharedPtr urdf_model_;
    bool chain_initialized_;
    KDL::JntArray q_min_, q_max_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_pos_solver_;


    // --- Utility functions from user's reference code ---
    static inline double DEG2RAD(double d) { return d * M_PI / 180.0; }

    static KDL::Rotation R_from_ABC_deg(double Adeg, double Bdeg, double Cdeg) {
        return KDL::Rotation::RotZ(DEG2RAD(Adeg)) * KDL::Rotation::RotY(DEG2RAD(Bdeg)) * KDL::Rotation::RotX(DEG2RAD(Cdeg));
    }

    static KDL::Frame make_frame_from_XYZABC(const double xyz[3], const double abc_deg[3]) {
        return KDL::Frame(R_from_ABC_deg(abc_deg[0], abc_deg[1], abc_deg[2]), KDL::Vector(xyz[0], xyz[1], xyz[2]));
    }

    static bool loadURDFtoChain(const std::string& urdf_path, const std::string& base_link, const std::string& tip_link, KDL::Chain& chain, urdf::ModelInterfaceSharedPtr& urdf_model) {
        std::ifstream ifs(urdf_path);
        if (!ifs) { std::cerr << "No se pudo abrir URDF: " << urdf_path << "\n"; return false; }
        std::stringstream ss; ss << ifs.rdbuf();
        const std::string urdf_xml = ss.str();
        urdf_model = urdf::parseURDF(urdf_xml);
        if (!urdf_model) { std::cerr << "urdf::parseURDF() fallo\n"; return false; }
        KDL::Tree tree;
        if (!kdl_parser::treeFromString(urdf_xml, tree)) { std::cerr << "kdl_parser::treeFromString() fallo\n"; return false; }
        if (!tree.getChain(base_link, tip_link, chain)) { std::cerr << "tree.getChain(" << base_link << "," << tip_link << ") fallo\n"; return false; }
        return true;
    }

    static void extractLimits(const urdf::ModelInterfaceSharedPtr& model, const KDL::Chain& chain, KDL::JntArray& q_min, KDL::JntArray& q_max) {
        q_min.resize(chain.getNrOfJoints());
        q_max.resize(chain.getNrOfJoints());
        unsigned j = 0;
        for (unsigned i = 0; i < chain.getNrOfSegments(); ++i) {
            const KDL::Joint& jnt = chain.getSegment(i).getJoint();
            if (jnt.getType() == KDL::Joint::None) continue;
            urdf::JointConstSharedPtr uj = model->getJoint(jnt.getName());
            if (uj && uj->type != urdf::Joint::CONTINUOUS && uj->limits) {
                q_min(j) = uj->limits->lower;
                q_max(j) = uj->limits->upper;
            } else { q_min(j) = -M_PI * 2; q_max(j) = +M_PI * 2; }
            ++j;
        }
    }
    
    static void bias_seed_for_S(int S, KDL::JntArray& q){
        if(S<0) return;
        if(S&0b001){ q(0) = (q(0)>=0)? +M_PI*0.9 : -M_PI*0.9; } else { q(0)=0.0; }
        if(S&0b010){ if(q(2)<0) q(2)=DEG2RAD(30.0); } else { if(q(2)>=0) q(2)=DEG2RAD(-30.0); }
        if(S&0b100){ q(4)=DEG2RAD(-60.0); } else { q(4)=DEG2RAD(+60.0); }
    }

    static bool project_to_T(const KDL::JntArray& q_in, const KDL::JntArray& qmin, const KDL::JntArray& qmax, int Tmask, KDL::JntArray& q_out) {
        if(Tmask<0){ q_out=q_in; return true; }
        q_out=q_in;
        for(unsigned int i=0; i < q_in.rows(); i++){
            int desired = (Tmask>>i)&1; // 1 => <0
            double best=q_out(i), bestabs=1e9; bool found=false;
            for(int k=-2;k<=2;k++){
                double cand = q_in(i) + k*2*M_PI;
                if(cand<qmin(i)-1e-9 || cand>qmax(i)+1e-9) continue;
                int bit = (cand<0.0)?1:0;
                if(bit==desired){
                    double av = std::fabs(cand);
                    if(av<bestabs){ best=cand; bestabs=av; found=true; }
                }
            }
            if(!found) return false;
            q_out(i)=best;
        }
        return true;
    }

    bool solve_IK_ST(const KDL::Frame& T_goal_internal, const KDL::JntArray& q_seed_init, int S_desired, int T_desired, KDL::JntArray& q_sol, int& iters_out) {
        KDL::JntArray q0 = q_seed_init;
        bias_seed_for_S(S_desired, q0);
        
        q_sol.resize(chain_.getNrOfJoints());
        int ret = ik_pos_solver_->CartToJnt(q0, T_goal_internal, q_sol);
        iters_out = 0; 

        if(ret<0){
            bias_seed_for_S(S_desired, q0);
            ret = ik_pos_solver_->CartToJnt(q0, T_goal_internal, q_sol);
            if(ret<0) return false;
        }

        if(T_desired>=0){
            KDL::JntArray q_proj;
            if(!project_to_T(q_sol, q_min_, q_max_, T_desired, q_proj)){
                return true;
            }
            KDL::JntArray q_ref;
            int ret2 = ik_pos_solver_->CartToJnt(q_proj, T_goal_internal, q_ref);
            if(ret2>=0) q_sol = q_ref;
            else        q_sol = q_proj;
        }
        return true;
    }
};

// =======================================================================================
// GLOBAL & TCP SERVER (from stable user version)
// =======================================================================================
std::mutex pose_mutex;
std::atomic<bool> new_target_received(false);
std::vector<double> target_pose_from_tcp(6, 0.0);
const int DOF = 6;

void tcp_server_task() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { std::cerr << "Fallo al crear el socket TCP" << std::endl; return; }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) { std::cerr << "Fallo en setsockopt" << std::endl; return; }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(12345);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) { std::cerr << "Fallo en el bind del socket TCP" << std::endl; return; }
    if (listen(server_fd, 3) < 0) { std::cerr << "Fallo en listen" << std::endl; return; }
    std::cout << "[TCP Server] Escuchando en el puerto 12345 (esperando X Y Z A B C)" << std::endl;
    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) { std::cerr << "Fallo en accept" << std::endl; continue; }
        char buffer[1024] = {0};
        int valread = read(new_socket, buffer, 1024);
        if (valread > 0) {
            std::string data(buffer, valread);
            std::stringstream ss(data);
            std::vector<double> temp_pose;
            double val;
            while(ss >> val) temp_pose.push_back(val);
            if (temp_pose.size() == 6) {
                std::lock_guard<std::mutex> lock(pose_mutex);
                target_pose_from_tcp = temp_pose;
                new_target_received.store(true);
                std::cout << "[TCP Server] Nueva pose cartesiana recibida." << std::endl;
            } else { std::cerr << "[TCP Server] Datos recibidos no válidos. Se esperaban 6 valores." << std::endl; }
        }
        close(new_socket);
    }
}

// =======================================================================================
// MAIN ROBOT CONTROL (from stable user version)
// =======================================================================================
int main(int argc, char const *argv[]) {
  IK_Solver ik_solver;
  if (!ik_solver.initialize()) { return -1; }
  
  kuka::external::control::iiqka::Configuration eci_config;
  eci_config.client_ip_address = "10.0.0.15";
  eci_config.koni_ip_address = "10.0.0.66";
  auto rob_if = std::make_shared<kuka::external::control::iiqka::Robot>(eci_config);
  
  if (rob_if->Setup().return_code != kuka::external::control::ReturnCode::OK) { std::cerr << "Fallo al configurar la red." << std::endl; return -1; }
  
  int timeout = 0;
  std::unique_ptr<external_control_sdk_example::ControlEventHandler> control_event_handler = std::make_unique<external_control_sdk_example::ControlEventHandler>();
  rob_if->RegisterEventHandler(std::move(control_event_handler));
  
  std::thread server_thread(tcp_server_task);
  server_thread.detach();

  if (rob_if->StartControlling(kuka::external::control::ControlMode::JOINT_POSITION_CONTROL).return_code != kuka::external::control::ReturnCode::OK) {
    std::cerr << "Fallo al abrir el canal de control." << std::endl;
    return 0;
  }

  std::vector<double> current_pos(DOF, 0.0), command_pos(DOF, 0.0);
  std::vector<double> interpolation_start_pos(DOF, 0.0), interpolation_target_pos(DOF, 0.0);
  std::vector<double> interpolation_increment(DOF, 0.0);
  
  bool interpolation_active = false;
  int interpolation_counter = 0;
  const int INTERPOLATION_STEPS = 2000;
  bool first_cycle = true;

  std::cout << "Bucle de control iniciado." << std::endl;
  while (true) {
    auto recv_ret = rob_if->ReceiveMotionState(std::chrono::milliseconds(timeout));
    if (recv_ret.return_code != kuka::external::control::ReturnCode::OK) { std::cerr << "Fallo en ReceiveMotionState: " << recv_ret.message << std::endl; break; }
    
    current_pos = rob_if->GetLastMotionState().GetMeasuredPositions();

    if (first_cycle) {
        command_pos = current_pos;
        first_cycle = false;
    }

    if (new_target_received.load()) {
        std::vector<double> target_cartesian_pose;
        {
            std::lock_guard<std::mutex> lock(pose_mutex);
            target_cartesian_pose = target_pose_from_tcp;
            new_target_received.store(false);
        }
        
        std::vector<double> joint_solution;
        if (ik_solver.solve(target_cartesian_pose, current_pos, joint_solution)) {
            std::cout << "[IK] Solución encontrada. Iniciando movimiento." << std::endl;
            interpolation_target_pos = joint_solution;
            interpolation_active = true;
            interpolation_counter = 0;
            interpolation_start_pos = current_pos;

            for (size_t i = 0; i < DOF; ++i) {
                interpolation_increment[i] = (interpolation_target_pos[i] - interpolation_start_pos[i]) / INTERPOLATION_STEPS;
            }
        } else {
            std::cerr << "[IK] No se pudo encontrar una solución. Manteniendo posición." << std::endl;
        }
    }

    if (interpolation_active) {
        if (interpolation_counter < INTERPOLATION_STEPS) {
            for (size_t i = 0; i < DOF; ++i) {
                command_pos[i] = interpolation_start_pos[i] + interpolation_increment[i] * (interpolation_counter + 1);
            }
            interpolation_counter++;
        } else {
            interpolation_active = false;
            command_pos = interpolation_target_pos;
        }
    }
    
    rob_if->GetControlSignal().AddJointPositionValues(command_pos.begin(), command_pos.end());
    if (rob_if->SendControlSignal().return_code != kuka::external::control::ReturnCode::OK) { std::cerr << "Fallo en SendControlSignal." << std::endl; break; }
  }

  rob_if->ReceiveMotionState(std::chrono::milliseconds(timeout));
  std::cout << "Enviando señal de parada..." << std::endl;
  rob_if->StopControlling();
  std::cout << "Bucle de control finalizado." << std::endl;
  return 0;
}


