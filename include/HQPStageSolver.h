/*--------------------------------------------------------------------------*/
/*------------------------- File HQPStageSolver.h --------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the class HQPStageSolver, a Solver for one level of an
 * HQPBlock with a stage structure, i.e., one level of the hierarchical
 * Model Predictive Control of a system with linear dynamics, which it
 * solves by a primal active-set method whose equality-constrained
 * subproblems are solved by a Riccati recursion with stage equalities.
 *
 * \author Antonio Frangioni \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Antonio Frangioni, Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*----------------------------- DEFINITIONS --------------------------------*/
/*--------------------------------------------------------------------------*/

#ifndef __HQPStageSolver
 #define __HQPStageSolver
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <Eigen/Dense>

#include "HQPBlock.h"

#include "Solver.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup HQPStageSolver_CLASSES Classes in HQPStageSolver.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*------------------------- CLASS HQPStageSolver ---------------------------*/
/*--------------------------------------------------------------------------*/
/// a Solver for one level of an HQPBlock with a stage structure
/** The HQPStageSolver class implements the Solver interface for the
 * problem \f$ (L_p) \f$ of the current level \f$ p \f$ of an HQPBlock [see
 * HQPBlock.h and HQPBlock::get_level()] that has a stage structure [see
 * HQPBlock::StageData], e.g., as the inner Solver of each level of an
 * HQPSolver [see HQPSolver.h]. The columns are the inputs \f$ u_k \f$, \f$ k
 * = 0 , \ldots , N - 1 \f$, and the states \f$ s_k \f$, \f$ k = 0 , \ldots
 * , N \f$ (those of \f$ s_0 \f$ possibly none), and the equality rows of
 * the level \f$ D \f$ of the dynamics are \f$ s_{k+1} = A_k s_k + B_k u_k +
 * c_k \f$. With the state augmented by the previous input, \f$ x_k = ( s_k
 * , u_{k-1} ) \f$, they read
 *
 * \f[
 *   x_{k+1} = \hat A_k x_k + \hat B_k u_k + \hat c_k \;\;,\qquad
 *   \hat A_k = \begin{pmatrix} A_k & 0 \\ 0 & 0 \end{pmatrix} \;,\;
 *   \hat B_k = \begin{pmatrix} B_k \\ I \end{pmatrix} \;,\;
 *   \hat c_k = \begin{pmatrix} c_k \\ 0 \end{pmatrix} \;\;,
 * \f]
 *
 * and each row of stage \f$ k \f$, which may have entries on \f$ u_k \f$,
 * \f$ s_{k+1} \f$, \f$ s_k \f$ and \f$ u_{k-1} \f$ only, is written as
 * \f$ \delta^\top x_k + \eta^\top u_k + \kappa \f$ by substituting the
 * dynamics for \f$ s_{k+1} \f$, which is exact for every \f$ x \f$ that
 * satisfies them. At every level \f$ p > D \f$ the dynamics are hard
 * equalities, their slacks being fixed, and at level \f$ D \f$ their
 * slacks \f$ v \f$ are additional inputs, \f$ s_{k+1} = A_k s_k + B_k u_k
 * + c_k + ( W^e a )^{-1} v_k \f$ with \f$ a \f$ the coefficient of \f$
 * s_{k+1} \f$ in each row, whose cost is \f$ \tfrac{1}{2} \| v_k \|^2
 * \f$ (they are 0 at the optimum only without regularization or with \f$
 * c_k = 0 \f$); hence \f$ (L_p) \f$ is the optimal control problem
 *
 * \f[
 *   \min \Big\{ \, \sum_{k \leq N} \ell_k( x_k , u_k ) \;:\;
 *   x_{k+1} = \hat A_k x_k + \hat B_k u_k + \hat c_k \,,\;
 *   E_k z_k = e_k \,,\; F_k z_k \leq f_k \, \Big\}
 * \f]
 *
 * with \f$ z_k = ( x_k , u_k ) \f$, where \f$ E_k z_k = e_k \f$ are the
 * equality rows of stage \f$ k \f$ of the levels \f$ q < p \f$, with the
 * right-hand side moved by their optimal slack \f$ ( W^e_q )^{-1} v^*_q
 * \f$, \f$ F_k z_k \leq f_k \f$ their inequality rows, with the bound
 * moved by \f$ ( W^i_q )^{-1} w^*_q \f$, and \f$ \ell_k \f$ the
 * regularization of the columns of stage \f$ k \f$ plus the terms of the
 * tasks of level \f$ p \f$ of stage \f$ k \f$, \f$ \tfrac{1}{2} ( W^e (
 * A z - b ) )^2 \f$ for an equality one and \f$ \tfrac{1}{2} \max \{ 0 ,
 * W^i ( C z - d ) \}^2 \f$ for an inequality one; the levels below \f$ p
 * \f$ are left out, whatever the state of their rows.
 *
 * The problem is solved by a primal active-set method that starts from the
 * current \f$ x \f$ of the HQPBlock, which in a cascade is the solution of
 * level \f$ p - 1 \f$, feasible for \f$ (L_p) \f$, with the working set
 * \f$ \mathcal{W} \f$ of the inequality rows of the levels \f$ q < p \f$
 * that are active there (those that are violated are made active first by
 * projection steps). Each iteration solves, by the recursion of StageEQP,
 * the equality-constrained problem with the rows of \f$ \mathcal{W} \f$ as
 * equalities and the inequality tasks of level \f$ p \f$ that are violated
 * at the current point, \f$ \mathcal{V} \f$, as least-squares terms, and
 * then
 *
 * - if the current point is not its solution, moves towards it up to the
 *   minimum of the objective of \f$ (L_p) \f$ along the step (a convex
 *   piecewise quadratic function, whose breakpoints are where the
 *   inequality tasks of level \f$ p \f$ change sign) or up to the first
 *   row out of \f$ \mathcal{W} \f$ that blocks it, which enters \f$
 *   \mathcal{W} \f$; \f$ \mathcal{V} \f$ is then the set of the tasks that
 *   are violated at the new point, so that the step is that of a semismooth
 *   Newton method for the tasks of level \f$ p \f$; a step along which the
 *   objective does not decrease, which only the rounding of an
 *   ill-conditioned subproblem gives, is taken to its end or to the first
 *   blocking row, so that the multipliers are those of the point where
 *   they are read;
 *
 * - otherwise, the row of \f$ \mathcal{W} \f$ with the most negative
 *   multiplier (if below \f$ - \f$ dblDualTol) leaves \f$ \mathcal{W} \f$,
 *   and if there is none the point is optimal.
 *
 * A row blocks a step only if the whole step would violate it by more than
 * half of dblFeasTol, so that a row that depends on the working set, which
 * the step moves only by rounding, does not; the rows of the working set
 * are scaled to largest coefficient 1, and a subproblem whose solution
 * violates them by more than dblFeasTol is solved again with a larger
 * tolerance of the rank decisions [see dblRankTol].
 *
 * The working set and \f$ \mathcal{V} \f$ of the last compute() are kept:
 * with intWarmStart == 1 the next compute() first solves the subproblem with
 * the rows active at its starting point and those of the previous working
 * set (by their index in the HQPBlock) and with the previous \f$ \mathcal{V}
 * \f$, and starts from its solution if it satisfies all the rows, from its
 * starting point otherwise; e.g., the same level in the next step of a
 * Model Predictive Control.
 *
 * The data are read through the methods of the HQPBlock (the matrices, the
 * right-hand sides, the weights, the regularization and its center, the
 * slacks of the higher levels at which they are fixed). The dynamics and
 * the rows written on the stages are computed the first time and again only
 * after a Modification that changes the coefficients of some row (e.g.,
 * HQPBlock::set_A()), or after an NBModification, while a change of the
 * right-hand sides only is read by the next compute().
 *
 * compute() throws std::invalid_argument if the Block is not an HQPBlock,
 * if it has no stage structure, if the current level is above that of the
 * dynamics, if the rows of the level of the dynamics are not the dynamics
 * described in HQPBlock::StageData or one of them has weight 0, or if a row
 * of a level not below the current one has an entry out of \f$ u_k \f$,
 * \f$ s_{k+1} \f$, \f$ s_k \f$ and \f$ u_{k-1} \f$, \f$ k \f$ being its
 * stage. */

class HQPStageSolver : public Solver
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/

 using Index = Block::Index;                ///< the index of Block
 using Vec_double = std::vector< double >;  ///< a vector of double

 using Mat = Eigen::MatrixXd;               ///< a dense matrix
 using Vec = Eigen::VectorXd;               ///< a dense vector

 /// public enum for the int parameters of HQPStageSolver
 enum int_par_type_HQPSS {
  intWarmStart = intLastAlgPar ,  ///< warm start from the previous compute()
  intLastHQPSSPar  ///< first allowed int parameter for derived classes
  };

 /// public enum for the double parameters of HQPStageSolver
 enum dbl_par_type_HQPSS {
  dblRankTol = dblLastAlgPar ,  ///< tolerance of the rank decisions
  dblFeasTol ,                  ///< tolerance on the rows
  dblDualTol ,                  ///< tolerance on the multipliers
  dblLastHQPSSPar  ///< first allowed double parameter for derived classes
  };

/*--------------------------------------------------------------------------*/
 /// an equality-constrained linear-quadratic problem over N stages
 /** The StageEQP class holds and solves the problem
  *
  * \f[
  *   \min \sum_{k < N} \Big( \tfrac{1}{2} x_k^\top Q_k x_k + u_k^\top S_k
  *   x_k + \tfrac{1}{2} u_k^\top R_k u_k + q_k^\top x_k + r_k^\top u_k
  *   \Big) + \tfrac{1}{2} x_N^\top Q_N x_N + q_N^\top x_N
  * \f]
  *
  * subject to \f$ x_{k+1} = A_k x_k + B_k u_k + c_k \f$ and \f$ G_k x_k +
  * H_k u_k = g_k \f$ for \f$ k < N \f$, and \f$ G_N x_N = g_N \f$, where
  * \f$ x_0 \f$ is free (its size may be 0) and all the sizes may change from
  * a stage to the next one. The rows may be dependent, also through the
  * dynamics, and solve() runs, backward from \f$ N \f$, the recursion of a
  * cost-to-go \f$ \tfrac{1}{2} x^\top P_k x + p_k^\top x \f$ and of a
  * constraint-to-go \f$ \Gamma_k x = \gamma_k \f$ (the rows that \f$ x_k \f$
  * must satisfy for the stages from \f$ k \f$ on to be feasible): at stage
  * \f$ k \f$, with \f$ P = P_{k+1} \f$ and \f$ \bar p = p_{k+1} + P c_k \f$,
  *
  * \f[
  *   Q_{xx} = Q_k + A_k^\top P A_k \,,\; Q_{uu} = R_k + B_k^\top P B_k
  *   \,,\; Q_{ux} = S_k + B_k^\top P A_k \,,\; q_x = q_k + A_k^\top \bar p
  *   \,,\; q_u = r_k + B_k^\top \bar p \;,
  * \f]
  *
  * the rows of stage \f$ k \f$ and those of \f$ \Gamma_{k+1} \f$ through
  * the dynamics are \f$ C_x x_k + C_u u_k = e \f$, with \f$ C_x = ( G_k ;
  * \Gamma_{k+1} A_k ) \f$, \f$ C_u = ( H_k ; \Gamma_{k+1} B_k ) \f$, \f$ e =
  * ( g_k ; \gamma_{k+1} - \Gamma_{k+1} c_k ) \f$, and the singular value
  * decomposition \f$ C_u = U_1 \Sigma_1 V_1^\top \f$, \f$ U = ( U_1 , U_2
  * ) \f$, \f$ V = ( V_1 , V_2 ) \f$, gives
  *
  * \f[
  *   u_k = K^c x_k + k^c + V_2 \zeta \;,\quad K^c = - V_1 \Sigma_1^{-1}
  *   U_1^\top C_x \;,\quad k^c = V_1 \Sigma_1^{-1} U_1^\top e \;,
  * \f]
  *
  * the rows \f$ U_2^\top C_x x_k = U_2^\top e \f$ being reduced to
  * independent ones, \f$ \Gamma_k = T_k U_2^\top C_x \f$, by a second
  * decomposition; minimizing over \f$ \zeta \f$, with \f$ Z = V_2^\top
  * Q_{uu} V_2 \f$,
  *
  * \f[
  *   K_k = K^c - V_2 Z^{-1} V_2^\top ( Q_{uu} K^c + Q_{ux} ) \;,\quad
  *   k_k = k^c - V_2 Z^{-1} V_2^\top ( Q_{uu} k^c + q_u ) \;,
  * \f]
  * \f[
  *   P_k = Q_{xx} + K_k^\top Q_{uu} K_k + K_k^\top Q_{ux} + Q_{ux}^\top K_k
  *   \;,\quad p_k = q_x + K_k^\top Q_{uu} k_k + K_k^\top q_u + Q_{ux}^\top
  *   k_k \;.
  * \f]
  *
  * A singular value is taken as 0 if it is below \f$ \tau \max \{
  * \sigma_1 , \| ( C_x , C_u ) \|_\infty \} \f$, \f$ \sigma_1 \f$ the
  * largest one of the matrix, the norm the largest absolute value of an
  * entry of the rows of the stage and \f$ \tau \f$ the largest of the
  * tolerance given to solve() and of the machine precision times the
  * largest size of the matrix. At stage 0, \f$ x_0 \f$ minimizes
  * \f$ \tfrac{1}{2} x^\top P_0 x + p_0^\top x \f$ on \f$ \Gamma_0 x =
  * \gamma_0 \f$, by the same decomposition, and a forward sweep gives the
  * trajectory. The multipliers of the rows follow from the stationarity
  * conditions: the costate of \f$ x_{k+1} \f$ is \f$ P_{k+1} x_{k+1} +
  * p_{k+1} + \Gamma_{k+1}^\top \nu_{k+1} \f$, and the multipliers \f$ \pi
  * \f$ of the rows \f$ C_x x_k + C_u u_k = e \f$, i.e., those of the rows
  * of stage \f$ k \f$ and \f$ \nu_{k+1} \f$, are, forward from the \f$ \nu_0
  * \f$ of stage 0,
  *
  * \f[
  *   \pi = - U_1 \Sigma_1^{-1} V_1^\top ( Q_{uu} u_k + Q_{ux} x_k + q_u )
  *   + U_2 T_k^\top \nu_k \;,
  * \f]
  *
  * with the sign of the Lagrangian \f$ f + \mu^\top ( G x + H u - g ) \f$;
  * the multipliers of dependent rows are those of minimum norm at each
  * stage. Each stage costs \f$ O( ( n_x + n_u + m_k )^3 ) \f$, the
  * constraint-to-go having at most \f$ n_x \f$ rows. A dependent row that is
  * not consistent with the others is dropped as well, and the largest
  * absolute value of the dropped combinations of the right-hand sides
  * (\f$ U_2^\top e \f$ beyond the rank of the second decomposition, and the
  * rows of \f$ \Gamma_0 \f$ if \f$ x_0 \f$ has size 0) is the inconsistency
  * of the rows. */

 class StageEQP
 {
  public:

  /// the number N of stages
  [[nodiscard]] Index num_stages( void ) const { return( A.size() ); }

  /// resizes the data to N stages, all the matrices and vectors empty
  void resize( Index N );

  /// solves the problem, with tol the tolerance of the rank decisions
  /** Solves the problem, writing the trajectory in x and u and the
   * multipliers of the rows of each stage in mu, and returns the
   * inconsistency of the rows [see above]. The sizes must agree: \f$ A_k
   * \f$ is \f$ n_{k+1} \times n_k \f$, \f$ B_k \f$ is \f$ n_{k+1} \times
   * m_k \f$ and so on, and \f$ H_N \f$ is ignored. If \f$ Z \f$ is not
   * positive definite (e.g., no regularization), the solution of minimum
   * norm of its system is taken. */

  double solve( double tol );

  std::vector< Mat > A , B , Q , S , R , G , H;  ///< the matrices
  std::vector< Vec > c , q , r , g;              ///< the vectors

  std::vector< Vec > x , u , mu;   ///< the solution and the multipliers

  Index dropped = 0;   ///< the dependent rows dropped by the last solve()
  Index to_go = 0;     ///< the rows of the constraints-to-go in it

  protected:

  /// what the recursion keeps of each stage for the forward sweep
  struct Stage {
   Mat K;          ///< the feedback
   Vec k;          ///< the feedforward
   Mat Quu , Qux;  ///< the Hessian blocks of the stage
   Vec qu;         ///< the gradient block in u
   Mat U , V;      ///< the decomposition of C_u
   Vec sigma;      ///< its non-zero singular values
   Mat T;          ///< the independent rows of U_2^T C_x
   Index m = 0;    ///< the number of rows of the stage
   };

  std::vector< Stage > v_st;   ///< one for each stage, N + 1 for x_0
 };

/*---------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------*/
 /// constructor: the HQPStageSolver has no Block

 HQPStageSolver( void ) : Solver() {}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// destructor: it does nothing

 ~HQPStageSolver() override = default;

/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
 /// sets the Block, which compute() requires to be an HQPBlock
 /** Sets the Block of the HQPStageSolver, forgetting the dynamics, the rows
  * written on the stages and the working set of the previous one. */

 void set_Block( Block * block ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// sets the int parameters of HQPStageSolver
 /** Sets the int parameters specific of HQPStageSolver:
  *
  * - intWarmStart [1]: 0, each compute() starts from the rows active at
  *   its starting point only; 1, it first tries the working set and the
  *   violated tasks of the previous compute(), by the index of the rows in
  *   the HQPBlock [see the general notes]; the other values are reserved
  *   for warm starts still to be defined, and rejected;
  *
  * - intMaxIter [10000]: the maximum number of subproblems solved by a
  *   compute(), after which it stops with kStopIter;
  *
  * - intLogVerb [0]: if positive, a line for each compute() is printed on
  *   the log, if larger than 1 also one for each iteration, if larger than
  *   2 also one for each rank tolerance tried [see dblRankTol]. */

 void set_par( idx_type par , int value ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// sets the double parameters of HQPStageSolver
 /** Sets the double parameters specific of HQPStageSolver:
  *
  * - dblRankTol [0]: the relative tolerance \f$ \tau \f$ of the rank
  *   decisions of the recursion [see StageEQP], which takes the machine
  *   precision times the size of the matrix if larger (as it is for 0); a
  *   subproblem whose solution violates a row of the working set by more
  *   than dblFeasTol is solved again with \f$ 100 \tau \f$ (at least
  *   1e-14), up to 1e-6, since rows that are dependent but for rounding
  *   make it ill-conditioned;
  *
  * - dblFeasTol [1e-9]: a row \f$ a^\top x \leq f \f$ is active if \f$ f
  *   - a^\top x \leq \f$ dblFeasTol \f$ ( 1 + | f | ) \f$, and violated if
  *   \f$ a^\top x - f > \f$ dblFeasTol \f$ ( 1 + | f | ) \f$;
  *
  * - dblDualTol [1e-9]: a row of the working set leaves it if its
  *   multiplier is below \f$ - \f$ dblDualTol. */

 void set_par( idx_type par , double value ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// receives a Modification of the HQPBlock
 /** Records whether the Modification changes the coefficients of some row
  * (a FunctionMod or a FunctionModVars, also inside a GroupModification),
  * or the whole HQPBlock (an NBModification, which also clears the working
  * set), and then the dynamics and the rows written on the stages are
  * computed anew by the next compute(); any other one is read from the
  * HQPBlock by the next compute(). */

 void add_Modification( sp_Mod & mod ) override;

/*--------------------- METHODS FOR SOLVING THE MODEL ----------------------*/
 /// solves the problem of the current level of the HQPBlock
 /** Solves \f$ (L_p) \f$ for the current level \f$ p \f$ of the HQPBlock
  * [see the general notes], returning kOK if optimal, kStopIter if
  * intMaxIter subproblems have been solved first, kError if the rows active
  * at the starting point cannot be satisfied (the projection steps fail)
  * and kBlockLocked if the Block cannot be read_lock()-ed. Throws
  * std::invalid_argument if the HQPBlock does not fit [see the general
  * notes]. */

 int compute( bool changedvars = true ) override;

/*---------------------- METHODS FOR READING RESULTS -----------------------*/
 /// returns the value of the Objective of the HQPBlock at the solution
 /** Returns the value of the Objective of the HQPBlock at the solution
  * found by the last compute() (+ infinity if none): the slacks of level
  * \f$ p \f$, those fixed of the higher levels and the regularization, all
  * multiplied by HQPBlock::get_objective_scale(). */

 [[nodiscard]] OFValue get_var_value( void ) override {
  return( f_has_sol ? f_value : Inf< OFValue >() );
  }

 /// returns get_var_value(), the solution being optimal

 [[nodiscard]] OFValue get_ub( void ) override { return( get_var_value() ); }

 /// returns get_var_value() if kOK, - infinity otherwise

 [[nodiscard]] OFValue get_lb( void ) override {
  return( f_status == kOK ? get_var_value() : - Inf< OFValue >() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns true if the last compute() found a solution

 [[nodiscard]] bool has_var_solution( void ) override { return( f_has_sol ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes the solution into the HQPBlock
 /** Writes into the ColVariable of the HQPBlock the \f$ x \f$ of the last
  * solution and the slacks of its level at it, \f$ v_p = W^e_p ( A_p x -
  * b_p ) \f$ and \f$ w_p = \max \{ 0 , W^i_p ( C_p x - d_p ) \} \f$. */

 void get_var_solution( Configuration * solc = nullptr ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number of subproblems solved by the last compute()

 [[nodiscard]] long get_elapsed_iterations( void ) const override {
  return( f_iter );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the size of the working set at the end of the last compute()

 [[nodiscard]] Index get_num_working( void ) const { return( v_W.size() ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number of violated tasks of the level at the solution

 [[nodiscard]] Index get_num_violated( void ) const { return( v_V.size() ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the largest inconsistency of the rows in the last compute()
 /** Returns the largest inconsistency of the rows [see StageEQP::solve()]
  * among the subproblems solved by the last compute(). */

 [[nodiscard]] double get_inconsistency( void ) const { return( f_incons ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns true if the last compute() started from the previous one
 /** Returns true if the last compute() started from the solution of the
  * subproblem of the working set of the previous one [see intWarmStart]. */

 [[nodiscard]] bool warm_started( void ) const { return( f_warm ); }

/*----------------- METHODS FOR HANDLING THE PARAMETERS --------------------*/
 /// returns the number of int parameters, those of Solver included

 [[nodiscard]] idx_type get_num_int_par( void ) const override {
  return( idx_type( intLastHQPSSPar ) );
  }

 /// returns the number of double parameters, those of Solver included

 [[nodiscard]] idx_type get_num_dbl_par( void ) const override {
  return( idx_type( dblLastHQPSSPar ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the default value of the int parameter par [see set_par()]

 [[nodiscard]] int get_dflt_int_par( idx_type par ) const override {
  if( par == intWarmStart )
   return( 1 );
  if( par == intMaxIter )
   return( 10000 );
  return( Solver::get_dflt_int_par( par ) );
  }

 /// returns the default value of the double parameter par

 [[nodiscard]] double get_dflt_dbl_par( idx_type par ) const override {
  if( par == dblRankTol )
   return( 0 );
  if( ( par == dblFeasTol ) || ( par == dblDualTol ) )
   return( 1e-9 );
  return( Solver::get_dflt_dbl_par( par ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the current value of the int parameter par

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == intWarmStart )
   return( f_warm_start );
  if( par == intMaxIter )
   return( f_max_iter );
  if( par == intLogVerb )
   return( f_verb );
  return( Solver::get_int_par( par ) );
  }

 /// returns the current value of the double parameter par

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dblRankTol )
   return( f_rank_tol );
  if( par == dblFeasTol )
   return( f_feas_tol );
  if( par == dblDualTol )
   return( f_dual_tol );
  return( Solver::get_dbl_par( par ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the index of the int parameter with the given name

 [[nodiscard]] idx_type int_par_str2idx( const std::string & name )
  const override {
  if( name == "intWarmStart" )
   return( intWarmStart );
  return( Solver::int_par_str2idx( name ) );
  }

 /// returns the index of the double parameter with the given name

 [[nodiscard]] idx_type dbl_par_str2idx( const std::string & name )
  const override {
  if( name == "dblRankTol" )
   return( dblRankTol );
  if( name == "dblFeasTol" )
   return( dblFeasTol );
  if( name == "dblDualTol" )
   return( dblDualTol );
  return( Solver::dbl_par_str2idx( name ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the name of the int parameter with the given index

 [[nodiscard]] const std::string & int_par_idx2str( idx_type idx )
  const override {
  static const std::string par = "intWarmStart";
  if( idx == intWarmStart )
   return( par );
  return( Solver::int_par_idx2str( idx ) );
  }

 /// returns the name of the double parameter with the given index

 [[nodiscard]] const std::string & dbl_par_idx2str( idx_type idx )
  const override {
  static const std::string pars[] = { "dblRankTol" , "dblFeasTol" ,
                                      "dblDualTol" };
  if( ( idx >= dblRankTol ) && ( idx < dblLastHQPSSPar ) )
   return( pars[ idx - dblRankTol ] );
  return( Solver::dbl_par_idx2str( idx ) );
  }

/*-------------------- PROTECTED PART OF THE CLASS -------------------------*/

 protected:

/*--------------------------- PROTECTED TYPES ------------------------------*/

 /// a row of the HQPBlock written on its stage
 /** A row \f$ a^\top x \f$ of the HQPBlock written on its stage \f$ k \f$
  * as \f$ \delta^\top x_k + \eta^\top u_k + \alpha^\top c_k \f$, \f$ \alpha
  * \f$ being its coefficients on \f$ s_{k+1} \f$, so that only \f$ c_k \f$
  * changes with the right-hand sides of the dynamics. */

 struct StageRow {
  Index stage = 0;     ///< the stage k of the row
  bool empty = true;   ///< the row has no entries
  bool fits = true;    ///< false if the row has entries out of its stage
  Eigen::RowVectorXd delta;  ///< the coefficients on x_k
  Eigen::RowVectorXd eta;    ///< the coefficients on u_k
  Eigen::RowVectorXd alpha;  ///< the coefficients on s_{k+1}
  };

/*-------------------------- PROTECTED METHODS -----------------------------*/

 /// computes the dynamics and the rows written on the stages
 /** Computes, from the stage structure and the matrices of the HQPBlock,
  * the sizes of the stages, the place of each column in them, the
  * dynamics and the rows of all the levels but that of the dynamics
  * written on their stages; throws std::invalid_argument if the level of
  * the dynamics does not fit [see the general notes]. */

 void build_stages( const HQPBlock & hb );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes the row h of the matrix M on the stages
 /** Writes the row h of M (a row of the HQPBlock whose stage, if given, is
  * given_stage, otherwise the size of the stage structure) on its stage. */

 StageRow stage_row( const HQPBlock::SparseMatrix & M , Index h ,
                     Index given_stage ) const;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// brings the trajectory of the StageEQP to the vector of the n columns

 void to_columns( Vec & z ) const;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// solves the subproblem of the working set and of the violated tasks
 /** Fills the StageEQP with the subproblem of the working set (the rows of
  * v_hard whose flag in work is true, the equality ones always) and of the
  * violated tasks of the level (those whose flag in viol is true), solves
  * it and writes its solution in zbar; updates the inconsistency and the
  * iterations. If the solution violates a row of the working set by more
  * than dblFeasTol, as rows dependent but for rounding can make it do, it
  * is solved again with the tolerance of the rank decisions multiplied by
  * 100, up to 1e-6. */

 void solve_eqp( const std::vector< char > & work ,
                 const std::vector< char > & viol , Vec & zbar );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// the slack \f$ f - a^\top z \f$ of the hard row t at z

 [[nodiscard]] double hard_slack( Index t , const Vec & z ) const;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// the value of the objective of the level at z, without the constant

 [[nodiscard]] double level_objective( const Vec & z ) const;

/*--------------------------- PROTECTED FIELDS -----------------------------*/

 int f_warm_start = 1;        ///< intWarmStart
 int f_max_iter = 10000;      ///< intMaxIter
 int f_verb = 0;              ///< intLogVerb
 double f_rank_tol = 0;       ///< dblRankTol
 double f_feas_tol = 1e-9;    ///< dblFeasTol
 double f_dual_tol = 1e-9;    ///< dblDualTol

 bool f_rebuild = true;       ///< the stages are to be computed anew

 // the stage structure, computed by build_stages()
 Index f_N = 0;                     ///< the number of stages
 Index f_dyn = 0;                   ///< the level of the dynamics
 std::vector< Index > v_ns;         ///< the size of s_k, k = 0 .. N
 std::vector< Index > v_nu;         ///< the size of u_k, k = 0 .. N - 1
 std::vector< Index > v_pos;        ///< the place of each column in s_k, u_k
 std::vector< Mat > v_A;            ///< the A_k of the dynamics
 std::vector< Mat > v_B;            ///< the B_k of the dynamics
 /// for each component of each s_{k+1}, its row in the level of the dynamics
 std::vector< std::vector< Index > > v_dyn_row;
 /// for each component of each s_{k+1}, its coefficient in that row
 std::vector< std::vector< double > > v_dyn_piv;
 std::vector< StageRow > v_eq_rows;  ///< the equality rows on the stages
 std::vector< StageRow > v_in_rows;  ///< the inequality rows on the stages

 // the data of the current compute()
 const HQPBlock * f_hb = nullptr;   ///< the HQPBlock
 Index f_p = 0;                     ///< its current level
 std::vector< Vec > v_c;            ///< the c_k of the dynamics
 /// the 1 / ( W^e a ) of each row of the dynamics, a its pivot
 std::vector< Vec > v_dyn_inv;
 std::vector< Mat > v_Bhat;         ///< the B of the augmented dynamics

 /// a row of a level above the current one, as a hard row
 struct HardRow {
  bool eq;      ///< an equality row (always in the working set)
  Index q;      ///< its level
  Index h;      ///< its index among the rows of its kind of its level
  Index i;      ///< its index among the rows of its kind of all levels
  double rhs;   ///< its right-hand side, moved by its fixed slack
  };
 std::vector< HardRow > v_hard;     ///< the hard rows
 std::vector< Index > v_task_in;    ///< the inequality tasks of the level

 StageEQP f_eqp;                    ///< the subproblem
 /// for each row of each stage of the subproblem, its place in v_hard
 std::vector< std::vector< Index > > v_eqp_rows;
 /// for each row of each stage of the subproblem, the factor it is divided by
 std::vector< Vec_double > v_eqp_scale;

 // the results
 int f_status = kUnEval;            ///< the status of the last compute()
 bool f_has_sol = false;            ///< a solution is there
 bool f_warm = false;               ///< the last compute() started warm
 long f_iter = 0;                   ///< the subproblems solved in it
 double f_incons = 0;               ///< the largest inconsistency in it
 double f_value = 0;                ///< the value of the Objective
 Vec f_z;                           ///< the solution
 Index f_sol_level = 0;             ///< the level of the solution

 // the working set and the violated tasks, kept from compute() to compute()
 Index f_ws_level = Inf< Index >();  ///< their level
 std::vector< Index > v_W;          ///< the inequality rows in it
 std::vector< Index > v_V;          ///< the violated tasks of the level

/*--------------------- PRIVATE PART OF THE CLASS --------------------------*/

 private:

/*-------------------------- PRIVATE METHODS -------------------------------*/

 SMSpp_insert_in_factory_h;  // insert HQPStageSolver in the Solver factory

/*--------------------------------------------------------------------------*/

 };  // end( class( HQPStageSolver ) )

/** @} end( group( HQPStageSolver_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* HQPStageSolver.h included */

/*--------------------------------------------------------------------------*/
/*---------------------- End File HQPStageSolver.h -------------------------*/
/*--------------------------------------------------------------------------*/
