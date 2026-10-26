/*--------------------------------------------------------------------------*/
/*---------------------------- File HQPBlock.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class HQPBlock, which implements the Block
 * concept [see Block.h] for a hierarchical (lexicographic) Quadratic
 * Program, i.e., a sequence of least-squares problems in the same variables
 * where each one is solved keeping all the previous ones at their optimum.
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

#ifndef __HQPBlock
 #define __HQPBlock
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <Eigen/Sparse>

#include "Block.h"

#include "ColVariable.h"

#include "DQuadFunction.h"

#include "FRealObjective.h"

#include "FRowConstraint.h"

#include "LinearFunction.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup HQPBlock_CLASSES Classes in HQPBlock.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*----------------------------- CLASS HQPBlock -----------------------------*/
/*--------------------------------------------------------------------------*/
/// implementation of the Block concept for a hierarchical QP
/** The HQPBlock class implements the Block concept [see Block.h] for a
 * hierarchical (lexicographic) Quadratic Program in \f$ n \f$ real
 * variables \f$ x \f$ with \f$ P \f$ levels (or tasks) \f$ p = 0 , \ldots ,
 * P - 1 \f$, level 0 having the highest priority. Level \f$ p \f$ has
 * \f$ m_p \geq 0 \f$ equality rows and \f$ l_p \geq 0 \f$ inequality rows
 *
 * \f[
 *   W^e_p ( A_p x - b_p ) = v_p \;\;,\qquad
 *   W^i_p ( C_p x - d_p ) \leq w_p \;\;,\qquad w_p \geq 0 \;\;,
 * \f]
 *
 * where \f$ A_p \in \mathbb{R}^{m_p \times n} \f$, \f$ b_p \in
 * \mathbb{R}^{m_p} \f$, \f$ C_p \in \mathbb{R}^{l_p \times n} \f$, \f$ d_p
 * \in \mathbb{R}^{l_p} \f$, \f$ W^e_p \f$ and \f$ W^i_p \f$ are diagonal
 * matrices of weights (the identity if not given), and \f$ v_p \in
 * \mathbb{R}^{m_p} \f$, \f$ w_p \in \mathbb{R}^{l_p} \f$ are the slacks of
 * the level. Denoting by \f$ ( v^*_q , w^*_q ) \f$ the optimal slacks of the
 * levels \f$ q < p \f$, level \f$ p \f$ is the problem
 *
 * \f[
 *   (L_p) \qquad \min \Big\{ \, \tfrac{1}{2} \| v_p \|^2 +
 *         \tfrac{1}{2} \| w_p \|^2 + \tfrac{\varepsilon}{2} \| x - c \|^2
 *         \;:\; v_q = v^*_q \,,\, w_q = w^*_q \;\; q < p \,,\;
 *         \mbox{the rows of all the levels} \, \Big\} \;\;,
 * \f]
 *
 * where \f$ \varepsilon \geq 0 \f$ is a regularization weight and \f$ c
 * \f$ its center (0 if not given), which a Solver can move, e.g., to the
 * solution of the previous level, to get a proximal term. Fixing \f$ v_q \f$
 * keeps \f$ A_q x \f$ at its optimal value, i.e., an equality task of a
 * higher level keeps its optimal residual, while fixing \f$ w_q \f$ gives
 * \f$ W^i_q ( C_q x - d_q ) \leq w^*_q \f$, i.e., an inequality task of a
 * higher level is never violated more than at its optimum. The optimal
 * slacks of each level are unique, while \f$ x \f$ is in general not (for
 * \f$ \varepsilon = 0 \f$), and with \f$ \varepsilon > 0 \f$ and \f$ c \f$
 * the solution of the previous level each \f$ (L_p) \f$ is, up to the
 * regularization of the slacks, the problem solved in the null space of the
 * equality tasks of the higher levels by the classical approach [see, e.g.,
 *
 *   O. Kanoun, F. Lamiraux, P.-B. Wieber, "Kinematic control of redundant
 *   manipulators: generalizing the task-priority framework to inequality
 *   task", IEEE Transactions on Robotics 27(4):785-792, 2011, doi
 *   10.1109/TRO.2011.2142450
 *
 * and
 *
 *   A. Escande, N. Mansard, P.-B. Wieber, "Hierarchical quadratic
 *   programming: Fast online humanoid-robot motion generation", The
 *   International Journal of Robotics Research 33(7):1006-1028, 2014, doi
 *   10.1177/0278364914521306].
 *
 * Unlike there, all the levels are the same problem in the original
 * variables: the abstract representation has the ColVariable \f$ x \f$
 * (free), \f$ v \f$ (free, one for each equality row of each level) and
 * \f$ w \f$ (non-negative, one for each inequality row), the FRowConstraint
 * with a LinearFunction
 *
 * \f[
 *   W^e_p A_p x - v_p = W^e_p b_p \;\;,\qquad
 *   W^i_p C_p x - w_p \leq W^i_p d_p
 * \f]
 *
 * for all the levels, always present, and an FRealObjective with a
 * DQuadFunction. Moving from a level to the next one is a matter of
 * Modification: the slacks of the level just solved are fixed at their
 * optimal values, and the slacks of the next level enter the Objective, so
 * that a Solver that can re-optimize (e.g., an active-set one) starts from
 * the solution of the previous level; see reset_levels() and next_level().
 * The slacks of the levels below the current one are free and out of the
 * Objective, so that their rows constrain nothing; alternatively, their
 * rows can be relaxed and their slacks fixed to 0 [see reset_levels()],
 * which gives a Hessian that is positive definite if \f$ \varepsilon > 0
 * \f$. The slacks of the levels above the current one stay in the Objective
 * with their coefficient \f$ 1 / 2 \f$, which, being them fixed, only adds a
 * constant to it (and keeps the Hessian non-singular on them).
 *
 * The data \f$ A_p \f$, \f$ b_p \f$, \f$ C_p \f$, \f$ d_p \f$ and the
 * weights can be changed with the methods of the class [see set_b() and the
 * others], which change the abstract representation by Modification, so
 * that, e.g., a new step of a Model Predictive Control scheme is a set of
 * Modification of the same Block rather than a new one. The data is meant
 * to be changed only through these methods: a change made directly to the
 * abstract representation is not reflected in the data of the HQPBlock.
 *
 * Only the data is serialized [see deserialize()]: the current level, the
 * fixed slacks and the center of the regularization are a state of the
 * solution process, not of the model. */

class HQPBlock : public Block
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/

 /// the row-major sparse matrix of the coefficients of a level
 using SparseMatrix = Eigen::SparseMatrix< double , Eigen::RowMajor >;

 using Vec_double = std::vector< double >;  ///< a vector of double

 /// the state of the slacks and of the rows of a level
 enum level_state {
  kFree = 0 ,  ///< the slacks are free and the rows enforced
  kFixed ,     ///< the slacks are fixed at their optimum, rows enforced
  kOff         ///< the slacks are fixed to 0 and the rows relaxed
  };

/*------------------------------ CONSTRUCTOR -------------------------------*/
 /// constructor of HQPBlock, taking a pointer to the father Block
 /** Constructor of HQPBlock. It accepts a pointer to the father Block
  * (defaulting to nullptr, both because the root Block has no father and so
  * that this can also be used as the void constructor required by the Block
  * factory). The HQPBlock is empty until load() or deserialize() is
  * called. */

 explicit HQPBlock( Block * father = nullptr )
  : Block( father ) , f_n( 0 ) , f_eps( 0 ) , f_level( 0 ) ,
    f_relax_lower( false ) , AR( 0 ) {}

/*------------------------------ DESTRUCTOR --------------------------------*/
 /// destructor of HQPBlock

 virtual ~HQPBlock() { guts_of_destructor(); }

/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
 /// loads the HQPBlock out of the data of its levels
 /** Loads the HQPBlock out of the number \p n of variables and of the data
  * of the levels, all the vectors having one element for each level (their
  * number being A.size()):
  *
  * - A[ p ] is the \f$ m_p \times n \f$ matrix \f$ A_p \f$ (with 0 rows if
  *   level p has no equality task), b[ p ] the vector \f$ b_p \f$;
  *
  * - C[ p ] is the \f$ l_p \times n \f$ matrix \f$ C_p \f$ (with 0 rows if
  *   level p has no inequality task), d[ p ] the vector \f$ d_p \f$;
  *
  * - we[ p ] and wi[ p ] are the diagonals of \f$ W^e_p \f$ and \f$ W^i_p
  *   \f$; either vector may be empty, and so any of its elements, meaning
  *   all ones;
  *
  * - eps is the weight \f$ \varepsilon \f$ of the regularization.
  *
  * As the && tell, the vectors become property of the HQPBlock. If the
  * abstract representation is there already it is constructed anew, and an
  * NBModification is issued. Exception is thrown if the sizes do not
  * agree. */

 void load( Index n , std::vector< SparseMatrix > && A ,
            std::vector< Vec_double > && b ,
            std::vector< SparseMatrix > && C ,
            std::vector< Vec_double > && d ,
            std::vector< Vec_double > && we = {} ,
            std::vector< Vec_double > && wi = {} , double eps = 0 );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// extends Block::deserialize( netCDF::NcGroup )
 /** Extends Block::deserialize( netCDF::NcGroup ) to the specific format of
  * an HQPBlock. Besides the mandatory "type" attribute of any :Block, the
  * group contains:
  *
  * - the dimension "NumVar", the number \f$ n \f$ of variables;
  *
  * - the dimension "NumLevels", the number \f$ P \f$ of levels;
  *
  * - the dimension "NumEqRows", the total number \f$ \sum_p m_p \f$ of
  *   equality rows, optional (0 if not there), and the variable "EqRows",
  *   of type netCDF::NcUint and indexed over "NumLevels", whose p-th entry
  *   is \f$ m_p \f$ (the rows are those of level 0 first, then those of
  *   level 1, ...), mandatory if NumEqRows > 0;
  *
  * - the dimension "NumEqElements", the number of non-zero entries of all
  *   the \f$ A_p \f$, and the variables "EqStart" (netCDF::NcUint, indexed
  *   over "NumEqRows"), "EqIndex" (netCDF::NcUint, indexed over
  *   "NumEqElements") and "EqValue" (netCDF::NcDouble, indexed over
  *   "NumEqElements"): the entries of row i are those in positions from
  *   EqStart[ i ] to EqStart[ i + 1 ] - 1 (to NumEqElements - 1 for the last
  *   row), EqIndex giving the column and EqValue the coefficient; a row with
  *   no entries is allowed, and all these are mandatory if NumEqRows > 0
  *   (NumEqElements may be 0);
  *
  * - the variable "EqRHS" (netCDF::NcDouble, indexed over "NumEqRows"), the
  *   vectors \f$ b_p \f$ one after the other, mandatory if NumEqRows > 0;
  *
  * - the variable "EqWeight" (netCDF::NcDouble, indexed over "NumEqRows"),
  *   the diagonals of the \f$ W^e_p \f$, optional (all ones if not there);
  *
  * - the dimensions "NumIneqRows" and "NumIneqElements" and the variables
  *   "IneqRows", "IneqStart", "IneqIndex", "IneqValue", "IneqRHS" and
  *   "IneqWeight", exactly as above for the \f$ C_p \f$, the \f$ d_p \f$ and
  *   the \f$ W^i_p \f$;
  *
  * - the scalar variable "Regularization" (netCDF::NcDouble), the weight
  *   \f$ \varepsilon \f$, optional (0 if not there). */

 void deserialize( const netCDF::NcGroup & group ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// loads the HQPBlock out of an istream: not implemented
 /** No text format is defined for HQPBlock, the netCDF one being the
  * only one [see deserialize()]: the method throws. */

 void load( std::istream & input , char frmt = 0 ) override
 {
  throw( std::logic_error( "HQPBlock::load: no text format, use the "
                           "netCDF one" ) );
  }

/*---------------------- Methods for handling Variable ---------------------*/
 /// generates the ColVariable x, v and w of the HQPBlock
 /** Generates the static ColVariable of the HQPBlock, in this order: the
  * \f$ n \f$ free ColVariable \f$ x \f$ (named "x"), the free slacks \f$ v
  * \f$ of all the equality rows (named "v") and the non-negative slacks
  * \f$ w \f$ of all the inequality rows (named "w"), the slacks of level 0
  * first, then those of level 1, ... */

 void generate_abstract_variables( Configuration * stvv = nullptr )
  override;

/*--------------------- Methods for handling Constraint --------------------*/
 /// generates the FRowConstraint of all the levels
 /** Generates the static FRowConstraint of the HQPBlock: those of the
  * equality rows (named "eq"), each one with a LinearFunction whose first
  * active Variable is its slack \f$ v_i \f$, and those of the inequality
  * rows (named "ineq"), the first active Variable being \f$ w_i \f$. */

 void generate_abstract_constraints( Configuration * stcc = nullptr )
  override;

/*---------------------- Methods for handling Objective --------------------*/
 /// generates the FRealObjective of the HQPBlock
 /** Generates the FRealObjective, a minimization one whose DQuadFunction
  * has as active Variable \f$ x \f$, then \f$ v \f$, then \f$ w \f$, in the
  * order of generate_abstract_variables(); the coefficients (linear and
  * quadratic) are those of the current level, i.e., \f$ ( - \varepsilon c ,
  * \varepsilon / 2 ) \f$ for \f$ x \f$, \f$ ( 0 , 1 / 2 ) \f$ for the slacks
  * in the Objective and \f$ ( 0 , 0 ) \f$ for the others. */

 void generate_objective( Configuration * objc = nullptr ) override;

/*-------------- Methods for checking the state of the HQPBlock ------------*/
 /// returns true if any part of the abstract representation is there

 bool anyone_there( void ) const override
 {
  return( AR ? true : Block::anyone_there() );
  }

/*----------------------- Methods for reading the data ---------------------*/
 /// returns the number \f$ n \f$ of variables

 [[nodiscard]] Index get_num_var( void ) const { return( f_n ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number \f$ P \f$ of levels

 [[nodiscard]] Index get_num_levels( void ) const { return( v_A.size() ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number \f$ m_p \f$ of equality rows of level p

 [[nodiscard]] Index get_num_eq_rows( Index p ) const {
  return( v_eq_start[ p + 1 ] - v_eq_start[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number \f$ l_p \f$ of inequality rows of level p

 [[nodiscard]] Index get_num_ineq_rows( Index p ) const {
  return( v_in_start[ p + 1 ] - v_in_start[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the index of the first equality row of level p among all
 /** Returns the index, among the equality rows of all the levels (hence
  * also among the ColVariable \f$ v \f$ and the FRowConstraint "eq"), of
  * the first equality row of level p; p == get_num_levels() is allowed,
  * giving the total number of equality rows. */

 [[nodiscard]] Index get_eq_start( Index p ) const {
  return( v_eq_start[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the index of the first inequality row of level p among all

 [[nodiscard]] Index get_ineq_start( Index p ) const {
  return( v_in_start[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the matrix \f$ A_p \f$

 [[nodiscard]] const SparseMatrix & get_A( Index p ) const {
  return( v_A[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the matrix \f$ C_p \f$

 [[nodiscard]] const SparseMatrix & get_C( Index p ) const {
  return( v_C[ p ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns \f$ b \f$ of all levels, \f$ b_p \f$ from get_eq_start( p ) on

 [[nodiscard]] const Vec_double & get_b( void ) const { return( v_b ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns \f$ d \f$ of all levels, \f$ d_p \f$ from get_ineq_start( p ) on

 [[nodiscard]] const Vec_double & get_d( void ) const { return( v_d ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the diagonals of all the \f$ W^e_p \f$, as get_b()

 [[nodiscard]] const Vec_double & get_we( void ) const { return( v_we ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the diagonals of all the \f$ W^i_p \f$, as get_d()

 [[nodiscard]] const Vec_double & get_wi( void ) const { return( v_wi ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the weight \f$ \varepsilon \f$ of the regularization

 [[nodiscard]] double get_regularization( void ) const { return( f_eps ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the center \f$ c \f$ of the regularization (empty means 0)

 [[nodiscard]] const Vec_double & get_center( void ) const {
  return( v_center );
  }

/*------------ Methods for reading the abstract representation -------------*/
 /// returns the ColVariable \f$ x \f$

 [[nodiscard]] const std::vector< ColVariable > & get_x( void ) const {
  return( v_x );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the slacks \f$ v \f$ of all the levels

 [[nodiscard]] const std::vector< ColVariable > & get_v( void ) const {
  return( v_v );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the slacks \f$ w \f$ of all the levels

 [[nodiscard]] const std::vector< ColVariable > & get_w( void ) const {
  return( v_w );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the FRowConstraint of the equality rows of all the levels

 [[nodiscard]] const std::vector< FRowConstraint > & get_eq( void ) const {
  return( v_eq );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the FRowConstraint of the inequality rows of all the levels

 [[nodiscard]] const std::vector< FRowConstraint > & get_ineq( void ) const
 {
  return( v_in );
  }

/*--------------------- Methods for handling the levels --------------------*/
 /// returns the current level
 /** Returns the level whose problem \f$ (L_p) \f$ the abstract
  * representation currently encodes [see reset_levels() and
  * next_level()]. */

 [[nodiscard]] Index get_level( void ) const { return( f_level ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the state of the slacks and rows of level p

 [[nodiscard]] level_state get_level_state( Index p ) const {
  return( level_state( v_lvl_state[ p ] ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns true if the rows of the levels below the current are relaxed

 [[nodiscard]] bool relaxes_lower( void ) const { return( f_relax_lower ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// brings the abstract representation to the problem of level 0
 /** Brings the abstract representation to \f$ (L_0) \f$: the slacks of
  * level 0 are free and in the Objective. If \p relax_lower == false, the
  * slacks of all the other levels are free and out of the Objective, their
  * rows being enforced (and constraining nothing); if relax_lower == true,
  * the rows of all the other levels are relaxed and their slacks fixed to 0,
  * in the Objective with coefficient \f$ 1 / 2 \f$. Only what differs from
  * the current state is changed, and the changes are issued as Modification
  * according to \p issueMod [see Observer::make_par()]. */

 void reset_levels( bool relax_lower = false ,
                    ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// moves the abstract representation from level p to level p + 1
 /** Moves the abstract representation from the current level p to level
  * p + 1, where p + 1 < get_num_levels() must hold. The slacks of level p
  * are fixed at the values they have at the current values of the
  * ColVariable \f$ x \f$, i.e.,
  *
  * \f[
  *   v_p = W^e_p ( A_p x - b_p ) \;\;,\qquad
  *   w_p = \max \{ \, 0 \,,\, W^i_p ( C_p x - d_p ) \, \} + \tau \;\;,
  * \f]
  *
  * so that the solution of \f$ (L_p) \f$, if written in the ColVariable,
  * is exactly feasible for \f$ (L_{p+1}) \f$ (\f$ \tau \geq 0 \f$, given
  * by \p tol, allows a small violation of the inequalities of level p in
  * the next levels); then the slacks of level p + 1 are made free (and its
  * rows enforced, if they were relaxed) and they enter the Objective. The
  * changes are issued as Modification according to \p issueMod. */

 void next_level( double tol = 0 , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// computes the slacks of level p at the current values of x
 /** Writes in \p v and \p w the slacks \f$ W^e_p ( A_p x - b_p ) \f$ and
  * \f$ \max \{ 0 , W^i_p ( C_p x - d_p ) \} \f$ of level p at the current
  * values of the ColVariable \f$ x \f$, and returns the value \f$ \|v\|^2 /
  * 2 + \|w\|^2 / 2 \f$ of level p there. */

 double level_slacks( Index p , Vec_double & v , Vec_double & w ) const;

/*-------------------------- Methods for changing data ---------------------*/
 /// changes \f$ b_p \f$
 /** Changes the vector \f$ b_p \f$ of level p into \p b (whose size must
  * be \f$ m_p \f$), the right-hand sides of the equality rows of level p
  * changing by Modification according to \p issueMod. */

 void set_b( Index p , const Vec_double & b , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ d_p \f$, as set_b() does for \f$ b_p \f$

 void set_d( Index p , const Vec_double & d , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ A_p \f$
 /** Changes the matrix \f$ A_p \f$ of level p into \p A, which must have
  * the same size. The coefficients of the LinearFunction of each row change
  * by Modification according to \p issueMod: those of the entries already
  * there, also if they become 0, with one C05FunctionModLinSbst, and the new
  * entries are added to the LinearFunction (the pattern of a row only
  * grows). */

 void set_A( Index p , const SparseMatrix & A ,
             ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ C_p \f$, as set_A() does for \f$ A_p \f$

 void set_C( Index p , const SparseMatrix & C ,
             ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the diagonal of \f$ W^e_p \f$ (coefficients and right-hand sides)

 void set_we( Index p , const Vec_double & we ,
              ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the diagonal of \f$ W^i_p \f$ (coefficients and right-hand sides)

 void set_wi( Index p , const Vec_double & wi ,
              ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the weight \f$ \varepsilon \f$ of the regularization

 void set_regularization( double eps , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the center \f$ c \f$ of the regularization
 /** Changes the center \f$ c \f$ of the regularization into \p c, whose
  * size must be \f$ n \f$ (or 0, meaning \f$ c = 0 \f$): the linear
  * coefficients \f$ - \varepsilon c \f$ of \f$ x \f$ in the Objective change
  * by Modification according to \p issueMod. The constant \f$ \varepsilon
  * \| c \|^2 / 2 \f$ is not added to the Objective. */

 void set_center( const Vec_double & c , ModParam issueMod = eModBlck );

/*---------------------------- Methods for printing ------------------------*/
 /// prints the HQPBlock on an ostream with the given verbosity
 /** Prints the sizes of the HQPBlock and, if \p vlvl > 0, also its data,
  * level by level. */

 void print( std::ostream & output , char vlvl = 0 ) const override;

/*--------------------------- Methods for saving ---------------------------*/
 /// extends Block::serialize( netCDF::NcGroup )
 /** Extends Block::serialize( netCDF::NcGroup ) to the specific format of
  * an HQPBlock [see deserialize()]. The weights are written only if one is
  * not 1, the regularization only if it is not 0. */

 void serialize( netCDF::NcGroup & group ) const override;

 using Block::serialize;  // the other versions, e.g., to a file

/*-------------------- PROTECTED PART OF THE CLASS -------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/

 void guts_of_destructor( void );

 void guts_of_load( Index n , std::vector< SparseMatrix > && A ,
                    std::vector< Vec_double > && b ,
                    std::vector< SparseMatrix > && C ,
                    std::vector< Vec_double > && d ,
                    std::vector< Vec_double > && we ,
                    std::vector< Vec_double > && wi , double eps );

 void check_level( Index p , const char * method ) const;

 void set_rows( bool eq , Index p , const SparseMatrix * M ,
                ModParam issueMod );

 void set_level_state( Index p , level_state s , double tol ,
                       ModParam issueMod );

 void set_level_in_objective( Index p , bool in , ModParam issueMod );

/*--------------------------- PROTECTED FIELDS -----------------------------*/

 Index f_n;                         ///< number of variables

 std::vector< SparseMatrix > v_A;   ///< the A_p
 std::vector< SparseMatrix > v_C;   ///< the C_p

 Vec_double v_b;                    ///< the b_p, one after the other
 Vec_double v_d;                    ///< the d_p, one after the other
 Vec_double v_we;                   ///< the weights of the equality rows
 Vec_double v_wi;                   ///< the weights of the inequality rows

 std::vector< Index > v_eq_start;   ///< first equality row of each level
 std::vector< Index > v_in_start;   ///< first inequality row of each level

 double f_eps;                      ///< weight of the regularization
 Vec_double v_center;               ///< center of the regularization

 std::vector< ColVariable > v_x;    ///< the variables x
 std::vector< ColVariable > v_v;    ///< the slacks of the equality rows
 std::vector< ColVariable > v_w;    ///< the slacks of the inequality rows

 std::vector< FRowConstraint > v_eq;  ///< the equality rows
 std::vector< FRowConstraint > v_in;  ///< the inequality rows

 /// for each row, the columns of the active Variable of its LinearFunction
 /** For each row (the equality ones first, then the inequality ones), the
  * column of x of each active Variable of its LinearFunction after the
  * first one, which is the slack. */
 std::vector< std::vector< Index > > v_cols;

 FRealObjective f_obj;              ///< the Objective

 Index f_level;                     ///< the current level
 bool f_relax_lower;                ///< rows of the lower levels relaxed
 std::vector< char > v_lvl_state;   ///< the level_state of each level
 std::vector< char > v_lvl_obj;     ///< slacks of each level in Objective

 unsigned char AR;     ///< bit-wise coded: what abstract is there

 static constexpr unsigned char HasVar = 1;
 ///< first bit of AR == 1 if the Variable have been constructed
 static constexpr unsigned char HasObj = 2;
 ///< second bit of AR == 1 if the Objective has been constructed
 static constexpr unsigned char HasCns = 4;
 ///< third bit of AR == 1 if the Constraint have been constructed

/*--------------------- PRIVATE PART OF THE CLASS --------------------------*/

 private:

/*-------------------------- PRIVATE METHODS -------------------------------*/

 SMSpp_insert_in_factory_h;  // insert HQPBlock in the Block factory

/*--------------------------------------------------------------------------*/

 };  // end( class( HQPBlock ) )

/** @} end( group( HQPBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* HQPBlock.h included */

/*--------------------------------------------------------------------------*/
/*------------------------- End File HQPBlock.h ----------------------------*/
/*--------------------------------------------------------------------------*/
