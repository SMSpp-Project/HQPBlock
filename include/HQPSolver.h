/*--------------------------------------------------------------------------*/
/*--------------------------- File HQPSolver.h -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the class HQPSolver, a Solver for the hierarchical
 * (lexicographic) Quadratic Program of an HQPBlock that solves its levels
 * one after the other with an inner Solver, moving from a level to the next
 * one by Modification of the same HQPBlock.
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

#ifndef __HQPSolver
 #define __HQPSolver
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "BlockSolverConfig.h"

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
/** @defgroup HQPSolver_CLASSES Classes in HQPSolver.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*---------------------------- CLASS HQPSolver -----------------------------*/
/*--------------------------------------------------------------------------*/
/// a Solver for the hierarchical QP of an HQPBlock, level by level
/** The HQPSolver class implements the Solver interface for the hierarchical
 * (lexicographic) Quadratic Program of an HQPBlock [see HQPBlock.h]: the
 * levels \f$ p = 0 , \ldots , P - 1 \f$ are solved one after the other,
 * each \f$ (L_p) \f$ being
 *
 * \f[
 *   \min \Big\{ \, \tfrac{1}{2} \| v_p \|^2 + \tfrac{1}{2} \| w_p \|^2 +
 *         \tfrac{\varepsilon}{2} \| x - c_p \|^2 \;:\;
 *         v_q = v^*_q \,,\, w_q = w^*_q \;\; q < p \,,\;
 *         \mbox{the rows of all the levels} \, \Big\} \;\;,
 * \f]
 *
 * by an inner Solver attached to the same HQPBlock. The inner Solver is
 * any Solver that can deal with a convex QP with linear constraints (e.g.,
 * a :MILPSolver); it is created through the Solver factory out of the
 * BlockSolverConfig whose filename is the string parameter str_HQP_BSCfg
 * [see set_par()], the first of its Solver that is in the factory being
 * used, together with its ComputeConfig.
 *
 * The inner Solver is attached to the HQPBlock with set_Block() but not
 * registered to it: it is the HQPSolver, registered to the HQPBlock, that
 * receives the Modification of the HQPBlock and passes them on to the inner
 * Solver. A compute() is:
 *
 * - the HQPBlock is brought to the problem of level 0 [see
 *   HQPBlock::reset_levels()], with the rows of the lower levels enforced
 *   or relaxed according to int_HQP_RelaxLower;
 *
 * - for each level p, the inner Solver solves \f$ (L_p) \f$ and writes its
 *   solution into the HQPBlock, whose slacks of level p are then fixed at
 *   their values at that \f$ x \f$ and the slacks of level p + 1 enter the
 *   Objective [see HQPBlock::next_level()].
 *
 * All the changes of a level are Modification of the same HQPBlock, so an
 * inner Solver that re-optimizes (e.g., an active-set one, which keeps its
 * working set) starts each level from the solution of the previous one, and
 * each compute() from the solution of the previous compute(), e.g., in a
 * Model Predictive Control scheme whose steps change \f$ b_p \f$, \f$ d_p
 * \f$, \f$ A_p \f$ and \f$ C_p \f$ with the methods of the HQPBlock.
 *
 * With int_HQP_Prox == 1 the center of the regularization of the HQPBlock
 * is \f$ c_0 = 0 \f$ at level 0 and \f$ c_p = x^*_{p-1} \f$, the solution
 * of the previous level, afterwards: if \f$ \varepsilon > 0 \f$, each level
 * then gives the solution of the classical approach that solves level p in
 * the null space of the equality tasks of the higher levels, with the
 * regularization of the step from the previous solution [see HQPBlock.h].
 * With int_HQP_Prox == 0 the center is 0 at all the levels.
 *
 * After compute() the solution of each level solved is available [see
 * get_level_x(), get_level_value(), get_level_v() and
 * get_level_w()], and the
 * HQPBlock is left in the state of the last level solved (its slacks of the
 * previous levels fixed), which the next compute() resets.
 *
 * Besides its own parameters, the HQPSolver listens to intLogVerb: if
 * positive, a line for each level is printed on the log [see set_log()]. */

class HQPSolver : public Solver
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/

 using Vec_double = std::vector< double >;  ///< a vector of double

 /// public enum for the int parameters of HQPSolver
 enum int_par_type_HQPS {
  int_HQP_Prox = intLastAlgPar ,  ///< center at the previous solution
  int_HQP_RelaxLower ,            ///< relax the rows of the lower levels
  intLastHQPSPar  ///< first allowed int parameter for derived classes
  };

 /// public enum for the double parameters of HQPSolver
 enum dbl_par_type_HQPS {
  dbl_HQP_SlackTol = dblLastAlgPar ,  ///< tolerance on the fixed w
  dblLastHQPSPar  ///< first allowed double parameter for derived classes
  };

 /// public enum for the string parameters of HQPSolver
 enum str_par_type_HQPS {
  str_HQP_BSCfg = strLastAlgPar ,  ///< BlockSolverConfig of the inner one
  strLastHQPSPar  ///< first allowed string parameter for derived classes
  };

/*---------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------*/
 /// constructor: the HQPSolver has no Block and no inner Solver

 HQPSolver( void ) : Solver() , f_bsc( nullptr ) , f_inner( nullptr ) ,
  f_prox( 0 ) , f_relax( 1 ) , f_verb( 0 ) , f_tol( 0 ) ,
  f_status( kUnEval ) {}

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// destructor: deletes the inner Solver and the BlockSolverConfig

 ~HQPSolver() override;

/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
 /// sets the Block, which compute() requires to be an HQPBlock
 /** Sets the Block of the HQPSolver; if the BlockSolverConfig of the inner
  * Solver is known, the inner Solver is created and attached to it. */

 void set_Block( Block * block ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// sets the int parameters of HQPSolver
 /** Sets the int parameters specific of HQPSolver:
  *
  * - int_HQP_Prox [0]: if 1, the center of the regularization of the
  *   HQPBlock is moved to the solution of the previous level (and to 0 at
  *   level 0), if 0 it is 0 at all the levels;
  *
  * - int_HQP_RelaxLower [1]: if 0, the rows of the levels below the
  *   current one are enforced, their slacks free and out of the Objective;
  *   if 1, they are relaxed and their slacks fixed to 0 [see
  *   HQPBlock::reset_levels()]. */

 void set_par( idx_type par , int value ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// sets the double parameters of HQPSolver
 /** Sets the double parameters specific of HQPSolver:
  *
  * - dbl_HQP_SlackTol [0]: the tolerance \f$ \tau \geq 0 \f$ added to the
  *   slacks \f$ w_p \f$ when they are fixed [see HQPBlock::next_level()]. */

 void set_par( idx_type par , double value ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// sets the string parameters of HQPSolver
 /** Sets the string parameters specific of HQPSolver:
  *
  * - str_HQP_BSCfg [""]: the name of the file out of which the
  *   BlockSolverConfig of the inner Solver is deserialized; the inner Solver
  *   is (re)created as soon as both it and the Block are there. */

 void set_par( idx_type par , std::string && value ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// receives a Modification of the HQPBlock and passes it to the inner one

 void add_Modification( sp_Mod & mod ) override {
  if( f_no_Mod )
   return;
  if( f_inner )
   f_inner->add_Modification( mod );
  }

/*--------------------- METHODS FOR SOLVING THE MODEL ----------------------*/
 /// solves the levels of the HQPBlock one after the other
 /** Solves the levels of the HQPBlock one after the other as described in
  * the general notes. Returns kOK if all the levels have been solved with
  * kOK, the first other status in [ kOK , kError ) returned by the inner
  * Solver if all the levels have been solved but some not with kOK, and
  * otherwise the status of the level where the process stopped (kError if
  * there is no HQPBlock or no inner Solver), the solutions of the levels
  * before that one being available. */

 int compute( bool changedvars = true ) override;

/*---------------------- METHODS FOR READING RESULTS -----------------------*/
 /// returns the value of the last level solved
 /** Returns \f$ \| v_p \|^2 / 2 + \| w_p \|^2 / 2 \f$ for the last level p
  * solved by the last compute() (+ infinity if none), the regularization
  * excluded. */

 [[nodiscard]] OFValue get_var_value( void ) override {
  return( v_val.empty() ? Inf< OFValue >() : v_val.back() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns true if at least level 0 has been solved

 [[nodiscard]] bool has_var_solution( void ) override {
  return( ! v_xl.empty() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes the solution of the last level solved into the HQPBlock
 /** Writes into the ColVariable of the HQPBlock the \f$ x \f$ of the last
  * level solved and the slacks of all the levels solved. */

 void get_var_solution( Configuration * solc = nullptr ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number of levels solved by the last compute()

 [[nodiscard]] Index get_num_solved_levels( void ) const {
  return( v_xl.size() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the status of the inner Solver at each level tried

 [[nodiscard]] const std::vector< int > & get_level_status( void ) const {
  return( v_stat );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the \f$ x \f$ of the solution of level p

 [[nodiscard]] const Vec_double & get_level_x( Index p ) const {
  return( v_xl.at( p ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns \f$ \| v_p \|^2 / 2 + \| w_p \|^2 / 2 \f$ at the solution of p

 [[nodiscard]] double get_level_value( Index p ) const {
  return( v_val.at( p ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the slacks \f$ v_p \f$ at the solution of level p
 /** Returns the slacks \f$ W^e_p ( A_p x - b_p ) \f$ of level p at the
  * \f$ x \f$ of the solution of level p, i.e., the values at which they are
  * fixed for the next levels. */

 [[nodiscard]] const Vec_double & get_level_v( Index p ) const {
  return( v_vl.at( p ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the slacks \f$ w_p \f$ at the solution of level p
 /** Returns the slacks \f$ \max \{ 0 , W^i_p ( C_p x - d_p ) \} \f$ of
  * level p at the \f$ x \f$ of the solution of level p, i.e., the values at
  * which they are fixed for the next levels (but for dbl_HQP_SlackTol). */

 [[nodiscard]] const Vec_double & get_level_w( Index p ) const {
  return( v_wl.at( p ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the inner Solver (nullptr if there is none)

 [[nodiscard]] Solver * get_inner_Solver( void ) const { return( f_inner ); }

/*----------------- METHODS FOR HANDLING THE PARAMETERS --------------------*/
 /// returns the number of int parameters, those of Solver included

 [[nodiscard]] idx_type get_num_int_par( void ) const override {
  return( idx_type( intLastHQPSPar ) );
  }

 /// returns the number of double parameters, those of Solver included

 [[nodiscard]] idx_type get_num_dbl_par( void ) const override {
  return( idx_type( dblLastHQPSPar ) );
  }

 /// returns the number of string parameters, those of Solver included

 [[nodiscard]] idx_type get_num_str_par( void ) const override {
  return( idx_type( strLastHQPSPar ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the default value of the int parameter par [see set_par()]

 [[nodiscard]] int get_dflt_int_par( idx_type par ) const override {
  if( par == int_HQP_Prox )
   return( 0 );
  if( par == int_HQP_RelaxLower )
   return( 1 );
  return( Solver::get_dflt_int_par( par ) );
  }

 /// returns the default value of the double parameter par

 [[nodiscard]] double get_dflt_dbl_par( idx_type par ) const override {
  if( par == dbl_HQP_SlackTol )
   return( 0 );
  return( Solver::get_dflt_dbl_par( par ) );
  }

 /// returns the default value of the string parameter par

 [[nodiscard]] const std::string & get_dflt_str_par( idx_type par )
  const override {
  static const std::string empty;
  if( par == str_HQP_BSCfg )
   return( empty );
  return( Solver::get_dflt_str_par( par ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the current value of the int parameter par

 [[nodiscard]] int get_int_par( idx_type par ) const override {
  if( par == int_HQP_Prox )
   return( f_prox );
  if( par == int_HQP_RelaxLower )
   return( f_relax );
  if( par == intLogVerb )
   return( f_verb );
  return( Solver::get_int_par( par ) );
  }

 /// returns the current value of the double parameter par

 [[nodiscard]] double get_dbl_par( idx_type par ) const override {
  if( par == dbl_HQP_SlackTol )
   return( f_tol );
  return( Solver::get_dbl_par( par ) );
  }

 /// returns the current value of the string parameter par

 [[nodiscard]] const std::string & get_str_par( idx_type par )
  const override {
  if( par == str_HQP_BSCfg )
   return( f_bsc_name );
  return( Solver::get_str_par( par ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the index of the int parameter with the given name

 [[nodiscard]] idx_type int_par_str2idx( const std::string & name )
  const override {
  if( name == "int_HQP_Prox" )
   return( int_HQP_Prox );
  if( name == "int_HQP_RelaxLower" )
   return( int_HQP_RelaxLower );
  return( Solver::int_par_str2idx( name ) );
  }

 /// returns the index of the double parameter with the given name

 [[nodiscard]] idx_type dbl_par_str2idx( const std::string & name )
  const override {
  if( name == "dbl_HQP_SlackTol" )
   return( dbl_HQP_SlackTol );
  return( Solver::dbl_par_str2idx( name ) );
  }

 /// returns the index of the string parameter with the given name

 [[nodiscard]] idx_type str_par_str2idx( const std::string & name )
  const override {
  if( name == "str_HQP_BSCfg" )
   return( str_HQP_BSCfg );
  return( Solver::str_par_str2idx( name ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /// returns the name of the int parameter with the given index

 [[nodiscard]] const std::string & int_par_idx2str( idx_type idx )
  const override {
  static const std::string pars[] = { "int_HQP_Prox" ,
                                      "int_HQP_RelaxLower" };
  if( ( idx >= int_HQP_Prox ) && ( idx < intLastHQPSPar ) )
   return( pars[ idx - int_HQP_Prox ] );
  return( Solver::int_par_idx2str( idx ) );
  }

 /// returns the name of the double parameter with the given index

 [[nodiscard]] const std::string & dbl_par_idx2str( idx_type idx )
  const override {
  static const std::string par = "dbl_HQP_SlackTol";
  if( idx == dbl_HQP_SlackTol )
   return( par );
  return( Solver::dbl_par_idx2str( idx ) );
  }

 /// returns the name of the string parameter with the given index

 [[nodiscard]] const std::string & str_par_idx2str( idx_type idx )
  const override {
  static const std::string par = "str_HQP_BSCfg";
  if( idx == str_HQP_BSCfg )
   return( par );
  return( Solver::str_par_idx2str( idx ) );
  }

/*-------------------- PROTECTED PART OF THE CLASS -------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/

 /// creates the inner Solver out of the BlockSolverConfig and attaches it
 /** Creates, through the Solver factory, the first Solver of the
  * BlockSolverConfig of str_HQP_BSCfg that is in the factory, configures it
  * with its ComputeConfig and attaches it to the Block (without registering
  * it); if none of them is in the factory there is no inner Solver. */

 void create_inner( void );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// detaches and deletes the inner Solver, if any

 void drop_inner( void );

/*--------------------------- PROTECTED FIELDS -----------------------------*/

 std::string f_bsc_name;      ///< filename of the inner BlockSolverConfig
 BlockSolverConfig * f_bsc;   ///< the inner BlockSolverConfig
 Solver * f_inner;            ///< the inner Solver

 int f_prox;                  ///< int_HQP_Prox
 int f_relax;                 ///< int_HQP_RelaxLower
 int f_verb;                  ///< intLogVerb
 double f_tol;                ///< dbl_HQP_SlackTol

 int f_status;                ///< status of the last compute()
 std::vector< int > v_stat;   ///< status of each level tried
 std::vector< Vec_double > v_xl;  ///< x of each level solved
 std::vector< Vec_double > v_vl;  ///< v of each level solved
 std::vector< Vec_double > v_wl;  ///< w of each level solved
 Vec_double v_val;            ///< value of each level solved

/*--------------------- PRIVATE PART OF THE CLASS --------------------------*/

 private:

/*-------------------------- PRIVATE METHODS -------------------------------*/

 SMSpp_insert_in_factory_h;  // insert HQPSolver in the Solver factory

/*--------------------------------------------------------------------------*/

 };  // end( class( HQPSolver ) )

/** @} end( group( HQPSolver_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* HQPSolver.h included */

/*--------------------------------------------------------------------------*/
/*------------------------- End File HQPSolver.h ---------------------------*/
/*--------------------------------------------------------------------------*/
