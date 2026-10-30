/*--------------------------------------------------------------------------*/
/*-------------------------- File HQPSolver.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the HQPSolver class, a Solver for the hierarchical
 * (lexicographic) Quadratic Program of an HQPBlock.
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
/*---------------------------- IMPLEMENTATION ------------------------------*/
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <iomanip>

#include "HQPSolver.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register HQPSolver in the Solver factory

SMSpp_insert_in_factory_cpp_0( HQPSolver );

/*--------------------------------------------------------------------------*/
/*---------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------*/
/*--------------------------------------------------------------------------*/

HQPSolver::~HQPSolver()
{
 drop_inner();
 delete f_bsc;
 }

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void HQPSolver::set_Block( Block * block )
{
 if( block == f_Block )
  return;

 drop_inner();
 Solver::set_Block( block );
 v_stat.clear();
 v_xl.clear();
 v_vl.clear();
 v_wl.clear();
 v_val.clear();
 f_status = kUnEval;

 if( f_Block && f_bsc )
  create_inner();
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , int value )
{
 switch( par ) {
  case( intProxCenter ):
   f_prox = value ? 1 : 0;
   break;
  case( intRelaxLower ):
   f_relax = value ? 1 : 0;
   break;
  case( intLevelScale ):
   f_scale = value ? 1 : 0;
   break;
  case( intLogVerb ):
   f_verb = value;
   break;
  default:
   Solver::set_par( par , value );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , double value )
{
 if( par != dblSlackTol ) {
  Solver::set_par( par , value );
  return;
  }
 if( ! ( value >= 0 ) )
  throw( std::invalid_argument( "HQPSolver::set_par: negative "
                                "dblSlackTol" ) );
 f_tol = value;
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , std::string && value )
{
 if( par != strInnerBSC ) {
  Solver::set_par( par , std::move( value ) );
  return;
  }

 BlockSolverConfig * cfg = nullptr;
 if( ! value.empty() ) {
  auto c = Configuration::deserialize( value );
  cfg = dynamic_cast< BlockSolverConfig * >( c );
  if( ! cfg ) {
   delete c;
   throw( std::invalid_argument( "HQPSolver::set_par: unable to create a "
                                 "BlockSolverConfig out of " + value ) );
   }
  }

 drop_inner();
 delete f_bsc;
 f_bsc = cfg;
 f_bsc_name = std::move( value );

 if( f_Block && f_bsc )
  create_inner();
 }

/*--------------------------------------------------------------------------*/
/*--------------------- METHODS FOR SOLVING THE MODEL ----------------------*/
/*--------------------------------------------------------------------------*/

int HQPSolver::compute( bool changedvars )
{
 lock();

 v_stat.clear();
 v_xl.clear();
 v_vl.clear();
 v_wl.clear();
 v_prev = std::move( v_obj );
 v_obj.clear();
 v_val.clear();

 // the inner Solvers dropped by an NBModification are created anew
 if( v_inner.empty() )
  create_inner();

 auto hb = dynamic_cast< HQPLevelBlock * >( f_Block );
 if( ( ! hb ) || v_inner.empty() || ( ! hb->get_num_levels() ) ) {
  unlock();
  return( f_status = kError );
  }

 // the Block is lock()-ed only while it is changed, since the inner Solver
 // has to read_lock() it in its compute()
 const bool owned = hb->is_owned_by( f_id );
 auto lock_block = [ & ]() {
  if( ( ! owned ) && ( ! hb->lock( f_id ) ) )
   throw( std::runtime_error( "HQPSolver::compute: unable to lock the "
                              "Block" ) );
  };
 auto unlock_block = [ & ]() {
  if( ! owned )
   hb->unlock( f_id );
  };

 const int verb = f_verb;
 const Index P = hb->get_num_levels();

 lock_block();
 hb->reset_levels( f_relax );
 hb->set_center( {} );
 unlock_block();

 f_status = kOK;
 f_iter = 0;
 for( Index p = 0 ; p < P ; ++p ) {
  // the inner Solver of a level is created the first time the level is
  // solved, and it loads the HQPBlock as it is then
  if( v_inner.size() <= p )
   v_inner.push_back( new_inner() );

  auto inner = v_inner[ p ];

  // the scale of the level, from the value of its Objective (the slacks
  // and the regularization) in the previous compute()
  const auto scale_of = []( double v ) {
   return( ( v > 1e-10 ) && ( v < Inf< double >() ) ? 1 / v : 1.0 );
   };
  const bool known = ( p < v_prev.size() );
  double scale = 1;
  if( f_scale ) {
   scale = known ? scale_of( v_prev[ p ] ) : 1.0;
   lock_block();
   hb->set_objective_scale( scale );
   unlock_block();
   }

  int st = inner->compute();
  f_iter += inner->get_elapsed_iterations();

  // the first time, a level of small value is solved again at its scale,
  // starting from the solution just found
  if( f_scale && ( ! known ) && ( st >= kOK ) && ( st < kError ) &&
      inner->has_var_solution() ) {
   const bool again = ( scale_of( inner->get_ub() ) > 10 );
   if( again ) {
    scale = scale_of( inner->get_ub() );
    lock_block();
    hb->set_objective_scale( scale );
    unlock_block();
    }
   if( again ) {
    st = inner->compute();
    f_iter += inner->get_elapsed_iterations();
    }
   }
  v_stat.push_back( st );
  if( ( st < kOK ) || ( st >= kError ) || ( ! inner->has_var_solution() ) )
  {
   f_status = ( ( st >= kOK ) && ( st < kError ) ) ? kError : st;
   if( f_log && verb )
    *f_log << "HQPSolver: level " << p << " stopped with status " << st
           << std::endl;
   break;
   }
  if( ( st != kOK ) && ( f_status == kOK ) )
   f_status = st;

  if( f_scale )
   v_obj.push_back( inner->get_ub() / scale );

  lock_block();
  inner->get_var_solution();

  Vec_double x;
  hb->get_x_values( x );
  Vec_double v , w;
  v_val.push_back( hb->level_slacks( p , v , w ) );
  v_vl.push_back( std::move( v ) );
  v_wl.push_back( std::move( w ) );

  if( f_log && verb )
   *f_log << "HQPSolver: level " << std::setw( 3 ) << p << " status "
          << std::setw( 3 ) << st << " value " << std::setprecision( 12 )
          << v_val.back() << std::endl;

  if( p + 1 < P ) {
   if( f_prox )
    hb->set_center( x );
   hb->next_level( f_tol );
   }
  v_xl.push_back( std::move( x ) );
  unlock_block();
  }

 if( f_scale ) {
  lock_block();
  hb->set_objective_scale( 1 );
  unlock_block();
  }

 unlock();
 return( f_status );

 }  // end( HQPSolver::compute )

/*--------------------------------------------------------------------------*/
/*---------------------- METHODS FOR READING RESULTS -----------------------*/
/*--------------------------------------------------------------------------*/

void HQPSolver::get_var_solution( Configuration * solc )
{
 auto hb = dynamic_cast< HQPLevelBlock * >( f_Block );
 if( ( ! hb ) || v_xl.empty() )
  return;

 // a fixed slack only takes the value it is fixed at
 hb->set_x_values( v_xl.back() );
 for( Index p = 0 ; p < v_xl.size() ; ++p )
  hb->set_slack_values( p , v_vl[ p ] , v_wl[ p ] );
 }  // end( HQPSolver::get_var_solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/
// the inner Solver of level 0; those of the other levels are created by
// compute()

void HQPSolver::create_inner( void )
{
 if( auto s = new_inner() )
  v_inner.push_back( s );
 }

/*--------------------------------------------------------------------------*/
// the first Solver of the BlockSolverConfig that is in the factory, with its
// ComputeConfig (if any), attached to the Block; nullptr if there is none

Solver * HQPSolver::new_inner( void )
{
 if( ! ( f_Block && f_bsc ) )
  return( nullptr );

 const auto & names = f_bsc->get_SolverNames();
 const auto & cfgs = f_bsc->get_SolverConfigs();
 for( Index i = 0 ; i < names.size() ; ++i ) {
  if( ! Solver::has_Solver( names[ i ] ) )
   continue;
  auto s = Solver::new_Solver( names[ i ] );
  if( ( i < cfgs.size() ) && cfgs[ i ] )
   s->set_ComputeConfig( cfgs[ i ] );
  s->set_Block( f_Block );
  // after the ComputeConfig, which may create Solvers of its own that the
  // log is passed to (e.g., the inner Solver of a LagrangianDualSolver)
  if( f_log )
   s->set_log( f_log );
  return( s );
  }
 return( nullptr );
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::drop_inner( void )
{
 for( auto s : v_inner ) {
  s->set_Block( nullptr );
  delete s;
  }
 v_inner.clear();
 }

/*--------------------------------------------------------------------------*/
/*------------------------ End File HQPSolver.cpp --------------------------*/
/*--------------------------------------------------------------------------*/
