/*--------------------------------------------------------------------------*/
/*--------------------------- File test_radial.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Tester of HQPBlock and HQPSolver on the hierarchical QPs of a
 * centralized Model Predictive Control of 8 robots (each one has to reach
 * the antipodal point of a circle, so that all cross in the middle; level 0
 * the dynamics, 1 the limits of the inputs, 2 their smoothness, 3 the
 * avoidance of the collisions, 4 the goals), at some steps of the run: the
 * files data/nc4/radial8_s<k>.nc4, written by tools/npz2nc4.py, carry with
 * the HQPBlock the solution after each level of the code that generated
 * them (group HQPReference), which solves each level in the null space of
 * the higher ones with a regularization of \f$ 10^{-6} \f$ on the step and
 * on the slacks.
 *
 * For each file every HQPSolver of BSPar.txt solves the HQPBlock, and the
 * values of its levels (\f$ \| v_p \|^2 / 2 + \| w_p \|^2 / 2 \f$, which are
 * unique) are compared with those at the solution in the file and with the
 * reference of test_common.h with the same regularization; x is printed but
 * not checked, since with a regularization of \f$ 10^{-6} \f$ it is fixed
 * only to about the square root of the accuracy of each solve. Then each
 * file after the first of a run of consecutive steps is also solved on the
 * HQPBlock of the previous step, changed into it by set_b(), set_d(),
 * set_A() and set_C(), i.e., as the Model Predictive Control does.
 *
 * Usage: HQPBlock_radial_test [ <directory of the files> ]
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 */
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <algorithm>
#include <filesystem>

#include "test_common.h"

/*--------------------------------------------------------------------------*/

using namespace hqp_test;

static const char * const config = "BSPar.txt";

static constexpr double VTOL = 1e-5;  // tolerance on the values

/*--------------------------------------------------------------------------*/
// the dense Instance of an HQPBlock

static Instance instance_of( const HQPBlock & block , const std::string & nm )
{
 Instance in;
 in.name = nm;
 in.n = block.get_num_var();
 in.eps = block.get_regularization();
 for( Block::Index p = 0 ; p < block.get_num_levels() ; ++p ) {
  in.A.push_back( Mat( block.get_A( p ) ) );
  in.C.push_back( Mat( block.get_C( p ) ) );
  const auto e = block.get_eq_start( p ) , i = block.get_ineq_start( p );
  in.b.push_back( Eigen::Map< const Vec >( block.get_b().data() + e ,
                                           block.get_num_eq_rows( p ) ) );
  in.d.push_back( Eigen::Map< const Vec >( block.get_d().data() + i ,
                                           block.get_num_ineq_rows( p ) ) );
  in.we.push_back( Eigen::Map< const Vec >( block.get_we().data() + e ,
                                            block.get_num_eq_rows( p ) ) );
  in.wi.push_back( Eigen::Map< const Vec >( block.get_wi().data() + i ,
                                            block.get_num_ineq_rows( p ) ) );
  }
 return( in );
 }

/*--------------------------------------------------------------------------*/
// the solution of each level in the file, and the values of the levels

static RefSolution file_solution( const std::string & file ,
                                  const Instance & in )
{
 RefSolution sol;
 netCDF::NcFile f( file , netCDF::NcFile::read );
 auto g = f.getGroup( "HQPReference" );
 if( g.isNull() ) {
  sol.ok = false;
  return( sol );
  }
 Mat X( in.levels() , in.n );
 Eigen::Matrix< double , Eigen::Dynamic , Eigen::Dynamic , Eigen::RowMajor >
  Xr( in.levels() , in.n );
 g.getVar( "X" ).getVar( Xr.data() );
 for( int p = 0 ; p < in.levels() ; ++p ) {
  Vec x = Xr.row( p ).transpose();
  const Vec v = in.we[ p ].cwiseProduct( in.A[ p ] * x - in.b[ p ] );
  const Vec w = in.wi[ p ].cwiseProduct( in.C[ p ] * x - in.d[ p ] )
   .cwiseMax( 0.0 );
  sol.value.push_back( ( v.squaredNorm() + w.squaredNorm() ) / 2 );
  sol.x.push_back( std::move( x ) );
  }
 return( sol );
 }

/*--------------------------------------------------------------------------*/
// solves with all the HQPSolver and compares; returns the number of errors

static int check( const std::string & what , std::vector< HQPSolver * > & sl ,
                  const RefSolution & fsol , const RefSolution & ref )
{
 int errors = 0;
 for( auto s : sl ) {
  if( ! s->get_inner_Solver() )
   continue;
  const int st = s->compute();
  if( st != Solver::kOK ) {
   std::cout << what << " [" << label( s ) << "]: status " << st
             << " -> KO" << std::endl;
   ++errors;
   continue;
   }
  // the tolerance on x is infinite: it is printed, not checked
  errors += compare( what + " vs file" , s , fsol , VTOL , 1e100 );
  errors += compare( what + " vs reference" , s , ref , VTOL , 1e100 );
  }
 return( errors );
 }

/*--------------------------------------------------------------------------*/

int main( int argc , char ** argv )
{
 const std::string dir = argc > 1 ? argv[ 1 ] : "../data/nc4";
 std::vector< std::string > files;
 if( std::filesystem::is_directory( dir ) )
  for( const auto & e : std::filesystem::directory_iterator( dir ) )
   if( e.path().filename().string().rfind( "radial" , 0 ) == 0 )
    files.push_back( e.path().string() );
 std::sort( files.begin() , files.end() );
 if( files.empty() ) {
  std::cout << "no radial*.nc4 in " << dir << ": skipped" << std::endl;
  return( 77 );
  }

 int errors = 0;
 HQPBlock * prev = nullptr;
 std::vector< HQPSolver * > prev_sl;
 int prev_step = -2;

 for( const auto & file : files ) {
  const std::string nm = std::filesystem::path( file ).stem().string();
  const int step = std::stoi( nm.substr( nm.rfind( 's' ) + 1 ) );
  auto block = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
  if( ! block ) {
   std::cout << nm << ": cannot read the HQPBlock -> KO" << std::endl;
   ++errors;
   continue;
   }
  const Instance in = instance_of( *block , nm );
  const RefSolution fsol = file_solution( file , in );
  const RefSolution ref = reference( in , true , in.eps );
  if( ! ( fsol.ok && ref.ok ) ) {
   std::cout << nm << ": no solution in the file, or the reference failed"
             << " -> KO" << std::endl;
   ++errors;
   }

  auto sl = attach( *block , config );
  errors += check( nm , sl , fsol , ref );

  // the same step reached by Modification of the HQPBlock of the previous
  if( prev && ( step == prev_step + 1 ) ) {
   for( Block::Index p = 0 ; p < block->get_num_levels() ; ++p ) {
    const auto e = block->get_eq_start( p ) , i = block->get_ineq_start( p );
    prev->set_b( p , std::vector< double >( block->get_b().begin() + e ,
                     block->get_b().begin() + e +
                     block->get_num_eq_rows( p ) ) );
    prev->set_d( p , std::vector< double >( block->get_d().begin() + i ,
                     block->get_d().begin() + i +
                     block->get_num_ineq_rows( p ) ) );
    prev->set_A( p , block->get_A( p ) );
    prev->set_C( p , block->get_C( p ) );
    }
   errors += check( nm + " after the step" , prev_sl , fsol , ref );
   }

  if( prev ) {
   prev->unregister_Solvers( true );
   delete prev;
   }
  prev = block;
  prev_sl = sl;
  prev_step = step;
  }

 if( prev ) {
  prev->unregister_Solvers( true );
  delete prev;
  }

 std::cout << ( errors ? "HQPBlock radial: FAILED"
                       : "HQPBlock radial: all tests passed" ) << std::endl;
 return( errors ? 1 : 0 );
 }

/*--------------------------------------------------------------------------*/
/*------------------------ End File test_radial.cpp ------------------------*/
/*--------------------------------------------------------------------------*/
