##############################################################################
################################ makefile ####################################
##############################################################################
#                                                                            #
#   makefile of HQPBlock                                                     #
#                                                                            #
#   Note that $(SMS++INC) is assumed to include any -I directive             #
#   corresponding to external libraries needed by SMS++, at least to the     #
#   extent in which they are needed by the parts of SMS++ used by            #
#   HQPBlock.                                                                #
#                                                                            #
#   Input:  $(CC)        = compiler command                                  #
#           $(SW)        = compiler options                                  #
#           $(SMS++INC)  = the -I$( core SMS++ directory )                   #
#           $(SMS++OBJ)  = the libSMS++ library itself                       #
#           $(HQPBkSDR)  = the directory where the source is                 #
#                                                                            #
#   Output: $(HQPBkOBJ)  = the final object(s) / library                     #
#           $(HQPBkH)    = the .h files to include                           #
#           $(HQPBkINC)  = the -I$( source directory )                       #
#                                                                            #
#                             Antonio Frangioni                              #
#                         Dipartimento di Informatica                        #
#                             Universita' di Pisa                            #
#                                                                            #
##############################################################################

# macros to be exported - - - - - - - - - - - - - - - - - - - - - - - - - - -

HQPBkOBJ = $(HQPBkSDR)/obj/HQPBlock.o $(HQPBkSDR)/obj/HQPDistBlock.o \
	$(HQPBkSDR)/obj/HQPSolver.o

HQPBkINC = -I$(HQPBkSDR)/include

HQPBkH   = $(HQPBkSDR)/include/HQPBlock.h \
	$(HQPBkSDR)/include/HQPDistBlock.h $(HQPBkSDR)/include/HQPSolver.h

# the C++20 module [see makefile-mod of SMS++] - - - - - - - - - - - - - - -
# the prefixes of the modules of the dependencies (e.g., UCBck) go in place
# of SMS++

$(eval $(call SMSppMODrule,HQPBk,$(HQPBkSDR),HQPBlock,\
	smspp.hqpblock,SMS++))

# clean - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

clean::
	rm -f $(HQPBkOBJ) $(HQPBkSDR)/*~

# dependencies: every .o from its .cpp + every recursively included .h- - - -

$(HQPBkSDR)/obj/HQPBlock.o: $(HQPBkSDR)/src/HQPBlock.cpp \
	$(HQPBkSDR)/include/HQPBlock.h $(SMS++H) $(SMS++OBJ)
	$(CC) -c $(HQPBkSDR)/src/HQPBlock.cpp -o $@ \
	$(HQPBkINC) $(SMS++INC) $(SW)

$(HQPBkSDR)/obj/HQPDistBlock.o: $(HQPBkSDR)/src/HQPDistBlock.cpp \
	$(HQPBkSDR)/include/HQPDistBlock.h $(HQPBkSDR)/include/HQPBlock.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(HQPBkSDR)/src/HQPDistBlock.cpp -o $@ \
	$(HQPBkINC) $(SMS++INC) $(SW)

$(HQPBkSDR)/obj/HQPSolver.o: $(HQPBkSDR)/src/HQPSolver.cpp \
	$(HQPBkSDR)/include/HQPSolver.h $(HQPBkSDR)/include/HQPBlock.h \
	$(SMS++H) $(SMS++OBJ)
	$(CC) -c $(HQPBkSDR)/src/HQPSolver.cpp -o $@ \
	$(HQPBkINC) $(SMS++INC) $(SW)

########################## End of makefile ###################################
