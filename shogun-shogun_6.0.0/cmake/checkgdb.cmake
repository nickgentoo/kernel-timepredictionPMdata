execute_process(COMMAND ${EXECUTABLE} ${FILE} ${ARGS} RESULT_VARIABLE RESULT)
if (RESULT)
	if (ENABLE_PYTHON_DEBUG AND GDB_COMMAND AND GDB_SCRIPT)
		execute_process(COMMAND ${GDB_COMMAND}
			--command=${GDB_SCRIPT}
			--args ${EXECUTABLE} ${FILE} ${ARGS}
				RESULT_VARIABLE GDB_RESULT)
	endif (ENABLE_PYTHON_DEBUG AND GDB_COMMAND AND GDB_SCRIPT)
	MESSAGE( FATAL_ERROR "FAIL:  ${ARGS}, EXITCODE:  ${RESULT}" )
endif (RESULT)
