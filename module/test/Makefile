ADDBUILD := $(if ${BUILD}, BUILD=${BUILD})
echo:
	@echo BUILD ${BUILD}
	@echo kdev:
	@${MAKE} ${ADDBUILD} -C kdev echo
	@echo unit:
	@${MAKE} ${ADDBUILD} -C unit echo

kdev_test:
	@${MAKE} ${ADDBUILD} -C kdev test

unit_test:
	@${MAKE} ${ADDBUILD} -C unit test

test: kdev_test unit_test
