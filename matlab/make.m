% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
function make()
try
	% This part is for OCTAVE
	if(exist('OCTAVE_VERSION', 'builtin'))
		mex libsvmread.c
		mex libsvmwrite.c
		setenv('CFLAGS', strcat(getenv('CFLAGS'), ' -fopenmp'))
		setenv('CXXFLAGS', strcat(getenv('CXXFLAGS'), ' -fopenmp'))
		mex -I.. -lgomp train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
		mex -I.. -lgomp predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		mex -largeArrayDims libsvmread.c
		mex -largeArrayDims libsvmwrite.c
		mex CFLAGS='$CFLAGS -fopenmp' CXXFLAGS='$CXXFLAGS -fopenmp' -I.. -largeArrayDims -lgomp train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
		mex CFLAGS='$CFLAGS -fopenmp' CXXFLAGS='$CXXFLAGS -fopenmp' -I.. -largeArrayDims -lgomp predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	end
catch err
	fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
	disp(err.message);
	fprintf('=> Please check README for detailed instructions.\n');
end
