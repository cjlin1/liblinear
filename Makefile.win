CXX = cl.exe
CFLAGS = /nologo /O2 /EHsc /I. /D _WIN64 /D _CRT_SECURE_NO_DEPRECATE
TARGET = windows

all: $(TARGET)\train.exe $(TARGET)\predict.exe lib

$(TARGET)\train.exe: tron.obj linear.obj train.c blas\*.c
	$(CXX) $(CFLAGS) -Fe$(TARGET)\train.exe tron.obj linear.obj train.c blas\*.c

$(TARGET)\predict.exe: tron.obj linear.obj predict.c blas\*.c
	$(CXX) $(CFLAGS) -Fe$(TARGET)\predict.exe tron.obj linear.obj predict.c blas\*.c

linear.obj: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c linear.cpp

tron.obj: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c tron.cpp

lib: linear.cpp linear.h linear.def tron.obj
	$(CXX) $(CFLAGS) -LD linear.cpp tron.obj blas\*.c -Fe$(TARGET)\liblinear -link -DEF:linear.def 

clean:
	 -erase /Q *.obj $(TARGET)\*.exe $(TARGET)\*.dll $(TARGET)\*.exp $(TARGET)\*.lib

