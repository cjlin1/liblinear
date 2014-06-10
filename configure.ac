#                                               -*- Autoconf -*- 
# Process this file with autoconf to produce a configure script. 
# Order is largely irrevellant, although it must start with AC_INIT and end with AC_OUTPUT 
# See http://autotoolset.sourceforge.net/tutorial.html 
# and http://www.openismus.com/documents/linux/automake/automake.shtml 

AC_INIT(LIBLINEAR, 1.2.1, bugs@afflib.org)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE
AM_MAINTAINER_MODE
AC_PREFIX_PROGRAM(liblinear)

# Programs we will be using
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AM_PROG_CC_C_O

LT_INIT

################################################################
## See if we are running on mingw

mingw="no"
case $host in
  *-*-*linux*-*) 
     AC_DEFINE([__LINUX__],1,[Linux operating system functions])
     ;;

  *mingw*)   
     LIBS="$LIBS -lpsapi -lws2_32 -lgdi32"  
     CPPFLAGS="-DUNICODE -D_UNICODE -D__MSVCRT_VERSION__=0x0601 -DWINVER=0x0500 -D_WIN32_WINNT=0x0500 -g $CPPFLAGS"
     CPPFLAGS="$CPPFLAGS --static "
     CFLAGS="$CFLAGS --static -static-libgcc -static-libstdc++"
     CXXFLAGS="$CXXFLAGS -Wno-format "  # compiler mingw-4.3.0 is broken on I64u formats
     CXXFLAGS="$CXXFLAGS --static -static-libgcc -static-libstdc++"
     LDFLAGS="$LDFLAGS --static"
     mingw="yes"
     ;;		 		     
esac



################################################################
# Specify our other headers

AC_HEADER_STDC

m4_include([m4/ax_openmp.m4])

#
# OpenMP
AX_OPENMP([AC_MSG_NOTICE([OpenMP Found])])
CFLAGS="$CFLAGS $OPENMP_CFLAGS"
CXXFLAGS="$CXXFLAGS $OPENMP_CFLAGS"

# -lm
AC_CHECK_HEADERS([math.h],,AC_MSG_ERROR([missing math.h]))
AC_CHECK_LIB([m],[fmax],,AC_MSG_ERROR([missing -lm]))
AC_CHECK_FUNCS([fmin fmax log exp fabs sqrt],,AC_MSG_ERROR([missing math functions]))

## libtool required for preparing the library
AC_CHECK_PROG(has_libtool, libtool, true, false)
if test x"$has_libtool" == "xfalse" ; then
  AC_MSG_ERROR([libhashdb requires libtool, please install libtool such as The GNU Portable Library Tool])
fi

## If compiling on mingw, add another -lgdi32 to be safe
## This is required becuase some libraries reference functions in gdi32 and, for whatever reason,
## the linker is not smart enough to re-check the library
if test x"$mingw" = x"yes" ; then
  LIBS="$LIBS -lgdi32"
else
  CFLAGS="$CFLAGS -fPIC"
  CXXFLAGS="$CXXFLAGS -fPIC"
fi


############## drop optimization flags if requested ################
############## or if GCC_NOOPT variable is set      ################
AC_ARG_WITH(noopt, AC_HELP_STRING([--with-noopt], [Drop -O C flags]))
if test x"${GCC_NOOPT}" != "x" ; then
   with_noopt="yes";
fi

if test "${with_noopt}" = "yes" ; then
  CFLAGS=`echo "$CFLAGS" | sed s/-O[[0-9]]//`             # note the double quoting! 
  CXXFLAGS=`echo "$CXXFLAGS" | sed s/-O[[0-9]]//`
fi

AC_OUTPUT
