#ifndef __LIBLINEAR_LOCALES__
#define __LIBLINEAR_LOCALES__

#include <locale.h>

#if __unix__
	#include <unistd.h> // For _POSIX_VERSION
#endif

#if _POSIX_VERSION >= 200809L

// If possible, use the thread-safe uselocale() function
typedef locale_t locale_handle ;

locale_handle set_c_locale();

void restore_locale(locale_handle locale);

#else

// But fall back to setlocale() if uselocale() is not available
typedef char *locale_handle;

locale_handle set_c_locale();

void restore_locale(locale_handle locale);

#endif /* _POSIX_VERSION */

#endif /* __LIBLINEAR_LOCALES__ */

