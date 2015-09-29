#include <linear_locales.h>

#if _POSIX_VERSION >= 200809L

locale_handle set_c_locale()
{
	locale_handle c_locale = newlocale(LC_ALL_MASK, "C", 0);
	locale_handle old_locale = uselocale(c_locale);
	return old_locale;
}

void restore_locale(locale_handle locale)
{
	locale_handle c_locale = uselocale(locale);
	if (c_locale && c_locale != LC_GLOBAL_LOCALE) {
		freelocale(c_locale);
	}
}

#else

locale_handle set_c_locale()
{
	locale_handle old_locale = setlocale(LC_ALL, NULL);
	if (old_locale) {
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");
	return old_locale;
}

void restore_locale(locale_handle locale)
{
	setlocale(LC_ALL, locale);
	free(locale);
}

#endif /* _POSIX_VERSION */

