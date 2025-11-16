// Function: .mbrtowc
// Address: 0x406d60
//
// attributes: thunk
size_t mbrtowc(wchar_t *pwc, const char *s, size_t n, mbstate_t *p)
{
  return mbrtowc(pwc, s, n, p);
}
