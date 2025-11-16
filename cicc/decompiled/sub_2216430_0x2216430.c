// Function: sub_2216430
// Address: 0x2216430
//
const wchar_t **__fastcall sub_2216430(const wchar_t **a1, size_t a2, __int64 a3, unsigned __int64 a4, wchar_t a5)
{
  wchar_t *v8; // rdi

  if ( a4 > a3 + 0xFFFFFFFFFFFFFFELL - *((_QWORD *)*a1 - 3) )
    sub_4262D8((__int64)"basic_string::_M_replace_aux");
  sub_22161B0(a1, a2, a3, a4);
  if ( !a4 )
    return a1;
  v8 = (wchar_t *)&(*a1)[a2];
  if ( a4 != 1 )
  {
    wmemset(v8, a5, a4);
    return a1;
  }
  *v8 = a5;
  return a1;
}
