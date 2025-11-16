// Function: sub_2251FC0
// Address: 0x2251fc0
//
__int64 __fastcall sub_2251FC0(_QWORD *a1, wchar_t *a2, size_t a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r12
  size_t v5; // r12
  const wchar_t *v7; // rsi

  v4 = a1[1];
  if ( a4 > v4 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::copy", a4, a1[1]);
  v5 = v4 - a4;
  if ( v5 > a3 )
    v5 = a3;
  if ( !v5 )
    return v5;
  v7 = (const wchar_t *)(*a1 + 4 * a4);
  if ( v5 != 1 )
  {
    wmemcpy(a2, v7, v5);
    return v5;
  }
  *a2 = *v7;
  return 1;
}
