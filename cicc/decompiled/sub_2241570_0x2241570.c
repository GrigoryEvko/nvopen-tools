// Function: sub_2241570
// Address: 0x2241570
//
__int64 __fastcall sub_2241570(_QWORD *a1, _BYTE *a2, size_t a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r12
  size_t v6; // r12
  _BYTE *v7; // rsi

  v4 = a1[1];
  if ( a4 > v4 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::copy", a4, a1[1]);
  v6 = v4 - a4;
  if ( v6 > a3 )
    v6 = a3;
  if ( !v6 )
    return v6;
  v7 = (_BYTE *)(*a1 + a4);
  if ( v6 != 1 )
  {
    memcpy(a2, v7, v6);
    return v6;
  }
  *a2 = *v7;
  return 1;
}
