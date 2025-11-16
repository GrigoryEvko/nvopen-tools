// Function: sub_2215320
// Address: 0x2215320
//
__int64 __fastcall sub_2215320(_QWORD *a1, _BYTE *a2, size_t a3, unsigned __int64 a4)
{
  unsigned __int64 v6; // r12
  size_t v7; // r12
  _BYTE *v8; // rsi

  v6 = *(_QWORD *)(*a1 - 24LL);
  if ( a4 > v6 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::copy");
  v7 = v6 - a4;
  if ( v7 > a3 )
    v7 = a3;
  if ( !v7 )
    return v7;
  v8 = (_BYTE *)(*a1 + a4);
  if ( v7 != 1 )
  {
    memcpy(a2, v8, v7);
    return v7;
  }
  *a2 = *v8;
  return 1;
}
