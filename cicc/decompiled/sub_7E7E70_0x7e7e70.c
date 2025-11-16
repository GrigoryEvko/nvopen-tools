// Function: sub_7E7E70
// Address: 0x7e7e70
//
_BYTE *__fastcall sub_7E7E70(__int64 a1, __int64 *a2)
{
  _BYTE *v3; // r12
  __int64 v5; // rsi

  v3 = sub_73DE50(a1, (__int64)a2);
  if ( dword_4F077C4 != 2 )
    return v3;
  v5 = *a2;
  if ( *a2 )
    sub_73E3D0((__int64)v3, v5, 1);
  sub_7E7D10(v3, v5);
  return v3;
}
