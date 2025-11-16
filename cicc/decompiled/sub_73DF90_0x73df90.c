// Function: sub_73DF90
// Address: 0x73df90
//
_BYTE *__fastcall sub_73DF90(__int64 a1, __int64 *a2)
{
  _BYTE *v2; // r12

  *(_QWORD *)(a1 + 16) = a2;
  a2[2] = 0;
  v2 = sub_73DBF0(0x5Bu, *a2, a1);
  sub_730580((__int64)a2, (__int64)v2);
  if ( (v2[25] & 3) != 0 )
    v2[58] |= 1u;
  return v2;
}
