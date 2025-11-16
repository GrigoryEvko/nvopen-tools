// Function: sub_1314E00
// Address: 0x1314e00
//
__int64 __fastcall sub_1314E00(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v3; // rax
  char v4; // cl
  __int64 v5; // rdx
  int v6; // eax
  unsigned int v7; // eax
  __int64 result; // rax

  v3 = qword_505FA40[(unsigned __int8)(*a3 >> 20)];
  if ( v3 > 0x7000000000000000LL )
  {
    result = 10392;
  }
  else
  {
    v4 = 7;
    if ( v3 < 0x4000 )
      v3 = 0x4000;
    _BitScanReverse64((unsigned __int64 *)&v5, 2 * v3 - 1);
    if ( (unsigned int)v5 >= 7 )
      v4 = v5;
    v6 = (((-1LL << (v4 - 3)) & (v3 - 1)) >> (v4 - 3)) & 3;
    if ( (unsigned int)v5 < 6 )
      LODWORD(v5) = 6;
    v7 = v6 + 4 * v5 - 23;
    if ( v7 < 0x24 )
      v7 = 36;
    result = 48LL * (v7 - 36) + 984;
  }
  _InterlockedAdd64((volatile signed __int64 *)(a2 + result), 1u);
  return result;
}
