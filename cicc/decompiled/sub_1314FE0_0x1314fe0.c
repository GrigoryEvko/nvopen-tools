// Function: sub_1314FE0
// Address: 0x1314fe0
//
__int64 __fastcall sub_1314FE0(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4)
{
  unsigned __int64 v6; // rax
  char v7; // cl
  __int64 v8; // rdi
  int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // rax
  char v12; // cl
  __int64 v13; // rax
  int v14; // edx
  unsigned int v15; // eax
  __int64 result; // rax

  v6 = qword_505FA40[(unsigned __int8)(*a3 >> 20)];
  if ( v6 > 0x7000000000000000LL )
  {
    v11 = 10384;
  }
  else
  {
    if ( v6 < 0x4000 )
      v6 = 0x4000;
    v7 = 7;
    _BitScanReverse64((unsigned __int64 *)&v8, 2 * v6 - 1);
    if ( (unsigned int)v8 >= 7 )
      v7 = v8;
    v9 = (((-1LL << (v7 - 3)) & (v6 - 1)) >> (v7 - 3)) & 3;
    if ( (unsigned int)v8 < 6 )
      LODWORD(v8) = 6;
    v10 = v9 + 4 * v8 - 23;
    if ( v10 < 0x24 )
      v10 = 36;
    v11 = 48LL * (v10 - 36) + 976;
  }
  _InterlockedAdd64((volatile signed __int64 *)(a2 + v11), 1u);
  if ( a4 > 0x7000000000000000LL )
  {
    result = 10392;
  }
  else
  {
    v12 = 7;
    if ( a4 < 0x4000 )
      a4 = 0x4000;
    _BitScanReverse64((unsigned __int64 *)&v13, 2 * a4 - 1);
    if ( (unsigned int)v13 >= 7 )
      v12 = v13;
    v14 = (((-1LL << (v12 - 3)) & (a4 - 1)) >> (v12 - 3)) & 3;
    if ( (unsigned int)v13 < 6 )
      LODWORD(v13) = 6;
    v15 = v14 + 4 * v13 - 23;
    if ( v15 < 0x24 )
      v15 = 36;
    result = 48LL * (v15 - 36) + 984;
  }
  _InterlockedAdd64((volatile signed __int64 *)(a2 + result), 1u);
  return result;
}
