// Function: sub_1AA62D0
// Address: 0x1aa62d0
//
__int64 __fastcall sub_1AA62D0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax
  __int64 ***v12; // r12
  __int64 *v13; // rax
  __int64 v14; // rsi

  result = *(_QWORD *)(a1 + 48);
  if ( !result )
    BUG();
  if ( *(_BYTE *)(result - 8) == 77 )
  {
    while ( *(_BYTE *)(result - 8) == 77 )
    {
      v12 = (__int64 ***)(result - 24);
      if ( (*(_BYTE *)(result - 1) & 0x40) != 0 )
        v13 = *(__int64 **)(result - 32);
      else
        v13 = (__int64 *)&v12[-3 * (*(_DWORD *)(result - 4) & 0xFFFFFFF)];
      v14 = *v13;
      if ( *v13 && v12 == (__int64 ***)v14 )
        v14 = sub_1599EF0(*v12);
      sub_164D160((__int64)v12, v14, a3, a4, a5, a6, a7, a8, a9, a10);
      if ( a2 )
        sub_14191F0(a2, (__int64)v12);
      sub_15F20C0(v12);
      result = *(_QWORD *)(a1 + 48);
      if ( !result )
        BUG();
    }
  }
  return result;
}
