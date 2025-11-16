// Function: sub_1713A90
// Address: 0x1713a90
//
__int64 __fastcall sub_1713A90(
        __int64 *a1,
        _BYTE *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  __int64 v13; // rax

  if ( *(_BYTE *)(*((_QWORD *)a2 - 3) + 16LL) > 0x10u )
    return 0;
  v10 = *((_QWORD *)a2 - 6);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 <= 0x17u )
    return 0;
  if ( v11 == 79 )
  {
    v13 = *(_QWORD *)(v10 + 8);
    if ( !v13 || *(_QWORD *)(v13 + 8) )
      return 0;
    return sub_1707160((__int64)a1, a2, v10, *(double *)a3.m128_u64, a4, a5);
  }
  else
  {
    if ( v11 != 77 )
      return 0;
    return sub_17127D0(a1, (__int64)a2, v10, a3, a4, a5, a6, a7, a8, a9, a10);
  }
}
