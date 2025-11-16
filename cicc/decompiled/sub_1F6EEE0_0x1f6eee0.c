// Function: sub_1F6EEE0
// Address: 0x1f6eee0
//
__int64 __fastcall sub_1F6EEE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const void **a4,
        __int64 a5,
        char a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  const void **v13; // [rsp+8h] [rbp-28h]

  v12 = a3;
  v13 = a4;
  if ( !(_BYTE)a3 )
  {
    if ( sub_1F58D20((__int64)&v12) && a6 )
      return 0;
    return sub_1D38BB0(a5, 0, a1, (unsigned int)v12, v13, 0, a7, a8, a9, 0);
  }
  if ( (unsigned __int8)(a3 - 14) > 0x5Fu || !a6 )
    return sub_1D38BB0(a5, 0, a1, (unsigned int)v12, v13, 0, a7, a8, a9, 0);
  if ( !*(_QWORD *)(a2 + 8LL * (unsigned __int8)a3 + 120) )
    return 0;
  if ( !*(_BYTE *)(a2 + 259LL * (unsigned __int8)a3 + 2526) )
    return sub_1D38BB0(a5, 0, a1, (unsigned int)v12, v13, 0, a7, a8, a9, 0);
  return 0;
}
