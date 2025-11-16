// Function: sub_1D392A0
// Address: 0x1d392a0
//
__int64 __fastcall sub_1D392A0(
        __int64 a1,
        int a2,
        __int64 a3,
        unsigned int a4,
        const void **a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  __int64 result; // rax
  __int64 v14; // [rsp-50h] [rbp-50h]
  __int64 v15; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v16; // [rsp-40h] [rbp-40h]
  char v17; // [rsp-38h] [rbp-38h]

  if ( (*(_BYTE *)(a6 + 26) & 8) != 0 )
    return 0;
  if ( (*(_BYTE *)(a10 + 26) & 8) != 0 )
    return 0;
  sub_1D14650((__int64)&v15, a2, *(_QWORD *)(a6 + 88) + 24LL, *(_QWORD *)(a10 + 88) + 24LL);
  if ( v17 )
    result = sub_1D38970(a1, (__int64)&v15, a3, a4, a5, 0, a7, a8, a9, 0);
  else
    result = 0;
  if ( v16 > 0x40 )
  {
    if ( v15 )
    {
      v14 = result;
      j_j___libc_free_0_0(v15);
      return v14;
    }
  }
  return result;
}
