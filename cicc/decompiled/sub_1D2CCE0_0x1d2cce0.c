// Function: sub_1D2CCE0
// Address: 0x1d2cce0
//
__int64 __fastcall sub_1D2CCE0(
        _QWORD *a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int64 v9; // rax
  __m128i v10; // xmm1
  int v11; // edx
  __int64 v12; // r9
  _OWORD v14[4]; // [rsp+0h] [rbp-40h] BYREF

  v9 = sub_1D29190((__int64)a1, a4, a5, a4, a5, a6);
  v10 = _mm_loadu_si128((const __m128i *)&a8);
  v14[0] = _mm_loadu_si128((const __m128i *)&a7);
  v14[1] = v10;
  return sub_1D23DE0(a1, a2, a3, v9, v11, v12, (__int64 *)v14, 2);
}
