// Function: sub_33F77A0
// Address: 0x33f77a0
//
__int64 __fastcall sub_33F77A0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __m128i *v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // rdx
  __int64 v12; // r9
  _OWORD v14[4]; // [rsp+0h] [rbp-40h] BYREF

  v9 = sub_33ED250((__int64)a1, a4, a5);
  v10 = _mm_loadu_si128((const __m128i *)&a8);
  v14[0] = _mm_loadu_si128((const __m128i *)&a7);
  v14[1] = v10;
  return sub_33E66D0(a1, a2, a3, (unsigned __int64)v9, v11, v12, (unsigned __int64 *)v14, 2);
}
