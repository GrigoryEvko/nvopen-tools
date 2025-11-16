// Function: sub_1D2CD40
// Address: 0x1d2cd40
//
__int64 __fastcall sub_1D2CD40(
        _QWORD *a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  __int64 v10; // rax
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  int v13; // edx
  __int64 v14; // r9
  _OWORD v16[5]; // [rsp+0h] [rbp-50h] BYREF

  v10 = sub_1D29190((__int64)a1, a4, a5, a4, a5, a6);
  v11 = _mm_loadu_si128((const __m128i *)&a8);
  v12 = _mm_loadu_si128((const __m128i *)&a9);
  v16[0] = _mm_loadu_si128((const __m128i *)&a7);
  v16[1] = v11;
  v16[2] = v12;
  return sub_1D23DE0(a1, a2, a3, v10, v13, v14, (__int64 *)v16, 3);
}
