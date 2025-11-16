// Function: sub_1D25E10
// Address: 0x1d25e10
//
__int64 __fastcall sub_1D25E10(
        _QWORD *a1,
        __int16 a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __int64 v8; // rax
  __m128i si128; // xmm0
  int v10; // edx
  __int64 v11; // r9
  __m128i v13[3]; // [rsp+0h] [rbp-30h] BYREF

  v13[0] = _mm_loadu_si128((const __m128i *)&a7);
  v8 = sub_1D25C30((__int64)a1, a4, a5);
  si128 = _mm_load_si128(v13);
  return sub_1D23DE0(a1, a2, a3, v8, v10, v11, (__int64 *)si128.m128i_i64[0], si128.m128i_i64[1]);
}
