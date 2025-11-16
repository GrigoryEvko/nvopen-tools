// Function: sub_33E6B00
// Address: 0x33e6b00
//
__int64 __fastcall sub_33E6B00(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  unsigned __int64 v9; // rax
  __m128i si128; // xmm0
  __int64 v11; // rdx
  __int64 v12; // r9
  __m128i v14[3]; // [rsp+0h] [rbp-30h] BYREF

  v14[0] = _mm_loadu_si128((const __m128i *)&a8);
  v9 = sub_33E5110(a1, a4, a5, a7, *((__int64 *)&a7 + 1));
  si128 = _mm_load_si128(v14);
  return sub_33E66D0(a1, a2, a3, v9, v11, v12, (unsigned __int64 *)si128.m128i_i64[0], si128.m128i_i64[1]);
}
