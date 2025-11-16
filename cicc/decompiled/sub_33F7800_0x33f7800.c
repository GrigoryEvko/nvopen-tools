// Function: sub_33F7800
// Address: 0x33f7800
//
__int64 __fastcall sub_33F7800(_QWORD *a1, int a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int128 a7)
{
  __m128i *v8; // rax
  __m128i si128; // xmm0
  __int64 v10; // rdx
  __int64 v11; // r9
  __m128i v13[3]; // [rsp+0h] [rbp-30h] BYREF

  v13[0] = _mm_loadu_si128((const __m128i *)&a7);
  v8 = sub_33ED250((__int64)a1, a4, a5);
  si128 = _mm_load_si128(v13);
  return sub_33E66D0(
           a1,
           a2,
           a3,
           (unsigned __int64)v8,
           v10,
           v11,
           (unsigned __int64 *)si128.m128i_i64[0],
           si128.m128i_i64[1]);
}
