// Function: sub_1D25BD0
// Address: 0x1d25bd0
//
__int64 __fastcall sub_1D25BD0(
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
  __m128i si128; // xmm0
  int v11; // edx
  __int64 v12; // r9
  __m128i v14[3]; // [rsp+0h] [rbp-30h] BYREF

  v14[0] = _mm_loadu_si128((const __m128i *)&a8);
  v9 = sub_1D252B0((__int64)a1, a4, a5, a7, *((__int64 *)&a7 + 1));
  si128 = _mm_load_si128(v14);
  return sub_1D23DE0(a1, a2, a3, v9, v11, v12, (__int64 *)si128.m128i_i64[0], si128.m128i_i64[1]);
}
