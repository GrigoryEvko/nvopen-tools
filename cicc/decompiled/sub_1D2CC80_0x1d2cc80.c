// Function: sub_1D2CC80
// Address: 0x1d2cc80
//
__int64 __fastcall sub_1D2CC80(_QWORD *a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int128 a7)
{
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // r9
  __m128i v12[3]; // [rsp+0h] [rbp-30h] BYREF

  v8 = sub_1D29190((__int64)a1, a4, a5, a4, a5, a6);
  v12[0] = _mm_loadu_si128((const __m128i *)&a7);
  return sub_1D23DE0(a1, a2, a3, v8, v9, v10, v12[0].m128i_i64, 1);
}
