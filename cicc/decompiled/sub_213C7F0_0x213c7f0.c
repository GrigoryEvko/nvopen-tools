// Function: sub_213C7F0
// Address: 0x213c7f0
//
__int64 *__fastcall sub_213C7F0(__int64 a1, __int64 *a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int64 v8; // rax
  unsigned int v9; // ecx
  __m128i v11; // [rsp+0h] [rbp-30h] BYREF
  __m128i v12; // [rsp+10h] [rbp-20h] BYREF

  v5 = a2[4];
  v6 = _mm_loadu_si128((const __m128i *)v5);
  v7 = _mm_loadu_si128((const __m128i *)(v5 + 40));
  v8 = *(_QWORD *)(v5 + 160);
  v11 = v6;
  v9 = *(_DWORD *)(v8 + 84);
  v12 = v7;
  sub_213C0E0(a1, (__int64)&v11, (__int64)&v12, v9, v6, *(double *)v7.m128i_i64, a5);
  return sub_1D2E370(
           *(_QWORD **)(a1 + 8),
           a2,
           v11.m128i_i64[0],
           v11.m128i_i64[1],
           v12.m128i_i64[0],
           v12.m128i_i64[1],
           *(_OWORD *)(a2[4] + 80),
           *(_OWORD *)(a2[4] + 120),
           *(_OWORD *)(a2[4] + 160));
}
