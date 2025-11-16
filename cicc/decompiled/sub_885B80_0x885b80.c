// Function: sub_885B80
// Address: 0x885b80
//
_QWORD *__fastcall sub_885B80(char *a1, size_t a2, unsigned __int8 a3, int a4)
{
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __int64 v10[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v11; // [rsp+10h] [rbp-50h]
  __m128i v12; // [rsp+20h] [rbp-40h]
  __m128i v13; // [rsp+30h] [rbp-30h]

  v6 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v8 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v10[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v11 = v6;
  v12 = v7;
  v13 = v8;
  v10[1] = *(_QWORD *)&dword_4F077C8;
  sub_878540(a1, a2, v10);
  return sub_885AD0(a3, (__int64)v10, a4, 0);
}
