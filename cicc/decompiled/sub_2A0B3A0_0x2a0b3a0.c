// Function: sub_2A0B3A0
// Address: 0x2a0b3a0
//
__int64 __fastcall sub_2A0B3A0(__int64 a1, const __m128i *a2)
{
  __m128i v3; // xmm1
  __m128i v4; // xmm0
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __int64 v7; // [rsp+0h] [rbp-100h]
  __int64 v8; // [rsp+30h] [rbp-D0h]

  LOBYTE(v7) = 0;
  LOBYTE(v8) = 0;
  v3 = _mm_loadu_si128(a2 + 2);
  v4 = _mm_loadu_si128(a2 + 3);
  v5 = _mm_loadu_si128(a2);
  *(_QWORD *)(a1 + 32) = v7;
  v6 = _mm_loadu_si128(a2 + 1);
  *(__m128i *)a1 = v5;
  *(_QWORD *)(a1 + 72) = v8;
  *(__m128i *)(a1 + 16) = v6;
  *(__m128i *)(a1 + 40) = v3;
  *(__m128i *)(a1 + 56) = v4;
  return a1;
}
