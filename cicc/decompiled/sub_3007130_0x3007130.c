// Function: sub_3007130
// Address: 0x3007130
//
__int64 __fastcall sub_3007130(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // rdi
  __m128i *v6; // rax
  __m128i si128; // xmm0

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_DWORD *)(v2 + 32);
  if ( *(_BYTE *)(v2 + 8) != 18 )
    return v3;
  v5 = sub_CA5BD0(a1, a2);
  v6 = *(__m128i **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 > 0xB0u )
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4457A20);
    v6[11].m128i_i8[0] = 10;
    *v6 = si128;
    v6[1] = _mm_load_si128((const __m128i *)&xmmword_4457A30);
    v6[2] = _mm_load_si128((const __m128i *)&xmmword_4457A40);
    v6[3] = _mm_load_si128((const __m128i *)&xmmword_4457A50);
    v6[4] = _mm_load_si128((const __m128i *)&xmmword_4457A60);
    v6[5] = _mm_load_si128((const __m128i *)&xmmword_4457A70);
    v6[6] = _mm_load_si128((const __m128i *)&xmmword_4457A80);
    v6[7] = _mm_load_si128((const __m128i *)&xmmword_4457A90);
    v6[8] = _mm_load_si128((const __m128i *)&xmmword_4457AA0);
    v6[9] = _mm_load_si128((const __m128i *)&xmmword_4457AB0);
    v6[10] = _mm_load_si128((const __m128i *)&xmmword_4457AC0);
    *(_QWORD *)(v5 + 32) += 177LL;
    return v3;
  }
  sub_CB6200(
    v5,
    "The code that requested the fixed number of elements has made the assumption that this vector is not scalable. This "
    "assumption was not correct, and this may lead to broken code\n",
    0xB1u);
  return v3;
}
