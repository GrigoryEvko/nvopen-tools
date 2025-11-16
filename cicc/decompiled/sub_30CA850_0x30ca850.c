// Function: sub_30CA850
// Address: 0x30ca850
//
__int64 __fastcall sub_30CA850(__int64 a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0

  v2 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0x21u )
    return sub_CB6200(a2, "Unimplemented InlineAdvisor print\n", 0x22u);
  si128 = _mm_load_si128((const __m128i *)&xmmword_44CCA90);
  v2[2].m128i_i16[0] = 2676;
  *v2 = si128;
  v2[1] = _mm_load_si128((const __m128i *)&xmmword_44CCAA0);
  *(_QWORD *)(a2 + 32) += 34LL;
  return 2676;
}
