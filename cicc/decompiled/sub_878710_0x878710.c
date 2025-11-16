// Function: sub_878710
// Address: 0x878710
//
__int64 __fastcall sub_878710(__int64 a1, __m128i *a2)
{
  __int64 v2; // rax
  int v3; // edx
  __int64 result; // rax

  *a2 = _mm_loadu_si128(xmmword_4F06660);
  a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  a2->m128i_i64[1] = *(_QWORD *)(a1 + 48);
  v2 = *(_QWORD *)a1;
  a2[1].m128i_i64[1] = a1;
  a2->m128i_i64[0] = v2;
  a2[1].m128i_i8[2] = (*(_BYTE *)(a1 + 81) >> 3) & 2 | a2[1].m128i_i8[2] & 0xFD;
  a2[2].m128i_i64[0] = *(_QWORD *)(a1 + 64);
  v3 = *(_BYTE *)(a1 + 81) & 0x20;
  result = v3 | a2[1].m128i_i8[1] & 0xDFu;
  a2[1].m128i_i8[1] = v3 | a2[1].m128i_i8[1] & 0xDF;
  return result;
}
