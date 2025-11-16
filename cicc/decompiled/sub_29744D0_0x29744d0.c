// Function: sub_29744D0
// Address: 0x29744d0
//
unsigned __int64 __fastcall sub_29744D0(__int64 a1, const __m128i *a2)
{
  __m128i v2; // xmm0

  v2 = _mm_loadu_si128(a2);
  *(_QWORD *)(a1 + 16) = a2[1].m128i_i64[0];
  *(__m128i *)a1 = v2;
  return sub_2973D40(a1);
}
