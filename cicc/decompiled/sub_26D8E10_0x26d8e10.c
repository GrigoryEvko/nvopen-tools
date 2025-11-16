// Function: sub_26D8E10
// Address: 0x26d8e10
//
__int64 __fastcall sub_26D8E10(__int64 a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 result; // rax

  v3 = *(__m128i **)(a1 + 8);
  if ( v3 == *(__m128i **)(a1 + 16) )
    return sub_104DF10((const __m128i **)a1, v3, a2);
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    v3[1] = _mm_loadu_si128(a2 + 1);
    v3 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v3 + 2;
  return result;
}
