// Function: sub_28C4BE0
// Address: 0x28c4be0
//
__int64 __fastcall sub_28C4BE0(__int64 a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 result; // rax

  v3 = *(__m128i **)(a1 + 8);
  if ( v3 == *(__m128i **)(a1 + 16) )
    return sub_103FC80((const __m128i **)a1, v3, a2);
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    result = a2[1].m128i_i64[0];
    v3[1].m128i_i64[0] = result;
    v3 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = (char *)v3 + 24;
  return result;
}
