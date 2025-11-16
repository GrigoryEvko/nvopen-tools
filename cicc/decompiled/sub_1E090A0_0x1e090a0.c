// Function: sub_1E090A0
// Address: 0x1e090a0
//
__int64 __fastcall sub_1E090A0(__int64 a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 result; // rax

  v3 = *(__m128i **)(a1 + 8);
  if ( v3 == *(__m128i **)(a1 + 16) )
    return sub_1E08EE0((const __m128i **)a1, v3, a2);
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    v3[1] = _mm_loadu_si128(a2 + 1);
    result = a2[2].m128i_i64[0];
    v3[2].m128i_i64[0] = result;
    v3 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = (char *)v3 + 40;
  return result;
}
