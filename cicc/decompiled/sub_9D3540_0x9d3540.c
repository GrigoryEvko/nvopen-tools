// Function: sub_9D3540
// Address: 0x9d3540
//
__int64 __fastcall sub_9D3540(__int64 a1, const __m128i *a2)
{
  __m128i *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax

  v3 = *(__m128i **)(a1 + 8);
  if ( v3 == *(__m128i **)(a1 + 16) )
    return sub_9D3300((const __m128i **)a1, v3, a2);
  if ( v3 )
  {
    *v3 = _mm_loadu_si128(a2);
    v4 = a2[1].m128i_i64[0];
    a2[1].m128i_i64[0] = 0;
    v3[1].m128i_i64[0] = v4;
    v5 = a2[1].m128i_i64[1];
    a2[1].m128i_i64[1] = 0;
    v3[1].m128i_i64[1] = v5;
    result = a2[2].m128i_i64[0];
    a2[2].m128i_i64[0] = 0;
    v3[2].m128i_i64[0] = result;
    v3 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = (char *)v3 + 40;
  return result;
}
