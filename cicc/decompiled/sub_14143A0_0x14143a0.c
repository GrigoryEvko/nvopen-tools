// Function: sub_14143A0
// Address: 0x14143a0
//
__m128i *__fastcall sub_14143A0(__int64 a1, __m128i *a2, const __m128i *a3)
{
  __m128i *result; // rax
  __m128i *v5; // rdi
  __int8 *v6; // r13
  __m128i v7[3]; // [rsp+8h] [rbp-38h] BYREF

  result = a2;
  v5 = *(__m128i **)(a1 + 8);
  v6 = &a2->m128i_i8[-*(_QWORD *)a1];
  if ( v5 == *(__m128i **)(a1 + 16) )
  {
    sub_1414220((const __m128i **)a1, a2, a3);
    return (__m128i *)&v6[*(_QWORD *)a1];
  }
  else if ( v5 == a2 )
  {
    if ( v5 )
    {
      *v5 = _mm_loadu_si128(a3);
      v5 = *(__m128i **)(a1 + 8);
      result = (__m128i *)&v6[*(_QWORD *)a1];
    }
    *(_QWORD *)(a1 + 8) = v5 + 1;
  }
  else
  {
    v7[0] = _mm_loadu_si128(a3);
    if ( v5 )
    {
      *v5 = _mm_loadu_si128(v5 - 1);
      v5 = *(__m128i **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v5 + 1;
    if ( a2 != &v5[-1] )
      memmove(&a2[1], a2, (char *)&v5[-1] - (char *)a2);
    *a2 = _mm_loadu_si128(v7);
    return (__m128i *)&v6[*(_QWORD *)a1];
  }
  return result;
}
