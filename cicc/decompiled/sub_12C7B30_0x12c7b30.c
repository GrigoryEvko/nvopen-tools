// Function: sub_12C7B30
// Address: 0x12c7b30
//
__m128i *__fastcall sub_12C7B30(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __m128i *result; // rax
  __int64 v8; // rcx

  v6 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v6 )
  {
    sub_12BE710(a1, 0, v6, a4, a5, a6);
    LODWORD(v6) = *(_DWORD *)(a1 + 8);
  }
  result = (__m128i *)(*(_QWORD *)a1 + 32LL * (unsigned int)v6);
  if ( result )
  {
    result->m128i_i64[0] = (__int64)result[1].m128i_i64;
    if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
    {
      result[1] = _mm_loadu_si128(a2 + 1);
    }
    else
    {
      result->m128i_i64[0] = a2->m128i_i64[0];
      result[1].m128i_i64[0] = a2[1].m128i_i64[0];
    }
    v8 = a2->m128i_i64[1];
    a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
    a2->m128i_i64[1] = 0;
    result->m128i_i64[1] = v8;
    a2[1].m128i_i8[0] = 0;
    LODWORD(v6) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v6 + 1;
  return result;
}
