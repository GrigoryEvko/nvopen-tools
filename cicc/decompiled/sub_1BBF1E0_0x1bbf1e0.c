// Function: sub_1BBF1E0
// Address: 0x1bbf1e0
//
__m128i *__fastcall sub_1BBF1E0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __m128i *result; // rax

  v6 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v6 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, a5, a6);
    v6 = *(unsigned int *)(a1 + 8);
  }
  result = (__m128i *)(*(_QWORD *)a1 + 24 * v6);
  *result = _mm_loadu_si128(a2);
  result[1].m128i_i64[0] = a2[1].m128i_i64[0];
  ++*(_DWORD *)(a1 + 8);
  return result;
}
