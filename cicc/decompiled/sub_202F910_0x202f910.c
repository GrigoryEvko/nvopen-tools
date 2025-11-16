// Function: sub_202F910
// Address: 0x202f910
//
__m128i *__fastcall sub_202F910(__int64 a1, unsigned __int64 a2, const __m128i *a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // rax
  __m128i *result; // rax
  __int64 v9; // rdx

  v7 = *(unsigned int *)(a1 + 12);
  *(_DWORD *)(a1 + 8) = 0;
  if ( a2 > v7 )
    sub_16CD150(a1, (const void *)(a1 + 16), a2, 16, a5, a6);
  *(_DWORD *)(a1 + 8) = a2;
  result = *(__m128i **)a1;
  v9 = *(_QWORD *)a1 + 16LL * (unsigned int)a2;
  if ( *(_QWORD *)a1 != v9 )
  {
    do
    {
      if ( result )
        *result = _mm_loadu_si128(a3);
      ++result;
    }
    while ( (__m128i *)v9 != result );
  }
  return result;
}
