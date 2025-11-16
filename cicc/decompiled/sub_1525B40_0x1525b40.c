// Function: sub_1525B40
// Address: 0x1525b40
//
__m128i *__fastcall sub_1525B40(__int64 a1, const __m128i *a2)
{
  __int64 v2; // rax
  __m128i *result; // rax

  v2 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v2 )
  {
    sub_16CD150(a1, a1 + 16, 0, 16);
    v2 = *(unsigned int *)(a1 + 8);
  }
  result = (__m128i *)(*(_QWORD *)a1 + 16 * v2);
  *result = _mm_loadu_si128(a2);
  ++*(_DWORD *)(a1 + 8);
  return result;
}
