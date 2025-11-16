// Function: sub_622D80
// Address: 0x622d80
//
__int64 __fastcall sub_622D80(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rcx
  __m128i v5; // xmm2
  __m128i v6; // xmm1
  __int64 v7; // [rsp-8h] [rbp-8h] BYREF

  v2 = *a1;
  result = *a2;
  if ( *a1 )
  {
    *a1 = result;
    v4 = a1[3];
    *(__m128i *)(&v7 - 3) = _mm_loadu_si128((const __m128i *)(a1 + 1));
    *(&v7 - 1) = v4;
    if ( result )
    {
      v5 = _mm_loadu_si128((const __m128i *)(a2 + 1));
      a1[3] = a2[3];
      *(__m128i *)(a1 + 1) = v5;
    }
    result = *(&v7 - 1);
    v6 = _mm_loadu_si128((const __m128i *)(&v7 - 3));
    *a2 = v2;
    a2[3] = result;
    *(__m128i *)(a2 + 1) = v6;
  }
  else
  {
    *a1 = result;
    if ( result )
    {
      result = a2[3];
      *(__m128i *)(a1 + 1) = _mm_loadu_si128((const __m128i *)(a2 + 1));
      a1[3] = result;
      *a2 = 0;
    }
  }
  return result;
}
