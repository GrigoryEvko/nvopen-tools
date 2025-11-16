// Function: sub_1044A90
// Address: 0x1044a90
//
__int64 *__fastcall sub_1044A90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 *result; // rax
  __int64 *i; // rdx
  __int64 *v8; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = *(unsigned int *)(a1 + 24);
  result = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[7 * v5]; i != result; result += 7 )
  {
    if ( result )
    {
      *result = -4096;
      result[1] = -4096;
      result[2] = -3;
      result[3] = 0;
      result[4] = 0;
      result[5] = 0;
      result[6] = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(__int64 **)v4;
      if ( *(_QWORD *)v4 == -4096 )
      {
        if ( *(_QWORD *)(v4 + 8) == -4096 && *(_QWORD *)(v4 + 16) == -3 && !*(_QWORD *)(v4 + 24) )
          goto LABEL_14;
      }
      else if ( result == (__int64 *)-8192LL
             && *(_QWORD *)(v4 + 8) == -8192
             && *(_QWORD *)(v4 + 16) == -4
             && !*(_QWORD *)(v4 + 24) )
      {
LABEL_14:
        if ( !*(_QWORD *)(v4 + 32) && !*(_QWORD *)(v4 + 40) && !*(_QWORD *)(v4 + 48) )
          goto LABEL_9;
      }
      sub_103F4F0(a1, (__int64 *)v4, &v8);
      result = v8;
      *v8 = *(_QWORD *)v4;
      *(__m128i *)(result + 1) = _mm_loadu_si128((const __m128i *)(v4 + 8));
      *(__m128i *)(result + 3) = _mm_loadu_si128((const __m128i *)(v4 + 24));
      *(__m128i *)(result + 5) = _mm_loadu_si128((const __m128i *)(v4 + 40));
      ++*(_DWORD *)(a1 + 16);
LABEL_9:
      v4 += 56;
    }
    while ( a3 != v4 );
  }
  return result;
}
