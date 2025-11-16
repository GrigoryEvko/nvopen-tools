// Function: sub_89F970
// Address: 0x89f970
//
__m128i *__fastcall sub_89F970(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __m128i *result; // rax
  __int64 v5; // r15
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  __int64 v12; // rcx
  __int64 i; // rax
  __int64 v14; // [rsp+8h] [rbp-58h]

  v2 = 0;
  do
  {
    result = (__m128i *)(*(_DWORD *)(a1 + 176) & 0x19000);
    if ( (_DWORD)result != 4096 || (result = *(__m128i **)(a1 + 168), !result[10].m128i_i64[1]) )
    {
      if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
        break;
      goto LABEL_3;
    }
    v5 = sub_892330(a1);
    if ( !v2 )
    {
      for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v2 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 104LL) + 96LL) + 80LL)
                       + 32LL);
    }
    v6 = *v2;
    v7 = *(_QWORD *)(a2 + 16);
    if ( v7 == *(_QWORD *)(a2 + 8) )
    {
      v14 = *v2;
      sub_738390((const __m128i **)a2);
      v6 = v14;
    }
    result = (__m128i *)(*(_QWORD *)a2 + 24 * v7);
    if ( result )
    {
      result[1].m128i_i8[0] &= 0xF0u;
      result->m128i_i64[0] = v6;
      result->m128i_i64[1] = v5;
    }
    *(_QWORD *)(a2 + 16) = v7 + 1;
    v2 = (__int64 *)v2[3];
    if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
      break;
LABEL_3:
    result = *(__m128i **)(a1 + 40);
    a1 = result[2].m128i_i64[0];
  }
  while ( a1 );
  v8 = *(_QWORD *)(a2 + 16);
  if ( v8 > 1 )
  {
    result = *(__m128i **)a2;
    v9 = *(_QWORD *)a2 + 24 * v8 - 24;
    if ( *(_QWORD *)a2 < v9 )
    {
      do
      {
        v10 = _mm_loadu_si128((const __m128i *)v9);
        v11 = _mm_loadu_si128(result);
        v9 -= 24LL;
        result = (__m128i *)((char *)result + 24);
        v12 = result[-1].m128i_i64[1];
        *(__m128i *)((char *)result - 24) = v10;
        result[-1].m128i_i64[1] = *(_QWORD *)(v9 + 40);
        *(__m128i *)(v9 + 24) = v11;
        *(_QWORD *)(v9 + 40) = v12;
      }
      while ( (unsigned __int64)result < v9 );
    }
  }
  return result;
}
