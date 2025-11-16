// Function: sub_277FD80
// Address: 0x277fd80
//
_QWORD *__fastcall sub_277FD80(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 v8; // r15
  __m128i *v9; // r14
  _QWORD *i; // rdx
  __m128i *v11; // rbx
  __int64 **v12; // rcx
  __m128i *v13; // rax
  _QWORD *j; // rdx
  __int64 **v15; // [rsp+8h] [rbp-48h]
  __m128i *v16; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(56LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 56 * v3;
    v9 = (__m128i *)(v4 + 56 * v3);
    for ( i = &result[7 * v7]; i != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -3;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
        result[5] = 0;
      }
    }
    if ( v9 != (__m128i *)v4 )
    {
      v11 = (__m128i *)v4;
      v12 = (__int64 **)&v16;
      do
      {
        if ( v11->m128i_i64[0] == -4096 )
        {
          if ( v11->m128i_i64[1] == -3 )
            goto LABEL_16;
        }
        else if ( v11->m128i_i64[0] == -8192 && v11->m128i_i64[1] == -4 )
        {
LABEL_16:
          if ( !v11[1].m128i_i64[0] && !v11[1].m128i_i64[1] && !v11[2].m128i_i64[0] && !v11[2].m128i_i64[1] )
            goto LABEL_13;
        }
        v15 = v12;
        sub_277DBB0(a1, v11->m128i_i64, v12);
        v13 = v16;
        v12 = v15;
        *v16 = _mm_loadu_si128(v11);
        v13[1] = _mm_loadu_si128(v11 + 1);
        v13[2] = _mm_loadu_si128(v11 + 2);
        v16[3].m128i_i64[0] = v11[3].m128i_i64[0];
        ++*(_DWORD *)(a1 + 16);
LABEL_13:
        v11 = (__m128i *)((char *)v11 + 56);
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -3;
        result[2] = 0;
        result[3] = 0;
        result[4] = 0;
        result[5] = 0;
      }
    }
  }
  return result;
}
