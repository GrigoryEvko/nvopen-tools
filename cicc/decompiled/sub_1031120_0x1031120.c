// Function: sub_1031120
// Address: 0x1031120
//
_QWORD *__fastcall sub_1031120(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  const __m128i *v10; // r15
  _QWORD *i; // rdx
  const __m128i *v12; // rbx
  unsigned __int64 v13; // rax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rdi
  int v17; // r10d
  __m128i *v18; // r9
  unsigned int v19; // ecx
  __m128i *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdx
  _QWORD *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(80LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 80 * v4;
    v10 = (const __m128i *)(v5 + 80 * v4);
    for ( i = &result[10 * v8]; i != result; result += 10 )
    {
      if ( result )
        *result = -4;
    }
    if ( v10 != (const __m128i *)v5 )
    {
      v12 = (const __m128i *)v5;
      do
      {
        v13 = v12->m128i_i64[0];
        if ( v12->m128i_i64[0] != -16 && v13 != -4 )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = v12->m128i_i64[0];
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = 1;
          v18 = 0;
          v19 = (v14 - 1) & (v13 ^ (v13 >> 9));
          v20 = (__m128i *)(v16 + 80LL * v19);
          v21 = v20->m128i_i64[0];
          if ( v13 != v20->m128i_i64[0] )
          {
            while ( v21 != -4 )
            {
              if ( v21 == -16 && !v18 )
                v18 = v20;
              v19 = v15 & (v17 + v19);
              v20 = (__m128i *)(v16 + 80LL * v19);
              v21 = v20->m128i_i64[0];
              if ( v13 == v20->m128i_i64[0] )
                goto LABEL_14;
              ++v17;
            }
            if ( v18 )
              v20 = v18;
          }
LABEL_14:
          v20->m128i_i64[0] = v12->m128i_i64[0];
          v20->m128i_i64[1] = v12->m128i_i64[1];
          v20[1].m128i_i64[0] = v12[1].m128i_i64[0];
          v20[1].m128i_i64[1] = v12[1].m128i_i64[1];
          v20[2].m128i_i64[0] = v12[2].m128i_i64[0];
          v22 = v12[2].m128i_i64[1];
          v12[2].m128i_i64[0] = 0;
          v12[1].m128i_i64[0] = 0;
          v12[1].m128i_i64[1] = 0;
          v20[2].m128i_i64[1] = v22;
          v20[3] = _mm_loadu_si128(v12 + 3);
          v20[4] = _mm_loadu_si128(v12 + 4);
          ++*(_DWORD *)(a1 + 16);
          v23 = v12[1].m128i_i64[0];
          if ( v23 )
            j_j___libc_free_0(v23, v12[2].m128i_i64[0] - v23);
        }
        v12 += 5;
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * v24]; j != result; result += 10 )
    {
      if ( result )
        *result = -4;
    }
  }
  return result;
}
