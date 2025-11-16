// Function: sub_981820
// Address: 0x981820
//
_DWORD *__fastcall sub_981820(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  const __m128i *v10; // r15
  _DWORD *i; // rdx
  const __m128i *v12; // rbx
  const __m128i *v13; // rax
  unsigned int v14; // eax
  int v15; // edx
  int v16; // ecx
  __int64 v17; // r8
  int v18; // r10d
  unsigned int *v19; // r9
  unsigned int v20; // esi
  unsigned int *v21; // rdx
  unsigned int v22; // edi
  const __m128i *v23; // rax
  const __m128i *v24; // rdi
  __int64 v25; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 40 * v4;
    v10 = (const __m128i *)(v5 + 40 * v4);
    for ( i = &result[10 * v8]; i != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
    v12 = (const __m128i *)(v5 + 24);
    if ( v10 != (const __m128i *)v5 )
    {
      while ( 1 )
      {
        v14 = v12[-2].m128i_u32[2];
        if ( v14 > 0xFFFFFFFD )
          goto LABEL_10;
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 1;
        v19 = 0;
        v20 = (v15 - 1) & (37 * v14);
        v21 = (unsigned int *)(v17 + 40LL * v20);
        v22 = *v21;
        if ( v14 != *v21 )
        {
          while ( v22 != -1 )
          {
            if ( !v19 && v22 == -2 )
              v19 = v21;
            v20 = v16 & (v18 + v20);
            v21 = (unsigned int *)(v17 + 40LL * v20);
            v22 = *v21;
            if ( v14 == *v21 )
              goto LABEL_15;
            ++v18;
          }
          if ( v19 )
            v21 = v19;
        }
LABEL_15:
        *v21 = v14;
        *((_QWORD *)v21 + 1) = v21 + 6;
        v23 = (const __m128i *)v12[-1].m128i_i64[0];
        if ( v12 == v23 )
        {
          *(__m128i *)(v21 + 6) = _mm_loadu_si128(v12);
        }
        else
        {
          *((_QWORD *)v21 + 1) = v23;
          *((_QWORD *)v21 + 3) = v12->m128i_i64[0];
        }
        *((_QWORD *)v21 + 2) = v12[-1].m128i_i64[1];
        v12[-1].m128i_i64[0] = (__int64)v12;
        v12[-1].m128i_i64[1] = 0;
        v12->m128i_i8[0] = 0;
        ++*(_DWORD *)(a1 + 16);
        v24 = (const __m128i *)v12[-1].m128i_i64[0];
        if ( v12 == v24 )
        {
LABEL_10:
          v13 = (const __m128i *)((char *)v12 + 40);
          if ( v10 == &v12[1] )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          j_j___libc_free_0(v24, v12->m128i_i64[0] + 1);
          v13 = (const __m128i *)((char *)v12 + 40);
          if ( v10 == &v12[1] )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 = v13;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * v25]; j != result; result += 10 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
