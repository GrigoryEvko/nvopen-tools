// Function: sub_2F08370
// Address: 0x2f08370
//
_DWORD *__fastcall sub_2F08370(__int64 a1, int a2)
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
  __int32 v14; // esi
  int v15; // eax
  int v16; // edx
  __int64 v17; // rdi
  int v18; // r10d
  int *v19; // r9
  unsigned int v20; // ecx
  __int32 *v21; // rax
  int v22; // r8d
  const __m128i *v23; // rdx
  __int32 v24; // edx
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
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
  result = (_DWORD *)sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 48 * v4;
    v10 = (const __m128i *)(v5 + 48 * v4);
    for ( i = &result[12 * v8]; i != result; result += 12 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
    v12 = (const __m128i *)(v5 + 24);
    if ( v10 != (const __m128i *)v5 )
    {
      while ( 1 )
      {
        v14 = v12[-2].m128i_i32[2];
        if ( (unsigned int)(v14 + 0x7FFFFFFF) > 0xFFFFFFFD )
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
        v21 = (__int32 *)(v17 + 48LL * v20);
        v22 = *v21;
        if ( *v21 != v14 )
        {
          while ( v22 != 0x7FFFFFFF )
          {
            if ( !v19 && v22 == 0x80000000 )
              v19 = v21;
            v20 = v16 & (v18 + v20);
            v21 = (__int32 *)(v17 + 48LL * v20);
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
        v24 = v12[1].m128i_i32[0];
        v12[-1].m128i_i64[0] = (__int64)v12;
        v12[-1].m128i_i64[1] = 0;
        v12->m128i_i8[0] = 0;
        v21[10] = v24;
        *((_BYTE *)v21 + 44) = v12[1].m128i_i8[4];
        ++*(_DWORD *)(a1 + 16);
        v25 = v12[-1].m128i_u64[0];
        if ( v12 == (const __m128i *)v25 )
        {
LABEL_10:
          v13 = v12 + 3;
          if ( v10 == (const __m128i *)&v12[1].m128i_u64[1] )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          j_j___libc_free_0(v25);
          v13 = v12 + 3;
          if ( v10 == (const __m128i *)&v12[1].m128i_u64[1] )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 = v13;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[12 * v26]; j != result; result += 12 )
    {
      if ( result )
        *result = 0x7FFFFFFF;
    }
  }
  return result;
}
