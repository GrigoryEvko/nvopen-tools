// Function: sub_26F8AB0
// Address: 0x26f8ab0
//
_QWORD *__fastcall sub_26F8AB0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  const __m128i *v9; // r9
  _QWORD *i; // rdx
  const __m128i *v11; // rax
  __int64 v12; // rdx
  int v13; // esi
  int v14; // esi
  __int64 v15; // r10
  __int64 *v16; // r14
  int v17; // r11d
  unsigned int j; // ecx
  __int64 *v19; // rdi
  __int64 v20; // r15
  __m128i v21; // xmm1
  __int64 v22; // rdx
  _QWORD *k; // rdx
  unsigned int v24; // ecx

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
  result = (_QWORD *)sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (const __m128i *)(v5 + 24 * v4);
    for ( i = &result[3 * v8]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
      }
    }
    if ( v9 != (const __m128i *)v5 )
    {
      v11 = (const __m128i *)v5;
      while ( 1 )
      {
        v12 = v11->m128i_i64[0];
        if ( v11->m128i_i64[0] == -4096 )
        {
          if ( v11->m128i_i64[1] != -1 )
            goto LABEL_12;
          v11 = (const __m128i *)((char *)v11 + 24);
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
        }
        else if ( v12 == -8192 && v11->m128i_i64[1] == -2 )
        {
          v11 = (const __m128i *)((char *)v11 + 24);
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
        }
        else
        {
LABEL_12:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = _mm_loadu_si128(v11);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 0;
          v17 = 1;
          for ( j = v14
                  & (((0xBF58476D1CE4E5B9LL * v11->m128i_i64[1]) >> 31)
                   ^ (484763065 * v11->m128i_i32[2])
                   ^ ((unsigned int)v12 >> 9)
                   ^ ((unsigned int)v12 >> 4)); ; j = v14 & v24 )
          {
            v19 = (__int64 *)(v15 + 24LL * j);
            v20 = *v19;
            if ( v12 == *v19 && v11->m128i_i64[1] == v19[1] )
              break;
            if ( v20 == -4096 )
            {
              if ( v19[1] == -1 )
              {
                if ( v16 )
                  v19 = v16;
                break;
              }
            }
            else if ( v20 == -8192 && v19[1] == -2 && !v16 )
            {
              v16 = (__int64 *)(v15 + 24LL * j);
            }
            v24 = v17 + j;
            ++v17;
          }
          v21 = _mm_loadu_si128(v11);
          v11 = (const __m128i *)((char *)v11 + 24);
          *(__m128i *)v19 = v21;
          *((_DWORD *)v19 + 4) = v11[-1].m128i_i32[2];
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, 24 * v4, 8);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * v22]; k != result; result += 3 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -1;
      }
    }
  }
  return result;
}
