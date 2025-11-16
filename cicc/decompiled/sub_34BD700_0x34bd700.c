// Function: sub_34BD700
// Address: 0x34bd700
//
_DWORD *__fastcall sub_34BD700(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // rdx
  const __m128i *v9; // r9
  _DWORD *i; // rdx
  const __m128i *v11; // rdx
  __int32 v12; // ecx
  int v13; // r8d
  __int32 v14; // esi
  __int64 v15; // rdi
  int v16; // r14d
  int v17; // r10d
  int *v18; // r11
  unsigned int j; // eax
  int *v20; // r8
  int v21; // r15d
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  _DWORD *k; // rdx

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
  result = (_DWORD *)sub_C7D670(24LL * v6, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (const __m128i *)(v5 + 24 * v4);
    for ( i = &result[6 * v8]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
    if ( v9 != (const __m128i *)v5 )
    {
      v11 = (const __m128i *)v5;
      while ( 1 )
      {
        v12 = v11->m128i_i32[0];
        if ( v11->m128i_i32[0] == -1 )
        {
          if ( v11->m128i_i32[1] != -1 )
            goto LABEL_12;
          v11 = (const __m128i *)((char *)v11 + 24);
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, 24 * v4, 4);
        }
        else if ( v12 == -2 && v11->m128i_i32[1] == -2 )
        {
          v11 = (const __m128i *)((char *)v11 + 24);
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, 24 * v4, 4);
        }
        else
        {
LABEL_12:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = v11->m128i_i64[0];
            BUG();
          }
          v14 = v11->m128i_i32[1];
          v16 = 1;
          v17 = v13 - 1;
          v18 = 0;
          for ( j = (v13 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v14) | ((unsigned __int64)(unsigned int)(37 * v12) << 32))) >> 31)
                   ^ (756364221 * v14)); ; j = v17 & v23 )
          {
            v15 = *(_QWORD *)(a1 + 8);
            v20 = (int *)(v15 + 24LL * j);
            v21 = *v20;
            if ( v12 == *v20 && v14 == v20[1] )
              break;
            if ( v21 == -1 )
            {
              if ( v20[1] == -1 )
              {
                if ( v18 )
                  v20 = v18;
                break;
              }
            }
            else if ( v21 == -2 && v20[1] == -2 && !v18 )
            {
              v18 = (int *)(v15 + 24LL * j);
            }
            v23 = v16 + j;
            ++v16;
          }
          v22 = v11->m128i_i64[0];
          v11 = (const __m128i *)((char *)v11 + 24);
          *(_QWORD *)v20 = v22;
          *(__m128i *)(v20 + 2) = _mm_loadu_si128(v11 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return (_DWORD *)sub_C7D6A0(v5, 24 * v4, 4);
        }
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, 24 * v4, 4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v24]; k != result; result += 6 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = -1;
      }
    }
  }
  return result;
}
