// Function: sub_18B5B40
// Address: 0x18b5b40
//
_QWORD *__fastcall sub_18B5B40(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r13
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  const __m128i *v7; // r9
  _QWORD *i; // rdx
  const __m128i *v9; // rax
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v12; // r8
  int v13; // edi
  __int64 v14; // r10
  int v15; // ebx
  __int64 *v16; // r14
  unsigned int j; // ecx
  __int64 *v18; // rsi
  __int64 v19; // r11
  __m128i v20; // xmm1
  _QWORD *k; // rdx
  unsigned int v22; // ecx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(const __m128i **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(24LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)((char *)v4 + 24 * v3);
    for ( i = &result[3 * *(unsigned int *)(a1 + 24)]; i != result; result += 3 )
    {
      if ( result )
      {
        *result = -4;
        result[1] = -1;
      }
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      while ( 1 )
      {
        v10 = v9->m128i_i64[0];
        if ( v9->m128i_i64[0] == -4 )
        {
          if ( v9->m128i_i64[1] != -1 )
            goto LABEL_12;
          v9 = (const __m128i *)((char *)v9 + 24);
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else if ( v10 == -8 && v9->m128i_i64[1] == -2 )
        {
          v9 = (const __m128i *)((char *)v9 + 24);
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = _mm_loadu_si128(v9);
            BUG();
          }
          v12 = v9->m128i_i64[1];
          v13 = v11 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          for ( j = (v11 - 1) & ((37 * v12) ^ ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)); ; j = v13 & v22 )
          {
            v18 = (__int64 *)(v14 + 24LL * j);
            v19 = *v18;
            if ( v10 == *v18 && v12 == v18[1] )
              break;
            if ( v19 == -4 )
            {
              if ( v18[1] == -1 )
              {
                if ( v16 )
                  v18 = v16;
                break;
              }
            }
            else if ( v19 == -8 && v18[1] == -2 && !v16 )
            {
              v16 = (__int64 *)(v14 + 24LL * j);
            }
            v22 = v15 + j;
            ++v15;
          }
          v20 = _mm_loadu_si128(v9);
          v9 = (const __m128i *)((char *)v9 + 24);
          *(__m128i *)v18 = v20;
          *((_DWORD *)v18 + 4) = v9[-1].m128i_i32[2];
          ++*(_DWORD *)(a1 + 16);
          if ( v7 == v9 )
            return (_QWORD *)j___libc_free_0(v4);
        }
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[3 * *(unsigned int *)(a1 + 24)]; k != result; result += 3 )
    {
      if ( result )
      {
        *result = -4;
        result[1] = -1;
      }
    }
  }
  return result;
}
