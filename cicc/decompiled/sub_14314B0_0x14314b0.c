// Function: sub_14314B0
// Address: 0x14314b0
//
_QWORD *__fastcall sub_14314B0(__int64 a1, int a2)
{
  __int64 v3; // r12
  const __m128i *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // rdi
  _QWORD *i; // rdx
  const __m128i *v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // r8
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // r12
  unsigned int v16; // r11d
  _QWORD *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // r9
  int v20; // r14d
  _QWORD *v21; // r15
  __int64 v22; // rdx
  _QWORD *j; // rdx

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
  result = (_QWORD *)sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = v10->m128i_i64[0];
        v12 = v10->m128i_u64[1];
        if ( v10->m128i_i64[0] || v12 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = _mm_loadu_si128(v10);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = v11 & v14;
          v17 = (_QWORD *)(v15 + 16LL * ((unsigned int)v11 & v14));
          v18 = v17[1];
          v19 = *v17;
          if ( v12 != v18 || v11 != v19 )
          {
            v20 = 1;
            v21 = 0;
            while ( 1 )
            {
              if ( !v19 )
              {
                if ( v18 == -1 )
                {
                  if ( v21 )
                    v17 = v21;
                  break;
                }
                if ( !v21 && v18 == -2 )
                  v21 = v17;
              }
              v16 = v14 & (v20 + v16);
              v17 = (_QWORD *)(v15 + 16LL * v16);
              v18 = v17[1];
              v19 = *v17;
              if ( v12 == v18 && v11 == v19 )
                break;
              ++v20;
            }
          }
          *(__m128i *)v17 = _mm_loadu_si128(v10);
          ++*(_DWORD *)(a1 + 16);
        }
        ++v10;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v22]; j != result; result += 2 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
  }
  return result;
}
