// Function: sub_D7B870
// Address: 0xd7b870
//
_QWORD *__fastcall sub_D7B870(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  const __m128i *v9; // r9
  _QWORD *i; // rdx
  const __m128i *v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // r13
  unsigned int v17; // r11d
  _QWORD *v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r8
  int v21; // r14d
  _QWORD *v22; // r15
  __int64 v23; // rdx
  _QWORD *j; // rdx
  __int64 v25; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v25 = 16LL * v4;
    v9 = (const __m128i *)(v5 + v25);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
    if ( v9 != (const __m128i *)v5 )
    {
      v11 = (const __m128i *)v5;
      do
      {
        v12 = v11->m128i_i64[0];
        v13 = v11->m128i_u64[1];
        if ( v11->m128i_i64[0] || v13 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = _mm_loadu_si128(v11);
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = v12 & v15;
          v18 = (_QWORD *)(v16 + 16LL * ((unsigned int)v12 & v15));
          v19 = v18[1];
          v20 = *v18;
          if ( v13 != v19 || v12 != v20 )
          {
            v21 = 1;
            v22 = 0;
            while ( 1 )
            {
              if ( !v20 )
              {
                if ( v19 == -1 )
                {
                  if ( v22 )
                    v18 = v22;
                  break;
                }
                if ( !v22 && v19 == -2 )
                  v22 = v18;
              }
              v17 = v15 & (v21 + v17);
              v18 = (_QWORD *)(v16 + 16LL * v17);
              v19 = v18[1];
              v20 = *v18;
              if ( v13 == v19 && v12 == v20 )
                break;
              ++v21;
            }
          }
          *(__m128i *)v18 = _mm_loadu_si128(v11);
          ++*(_DWORD *)(a1 + 16);
        }
        ++v11;
      }
      while ( v9 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v25, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v23]; j != result; result += 2 )
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
