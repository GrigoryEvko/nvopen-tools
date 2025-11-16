// Function: sub_17892C0
// Address: 0x17892c0
//
__int64 __fastcall sub_17892C0(__int64 a1, int a2)
{
  __int64 v2; // r13
  const __m128i *v4; // r12
  unsigned __int64 v5; // rax
  __int64 result; // rax
  const __m128i *v7; // rdi
  __int64 i; // rdx
  const __m128i *v9; // rax
  __int64 v10; // rcx
  int v11; // edx
  int v12; // r9d
  __int64 v13; // r8
  int v14; // r11d
  __m128i *v15; // r13
  unsigned int j; // edx
  __m128i *v17; // rsi
  unsigned int v18; // edx
  __m128i v19; // xmm1
  __int64 v20; // rdx
  __int64 k; // rdx
  __int64 v22; // r10

  v2 = *(unsigned int *)(a1 + 24);
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
  result = sub_22077B0(24LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = (const __m128i *)((char *)v4 + 24 * v2);
    for ( i = result + 24LL * *(unsigned int *)(a1 + 24); i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        while ( 1 )
        {
          v10 = v9->m128i_i64[0];
          if ( v9->m128i_i64[0] || v9->m128i_i64[1] > 1uLL )
            break;
          v9 = (const __m128i *)((char *)v9 + 24);
          if ( v7 == v9 )
            return j___libc_free_0(v4);
        }
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
        {
          MEMORY[0] = _mm_loadu_si128(v9);
          BUG();
        }
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = 1;
        v15 = 0;
        for ( j = (v11 - 1)
                & (((unsigned int)v10 >> 9)
                 ^ ((unsigned int)v10 >> 4)
                 ^ ((unsigned int)(v9->m128i_i32[2] ^ v9->m128i_i32[3]) >> 3)); ; j = v12 & v18 )
        {
          v17 = (__m128i *)(v13 + 24LL * j);
          if ( v10 == v17->m128i_i64[0] && v9->m128i_i64[1] == v17->m128i_i64[1] )
            break;
          if ( !v17->m128i_i64[0] )
          {
            v22 = v17->m128i_i64[1];
            if ( !v22 )
            {
              if ( v15 )
                v17 = v15;
              break;
            }
            if ( !v15 && v22 == 1 )
              v15 = (__m128i *)(v13 + 24LL * j);
          }
          v18 = v14 + j;
          ++v14;
        }
        v19 = _mm_loadu_si128(v9);
        v9 = (const __m128i *)((char *)v9 + 24);
        *v17 = v19;
        v17[1].m128i_i64[0] = v9[-1].m128i_i64[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != v9 );
    }
    return j___libc_free_0(v4);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v20; k != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
  }
  return result;
}
