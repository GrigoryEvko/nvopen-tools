// Function: sub_116D1D0
// Address: 0x116d1d0
//
__int64 __fastcall sub_116D1D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  const __m128i *v10; // rdi
  __int64 i; // rdx
  const __m128i *v12; // rax
  __int64 v13; // rcx
  int v14; // esi
  int v15; // esi
  __int64 v16; // r9
  __m128i *v17; // r14
  int v18; // r13d
  unsigned int j; // edx
  __m128i *v20; // r10
  unsigned int v21; // edx
  __m128i v22; // xmm1
  __int64 v23; // rdx
  __int64 k; // rdx
  __int64 v25; // r11

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
  result = sub_C7D670(24LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 24 * v4;
    v10 = (const __m128i *)(v5 + 24 * v4);
    for ( i = result + 24 * v8; i != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 12) = 0;
      }
    }
    if ( v10 != (const __m128i *)v5 )
    {
      v12 = (const __m128i *)v5;
      do
      {
        while ( 1 )
        {
          v13 = v12->m128i_i64[0];
          if ( v12->m128i_i64[0] || v12->m128i_i64[1] > 1uLL )
            break;
          v12 = (const __m128i *)((char *)v12 + 24);
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = _mm_loadu_si128(v12);
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 0;
        v18 = 1;
        for ( j = v15
                & (((unsigned int)v13 >> 9)
                 ^ ((unsigned int)v13 >> 4)
                 ^ ((unsigned int)(v12->m128i_i32[2] ^ v12->m128i_i32[3]) >> 3)); ; j = v15 & v21 )
        {
          v20 = (__m128i *)(v16 + 24LL * j);
          if ( v13 == v20->m128i_i64[0] && v12->m128i_i64[1] == v20->m128i_i64[1] )
            break;
          if ( !v20->m128i_i64[0] )
          {
            v25 = v20->m128i_i64[1];
            if ( !v25 )
            {
              if ( v17 )
                v20 = v17;
              break;
            }
            if ( !v17 && v25 == 1 )
              v17 = (__m128i *)(v16 + 24LL * j);
          }
          v21 = v18 + j;
          ++v18;
        }
        v22 = _mm_loadu_si128(v12);
        v12 = (const __m128i *)((char *)v12 + 24);
        *v20 = v22;
        v20[1].m128i_i64[0] = v12[-1].m128i_i64[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 24 * v23; k != result; result += 24 )
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
