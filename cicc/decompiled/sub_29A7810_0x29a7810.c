// Function: sub_29A7810
// Address: 0x29a7810
//
__int64 __fastcall sub_29A7810(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  const __m128i *v10; // rcx
  __int64 i; // rdx
  const __m128i *v12; // rsi
  __int64 v13; // rdx
  int v14; // eax
  int v15; // edi
  __int64 v16; // r10
  unsigned int v17; // r9d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 k; // rdx
  int j; // eax
  int v22; // r11d
  __int64 v23; // rcx
  _DWORD *v24; // rdi

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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 32LL * v4;
    v10 = (const __m128i *)(v5 + v9);
    for ( i = result + 32 * v8; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
        *(_DWORD *)(result + 24) = 0;
      }
    }
    if ( v10 != (const __m128i *)v5 )
    {
      v12 = (const __m128i *)v5;
      do
      {
        v13 = v12->m128i_i64[0];
        if ( v12->m128i_i64[0] )
        {
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            v23 = 7;
            v24 = 0;
            while ( v23 )
            {
              *v24 = v12->m128i_i32[0];
              v12 = (const __m128i *)((char *)v12 + 4);
              ++v24;
              --v23;
            }
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v18 = v16 + 32LL * v17;
          if ( *(_QWORD *)v18 && v13 != *(_QWORD *)v18 )
          {
            for ( j = 1; ; j = v22 )
            {
              v22 = j + 1;
              v17 = v15 & (j + v17);
              v18 = v16 + 32LL * v17;
              if ( !*(_QWORD *)v18 || v13 == *(_QWORD *)v18 )
                break;
            }
          }
          *(__m128i *)v18 = _mm_loadu_si128(v12);
          *(_QWORD *)(v18 + 16) = v12[1].m128i_i64[0];
          *(_DWORD *)(v18 + 24) = v12[1].m128i_i32[2];
          ++*(_DWORD *)(a1 + 16);
        }
        v12 += 2;
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32 * v19; k != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -4096;
        *(_DWORD *)(result + 24) = 0;
      }
    }
  }
  return result;
}
