// Function: sub_24D53B0
// Address: 0x24d53b0
//
__int64 *__fastcall sub_24D53B0(__int64 a1, __int64 a2, const __m128i *a3)
{
  bool v5; // zf
  __int64 *result; // rax
  __int64 v7; // rdx
  __int64 *i; // rdx
  const __m128i *v9; // rbx
  __int64 v10; // rdi
  int v11; // esi
  int v12; // r10d
  __int64 *v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  const __m128i *v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  int v20; // edx

  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(__int64 **)(a1 + 16);
    v7 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (__int64 *)(a1 + 16);
    v7 = 40;
  }
  for ( i = &result[v7]; i != result; result += 5 )
  {
    if ( result )
      *result = -4096;
  }
  v9 = (const __m128i *)(a2 + 24);
  if ( (const __m128i *)a2 != a3 )
  {
    while ( 1 )
    {
      v19 = v9[-2].m128i_i64[1];
      if ( v19 != -4096 && v19 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 7;
        }
        else
        {
          v20 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v20 )
          {
            MEMORY[0] = v9[-2].m128i_i64[1];
            BUG();
          }
          v11 = v20 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v15 = (__int64 *)(v10 + 40LL * v14);
        v16 = *v15;
        if ( v19 != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (__int64 *)(v10 + 40LL * v14);
            v16 = *v15;
            if ( v19 == *v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        *v15 = v19;
        v15[1] = (__int64)(v15 + 3);
        v17 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v17 == v9 )
        {
          *(__m128i *)(v15 + 3) = _mm_loadu_si128(v9);
        }
        else
        {
          v15[1] = (__int64)v17;
          v15[3] = v9->m128i_i64[0];
        }
        v15[2] = v9[-1].m128i_i64[1];
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v18 = v9[-1].m128i_u64[0];
        if ( (const __m128i *)v18 != v9 )
          j_j___libc_free_0(v18);
      }
      result = &v9[2].m128i_i64[1];
      if ( a3 == &v9[1] )
        break;
      v9 = (const __m128i *)((char *)v9 + 40);
    }
  }
  return result;
}
