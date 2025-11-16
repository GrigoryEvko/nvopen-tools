// Function: sub_2D2DD70
// Address: 0x2d2dd70
//
_DWORD *__fastcall sub_2D2DD70(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // r15
  const __m128i *v8; // r13
  _DWORD *i; // rdx
  const __m128i *v10; // rbx
  __int32 v11; // eax
  int v12; // edx
  int v13; // ecx
  __int64 v14; // r8
  unsigned __int64 v15; // r9
  int v16; // r10d
  unsigned int v17; // esi
  __m128i *v18; // rdx
  __int32 v19; // edi
  __int64 v20; // rax
  unsigned __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 j; // rax
  _DWORD *k; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(216LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 216 * v3;
    v8 = (const __m128i *)(v4 + 216 * v3);
    for ( i = &result[54 * *(unsigned int *)(a1 + 24)]; i != result; result += 54 )
    {
      if ( result )
        *result = -1;
    }
    if ( v8 != (const __m128i *)v4 )
    {
      v10 = (const __m128i *)v4;
      do
      {
        while ( 1 )
        {
          v11 = v10->m128i_i32[0];
          if ( v10->m128i_i32[0] <= 0xFFFFFFFD )
          {
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = 0;
              BUG();
            }
            v13 = v12 - 1;
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 0;
            v16 = 1;
            v17 = (v12 - 1) & (37 * v11);
            v18 = (__m128i *)(v14 + 216LL * v17);
            v19 = v18->m128i_i32[0];
            if ( v11 != v18->m128i_i32[0] )
            {
              while ( v19 != -1 )
              {
                if ( !v15 && v19 == -2 )
                  v15 = (unsigned __int64)v18;
                v17 = v13 & (v16 + v17);
                v18 = (__m128i *)(v14 + 216LL * v17);
                v19 = v18->m128i_i32[0];
                if ( v11 == v18->m128i_i32[0] )
                  goto LABEL_14;
                ++v16;
              }
              if ( v15 )
                v18 = (__m128i *)v15;
            }
LABEL_14:
            v18->m128i_i32[0] = v11;
            v20 = v10[13].m128i_i64[0];
            v21 = (unsigned __int64)&v10->m128i_u64[1];
            v18[12].m128i_i64[1] = 0;
            v18[13].m128i_i64[0] = v20;
            v18[12].m128i_i32[2] = v10[12].m128i_i32[2];
            v18[12].m128i_i32[3] = v10[12].m128i_i32[3];
            v18[13].m128i_i64[0] = v10[13].m128i_i64[0];
            if ( v10[12].m128i_i32[2] )
            {
              v18[1] = _mm_loadu_si128(v10 + 1);
              v18[2] = _mm_loadu_si128(v10 + 2);
              v18[3] = _mm_loadu_si128(v10 + 3);
              v18[4] = _mm_loadu_si128(v10 + 4);
              v18[5] = _mm_loadu_si128(v10 + 5);
              v18[6] = _mm_loadu_si128(v10 + 6);
              v18[7] = _mm_loadu_si128(v10 + 7);
              v18[8] = _mm_loadu_si128(v10 + 8);
              v18[9] = _mm_loadu_si128(v10 + 9);
              v18[10] = _mm_loadu_si128(v10 + 10);
              v18[11] = _mm_loadu_si128(v10 + 11);
              v18[12].m128i_i32[0] = v10[12].m128i_i32[0];
              v10[12].m128i_i32[2] = 0;
              v10->m128i_i64[1] = 0;
              v10[12].m128i_i64[0] = 0;
              memset(
                (void *)((unsigned __int64)&v10[1] & 0xFFFFFFFFFFFFFFF8LL),
                0,
                8LL * (((unsigned int)v21 - (((_DWORD)v10 + 16) & 0xFFFFFFF8) + 192) >> 3));
              v22 = 0;
            }
            else
            {
              for ( j = 2; j != 34; j += 2 )
              {
                v18->m128i_i32[j] = v10->m128i_i32[j];
                v18->m128i_i32[j + 1] = v10->m128i_i32[j + 1];
              }
              do
              {
                v22 = v10->m128i_u32[j];
                v18->m128i_i32[j++] = v22;
              }
              while ( j != 50 );
            }
            ++*(_DWORD *)(a1 + 16);
            if ( v10[12].m128i_i32[2] )
              break;
          }
          v10 = (const __m128i *)((char *)v10 + 216);
          if ( v8 == v10 )
            return (_DWORD *)sub_C7D6A0(v4, v7, 8);
        }
        v23 = (__int64)&v10->m128i_i64[1];
        v10 = (const __m128i *)((char *)v10 + 216);
        sub_2D2A3E0(v23, (char *)sub_2D227B0, 0, v22, v21, v15);
      }
      while ( v8 != v10 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v7, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[54 * *(unsigned int *)(a1 + 24)]; k != result; result += 54 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
