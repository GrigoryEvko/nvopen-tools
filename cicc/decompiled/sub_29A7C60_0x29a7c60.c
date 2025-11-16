// Function: sub_29A7C60
// Address: 0x29a7c60
//
void __fastcall sub_29A7C60(__int64 a1)
{
  const __m128i *v1; // rbx
  const __m128i *v2; // r13
  __int64 v4; // r8
  int v5; // r10d
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // esi
  int v10; // eax
  __int64 v11; // rdi
  int v12; // esi
  __int64 v13; // r8
  int v14; // edx
  unsigned int v15; // ecx
  int j; // eax
  int v17; // r9d
  int v18; // edi
  __m128i v19; // xmm0
  int v20; // eax
  __int64 v21; // rdi
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // ecx
  int i; // eax
  int v26; // r9d

  v1 = *(const __m128i **)(a1 + 32);
  v2 = &v1[2 * *(unsigned int *)(a1 + 40)];
  if ( v2 != v1 )
  {
    while ( 1 )
    {
      v9 = *(_DWORD *)(a1 + 24);
      if ( !v9 )
        break;
      v4 = *(_QWORD *)(a1 + 8);
      v5 = 1;
      v6 = (v9 - 1) & (((unsigned int)v1->m128i_i64[0] >> 9) ^ ((unsigned int)v1->m128i_i64[0] >> 4));
      v7 = (__int64 *)(v4 + 32LL * v6);
      v8 = *v7;
      if ( v1->m128i_i64[0] == *v7 )
      {
LABEL_4:
        v1 += 2;
        if ( v2 == v1 )
          return;
      }
      else
      {
        while ( v8 )
        {
          v6 = (v9 - 1) & (v5 + v6);
          v7 = (__int64 *)(v4 + 32LL * v6);
          v8 = *v7;
          if ( v1->m128i_i64[0] == *v7 )
            goto LABEL_4;
          ++v5;
        }
        v18 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v14 = v18 + 1;
        if ( 4 * (v18 + 1) < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(a1 + 20) - v14 <= v9 >> 3 )
          {
            sub_29A7810(a1, v9);
            v20 = *(_DWORD *)(a1 + 24);
            if ( !v20 )
            {
LABEL_30:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v21 = v1->m128i_i64[0];
            v22 = v20 - 1;
            v23 = *(_QWORD *)(a1 + 8);
            v14 = *(_DWORD *)(a1 + 16) + 1;
            v24 = (v20 - 1) & (((unsigned int)v1->m128i_i64[0] >> 9) ^ ((unsigned int)v1->m128i_i64[0] >> 4));
            v7 = (__int64 *)(v23 + 32LL * v24);
            if ( *v7 && v21 != *v7 )
            {
              for ( i = 1; ; i = v26 )
              {
                v26 = i + 1;
                v24 = v22 & (i + v24);
                v7 = (__int64 *)(v23 + 32LL * v24);
                if ( !*v7 || v21 == *v7 )
                  break;
              }
            }
          }
          goto LABEL_19;
        }
LABEL_7:
        sub_29A7810(a1, 2 * v9);
        v10 = *(_DWORD *)(a1 + 24);
        if ( !v10 )
          goto LABEL_30;
        v11 = v1->m128i_i64[0];
        v12 = v10 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = *(_DWORD *)(a1 + 16) + 1;
        v15 = (v10 - 1) & (((unsigned int)v1->m128i_i64[0] >> 9) ^ ((unsigned int)v1->m128i_i64[0] >> 4));
        v7 = (__int64 *)(v13 + 32LL * v15);
        if ( *v7 && *v7 != v11 )
        {
          for ( j = 1; ; j = v17 )
          {
            v17 = j + 1;
            v15 = v12 & (j + v15);
            v7 = (__int64 *)(v13 + 32LL * v15);
            if ( !*v7 || v11 == *v7 )
              break;
          }
        }
LABEL_19:
        *(_DWORD *)(a1 + 16) = v14;
        if ( *v7 )
          --*(_DWORD *)(a1 + 20);
        v19 = _mm_loadu_si128(v1);
        v1 += 2;
        *(__m128i *)v7 = v19;
        v7[2] = v1[-1].m128i_i64[0];
        *((_DWORD *)v7 + 6) = v1[-1].m128i_i32[2];
        if ( v2 == v1 )
          return;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_7;
  }
}
