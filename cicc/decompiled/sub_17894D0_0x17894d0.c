// Function: sub_17894D0
// Address: 0x17894d0
//
_QWORD *__fastcall sub_17894D0(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // esi
  _QWORD *v5; // r10
  __int64 v6; // rdi
  int v7; // r9d
  unsigned int i; // eax
  _QWORD *v9; // r8
  _QWORD *v10; // r13
  unsigned int v11; // eax
  __int64 v13; // r11
  int v14; // eax
  int v15; // edx
  _QWORD *v16; // r9
  __int64 v17; // rsi
  int v18; // r10d
  unsigned int k; // eax
  unsigned int v20; // eax
  int v21; // edx
  __m128i v22; // xmm0
  int v23; // eax
  int v24; // eax
  int v25; // esi
  __int64 v26; // rcx
  int v27; // r9d
  unsigned int j; // eax
  unsigned int v29; // eax
  __int64 v30; // rdi
  __int64 v31; // rdi

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = 0;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  for ( i = (v4 - 1)
          & (((unsigned int)a2->m128i_i64[0] >> 9)
           ^ ((unsigned int)a2->m128i_i64[0] >> 4)
           ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; i = (v4 - 1) & v11 )
  {
    v9 = (_QWORD *)(v6 + 24LL * i);
    v10 = (_QWORD *)*v9;
    if ( a2->m128i_i64[0] == *v9 && a2->m128i_i64[1] == v9[1] )
      return (_QWORD *)(v6 + 24LL * i);
    if ( !v10 )
      break;
LABEL_5:
    v11 = v7 + i;
    ++v7;
  }
  v13 = v9[1];
  if ( v13 )
  {
    if ( v13 == 1 && !v5 )
      v5 = (_QWORD *)(v6 + 24LL * i);
    goto LABEL_5;
  }
  v23 = *(_DWORD *)(a1 + 16);
  if ( v5 )
    v9 = v5;
  ++*(_QWORD *)a1;
  v21 = v23 + 1;
  if ( 4 * (v23 + 1) < 3 * v4 )
  {
    if ( v4 - *(_DWORD *)(a1 + 20) - v21 <= v4 >> 3 )
    {
      sub_17892C0(a1, v4);
      v24 = *(_DWORD *)(a1 + 24);
      if ( v24 )
      {
        v25 = v24 - 1;
        v27 = 1;
        for ( j = (v24 - 1)
                & (((unsigned int)a2->m128i_i64[0] >> 9)
                 ^ ((unsigned int)a2->m128i_i64[0] >> 4)
                 ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; j = v25 & v29 )
        {
          v26 = *(_QWORD *)(a1 + 8);
          v9 = (_QWORD *)(v26 + 24LL * j);
          if ( a2->m128i_i64[0] == *v9 && a2->m128i_i64[1] == v9[1] )
            break;
          if ( !*v9 )
          {
            v31 = v9[1];
            if ( !v31 )
            {
              if ( v10 )
                v9 = v10;
              v21 = *(_DWORD *)(a1 + 16) + 1;
              goto LABEL_21;
            }
            if ( v31 == 1 && !v10 )
              v10 = (_QWORD *)(v26 + 24LL * j);
          }
          v29 = v27 + j;
          ++v27;
        }
        goto LABEL_20;
      }
LABEL_53:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    goto LABEL_21;
  }
LABEL_14:
  sub_17892C0(a1, 2 * v4);
  v14 = *(_DWORD *)(a1 + 24);
  if ( !v14 )
    goto LABEL_53;
  v15 = v14 - 1;
  v16 = 0;
  v18 = 1;
  for ( k = (v14 - 1)
          & (((unsigned int)a2->m128i_i64[0] >> 9)
           ^ ((unsigned int)a2->m128i_i64[0] >> 4)
           ^ ((unsigned int)(a2->m128i_i32[2] ^ a2->m128i_i32[3]) >> 3)); ; k = v15 & v20 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    v9 = (_QWORD *)(v17 + 24LL * k);
    if ( a2->m128i_i64[0] == *v9 && a2->m128i_i64[1] == v9[1] )
      break;
    if ( !*v9 )
    {
      v30 = v9[1];
      if ( !v30 )
      {
        if ( v16 )
          v9 = v16;
        v21 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_21;
      }
      if ( v30 == 1 && !v16 )
        v16 = (_QWORD *)(v17 + 24LL * k);
    }
    v20 = v18 + k;
    ++v18;
  }
LABEL_20:
  v21 = *(_DWORD *)(a1 + 16) + 1;
LABEL_21:
  *(_DWORD *)(a1 + 16) = v21;
  if ( *v9 || v9[1] )
    --*(_DWORD *)(a1 + 20);
  v22 = _mm_loadu_si128(a2);
  v9[2] = 0;
  *(__m128i *)v9 = v22;
  return v9;
}
