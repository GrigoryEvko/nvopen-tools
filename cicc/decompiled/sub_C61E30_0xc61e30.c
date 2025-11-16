// Function: sub_C61E30
// Address: 0xc61e30
//
_DWORD *__fastcall sub_C61E30(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // edi
  _DWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r14
  _DWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // edx
  int v16; // edx
  __int64 v17; // r9
  int v18; // r11d
  __m128i *v19; // r10
  __int64 v20; // rsi
  __m128i *v21; // rdi
  __int32 v22; // r8d
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdx
  _DWORD *j; // rdx

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
  result = (_DWORD *)sub_C7D670((unsigned __int64)v6 << 7, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4 << 7;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = &result[32 * v8]; i != result; result += 32 )
    {
      if ( result )
        *result = -1;
    }
    v12 = v5 + 48;
    if ( v10 != v5 )
    {
      while ( 1 )
      {
        v14 = *(_DWORD *)(v12 - 48);
        if ( v14 > 0xFFFFFFFD )
          goto LABEL_10;
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 1;
        v19 = 0;
        v20 = v16 & (37 * v14);
        v21 = (__m128i *)(v17 + (v20 << 7));
        v22 = v21->m128i_i32[0];
        if ( v14 != v21->m128i_i32[0] )
        {
          while ( v22 != -1 )
          {
            if ( !v19 && v22 == -2 )
              v19 = v21;
            v20 = v16 & (unsigned int)(v18 + v20);
            v21 = (__m128i *)(v17 + ((unsigned __int64)(unsigned int)v20 << 7));
            v22 = v21->m128i_i32[0];
            if ( v14 == v21->m128i_i32[0] )
              goto LABEL_15;
            ++v18;
          }
          if ( v19 )
            v21 = v19;
        }
LABEL_15:
        v21->m128i_i32[0] = v14;
        v21->m128i_i64[1] = *(_QWORD *)(v12 - 40);
        v21[1].m128i_i64[0] = *(_QWORD *)(v12 - 32);
        v21[1].m128i_i8[8] = *(_BYTE *)(v12 - 24);
        v21[2].m128i_i64[0] = (__int64)v21[3].m128i_i64;
        v23 = *(_QWORD *)(v12 - 16);
        if ( v12 == v23 )
        {
          v21[3] = _mm_loadu_si128((const __m128i *)v12);
        }
        else
        {
          v21[2].m128i_i64[0] = v23;
          v21[3].m128i_i64[0] = *(_QWORD *)v12;
        }
        v21[2].m128i_i64[1] = *(_QWORD *)(v12 - 8);
        *(_QWORD *)(v12 - 16) = v12;
        *(_QWORD *)(v12 - 8) = 0;
        *(_BYTE *)v12 = 0;
        v21[4].m128i_i64[0] = (__int64)v21[5].m128i_i64;
        v21[4].m128i_i64[1] = 0x300000000LL;
        if ( *(_DWORD *)(v12 + 24) )
        {
          v20 = v12 + 16;
          sub_C5F8E0((__int64)v21[4].m128i_i64, (char **)(v12 + 16));
        }
        ++*(_DWORD *)(a1 + 16);
        v24 = *(_QWORD *)(v12 + 16);
        if ( v24 != v12 + 32 )
          _libc_free(v24, v20);
        v25 = *(_QWORD *)(v12 - 16);
        if ( v12 == v25 )
        {
LABEL_10:
          v13 = v12 + 128;
          if ( v10 == v12 + 80 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          j_j___libc_free_0(v25, *(_QWORD *)v12 + 1LL);
          v13 = v12 + 128;
          if ( v10 == v12 + 80 )
            return (_DWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v12 = v13;
      }
    }
    return (_DWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[32 * v26]; j != result; result += 32 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
