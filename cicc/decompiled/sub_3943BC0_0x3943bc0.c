// Function: sub_3943BC0
// Address: 0x3943bc0
//
void __fastcall sub_3943BC0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rax
  _QWORD *v5; // rax
  const __m128i *v6; // r14
  _QWORD *i; // rdx
  const __m128i *v8; // rbx
  unsigned __int64 v9; // r12
  int v11; // r15d
  int v12; // r15d
  int v13; // eax
  const void *v14; // rdi
  size_t v15; // rdx
  __int64 v16; // r10
  int v17; // r11d
  unsigned int j; // r8d
  __int64 v19; // rcx
  const void *v20; // rsi
  unsigned int v21; // r8d
  int v22; // eax
  _QWORD *k; // rdx
  size_t v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  int v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+1Ch] [rbp-44h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  v5 = (_QWORD *)sub_22077B0(24LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = v5;
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = (const __m128i *)(v3 + 24 * v2);
    for ( i = &v5[3 * *(unsigned int *)(a1 + 24)]; i != v5; v5 += 3 )
    {
      if ( v5 )
      {
        *v5 = -1;
        v5[1] = 0;
      }
    }
    if ( v6 != (const __m128i *)v3 )
    {
      v8 = (const __m128i *)v3;
      v9 = v3;
      do
      {
        if ( v8->m128i_i64[0] != -1 && v8->m128i_i64[0] != -2 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = _mm_loadu_si128(v8);
            BUG();
          }
          v12 = v11 - 1;
          v29 = *(_QWORD *)(a1 + 8);
          v13 = sub_16D3930(v8->m128i_i64[0], v8->m128i_i64[1]);
          v14 = (const void *)v8->m128i_i64[0];
          v15 = v8->m128i_u64[1];
          v16 = 0;
          v17 = 1;
          for ( j = v12 & v13; ; j = v12 & v21 )
          {
            v19 = v29 + 24LL * j;
            v20 = *(const void **)v19;
            if ( *(_QWORD *)v19 == -1 )
              break;
            if ( v20 == (const void *)-2LL )
            {
              if ( v14 == (const void *)-2LL )
                goto LABEL_22;
              if ( !v16 )
                v16 = v29 + 24LL * j;
            }
            else if ( *(_QWORD *)(v19 + 8) == v15 )
            {
              v26 = v17;
              v25 = v16;
              v27 = j;
              if ( !v15 )
                goto LABEL_22;
              v28 = v29 + 24LL * j;
              v24 = v15;
              v22 = memcmp(v14, v20, v15);
              v19 = v28;
              if ( !v22 )
                goto LABEL_22;
              v15 = v24;
              j = v27;
              v16 = v25;
              v17 = v26;
            }
            v21 = v17 + j;
            ++v17;
          }
          if ( v14 != (const void *)-1LL && v16 )
            v19 = v16;
LABEL_22:
          *(__m128i *)v19 = _mm_loadu_si128(v8);
          *(_QWORD *)(v19 + 16) = v8[1].m128i_i64[0];
          ++*(_DWORD *)(a1 + 16);
        }
        v8 = (const __m128i *)((char *)v8 + 24);
      }
      while ( v6 != v8 );
      v3 = v9;
    }
    j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &v5[3 * *(unsigned int *)(a1 + 24)]; k != v5; v5 += 3 )
    {
      if ( v5 )
      {
        *v5 = -1;
        v5[1] = 0;
      }
    }
  }
}
