// Function: sub_36FD760
// Address: 0x36fd760
//
void __fastcall sub_36FD760(__int64 a1)
{
  __m128i *v2; // r13
  __m128i *v3; // r9
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  const __m128i *v6; // r13
  __m128i *v7; // r14
  size_t v8; // rbx
  const void *v9; // rcx
  int v10; // eax
  size_t v11; // r8
  const void *v12; // rdi
  __m128i *v13; // r15
  __m128i *v14; // rdx
  const __m128i *v15; // rbx
  __m128i v16; // xmm0
  __m128i *v17; // r15
  __m128i *v18; // rbx
  const void *v19; // r14
  size_t v20; // r15
  int v21; // eax
  __int64 v22; // rdx
  unsigned int v23; // r8d
  _QWORD *v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rsi
  __m128i *v27; // rbx
  __m128i *v28; // rdi
  __int64 v29; // rax
  unsigned int v30; // r8d
  _QWORD *v31; // r9
  _QWORD *v32; // rcx
  __int64 *v33; // rax
  __int64 *v34; // rax
  __m128i *v35; // [rsp+0h] [rbp-50h]
  size_t n; // [rsp+8h] [rbp-48h]
  size_t na; // [rsp+8h] [rbp-48h]
  const void *v38; // [rsp+10h] [rbp-40h]
  _QWORD *v39; // [rsp+10h] [rbp-40h]
  __m128i *src; // [rsp+18h] [rbp-38h]
  unsigned int srca; // [rsp+18h] [rbp-38h]

  v2 = *(__m128i **)(a1 + 32);
  v3 = *(__m128i **)(a1 + 24);
  if ( v2 == v3 )
    goto LABEL_45;
  v4 = (char *)v2 - (char *)v3;
  src = *(__m128i **)(a1 + 24);
  _BitScanReverse64(&v5, v2 - v3);
  sub_36FD510(src, v2, 2LL * (int)(63 - (v5 ^ 0x3F)));
  if ( v4 > 256 )
  {
    v27 = src + 16;
    sub_A3B670(src, src + 16);
    if ( v2 != &src[16] )
    {
      do
      {
        v28 = v27++;
        sub_A3B600(v28);
      }
      while ( v2 != v27 );
    }
  }
  else
  {
    sub_A3B670(src, v2);
  }
  v6 = *(const __m128i **)(a1 + 32);
  v3 = *(__m128i **)(a1 + 24);
  if ( v6 == v3 )
  {
LABEL_45:
    v6 = v3;
    goto LABEL_25;
  }
  v7 = v3 + 1;
  if ( v6 == &v3[1] )
    goto LABEL_25;
  v8 = v3->m128i_u64[1];
  v9 = (const void *)v3->m128i_i64[0];
  while ( 1 )
  {
    v11 = v8;
    v8 = v7->m128i_u64[1];
    v12 = v9;
    v13 = v7 - 1;
    v9 = (const void *)v7->m128i_i64[0];
    if ( v8 == v11 )
    {
      if ( !v8 )
        break;
      v35 = v3;
      n = v11;
      v38 = (const void *)v7->m128i_i64[0];
      v10 = memcmp(v12, (const void *)v7->m128i_i64[0], v7->m128i_u64[1]);
      v9 = v38;
      v11 = n;
      v3 = v35;
      if ( !v10 )
        break;
    }
    if ( v6 == ++v7 )
      goto LABEL_25;
  }
  if ( v6 != v13 )
  {
    v14 = v13 + 2;
    if ( v6 != &v13[2] )
    {
      v15 = v13 + 2;
      while ( 1 )
      {
        if ( v15->m128i_i64[1] == v11 && (!v11 || !memcmp(v12, (const void *)v15->m128i_i64[0], v11)) )
        {
          if ( v6 == ++v15 )
            goto LABEL_20;
        }
        else
        {
          v16 = _mm_loadu_si128(v15++);
          *++v13 = v16;
          if ( v6 == v15 )
          {
LABEL_20:
            v17 = v13 + 1;
            v14 = *(__m128i **)(a1 + 32);
            if ( v6 != v17 )
            {
              if ( v6 != v14 )
              {
                memmove(v17, v6, (char *)v14 - (char *)v6);
                v14 = *(__m128i **)(a1 + 32);
              }
LABEL_23:
              v3 = *(__m128i **)(a1 + 24);
              v6 = (__m128i *)((char *)v17 + (char *)v14 - (char *)v6);
              if ( v6 != v14 )
                *(_QWORD *)(a1 + 32) = v6;
              goto LABEL_25;
            }
LABEL_47:
            v3 = *(__m128i **)(a1 + 24);
            v6 = v14;
            goto LABEL_25;
          }
        }
        v11 = v13->m128i_u64[1];
        v12 = (const void *)v13->m128i_i64[0];
      }
    }
    v17 = v7;
    if ( v7 != v6 )
      goto LABEL_23;
    goto LABEL_47;
  }
LABEL_25:
  v18 = v3;
  if ( v3 != v6 )
  {
    while ( 1 )
    {
      v19 = (const void *)v18->m128i_i64[0];
      v20 = v18->m128i_u64[1];
      v21 = sub_C92610();
      v22 = (unsigned int)sub_C92740(a1, v19, v20, v21);
      v23 = v22;
      v24 = (_QWORD *)(*(_QWORD *)a1 + 8 * v22);
      v25 = *v24;
      if ( !*v24 )
        goto LABEL_37;
      if ( v25 == -8 )
        break;
LABEL_30:
      v26 = *(_QWORD *)(v25 + 8);
      if ( *(_BYTE *)(v26 + 89) )
      {
        if ( v6 == ++v18 )
          return;
      }
      else
      {
        ++v18;
        sub_36FCCC0(a1, v26);
        if ( v6 == v18 )
          return;
      }
    }
    --*(_DWORD *)(a1 + 16);
LABEL_37:
    v39 = v24;
    srca = v23;
    v29 = sub_C7D670(v20 + 17, 8);
    v30 = srca;
    v31 = v39;
    v32 = (_QWORD *)v29;
    if ( v20 )
    {
      na = v29;
      memcpy((void *)(v29 + 16), v19, v20);
      v30 = srca;
      v31 = v39;
      v32 = (_QWORD *)na;
    }
    *((_BYTE *)v32 + v20 + 16) = 0;
    *v32 = v20;
    v32[1] = 0;
    *v31 = v32;
    ++*(_DWORD *)(a1 + 12);
    v33 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_C929D0((__int64 *)a1, v30));
    v25 = *v33;
    if ( *v33 == -8 || !v25 )
    {
      v34 = v33 + 1;
      do
      {
        do
          v25 = *v34++;
        while ( v25 == -8 );
      }
      while ( !v25 );
    }
    goto LABEL_30;
  }
}
