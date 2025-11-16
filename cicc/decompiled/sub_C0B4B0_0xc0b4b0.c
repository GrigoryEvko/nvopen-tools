// Function: sub_C0B4B0
// Address: 0xc0b4b0
//
char __fastcall sub_C0B4B0(__int64 a1, const __m128i *a2, const __m128i *a3)
{
  __m128i *v3; // rax
  const __m128i *v5; // rbx
  unsigned int v6; // r13d
  __int64 v7; // r15
  int v8; // r14d
  __int64 v9; // rcx
  int v10; // eax
  __int64 v11; // rax
  __m128i v12; // xmm0
  unsigned int v13; // r13d
  int v14; // eax
  char *v15; // rdi
  size_t v16; // rdx
  int v17; // r9d
  unsigned int j; // r8d
  __int64 v19; // r14
  const void *v20; // rsi
  unsigned int v21; // r8d
  int v22; // eax
  int v23; // r14d
  int v24; // r14d
  __int64 v25; // r13
  int v26; // eax
  size_t v27; // rdx
  char *v28; // rdi
  __int64 v29; // r8
  int v30; // r9d
  unsigned int v31; // r15d
  const void *v32; // rsi
  bool v33; // al
  unsigned int v34; // r15d
  int v35; // r14d
  __int64 v36; // r13
  int v37; // eax
  size_t v38; // rdx
  int v39; // r9d
  unsigned int i; // r15d
  const void *v41; // rsi
  bool v42; // al
  int v43; // eax
  unsigned int v44; // r15d
  int v45; // eax
  __int64 v47; // [rsp+0h] [rbp-80h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  size_t v49; // [rsp+10h] [rbp-70h]
  size_t v50; // [rsp+18h] [rbp-68h]
  size_t v51; // [rsp+18h] [rbp-68h]
  __int64 v52; // [rsp+28h] [rbp-58h]
  __m128i v53; // [rsp+30h] [rbp-50h] BYREF
  int v54; // [rsp+44h] [rbp-3Ch]
  const __m128i *v55; // [rsp+48h] [rbp-38h]

  LOBYTE(v3) = a1 + 48;
  v55 = a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v48 = a1 + 48;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  if ( a2 == a3 )
    return (char)v3;
  v5 = a2;
  v6 = 0;
  v47 = a1 + 32;
  v7 = 0;
LABEL_3:
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
LABEL_5:
    sub_BA8070(a1, 2 * v6);
    v8 = *(_DWORD *)(a1 + 24);
    v9 = 0;
    if ( !v8 )
      goto LABEL_6;
    v35 = v8 - 1;
    v36 = *(_QWORD *)(a1 + 8);
    v37 = sub_C94890(v5->m128i_i64[0], v5->m128i_i64[1]);
    v38 = v5->m128i_u64[1];
    v28 = (char *)v5->m128i_i64[0];
    v29 = 0;
    v39 = 1;
    for ( i = v35 & v37; ; i = v35 & v44 )
    {
      v9 = v36 + 16LL * i;
      v41 = *(const void **)v9;
      if ( *(_QWORD *)v9 == -1 )
        goto LABEL_57;
      v42 = v28 + 2 == 0;
      if ( v41 != (const void *)-2LL )
      {
        if ( v38 != *(_QWORD *)(v9 + 8) )
          goto LABEL_49;
        v54 = v39;
        v53.m128i_i64[0] = v29;
        if ( !v38 )
          goto LABEL_6;
        v50 = v38;
        v43 = memcmp(v28, v41, v38);
        v38 = v50;
        v9 = v36 + 16LL * i;
        v29 = v53.m128i_i64[0];
        v39 = v54;
        v42 = v43 == 0;
      }
      if ( v42 )
        goto LABEL_6;
      if ( !v29 && v41 == (const void *)-2LL )
        v29 = v9;
LABEL_49:
      v44 = v39 + i;
      ++v39;
    }
  }
  v13 = v6 - 1;
  v14 = sub_C94890(v5->m128i_i64[0], v5->m128i_i64[1]);
  v15 = (char *)v5->m128i_i64[0];
  v16 = v5->m128i_u64[1];
  v9 = 0;
  v17 = 1;
  for ( j = v13 & v14; ; j = v13 & v21 )
  {
    v19 = v7 + 16LL * j;
    v20 = *(const void **)v19;
    LOBYTE(v3) = v15 + 1 == 0;
    if ( *(_QWORD *)v19 != -1 )
    {
      LOBYTE(v3) = v15 + 2 == 0;
      if ( v20 != (const void *)-2LL )
      {
        if ( v16 != *(_QWORD *)(v19 + 8) )
          goto LABEL_17;
        v52 = v9;
        v54 = v17;
        v53.m128i_i32[0] = j;
        if ( !v16 )
          goto LABEL_24;
        v49 = v16;
        LODWORD(v3) = memcmp(v15, v20, v16);
        v16 = v49;
        j = v53.m128i_i32[0];
        v17 = v54;
        v9 = v52;
        LOBYTE(v3) = (_DWORD)v3 == 0;
      }
    }
    if ( (_BYTE)v3 )
    {
LABEL_24:
      if ( v55 == ++v5 )
        return (char)v3;
LABEL_25:
      v7 = *(_QWORD *)(a1 + 8);
      v6 = *(_DWORD *)(a1 + 24);
      goto LABEL_3;
    }
    if ( v20 == (const void *)-1LL )
      break;
LABEL_17:
    if ( v20 == (const void *)-2LL && !v9 )
      v9 = v19;
    v21 = v17 + j;
    ++v17;
  }
  v22 = *(_DWORD *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
    v9 = v19;
  ++*(_QWORD *)a1;
  v10 = v22 + 1;
  if ( 4 * v10 >= 3 * v6 )
    goto LABEL_5;
  if ( v6 - (v10 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
    goto LABEL_7;
  sub_BA8070(a1, v6);
  v23 = *(_DWORD *)(a1 + 24);
  v9 = 0;
  if ( !v23 )
    goto LABEL_6;
  v24 = v23 - 1;
  v25 = *(_QWORD *)(a1 + 8);
  v26 = sub_C94890(v5->m128i_i64[0], v5->m128i_i64[1]);
  v27 = v5->m128i_u64[1];
  v28 = (char *)v5->m128i_i64[0];
  v29 = 0;
  v30 = 1;
  v31 = v24 & v26;
  while ( 2 )
  {
    v9 = v25 + 16LL * v31;
    v32 = *(const void **)v9;
    if ( *(_QWORD *)v9 != -1 )
    {
      v33 = v28 + 2 == 0;
      if ( v32 != (const void *)-2LL )
      {
        if ( *(_QWORD *)(v9 + 8) != v27 )
        {
LABEL_36:
          if ( v29 || v32 != (const void *)-2LL )
            v9 = v29;
          v34 = v30 + v31;
          v29 = v9;
          ++v30;
          v31 = v24 & v34;
          continue;
        }
        v54 = v30;
        v53.m128i_i64[0] = v29;
        if ( !v27 )
          goto LABEL_6;
        v51 = v27;
        v45 = memcmp(v28, v32, v27);
        v27 = v51;
        v9 = v25 + 16LL * v31;
        v29 = v53.m128i_i64[0];
        v30 = v54;
        v33 = v45 == 0;
      }
      if ( v33 )
        goto LABEL_6;
      if ( v32 == (const void *)-1LL )
        goto LABEL_54;
      goto LABEL_36;
    }
    break;
  }
LABEL_57:
  if ( v28 == (char *)-1LL )
    goto LABEL_6;
LABEL_54:
  if ( v29 )
    v9 = v29;
LABEL_6:
  v10 = *(_DWORD *)(a1 + 16) + 1;
LABEL_7:
  *(_DWORD *)(a1 + 16) = v10;
  if ( *(_QWORD *)v9 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v9 = _mm_loadu_si128(v5);
  v11 = *(unsigned int *)(a1 + 40);
  v12 = _mm_loadu_si128(v5);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v53 = v12;
    sub_C8D5F0(v47, v48, v11 + 1, 16);
    v11 = *(unsigned int *)(a1 + 40);
    v12 = _mm_load_si128(&v53);
  }
  v3 = (__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v11);
  ++v5;
  *v3 = v12;
  ++*(_DWORD *)(a1 + 40);
  if ( v55 != v5 )
    goto LABEL_25;
  return (char)v3;
}
