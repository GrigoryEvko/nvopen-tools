// Function: sub_29DF010
// Address: 0x29df010
//
char __fastcall sub_29DF010(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // r13d
  _QWORD *v6; // r15
  size_t v7; // r8
  __int64 i; // r9
  int v9; // r14d
  size_t v10; // r10
  int v11; // eax
  __int64 v12; // rax
  unsigned int v13; // r13d
  int v14; // eax
  int v15; // r11d
  unsigned int j; // ecx
  size_t v17; // r14
  const void *v18; // rsi
  unsigned int v19; // ecx
  int v20; // eax
  int v21; // r14d
  __int64 v22; // r13
  int v23; // r14d
  unsigned int v24; // eax
  int v25; // r11d
  size_t v26; // rcx
  const void *v27; // rsi
  bool v28; // al
  unsigned int v29; // r9d
  __int64 v30; // r13
  int v31; // r14d
  unsigned int v32; // eax
  int v33; // r11d
  const void *v34; // rsi
  bool v35; // al
  int v36; // eax
  unsigned int v37; // r9d
  int v38; // eax
  size_t v40; // [rsp+8h] [rbp-68h]
  size_t v41; // [rsp+8h] [rbp-68h]
  size_t v42; // [rsp+8h] [rbp-68h]
  size_t v43; // [rsp+10h] [rbp-60h]
  size_t v44; // [rsp+10h] [rbp-60h]
  size_t v45; // [rsp+18h] [rbp-58h]
  int v46; // [rsp+24h] [rbp-4Ch]
  int v47; // [rsp+24h] [rbp-4Ch]
  int v48; // [rsp+24h] [rbp-4Ch]
  size_t nc; // [rsp+28h] [rbp-48h]
  unsigned int n; // [rsp+28h] [rbp-48h]
  size_t na; // [rsp+28h] [rbp-48h]
  size_t nb; // [rsp+28h] [rbp-48h]
  size_t v53; // [rsp+30h] [rbp-40h]
  size_t v54; // [rsp+30h] [rbp-40h]
  size_t v55; // [rsp+30h] [rbp-40h]
  size_t v56; // [rsp+30h] [rbp-40h]
  unsigned int v57; // [rsp+30h] [rbp-40h]
  unsigned int v58; // [rsp+30h] [rbp-40h]

  LOBYTE(v3) = a1 + 48;
  if ( a2 == a3 )
    return (char)v3;
  v4 = a2;
  do
  {
LABEL_3:
    v5 = *(_DWORD *)(a1 + 24);
    v6 = *(_QWORD **)v4;
    v7 = *(_QWORD *)(v4 + 8);
    if ( !v5 )
    {
      ++*(_QWORD *)a1;
LABEL_5:
      v53 = v7;
      sub_BA8070(a1, 2 * v5);
      v9 = *(_DWORD *)(a1 + 24);
      v10 = 0;
      v7 = v53;
      if ( !v9 )
        goto LABEL_6;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = v9 - 1;
      v32 = sub_C94890(v6, v53);
      v7 = v53;
      v33 = 1;
      v26 = 0;
      for ( i = v31 & v32; ; i = v31 & v37 )
      {
        v10 = v30 + 16LL * (unsigned int)i;
        v34 = *(const void **)v10;
        if ( *(_QWORD *)v10 == -1 )
          goto LABEL_57;
        v35 = (_QWORD *)((char *)v6 + 2) == 0;
        if ( v34 != (const void *)-2LL )
        {
          if ( v7 != *(_QWORD *)(v10 + 8) )
            goto LABEL_49;
          v47 = v33;
          na = v26;
          v57 = i;
          if ( !v7 )
            goto LABEL_6;
          v41 = v30 + 16LL * (unsigned int)i;
          v43 = v7;
          v36 = memcmp(v6, v34, v7);
          v7 = v43;
          v10 = v41;
          i = v57;
          v26 = na;
          v33 = v47;
          v35 = v36 == 0;
        }
        if ( v35 )
          goto LABEL_6;
        if ( v34 == (const void *)-2LL && !v26 )
          v26 = v10;
LABEL_49:
        v37 = v33 + i;
        ++v33;
      }
    }
    nc = *(_QWORD *)(v4 + 8);
    v13 = v5 - 1;
    v54 = *(_QWORD *)(a1 + 8);
    v14 = sub_C94890(*(_QWORD **)v4, nc);
    v7 = nc;
    v15 = 1;
    v10 = 0;
    for ( j = v13 & v14; ; j = v13 & v19 )
    {
      v17 = v54 + 16LL * j;
      v18 = *(const void **)v17;
      LOBYTE(v3) = (_QWORD *)((char *)v6 + 1) == 0;
      if ( *(_QWORD *)v17 != -1 )
      {
        LOBYTE(v3) = (_QWORD *)((char *)v6 + 2) == 0;
        if ( v18 != (const void *)-2LL )
        {
          if ( v7 != *(_QWORD *)(v17 + 8) )
            goto LABEL_17;
          v45 = v10;
          v46 = v15;
          n = j;
          if ( !v7 )
            goto LABEL_24;
          v40 = v7;
          LODWORD(v3) = memcmp(v6, v18, v7);
          v7 = v40;
          j = n;
          v15 = v46;
          v10 = v45;
          LOBYTE(v3) = (_DWORD)v3 == 0;
        }
      }
      if ( (_BYTE)v3 )
      {
LABEL_24:
        v4 += 32;
        if ( a3 == v4 )
          return (char)v3;
        goto LABEL_3;
      }
      if ( v18 == (const void *)-1LL )
        break;
LABEL_17:
      if ( v18 == (const void *)-2LL && !v10 )
        v10 = v17;
      v19 = v15 + j;
      ++v15;
    }
    v20 = *(_DWORD *)(a1 + 16);
    v5 = *(_DWORD *)(a1 + 24);
    if ( !v10 )
      v10 = v17;
    ++*(_QWORD *)a1;
    v11 = v20 + 1;
    if ( 4 * v11 >= 3 * v5 )
      goto LABEL_5;
    if ( v5 - (v11 + *(_DWORD *)(a1 + 20)) > v5 >> 3 )
      goto LABEL_7;
    v55 = v7;
    sub_BA8070(a1, v5);
    v21 = *(_DWORD *)(a1 + 24);
    v10 = 0;
    v7 = v55;
    if ( !v21 )
      goto LABEL_6;
    v22 = *(_QWORD *)(a1 + 8);
    v23 = v21 - 1;
    v24 = sub_C94890(v6, v55);
    v7 = v55;
    v25 = 1;
    v26 = 0;
    i = v23 & v24;
    while ( 2 )
    {
      v10 = v22 + 16LL * (unsigned int)i;
      v27 = *(const void **)v10;
      if ( *(_QWORD *)v10 != -1 )
      {
        v28 = (_QWORD *)((char *)v6 + 2) == 0;
        if ( v27 != (const void *)-2LL )
        {
          if ( v7 != *(_QWORD *)(v10 + 8) )
          {
LABEL_36:
            if ( v26 || v27 != (const void *)-2LL )
              v10 = v26;
            v29 = v25 + i;
            v26 = v10;
            ++v25;
            i = v23 & v29;
            continue;
          }
          v48 = v25;
          nb = v26;
          v58 = i;
          if ( !v7 )
            goto LABEL_6;
          v42 = v22 + 16LL * (unsigned int)i;
          v44 = v7;
          v38 = memcmp(v6, v27, v7);
          v7 = v44;
          v10 = v42;
          i = v58;
          v26 = nb;
          v25 = v48;
          v28 = v38 == 0;
        }
        if ( v28 )
          goto LABEL_6;
        if ( v27 == (const void *)-1LL )
          goto LABEL_54;
        goto LABEL_36;
      }
      break;
    }
LABEL_57:
    if ( v6 == (_QWORD *)-1LL )
      goto LABEL_6;
LABEL_54:
    if ( v26 )
      v10 = v26;
LABEL_6:
    v11 = *(_DWORD *)(a1 + 16) + 1;
LABEL_7:
    *(_DWORD *)(a1 + 16) = v11;
    if ( *(_QWORD *)v10 != -1 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v10 = v6;
    *(_QWORD *)(v10 + 8) = v7;
    v12 = *(unsigned int *)(a1 + 40);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      v56 = v7;
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v12 + 1, 0x10u, v7, i);
      v12 = *(unsigned int *)(a1 + 40);
      v7 = v56;
    }
    v3 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 16 * v12);
    v4 += 32;
    *v3 = v6;
    v3[1] = v7;
    ++*(_DWORD *)(a1 + 40);
  }
  while ( a3 != v4 );
  return (char)v3;
}
