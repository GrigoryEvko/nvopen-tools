// Function: sub_C1DD70
// Address: 0xc1dd70
//
__int64 __fastcall sub_C1DD70(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r15
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  char *v7; // r8
  size_t v8; // rax
  size_t v9; // rdx
  __int64 v10; // r9
  unsigned int v11; // r12d
  __int64 v12; // rdi
  int v13; // r14d
  size_t v14; // r11
  int v15; // eax
  int v18; // eax
  unsigned int v19; // r10d
  int i; // r12d
  size_t v21; // r14
  const void *v22; // rcx
  bool v23; // al
  unsigned int v24; // r10d
  int v25; // eax
  int v26; // eax
  int v27; // r14d
  size_t v28; // rdi
  size_t v29; // rsi
  __int64 v30; // r12
  int v31; // eax
  size_t v32; // r10
  unsigned int v33; // r14d
  const void *v34; // rcx
  bool v35; // al
  size_t v36; // rdi
  size_t v37; // rsi
  __int64 v38; // r12
  int v39; // eax
  unsigned int v40; // r14d
  const void *v41; // rcx
  bool v42; // al
  int v43; // eax
  int v44; // eax
  __int64 v45; // [rsp+0h] [rbp-80h]
  size_t v46; // [rsp+8h] [rbp-78h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  const void *v49; // [rsp+10h] [rbp-70h]
  size_t v50; // [rsp+10h] [rbp-70h]
  size_t v51; // [rsp+10h] [rbp-70h]
  char *v52; // [rsp+18h] [rbp-68h]
  const void *v53; // [rsp+18h] [rbp-68h]
  const void *v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+20h] [rbp-60h]
  size_t v56; // [rsp+20h] [rbp-60h]
  int v57; // [rsp+20h] [rbp-60h]
  int v58; // [rsp+20h] [rbp-60h]
  void *s1; // [rsp+28h] [rbp-58h]
  char *s1c; // [rsp+28h] [rbp-58h]
  unsigned int s1a; // [rsp+28h] [rbp-58h]
  void *s1b; // [rsp+28h] [rbp-58h]
  char *s1d; // [rsp+28h] [rbp-58h]
  char *s1e; // [rsp+28h] [rbp-58h]
  size_t n; // [rsp+30h] [rbp-50h]
  size_t ne; // [rsp+30h] [rbp-50h]
  unsigned int na; // [rsp+30h] [rbp-50h]
  size_t nb; // [rsp+30h] [rbp-50h]
  size_t nf; // [rsp+30h] [rbp-50h]
  size_t ng; // [rsp+30h] [rbp-50h]
  size_t nc; // [rsp+30h] [rbp-50h]
  size_t nd; // [rsp+30h] [rbp-50h]
  size_t v75; // [rsp+48h] [rbp-38h]
  size_t v76; // [rsp+48h] [rbp-38h]
  size_t v77; // [rsp+48h] [rbp-38h]
  size_t v78; // [rsp+48h] [rbp-38h]
  int v79; // [rsp+48h] [rbp-38h]
  size_t v80; // [rsp+48h] [rbp-38h]
  int v81; // [rsp+48h] [rbp-38h]

  v3 = 0;
  v5 = 0;
  if ( !a3 )
    goto LABEL_14;
  do
  {
LABEL_2:
    v6 = qword_4F83A48;
    if ( qword_4F83A48 <= v3 )
      goto LABEL_13;
    v7 = (char *)(v5 + a2);
    if ( v5 + a2 )
    {
      v8 = strlen((const char *)(v5 + a2));
      v7 = (char *)(v5 + a2);
      v9 = v8;
      v10 = v8 + 1;
    }
    else
    {
      v10 = 1;
      v9 = 0;
    }
    v11 = *(_DWORD *)(a1 + 32);
    if ( !v11 )
    {
      ++*(_QWORD *)(a1 + 8);
      v12 = a1 + 8;
LABEL_7:
      s1 = (void *)v10;
      n = v9;
      v75 = (size_t)v7;
      sub_BA8070(v12, 2 * v11);
      v13 = *(_DWORD *)(a1 + 32);
      v14 = 0;
      v7 = (char *)v75;
      v9 = n;
      v10 = (__int64)s1;
      if ( !v13 )
        goto LABEL_8;
      v36 = v75;
      v37 = n;
      ng = v75;
      v38 = *(_QWORD *)(a1 + 16);
      v80 = v9;
      v39 = sub_C94890(v36, v37);
      v9 = v80;
      v32 = 0;
      v7 = (char *)ng;
      v10 = (__int64)s1;
      v81 = v13 - 1;
      v58 = 1;
      v40 = (v13 - 1) & v39;
      while ( 1 )
      {
        v14 = v38 + 16LL * v40;
        v41 = *(const void **)v14;
        if ( *(_QWORD *)v14 == -1 )
          goto LABEL_63;
        v42 = v7 + 2 == 0;
        if ( v41 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v14 + 8) )
            goto LABEL_55;
          nc = v32;
          if ( !v9 )
            goto LABEL_8;
          v47 = v10;
          v50 = v9;
          v53 = *(const void **)v14;
          s1d = v7;
          v43 = memcmp(v7, v41, v9);
          v7 = s1d;
          v41 = v53;
          v9 = v50;
          v10 = v47;
          v14 = v38 + 16LL * v40;
          v32 = nc;
          v42 = v43 == 0;
        }
        if ( v42 )
          goto LABEL_8;
        if ( v41 == (const void *)-2LL && !v32 )
          v32 = v14;
LABEL_55:
        v40 = v81 & (v58 + v40);
        ++v58;
      }
    }
    v55 = v10;
    s1c = v7;
    ne = v9;
    v76 = *(_QWORD *)(a1 + 16);
    v18 = sub_C94890(v7, v9);
    v7 = s1c;
    v10 = v55;
    v14 = 0;
    v9 = ne;
    na = v11 - 1;
    v19 = (v11 - 1) & v18;
    for ( i = 1; ; ++i )
    {
      v21 = v76 + 16LL * v19;
      v22 = *(const void **)v21;
      v23 = v7 + 1 == 0;
      if ( *(_QWORD *)v21 != -1 )
      {
        v23 = v7 + 2 == 0;
        if ( v22 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v21 + 8) )
            goto LABEL_20;
          v56 = v14;
          s1a = v19;
          if ( !v9 )
            goto LABEL_27;
          v45 = v10;
          v46 = v9;
          v49 = *(const void **)v21;
          v52 = v7;
          v25 = memcmp(v7, v22, v9);
          v7 = v52;
          v22 = v49;
          v9 = v46;
          v10 = v45;
          v19 = s1a;
          v14 = v56;
          v23 = v25 == 0;
        }
      }
      if ( v23 )
      {
LABEL_27:
        v5 += v10;
        ++v3;
        if ( a3 <= v5 )
        {
          v6 = qword_4F83A48;
          if ( a3 == v5 )
            goto LABEL_14;
LABEL_13:
          if ( v3 == v6 )
            goto LABEL_14;
          sub_C1AFD0();
          return 5;
        }
        goto LABEL_2;
      }
      if ( v22 == (const void *)-1LL )
        break;
LABEL_20:
      if ( v22 == (const void *)-2LL && !v14 )
        v14 = v21;
      v24 = i + v19;
      v19 = na & v24;
    }
    v26 = *(_DWORD *)(a1 + 24);
    v11 = *(_DWORD *)(a1 + 32);
    v12 = a1 + 8;
    if ( !v14 )
      v14 = v21;
    ++*(_QWORD *)(a1 + 8);
    v15 = v26 + 1;
    if ( 4 * v15 >= 3 * v11 )
      goto LABEL_7;
    if ( v11 - (v15 + *(_DWORD *)(a1 + 28)) > v11 >> 3 )
      goto LABEL_9;
    s1b = (void *)v10;
    nb = v9;
    v77 = (size_t)v7;
    sub_BA8070(v12, v11);
    v27 = *(_DWORD *)(a1 + 32);
    v14 = 0;
    v7 = (char *)v77;
    v9 = nb;
    v10 = (__int64)s1b;
    if ( !v27 )
      goto LABEL_8;
    v28 = v77;
    v29 = nb;
    nf = v77;
    v30 = *(_QWORD *)(a1 + 16);
    v78 = v9;
    v31 = sub_C94890(v28, v29);
    v9 = v78;
    v32 = 0;
    v7 = (char *)nf;
    v10 = (__int64)s1b;
    v79 = v27 - 1;
    v57 = 1;
    v33 = (v27 - 1) & v31;
    while ( 2 )
    {
      v14 = v30 + 16LL * v33;
      v34 = *(const void **)v14;
      if ( *(_QWORD *)v14 != -1 )
      {
        v35 = v7 + 2 == 0;
        if ( v34 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v14 + 8) )
          {
LABEL_42:
            if ( v32 || v34 != (const void *)-2LL )
              v14 = v32;
            v32 = v14;
            v33 = v79 & (v57 + v33);
            ++v57;
            continue;
          }
          nd = v32;
          if ( !v9 )
            goto LABEL_8;
          v48 = v10;
          v51 = v9;
          v54 = *(const void **)v14;
          s1e = v7;
          v44 = memcmp(v7, v34, v9);
          v7 = s1e;
          v34 = v54;
          v9 = v51;
          v10 = v48;
          v14 = v30 + 16LL * v33;
          v32 = nd;
          v35 = v44 == 0;
        }
        if ( v35 )
          goto LABEL_8;
        if ( v34 == (const void *)-1LL )
          goto LABEL_60;
        goto LABEL_42;
      }
      break;
    }
LABEL_63:
    if ( v7 == (char *)-1LL )
      goto LABEL_8;
LABEL_60:
    if ( v32 )
      v14 = v32;
LABEL_8:
    v15 = *(_DWORD *)(a1 + 24) + 1;
LABEL_9:
    *(_DWORD *)(a1 + 24) = v15;
    if ( *(_QWORD *)v14 != -1 )
      --*(_DWORD *)(a1 + 28);
    *(_QWORD *)v14 = v7;
    v5 += v10;
    ++v3;
    *(_QWORD *)(v14 + 8) = v9;
  }
  while ( a3 > v5 );
  v6 = qword_4F83A48;
  if ( a3 != v5 )
    goto LABEL_13;
LABEL_14:
  sub_C1AFD0();
  return 0;
}
