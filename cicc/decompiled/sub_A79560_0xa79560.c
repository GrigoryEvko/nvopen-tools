// Function: sub_A79560
// Address: 0xa79560
//
char __fastcall sub_A79560(__int64 a1, const void *a2, __int64 a3)
{
  int v3; // eax
  unsigned int v4; // r13d
  __int64 *v5; // rbx
  size_t v6; // rdx
  __int64 v7; // r14
  __int64 v8; // rax
  size_t v9; // rdx
  unsigned int v10; // r8d
  char *v11; // r13
  __int64 v12; // r9
  size_t v13; // rsi
  int v14; // eax
  __int64 v15; // r11
  int v16; // ecx
  int v17; // r8d
  unsigned int i; // r10d
  const void *v19; // rsi
  bool v20; // al
  unsigned int v21; // r10d
  int v22; // eax
  int v23; // edx
  int v24; // eax
  int v25; // ecx
  int v26; // r8d
  unsigned int j; // r10d
  __int64 v28; // r11
  const void *v29; // rsi
  unsigned int v30; // r10d
  int v31; // eax
  int v32; // eax
  size_t v33; // rsi
  int v34; // eax
  int v35; // ecx
  int v36; // r8d
  unsigned int v37; // r10d
  const void *v38; // rsi
  bool v39; // al
  unsigned int v40; // r10d
  int v41; // eax
  int v42; // eax
  size_t v44; // [rsp+8h] [rbp-98h]
  int v45; // [rsp+8h] [rbp-98h]
  int v46; // [rsp+8h] [rbp-98h]
  __int64 v47; // [rsp+10h] [rbp-90h]
  __int64 v48; // [rsp+10h] [rbp-90h]
  __int64 v49; // [rsp+18h] [rbp-88h]
  unsigned int v50; // [rsp+18h] [rbp-88h]
  unsigned int v51; // [rsp+18h] [rbp-88h]
  __int64 v52; // [rsp+20h] [rbp-80h]
  int v53; // [rsp+20h] [rbp-80h]
  int v54; // [rsp+20h] [rbp-80h]
  int v55; // [rsp+28h] [rbp-78h]
  __int64 v56; // [rsp+28h] [rbp-78h]
  __int64 v57; // [rsp+28h] [rbp-78h]
  __int64 v58; // [rsp+30h] [rbp-70h]
  size_t v59; // [rsp+38h] [rbp-68h]
  int v60; // [rsp+38h] [rbp-68h]
  unsigned int v61; // [rsp+38h] [rbp-68h]
  size_t v62; // [rsp+38h] [rbp-68h]
  size_t v63; // [rsp+38h] [rbp-68h]
  size_t v64; // [rsp+38h] [rbp-68h]
  int n; // [rsp+40h] [rbp-60h]
  size_t nc; // [rsp+40h] [rbp-60h]
  int na; // [rsp+40h] [rbp-60h]
  int nb; // [rsp+40h] [rbp-60h]
  size_t v69; // [rsp+48h] [rbp-58h]
  __int64 v70; // [rsp+48h] [rbp-58h]
  __int64 v71; // [rsp+48h] [rbp-58h]
  size_t v72; // [rsp+48h] [rbp-58h]
  __int64 v73; // [rsp+48h] [rbp-58h]
  __int64 v74; // [rsp+60h] [rbp-40h]

  LOBYTE(v3) = a1 + 32;
  v4 = a3;
  v5 = (__int64 *)(a1 + 64);
  *(_DWORD *)(a1 + 8) = a3;
  v6 = 8 * a3;
  *(_QWORD *)a1 = 0;
  v58 = a1 + 32;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_OWORD *)(a1 + 12) = 0;
  if ( v6 )
    LOBYTE(v3) = (unsigned __int8)memmove((void *)(a1 + 64), a2, v6);
  v7 = a1 + 8LL * v4 + 64;
  if ( (__int64 *)v7 == v5 )
    return v3;
  do
  {
    if ( !sub_A71840((__int64)v5) )
    {
      v22 = sub_A71AE0(v5);
      v23 = v22 + 7;
      if ( v22 >= 0 )
        v23 = v22;
      v3 = 1 << (v22 % 8);
      *(_BYTE *)(a1 + (v23 >> 3) + 12) |= v3;
      goto LABEL_16;
    }
    v8 = sub_A71FD0(v5);
    v10 = *(_DWORD *)(a1 + 56);
    v11 = (char *)v8;
    v74 = *v5;
    if ( !v10 )
    {
      ++*(_QWORD *)(a1 + 32);
LABEL_7:
      v69 = v9;
      sub_A792C0(v58, 2 * v10);
      v12 = 0;
      v9 = v69;
      n = *(_DWORD *)(a1 + 56);
      if ( !n )
        goto LABEL_43;
      v13 = v69;
      v59 = v69;
      v70 = *(_QWORD *)(a1 + 40);
      v14 = sub_C94890(v11, v13);
      v9 = v59;
      v15 = 0;
      v16 = 1;
      v17 = n - 1;
      for ( i = (n - 1) & v14; ; i = v17 & v21 )
      {
        v12 = v70 + 24LL * i;
        v19 = *(const void **)v12;
        if ( *(_QWORD *)v12 == -1 )
          goto LABEL_40;
        v20 = v11 + 2 == 0;
        if ( v19 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v12 + 8) )
            goto LABEL_12;
          v46 = v16;
          v48 = v15;
          v51 = i;
          v54 = v17;
          if ( !v9 )
            goto LABEL_43;
          v57 = v70 + 24LL * i;
          v64 = v9;
          v42 = memcmp(v11, v19, v9);
          v9 = v64;
          v12 = v57;
          v17 = v54;
          i = v51;
          v15 = v48;
          v20 = v42 == 0;
          v16 = v46;
        }
        if ( v20 )
          goto LABEL_43;
        if ( v19 == (const void *)-2LL && !v15 )
          v15 = v12;
LABEL_12:
        v21 = v16 + i;
        ++v16;
      }
    }
    v60 = *(_DWORD *)(a1 + 56);
    nc = v9;
    v71 = *(_QWORD *)(a1 + 40);
    v24 = sub_C94890(v8, v9);
    v9 = nc;
    v12 = 0;
    v25 = 1;
    v26 = v60 - 1;
    for ( j = (v60 - 1) & v24; ; j = v26 & v30 )
    {
      v28 = v71 + 24LL * j;
      LOBYTE(v3) = v11 + 1 == 0;
      v29 = *(const void **)v28;
      if ( *(_QWORD *)v28 != -1 )
      {
        LOBYTE(v3) = v11 + 2 == 0;
        if ( v29 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v28 + 8) )
            goto LABEL_22;
          v49 = v71 + 24LL * j;
          v52 = v12;
          v55 = v25;
          v61 = j;
          na = v26;
          if ( !v9 )
            goto LABEL_16;
          v44 = v9;
          v3 = memcmp(v11, v29, v9);
          v9 = v44;
          v26 = na;
          j = v61;
          v25 = v55;
          v12 = v52;
          LOBYTE(v3) = v3 == 0;
          v28 = v49;
        }
      }
      if ( (_BYTE)v3 )
        goto LABEL_16;
      if ( v29 == (const void *)-1LL )
        break;
LABEL_22:
      if ( v29 == (const void *)-2LL && !v12 )
        v12 = v28;
      v30 = v25 + j;
      ++v25;
    }
    v31 = *(_DWORD *)(a1 + 48);
    v10 = *(_DWORD *)(a1 + 56);
    if ( !v12 )
      v12 = v28;
    ++*(_QWORD *)(a1 + 32);
    v32 = v31 + 1;
    if ( 4 * v32 >= 3 * v10 )
      goto LABEL_7;
    if ( v10 - (v32 + *(_DWORD *)(a1 + 52)) > v10 >> 3 )
      goto LABEL_34;
    v72 = v9;
    sub_A792C0(v58, v10);
    v12 = 0;
    v9 = v72;
    nb = *(_DWORD *)(a1 + 56);
    if ( !nb )
      goto LABEL_43;
    v33 = v72;
    v62 = v72;
    v73 = *(_QWORD *)(a1 + 40);
    v34 = sub_C94890(v11, v33);
    v9 = v62;
    v15 = 0;
    v35 = 1;
    v36 = nb - 1;
    v37 = (nb - 1) & v34;
    while ( 2 )
    {
      v12 = v73 + 24LL * v37;
      v38 = *(const void **)v12;
      if ( *(_QWORD *)v12 != -1 )
      {
        v39 = v11 + 2 == 0;
        if ( v38 != (const void *)-2LL )
        {
          if ( v9 != *(_QWORD *)(v12 + 8) )
          {
LABEL_46:
            if ( v15 || v38 != (const void *)-2LL )
              v12 = v15;
            v40 = v35 + v37;
            v15 = v12;
            ++v35;
            v37 = v36 & v40;
            continue;
          }
          v45 = v35;
          v47 = v15;
          v50 = v37;
          v53 = v36;
          if ( !v9 )
            goto LABEL_43;
          v56 = v73 + 24LL * v37;
          v63 = v9;
          v41 = memcmp(v11, v38, v9);
          v9 = v63;
          v12 = v56;
          v36 = v53;
          v37 = v50;
          v15 = v47;
          v39 = v41 == 0;
          v35 = v45;
        }
        if ( v39 )
          goto LABEL_43;
        if ( v38 == (const void *)-1LL )
          goto LABEL_41;
        goto LABEL_46;
      }
      break;
    }
LABEL_40:
    if ( v11 == (char *)-1LL )
      goto LABEL_43;
LABEL_41:
    if ( v15 )
      v12 = v15;
LABEL_43:
    v32 = *(_DWORD *)(a1 + 48) + 1;
LABEL_34:
    *(_DWORD *)(a1 + 48) = v32;
    if ( *(_QWORD *)v12 != -1 )
      --*(_DWORD *)(a1 + 52);
    LOBYTE(v3) = v74;
    *(_QWORD *)v12 = v11;
    *(_QWORD *)(v12 + 8) = v9;
    *(_QWORD *)(v12 + 16) = v74;
LABEL_16:
    ++v5;
  }
  while ( (__int64 *)v7 != v5 );
  return v3;
}
