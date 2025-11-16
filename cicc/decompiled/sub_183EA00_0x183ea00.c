// Function: sub_183EA00
// Address: 0x183ea00
//
char *__fastcall sub_183EA00(const char *a1, size_t a2, __int64 a3, __int64 a4)
{
  char *v4; // r12
  size_t v5; // rbx
  char *v6; // rcx
  int v7; // r14d
  char *v8; // rsi
  size_t v9; // r8
  __int64 v10; // rax
  char *v11; // r13
  size_t v12; // r15
  int v13; // r14d
  __int64 v14; // rax
  void *v15; // r9
  void *v16; // rax
  char *v17; // rcx
  int v18; // r13d
  size_t v19; // r15
  __int64 v20; // rax
  void *v21; // r14
  size_t v22; // r9
  __int64 *v23; // r13
  __int64 *v24; // r14
  int v25; // eax
  char *v26; // rsi
  __int64 v27; // r15
  const char *v28; // r12
  size_t v29; // rdx
  size_t v30; // rbx
  size_t v31; // rdx
  size_t v32; // r15
  __int64 v33; // r15
  const char *v34; // r12
  size_t v35; // rdx
  size_t v36; // rbx
  size_t v37; // rdx
  size_t v38; // r15
  int v39; // eax
  char *v40; // rsi
  __int64 v41; // r15
  __int64 *v42; // rdx
  __int64 v43; // r14
  __int64 *v44; // rdx
  __int64 v45; // r13
  char *v46; // rax
  char *v47; // rsi
  unsigned __int64 v48; // r13
  char *v49; // rcx
  _BYTE *v50; // rax
  _BYTE *v51; // rsi
  size_t v52; // rbx
  int v54; // eax
  int v55; // r13d
  void *v56; // r10
  void *v57; // rax
  unsigned __int64 v58; // r13
  _BYTE *v59; // rax
  unsigned __int64 v60; // r14
  __int64 v61; // rax
  char *v62; // rcx
  _BYTE *v63; // rax
  _BYTE *v64; // rsi
  size_t v65; // rbx
  char *v66; // rsi
  char *v67; // rsi
  __int64 v68; // rax
  int v69; // eax
  char *v70; // [rsp+8h] [rbp-88h]
  char *v71; // [rsp+10h] [rbp-80h]
  size_t v72; // [rsp+10h] [rbp-80h]
  size_t v73; // [rsp+10h] [rbp-80h]
  size_t nb; // [rsp+18h] [rbp-78h]
  size_t nc; // [rsp+18h] [rbp-78h]
  size_t n; // [rsp+18h] [rbp-78h]
  size_t nd; // [rsp+18h] [rbp-78h]
  size_t ne; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  size_t nf; // [rsp+18h] [rbp-78h]
  char *v81; // [rsp+20h] [rbp-70h]
  char *v82; // [rsp+20h] [rbp-70h]
  char *v83; // [rsp+20h] [rbp-70h]
  bool v84; // [rsp+28h] [rbp-68h]
  __int64 *v86; // [rsp+30h] [rbp-60h]
  __int64 v87; // [rsp+38h] [rbp-58h]
  __int64 *v88; // [rsp+38h] [rbp-58h]
  __int64 v89; // [rsp+40h] [rbp-50h] BYREF
  char *v90; // [rsp+48h] [rbp-48h]
  char *v91; // [rsp+50h] [rbp-40h]

  v4 = (char *)a1;
  v5 = a2;
  v6 = *(char **)(a2 + 56);
  v7 = *(_DWORD *)(a2 + 40);
  v8 = *(char **)(a2 + 48);
  v87 = a3;
  v9 = v6 - v8;
  v81 = (char *)(v6 - v8);
  if ( v6 == v8 )
  {
    v12 = 0;
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_140;
    a1 = (const char *)(v6 - v8);
    v10 = sub_22077B0(v6 - v8);
    v6 = *(char **)(v5 + 56);
    v8 = *(char **)(v5 + 48);
    v11 = (char *)v10;
    v9 = v6 - v8;
    v12 = v6 - v8;
  }
  if ( v8 != v6 )
  {
    a1 = v11;
    v71 = v6;
    nb = v9;
    memmove(v11, v8, v12);
    v6 = v71;
    v9 = nb;
  }
  if ( v7 == *(_DWORD *)v87 )
  {
    a1 = *(const char **)(v87 + 8);
    if ( v9 == *(_QWORD *)(v87 + 16) - (_QWORD)a1 )
    {
      if ( v9 )
      {
        v8 = v11;
        if ( memcmp(a1, v11, v9) )
        {
          v13 = *(_DWORD *)(v5 + 40);
          goto LABEL_8;
        }
      }
      else if ( !v11 )
      {
        goto LABEL_80;
      }
      v8 = v81;
      a1 = v11;
      j_j___libc_free_0(v11, v81);
      goto LABEL_80;
    }
  }
  v13 = *(_DWORD *)(v5 + 40);
  if ( !v9 )
  {
    v15 = 0;
    if ( v6 != v8 )
      goto LABEL_10;
    goto LABEL_66;
  }
LABEL_8:
  if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_140;
  a1 = (const char *)v12;
  v14 = sub_22077B0(v12);
  v8 = *(char **)(v5 + 48);
  v15 = (void *)v14;
  v9 = *(_QWORD *)(v5 + 56) - (_QWORD)v8;
  if ( *(char **)(v5 + 56) != v8 )
  {
LABEL_10:
    nc = v9;
    v16 = memmove(v15, v8, v9);
    v84 = 0;
    v9 = nc;
    v15 = v16;
    if ( *(_DWORD *)a4 != v13 )
    {
LABEL_11:
      v8 = (char *)v12;
      a1 = (const char *)v15;
      j_j___libc_free_0(v15, v12);
      goto LABEL_12;
    }
    goto LABEL_92;
  }
LABEL_66:
  if ( *(_DWORD *)a4 != v13 )
  {
LABEL_67:
    v84 = 0;
    goto LABEL_68;
  }
LABEL_92:
  a1 = *(const char **)(a4 + 8);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)a1 != v9 )
    goto LABEL_67;
  if ( v9 )
  {
    nd = (size_t)v15;
    v54 = memcmp(a1, v15, v9);
    v15 = (void *)nd;
    v84 = v54 == 0;
    goto LABEL_11;
  }
  v84 = 1;
LABEL_68:
  if ( v15 )
    goto LABEL_11;
LABEL_12:
  if ( v11 )
  {
    v8 = v81;
    a1 = v11;
    j_j___libc_free_0(v11, v81);
  }
  if ( v84 )
  {
LABEL_80:
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 40);
    v48 = *(_QWORD *)(v5 + 56) - *(_QWORD *)(v5 + 48);
    *((_QWORD *)v4 + 1) = 0;
    *((_QWORD *)v4 + 2) = 0;
    *((_QWORD *)v4 + 3) = 0;
    if ( v48 )
    {
      if ( v48 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_140;
      v49 = (char *)sub_22077B0(v48);
    }
    else
    {
      v48 = 0;
      v49 = 0;
    }
    *((_QWORD *)v4 + 1) = v49;
    *((_QWORD *)v4 + 2) = v49;
    *((_QWORD *)v4 + 3) = &v49[v48];
    v50 = *(_BYTE **)(v5 + 56);
    v51 = *(_BYTE **)(v5 + 48);
    v52 = v50 - v51;
    if ( v50 == v51 )
    {
LABEL_85:
      *((_QWORD *)v4 + 2) = &v49[v52];
      return v4;
    }
LABEL_84:
    v49 = (char *)memmove(v49, v51, v52);
    goto LABEL_85;
  }
  v17 = *(char **)(v5 + 24);
  v8 = *(char **)(v5 + 16);
  v18 = *(_DWORD *)(v5 + 8);
  v19 = v17 - v8;
  v82 = (char *)(v17 - v8);
  if ( v17 == v8 )
  {
    v22 = 0;
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_140;
    v20 = sub_22077B0(v19);
    v17 = *(char **)(v5 + 24);
    v8 = *(char **)(v5 + 16);
    v21 = (void *)v20;
    v19 = v17 - v8;
    v22 = v17 - v8;
  }
  if ( v17 == v8 )
  {
    if ( *(_DWORD *)v87 != v18 || (a1 = *(const char **)(v87 + 8), *(_QWORD *)(v87 + 16) - (_QWORD)a1 != v19) )
    {
      if ( !v21 )
        goto LABEL_21;
      goto LABEL_20;
    }
  }
  else
  {
    v70 = v17;
    v72 = v22;
    memmove(v21, v8, v22);
    v22 = v72;
    v17 = v70;
    if ( *(_DWORD *)v87 != v18 )
      goto LABEL_20;
    a1 = *(const char **)(v87 + 8);
    if ( v19 != *(_QWORD *)(v87 + 16) - (_QWORD)a1 )
      goto LABEL_20;
  }
  if ( !v19 )
  {
    v55 = *(_DWORD *)(v5 + 8);
    v56 = 0;
    goto LABEL_103;
  }
  v8 = (char *)v21;
  na = v22;
  if ( memcmp(a1, v21, v19) )
  {
LABEL_20:
    j_j___libc_free_0(v21, v82);
    goto LABEL_21;
  }
  v55 = *(_DWORD *)(v5 + 8);
  if ( na > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_140;
  a1 = (const char *)na;
  v68 = sub_22077B0(na);
  v17 = *(char **)(v5 + 24);
  v8 = *(char **)(v5 + 16);
  v22 = na;
  v56 = (void *)v68;
  v19 = v17 - v8;
LABEL_103:
  if ( v8 == v17 )
  {
    if ( *(_DWORD *)a4 != v55 )
      goto LABEL_134;
  }
  else
  {
    ne = v22;
    v57 = memmove(v56, v8, v19);
    v22 = ne;
    v56 = v57;
    if ( *(_DWORD *)a4 != v55 )
    {
LABEL_105:
      v8 = (char *)v22;
      a1 = (const char *)v56;
      j_j___libc_free_0(v56, v22);
      goto LABEL_106;
    }
  }
  a1 = *(const char **)(a4 + 8);
  if ( v19 == *(_QWORD *)(a4 + 16) - (_QWORD)a1 )
  {
    if ( v19 )
    {
      v73 = v22;
      nf = (size_t)v56;
      v69 = memcmp(a1, v56, v19);
      v56 = (void *)nf;
      v22 = v73;
      v84 = v69 == 0;
      goto LABEL_105;
    }
    v84 = 1;
  }
LABEL_134:
  if ( v56 )
    goto LABEL_105;
LABEL_106:
  if ( v21 )
  {
    v8 = v82;
    a1 = (const char *)v21;
    j_j___libc_free_0(v21, v82);
  }
  if ( v84 )
  {
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 8);
    v58 = *(_QWORD *)(v5 + 24) - *(_QWORD *)(v5 + 16);
    *((_QWORD *)v4 + 1) = 0;
    *((_QWORD *)v4 + 2) = 0;
    *((_QWORD *)v4 + 3) = 0;
    if ( !v58 )
    {
      v49 = 0;
LABEL_112:
      *((_QWORD *)v4 + 1) = v49;
      *((_QWORD *)v4 + 2) = v49;
      *((_QWORD *)v4 + 3) = &v49[v58];
      v59 = *(_BYTE **)(v5 + 24);
      v51 = *(_BYTE **)(v5 + 16);
      v52 = v59 - v51;
      if ( v59 == v51 )
        goto LABEL_85;
      goto LABEL_84;
    }
    if ( v58 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v49 = (char *)sub_22077B0(v58);
      goto LABEL_112;
    }
LABEL_140:
    sub_4261EA(a1, v8, a3);
  }
LABEL_21:
  v89 = 0;
  v90 = 0;
  a1 = *(const char **)(a4 + 16);
  v23 = *(__int64 **)(a4 + 8);
  v91 = 0;
  v86 = (__int64 *)a1;
  v24 = *(__int64 **)(v87 + 8);
  v88 = *(__int64 **)(v87 + 16);
  if ( v24 != v88 && v23 != (__int64 *)a1 )
  {
    v83 = v4;
    n = v5;
    while ( 1 )
    {
      v27 = *v24;
      v28 = sub_1649960(*v23);
      v30 = v29;
      a1 = sub_1649960(v27);
      v32 = v31;
      if ( v31 <= v30 )
      {
        if ( !v31 || (v25 = memcmp(a1, v28, v31)) == 0 )
        {
          if ( v32 == v30 )
            goto LABEL_38;
LABEL_27:
          if ( v32 >= v30 )
            goto LABEL_38;
          goto LABEL_28;
        }
      }
      else
      {
        if ( !v30 )
          goto LABEL_38;
        v25 = memcmp(a1, v28, v30);
        if ( !v25 )
          goto LABEL_27;
      }
      if ( v25 >= 0 )
      {
LABEL_38:
        v33 = *v23;
        v34 = sub_1649960(*v24);
        v36 = v35;
        a1 = sub_1649960(v33);
        v38 = v37;
        if ( v37 > v36 )
        {
          if ( !v36 )
            goto LABEL_73;
          v39 = memcmp(a1, v34, v36);
          if ( v39 )
            goto LABEL_72;
LABEL_42:
          if ( v38 >= v36 )
            goto LABEL_73;
        }
        else
        {
          if ( !v37 || (v39 = memcmp(a1, v34, v37)) == 0 )
          {
            if ( v38 != v36 )
              goto LABEL_42;
LABEL_73:
            v47 = v90;
            if ( v90 == v91 )
            {
              a1 = (const char *)&v89;
              sub_14F2380((__int64)&v89, v90, v24);
            }
            else
            {
              if ( v90 )
              {
                *(_QWORD *)v90 = *v24;
                v47 = v90;
              }
              v90 = v47 + 8;
            }
            ++v24;
LABEL_47:
            ++v23;
            if ( v88 == v24 )
              goto LABEL_48;
            goto LABEL_33;
          }
LABEL_72:
          if ( v39 >= 0 )
            goto LABEL_73;
        }
        v40 = v90;
        if ( v90 == v91 )
        {
          a1 = (const char *)&v89;
          sub_14F2380((__int64)&v89, v90, v23);
        }
        else
        {
          if ( v90 )
          {
            *(_QWORD *)v90 = *v23;
            v40 = v90;
          }
          v90 = v40 + 8;
        }
        goto LABEL_47;
      }
LABEL_28:
      v26 = v90;
      if ( v90 == v91 )
      {
        a1 = (const char *)&v89;
        sub_14F2380((__int64)&v89, v90, v24);
      }
      else
      {
        if ( v90 )
        {
          *(_QWORD *)v90 = *v24;
          v26 = v90;
        }
        v90 = v26 + 8;
      }
      if ( v88 == ++v24 )
      {
LABEL_48:
        v4 = v83;
        v5 = n;
        v8 = v90;
        goto LABEL_49;
      }
LABEL_33:
      if ( v86 == v23 )
        goto LABEL_48;
    }
  }
  v8 = 0;
LABEL_49:
  v41 = v88 - v24;
  if ( (char *)v88 - (char *)v24 > 0 )
  {
    do
    {
      while ( v91 == v8 )
      {
        v42 = v24;
        a1 = (const char *)&v89;
        ++v24;
        sub_14F2380((__int64)&v89, v8, v42);
        v8 = v90;
        if ( !--v41 )
          goto LABEL_56;
      }
      if ( v8 )
      {
        *(_QWORD *)v8 = *v24;
        v8 = v90;
      }
      v8 += 8;
      ++v24;
      v90 = v8;
      --v41;
    }
    while ( v41 );
  }
LABEL_56:
  v43 = v86 - v23;
  if ( (char *)v86 - (char *)v23 > 0 )
  {
    do
    {
      while ( v8 == v91 )
      {
        v44 = v23;
        a1 = (const char *)&v89;
        ++v23;
        sub_14F2380((__int64)&v89, v8, v44);
        v8 = v90;
        if ( !--v43 )
          goto LABEL_63;
      }
      if ( v8 )
      {
        *(_QWORD *)v8 = *v23;
        v8 = v90;
      }
      v8 += 8;
      ++v23;
      v90 = v8;
      --v43;
    }
    while ( v43 );
  }
LABEL_63:
  v45 = v89;
  a3 = (unsigned int)dword_4FAA560;
  if ( (unsigned int)dword_4FAA560 < (unsigned __int64)((__int64)&v8[-v89] >> 3) )
  {
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 40);
    v60 = *(_QWORD *)(v5 + 56) - *(_QWORD *)(v5 + 48);
    *((_QWORD *)v4 + 1) = 0;
    *((_QWORD *)v4 + 2) = 0;
    *((_QWORD *)v4 + 3) = 0;
    if ( v60 )
    {
      if ( v60 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_140;
      v61 = sub_22077B0(v60);
      v45 = v89;
      v62 = (char *)v61;
    }
    else
    {
      v60 = 0;
      v62 = 0;
    }
    *((_QWORD *)v4 + 1) = v62;
    *((_QWORD *)v4 + 2) = v62;
    *((_QWORD *)v4 + 3) = &v62[v60];
    v63 = *(_BYTE **)(v5 + 56);
    v64 = *(_BYTE **)(v5 + 48);
    v65 = v63 - v64;
    if ( v63 != v64 )
      v62 = (char *)memmove(v62, v64, v65);
    v66 = v91;
    *((_QWORD *)v4 + 2) = &v62[v65];
    v67 = &v66[-v45];
    if ( v45 )
      j_j___libc_free_0(v45, v67);
  }
  else
  {
    v46 = v91;
    *(_DWORD *)v4 = 1;
    *((_QWORD *)v4 + 1) = v45;
    *((_QWORD *)v4 + 2) = v8;
    *((_QWORD *)v4 + 3) = v46;
  }
  return v4;
}
