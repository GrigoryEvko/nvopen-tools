// Function: sub_25A93A0
// Address: 0x25a93a0
//
unsigned __int64 *__fastcall sub_25A93A0(unsigned __int64 *a1, size_t a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v4; // r12
  size_t v5; // rbx
  char *v6; // rcx
  int v7; // r14d
  char *v8; // rsi
  size_t v9; // r8
  __int64 v10; // rax
  unsigned __int64 *v11; // r13
  size_t v12; // r15
  int v13; // r14d
  __int64 v14; // rax
  void *v15; // r9
  void *v16; // rax
  char *v17; // rcx
  int v18; // r13d
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  void *v21; // r14
  size_t v22; // r9
  __int64 *v23; // rcx
  __int64 *v24; // r13
  __int64 *v25; // r14
  char *v26; // rsi
  __int64 v27; // rbx
  const char *v28; // rax
  size_t v29; // rdx
  size_t v30; // r15
  const char *v31; // r12
  const char *v32; // rax
  size_t v33; // rdx
  size_t v34; // rbx
  int v35; // eax
  __int64 v36; // r12
  const char *v37; // rax
  size_t v38; // rdx
  size_t v39; // r15
  const char *v40; // rbx
  const char *v41; // rax
  size_t v42; // rdx
  size_t v43; // r12
  bool v44; // cc
  size_t v45; // rdx
  int v46; // eax
  char *v47; // rsi
  __int64 v48; // r15
  __int64 *v49; // rdx
  __int64 v50; // r14
  __int64 *v51; // rdx
  unsigned __int64 v52; // r13
  char *v53; // rax
  char *v54; // rsi
  unsigned __int64 v55; // r13
  char *v56; // rcx
  _BYTE *v57; // rax
  _BYTE *v58; // rsi
  size_t v59; // rbx
  int v61; // eax
  int v62; // r13d
  void *v63; // r10
  void *v64; // rax
  unsigned __int64 v65; // r13
  _BYTE *v66; // rax
  unsigned __int64 v67; // r14
  __int64 v68; // rax
  char *v69; // rcx
  _BYTE *v70; // rax
  _BYTE *v71; // rsi
  size_t v72; // rbx
  __int64 v73; // rax
  int v74; // eax
  char *v75; // [rsp+8h] [rbp-88h]
  size_t v76; // [rsp+10h] [rbp-80h]
  size_t v77; // [rsp+10h] [rbp-80h]
  size_t v78; // [rsp+10h] [rbp-80h]
  size_t nb; // [rsp+18h] [rbp-78h]
  size_t nc; // [rsp+18h] [rbp-78h]
  size_t n; // [rsp+18h] [rbp-78h]
  size_t nd; // [rsp+18h] [rbp-78h]
  size_t ne; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  size_t nf; // [rsp+18h] [rbp-78h]
  char *v86; // [rsp+20h] [rbp-70h]
  char *v87; // [rsp+20h] [rbp-70h]
  unsigned __int64 *v88; // [rsp+20h] [rbp-70h]
  bool v89; // [rsp+28h] [rbp-68h]
  __int64 *v91; // [rsp+30h] [rbp-60h]
  __int64 v92; // [rsp+38h] [rbp-58h]
  __int64 *v93; // [rsp+38h] [rbp-58h]
  unsigned __int64 v94; // [rsp+40h] [rbp-50h] BYREF
  char *v95; // [rsp+48h] [rbp-48h]
  char *v96; // [rsp+50h] [rbp-40h]

  v4 = a1;
  v5 = a2;
  v6 = *(char **)(a2 + 56);
  v7 = *(_DWORD *)(a2 + 40);
  v8 = *(char **)(a2 + 48);
  v92 = a3;
  v9 = v6 - v8;
  v86 = (char *)(v6 - v8);
  if ( v6 == v8 )
  {
    v12 = 0;
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_139;
    a1 = (unsigned __int64 *)(v6 - v8);
    v10 = sub_22077B0(v6 - v8);
    v6 = *(char **)(v5 + 56);
    v8 = *(char **)(v5 + 48);
    v11 = (unsigned __int64 *)v10;
    v9 = v6 - v8;
    v12 = v6 - v8;
  }
  if ( v8 != v6 )
  {
    a1 = v11;
    v76 = v9;
    nb = (size_t)v6;
    memmove(v11, v8, v12);
    v9 = v76;
    v6 = (char *)nb;
  }
  if ( v7 == *(_DWORD *)v92 )
  {
    a1 = *(unsigned __int64 **)(v92 + 8);
    if ( *(_QWORD *)(v92 + 16) - (_QWORD)a1 == v9 )
    {
      if ( v9 )
      {
        v8 = (char *)v11;
        if ( memcmp(a1, v11, v9) )
        {
          v13 = *(_DWORD *)(v5 + 40);
          goto LABEL_8;
        }
      }
      else if ( !v11 )
      {
        goto LABEL_77;
      }
      v8 = v86;
      a1 = v11;
      j_j___libc_free_0((unsigned __int64)v11);
      goto LABEL_77;
    }
  }
  v13 = *(_DWORD *)(v5 + 40);
  if ( !v9 )
  {
    v15 = 0;
    if ( v6 != v8 )
      goto LABEL_10;
    goto LABEL_65;
  }
LABEL_8:
  if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_139;
  a1 = (unsigned __int64 *)v12;
  v14 = sub_22077B0(v12);
  v8 = *(char **)(v5 + 48);
  v15 = (void *)v14;
  v9 = *(_QWORD *)(v5 + 56) - (_QWORD)v8;
  if ( *(char **)(v5 + 56) != v8 )
  {
LABEL_10:
    nc = v9;
    v16 = memmove(v15, v8, v9);
    v89 = 0;
    v9 = nc;
    v15 = v16;
    if ( *(_DWORD *)a4 != v13 )
    {
LABEL_11:
      v8 = (char *)v12;
      a1 = (unsigned __int64 *)v15;
      j_j___libc_free_0((unsigned __int64)v15);
      goto LABEL_12;
    }
    goto LABEL_92;
  }
LABEL_65:
  if ( *(_DWORD *)a4 != v13 )
  {
LABEL_66:
    v89 = 0;
    goto LABEL_67;
  }
LABEL_92:
  a1 = *(unsigned __int64 **)(a4 + 8);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)a1 != v9 )
    goto LABEL_66;
  if ( v9 )
  {
    nd = (size_t)v15;
    v61 = memcmp(a1, v15, v9);
    v15 = (void *)nd;
    v89 = v61 == 0;
    goto LABEL_11;
  }
  v89 = 1;
LABEL_67:
  if ( v15 )
    goto LABEL_11;
LABEL_12:
  if ( v11 )
  {
    v8 = v86;
    a1 = v11;
    j_j___libc_free_0((unsigned __int64)v11);
  }
  if ( v89 )
  {
LABEL_77:
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 40);
    v55 = *(_QWORD *)(v5 + 56) - *(_QWORD *)(v5 + 48);
    v4[1] = 0;
    v4[2] = 0;
    v4[3] = 0;
    if ( v55 )
    {
      if ( v55 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_139;
      v56 = (char *)sub_22077B0(v55);
    }
    else
    {
      v55 = 0;
      v56 = 0;
    }
    v4[1] = (unsigned __int64)v56;
    v4[2] = (unsigned __int64)v56;
    v4[3] = (unsigned __int64)&v56[v55];
    v57 = *(_BYTE **)(v5 + 56);
    v58 = *(_BYTE **)(v5 + 48);
    v59 = v57 - v58;
    if ( v57 == v58 )
    {
LABEL_82:
      v4[2] = (unsigned __int64)&v56[v59];
      return v4;
    }
LABEL_81:
    v56 = (char *)memmove(v56, v58, v59);
    goto LABEL_82;
  }
  v17 = *(char **)(v5 + 24);
  v8 = *(char **)(v5 + 16);
  v18 = *(_DWORD *)(v5 + 8);
  v19 = v17 - v8;
  v87 = (char *)(v17 - v8);
  if ( v17 == v8 )
  {
    v22 = 0;
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_139;
    v20 = sub_22077B0(v19);
    v17 = *(char **)(v5 + 24);
    v8 = *(char **)(v5 + 16);
    v21 = (void *)v20;
    v19 = v17 - v8;
    v22 = v17 - v8;
  }
  if ( v8 == v17 )
  {
    if ( *(_DWORD *)v92 != v18 || (a1 = *(unsigned __int64 **)(v92 + 8), *(_QWORD *)(v92 + 16) - (_QWORD)a1 != v19) )
    {
      if ( !v21 )
        goto LABEL_21;
      goto LABEL_20;
    }
  }
  else
  {
    v75 = v17;
    v77 = v22;
    memmove(v21, v8, v22);
    v22 = v77;
    v17 = v75;
    if ( v18 != *(_DWORD *)v92 )
      goto LABEL_20;
    a1 = *(unsigned __int64 **)(v92 + 8);
    if ( v19 != *(_QWORD *)(v92 + 16) - (_QWORD)a1 )
      goto LABEL_20;
  }
  if ( !v19 )
  {
    v62 = *(_DWORD *)(v5 + 8);
    v63 = 0;
    goto LABEL_103;
  }
  v8 = (char *)v21;
  na = v22;
  if ( memcmp(a1, v21, v19) )
  {
LABEL_20:
    j_j___libc_free_0((unsigned __int64)v21);
    goto LABEL_21;
  }
  v62 = *(_DWORD *)(v5 + 8);
  if ( na > 0x7FFFFFFFFFFFFFF8LL )
    goto LABEL_139;
  a1 = (unsigned __int64 *)na;
  v73 = sub_22077B0(na);
  v17 = *(char **)(v5 + 24);
  v8 = *(char **)(v5 + 16);
  v22 = na;
  v63 = (void *)v73;
  v19 = v17 - v8;
LABEL_103:
  if ( v8 == v17 )
  {
    if ( *(_DWORD *)a4 != v62 )
      goto LABEL_133;
  }
  else
  {
    ne = v22;
    v64 = memmove(v63, v8, v19);
    v22 = ne;
    v63 = v64;
    if ( *(_DWORD *)a4 != v62 )
    {
LABEL_105:
      v8 = (char *)v22;
      a1 = (unsigned __int64 *)v63;
      j_j___libc_free_0((unsigned __int64)v63);
      goto LABEL_106;
    }
  }
  a1 = *(unsigned __int64 **)(a4 + 8);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)a1 == v19 )
  {
    if ( v19 )
    {
      v78 = v22;
      nf = (size_t)v63;
      v74 = memcmp(a1, v63, v19);
      v63 = (void *)nf;
      v22 = v78;
      v89 = v74 == 0;
      goto LABEL_105;
    }
    v89 = 1;
  }
LABEL_133:
  if ( v63 )
    goto LABEL_105;
LABEL_106:
  if ( v21 )
  {
    v8 = v87;
    a1 = (unsigned __int64 *)v21;
    j_j___libc_free_0((unsigned __int64)v21);
  }
  if ( v89 )
  {
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 8);
    v65 = *(_QWORD *)(v5 + 24) - *(_QWORD *)(v5 + 16);
    v4[1] = 0;
    v4[2] = 0;
    v4[3] = 0;
    if ( !v65 )
    {
      v56 = 0;
LABEL_112:
      v4[1] = (unsigned __int64)v56;
      v4[2] = (unsigned __int64)v56;
      v4[3] = (unsigned __int64)&v56[v65];
      v66 = *(_BYTE **)(v5 + 24);
      v58 = *(_BYTE **)(v5 + 16);
      v59 = v66 - v58;
      if ( v66 == v58 )
        goto LABEL_82;
      goto LABEL_81;
    }
    if ( v65 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v56 = (char *)sub_22077B0(v65);
      goto LABEL_112;
    }
LABEL_139:
    sub_4261EA(a1, v8, a3);
  }
LABEL_21:
  v94 = 0;
  v95 = 0;
  v23 = *(__int64 **)(a4 + 16);
  v24 = *(__int64 **)(a4 + 8);
  v96 = 0;
  v91 = v23;
  a1 = *(unsigned __int64 **)(v92 + 16);
  v25 = *(__int64 **)(v92 + 8);
  v93 = (__int64 *)a1;
  if ( v24 != v23 && v25 != (__int64 *)a1 )
  {
    v88 = v4;
    n = v5;
    while ( 1 )
    {
      v27 = *v25;
      v28 = sub_BD5D20(*v24);
      v30 = v29;
      v31 = v28;
      v32 = sub_BD5D20(v27);
      v34 = v33;
      a1 = (unsigned __int64 *)v32;
      if ( v30 <= v33 )
        v33 = v30;
      if ( v33 && (v35 = memcmp(v32, v31, v33)) != 0 )
      {
        if ( v35 < 0 )
        {
          v26 = v95;
          if ( v95 == v96 )
            goto LABEL_86;
LABEL_25:
          if ( v26 )
          {
            *(_QWORD *)v26 = *v25;
            v26 = v95;
          }
          v95 = v26 + 8;
LABEL_28:
          if ( v93 == ++v25 )
            goto LABEL_47;
          goto LABEL_29;
        }
      }
      else if ( v30 != v34 && v30 > v34 )
      {
        v26 = v95;
        if ( v95 != v96 )
          goto LABEL_25;
LABEL_86:
        a1 = &v94;
        sub_9CC5C0((__int64)&v94, v26, v25);
        goto LABEL_28;
      }
      v36 = *v24;
      v37 = sub_BD5D20(*v25);
      v39 = v38;
      v40 = v37;
      v41 = sub_BD5D20(v36);
      v43 = v42;
      v44 = v42 <= v39;
      v45 = v39;
      a1 = (unsigned __int64 *)v41;
      if ( v44 )
        v45 = v43;
      if ( v45 && (v46 = memcmp(v41, v40, v45)) != 0 )
      {
        if ( v46 < 0 )
          goto LABEL_42;
      }
      else if ( v43 != v39 && v43 < v39 )
      {
LABEL_42:
        v47 = v95;
        if ( v95 == v96 )
        {
          a1 = &v94;
          sub_9CC5C0((__int64)&v94, v95, v24);
        }
        else
        {
          if ( v95 )
          {
            *(_QWORD *)v95 = *v24;
            v47 = v95;
          }
          v95 = v47 + 8;
        }
        goto LABEL_46;
      }
      v54 = v95;
      if ( v95 == v96 )
      {
        a1 = &v94;
        sub_9CC5C0((__int64)&v94, v95, v25);
      }
      else
      {
        if ( v95 )
        {
          *(_QWORD *)v95 = *v25;
          v54 = v95;
        }
        v95 = v54 + 8;
      }
      ++v25;
LABEL_46:
      ++v24;
      if ( v93 == v25 )
      {
LABEL_47:
        v4 = v88;
        v5 = n;
        v8 = v95;
        goto LABEL_48;
      }
LABEL_29:
      if ( v91 == v24 )
        goto LABEL_47;
    }
  }
  v8 = 0;
LABEL_48:
  v48 = v93 - v25;
  if ( (char *)v93 - (char *)v25 > 0 )
  {
    do
    {
      while ( v96 == v8 )
      {
        v49 = v25;
        a1 = &v94;
        ++v25;
        sub_9CC5C0((__int64)&v94, v8, v49);
        v8 = v95;
        if ( !--v48 )
          goto LABEL_55;
      }
      if ( v8 )
      {
        *(_QWORD *)v8 = *v25;
        v8 = v95;
      }
      v8 += 8;
      ++v25;
      v95 = v8;
      --v48;
    }
    while ( v48 );
  }
LABEL_55:
  v50 = v91 - v24;
  if ( (char *)v91 - (char *)v24 > 0 )
  {
    do
    {
      while ( v8 == v96 )
      {
        v51 = v24;
        a1 = &v94;
        ++v24;
        sub_9CC5C0((__int64)&v94, v8, v51);
        v8 = v95;
        if ( !--v50 )
          goto LABEL_62;
      }
      if ( v8 )
      {
        *(_QWORD *)v8 = *v24;
        v8 = v95;
      }
      v8 += 8;
      ++v24;
      v95 = v8;
      --v50;
    }
    while ( v50 );
  }
LABEL_62:
  v52 = v94;
  a3 = (unsigned int)qword_4FEFCA8;
  if ( (unsigned int)qword_4FEFCA8 < (unsigned __int64)((__int64)&v8[-v94] >> 3) )
  {
    *(_DWORD *)v4 = *(_DWORD *)(v5 + 40);
    v67 = *(_QWORD *)(v5 + 56) - *(_QWORD *)(v5 + 48);
    v4[1] = 0;
    v4[2] = 0;
    v4[3] = 0;
    if ( v67 )
    {
      if ( v67 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_139;
      v68 = sub_22077B0(v67);
      v52 = v94;
      v69 = (char *)v68;
    }
    else
    {
      v67 = 0;
      v69 = 0;
    }
    v4[1] = (unsigned __int64)v69;
    v4[2] = (unsigned __int64)v69;
    v4[3] = (unsigned __int64)&v69[v67];
    v70 = *(_BYTE **)(v5 + 56);
    v71 = *(_BYTE **)(v5 + 48);
    v72 = v70 - v71;
    if ( v70 != v71 )
      v69 = (char *)memmove(v69, v71, v72);
    v4[2] = (unsigned __int64)&v69[v72];
    if ( v52 )
      j_j___libc_free_0(v52);
  }
  else
  {
    v53 = v96;
    *(_DWORD *)v4 = 1;
    v4[1] = v52;
    v4[2] = (unsigned __int64)v8;
    v4[3] = (unsigned __int64)v53;
  }
  return v4;
}
