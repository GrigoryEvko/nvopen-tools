// Function: sub_25A7990
// Address: 0x25a7990
//
void __fastcall sub_25A7990(unsigned __int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  size_t *v8; // rbx
  unsigned __int64 v9; // rdx
  _DWORD *v10; // r12
  size_t v11; // rcx
  _BYTE *v12; // rsi
  _BYTE *v13; // rax
  unsigned __int64 v14; // r8
  void *v15; // r13
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  _DWORD *v18; // r9
  _DWORD *v19; // rax
  bool v20; // al
  size_t v21; // rbx
  _BYTE *v22; // rax
  int v23; // r13d
  unsigned __int64 v24; // r15
  __int64 v25; // rax
  void *v26; // r8
  size_t v27; // rcx
  _BYTE *v28; // rax
  unsigned __int64 v29; // r9
  void *v30; // r13
  _BYTE *v31; // rax
  unsigned __int64 v32; // r9
  _DWORD *v33; // r10
  _DWORD *v34; // rax
  bool v35; // al
  _BYTE *v36; // rax
  int v37; // ebx
  unsigned __int64 v38; // r8
  void *v39; // r9
  size_t v40; // rbx
  size_t v41; // rax
  size_t v42; // rdx
  void *v43; // rdi
  __int64 v44; // rax
  unsigned int v45; // ecx
  _DWORD *v46; // r13
  __int64 v47; // rsi
  _BYTE *v48; // rax
  unsigned __int64 v49; // r15
  __int64 v50; // rax
  size_t v51; // r13
  __int64 v52; // rax
  __int64 v53; // rsi
  unsigned int v54; // ecx
  __int64 v55; // r8
  _BYTE *v56; // rax
  unsigned __int64 v57; // r13
  _BYTE *v58; // rax
  unsigned __int64 v59; // r13
  size_t v60; // r15
  int v61; // eax
  int v62; // eax
  unsigned int v63; // eax
  __int64 v64; // r8
  __int64 v65; // r9
  size_t v66; // rbx
  size_t v67; // rax
  size_t v68; // rdx
  void *v69; // rdi
  int v70; // eax
  size_t v71; // rdx
  int v72; // eax
  _BYTE *v73; // rax
  __int64 v74; // rax
  size_t v75; // r13
  void *v76; // rdi
  int v77; // r9d
  int v78; // r10d
  size_t n; // [rsp+18h] [rbp-78h]
  size_t nd; // [rsp+18h] [rbp-78h]
  size_t na; // [rsp+18h] [rbp-78h]
  size_t ne; // [rsp+18h] [rbp-78h]
  size_t nb; // [rsp+18h] [rbp-78h]
  size_t nf; // [rsp+18h] [rbp-78h]
  size_t nc; // [rsp+18h] [rbp-78h]
  size_t ng; // [rsp+18h] [rbp-78h]
  char *v87; // [rsp+28h] [rbp-68h]
  char *v88; // [rsp+28h] [rbp-68h]
  size_t v89; // [rsp+28h] [rbp-68h]
  int v90; // [rsp+30h] [rbp-60h]
  int v91; // [rsp+30h] [rbp-60h]
  bool v92; // [rsp+30h] [rbp-60h]
  bool v93; // [rsp+30h] [rbp-60h]
  int v94; // [rsp+30h] [rbp-60h]
  int v95; // [rsp+30h] [rbp-60h]
  bool v96; // [rsp+30h] [rbp-60h]
  bool v97; // [rsp+30h] [rbp-60h]
  size_t v98; // [rsp+30h] [rbp-60h]
  unsigned __int64 v99; // [rsp+30h] [rbp-60h]
  __int64 v100; // [rsp+30h] [rbp-60h]
  unsigned __int64 v101; // [rsp+30h] [rbp-60h]
  size_t v102; // [rsp+30h] [rbp-60h]
  size_t v103; // [rsp+30h] [rbp-60h]
  _DWORD *v104; // [rsp+30h] [rbp-60h]
  _DWORD *v105; // [rsp+30h] [rbp-60h]
  unsigned __int64 v106; // [rsp+30h] [rbp-60h]
  size_t v107; // [rsp+30h] [rbp-60h]
  unsigned int v108; // [rsp+38h] [rbp-58h]
  unsigned int v109; // [rsp+38h] [rbp-58h]
  unsigned __int64 v110; // [rsp+38h] [rbp-58h]
  unsigned __int64 v111; // [rsp+38h] [rbp-58h]
  unsigned int v112; // [rsp+40h] [rbp-50h] BYREF
  void *s1; // [rsp+48h] [rbp-48h]
  char *v114; // [rsp+50h] [rbp-40h]

  v8 = (size_t *)a1;
  if ( *(_BYTE *)a2 != 31 )
  {
    if ( *(_BYTE *)a2 != 32 )
    {
      v40 = *(_QWORD *)(a3 + 8);
      if ( v40 <= *(_QWORD *)(a3 + 16) )
      {
        if ( !v40 )
        {
LABEL_93:
          *(_QWORD *)(a3 + 8) = v40;
          return;
        }
        v43 = *(void **)a3;
        v42 = *(_QWORD *)(a3 + 8);
        goto LABEL_91;
      }
LABEL_158:
      *(_QWORD *)(a3 + 8) = 0;
      sub_C8D290(a3, (const void *)(a3 + 24), v40, 1u, a5, a6);
      if ( v40 )
        memset(*(void **)a3, 1, v40);
      goto LABEL_93;
    }
    v9 = **(_QWORD **)(a2 - 8) & 0xFFFFFFFFFFFFFFF9LL;
    if ( a4 )
    {
      a1 = (unsigned __int64)&v112;
      sub_25A73A0((__int64)&v112, v8, v9);
      v10 = s1;
      v27 = *v8;
      v109 = v112;
      v88 = v114;
      goto LABEL_32;
    }
    v52 = *(unsigned int *)(a1 + 32);
    v53 = *(_QWORD *)(a1 + 16);
    if ( (_DWORD)v52 )
    {
      v54 = (v52 - 1) & (v9 ^ (v9 >> 9));
      v55 = v53 + 40LL * v54;
      a1 = *(_QWORD *)v55;
      if ( v9 == *(_QWORD *)v55 )
      {
LABEL_75:
        if ( v55 != v53 + 40 * v52 )
        {
          v12 = *(_BYTE **)(v55 + 16);
          v100 = v55;
          v109 = *(_DWORD *)(v55 + 8);
          v56 = *(_BYTE **)(v55 + 24);
          v57 = v56 - v12;
          if ( v56 == v12 )
          {
            v9 = 0;
            v10 = 0;
          }
          else
          {
            if ( v57 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_183;
            v10 = (_DWORD *)sub_22077B0(v57);
            v56 = *(_BYTE **)(v100 + 24);
            v12 = *(_BYTE **)(v100 + 16);
            v9 = v56 - v12;
          }
          a1 = (unsigned __int64)v10 + v57;
          if ( v12 != v56 )
          {
            a1 = (unsigned __int64)v10;
            v101 = v9;
            memmove(v10, v12, v9);
            v9 = v101;
          }
          v27 = *v8;
          v88 = (char *)v10 + v9;
          goto LABEL_32;
        }
      }
      else
      {
        v78 = 1;
        while ( a1 != -2 )
        {
          v54 = (v52 - 1) & (v78 + v54);
          v55 = v53 + 40LL * v54;
          a1 = *(_QWORD *)v55;
          if ( v9 == *(_QWORD *)v55 )
            goto LABEL_75;
          ++v78;
        }
      }
    }
    v27 = *v8;
    v12 = *(_BYTE **)(*v8 + 80);
    v109 = *(_DWORD *)(*v8 + 72);
    v73 = *(_BYTE **)(*v8 + 88);
    v9 = v73 - v12;
    if ( v73 == v12 )
    {
      v75 = 0;
      v10 = 0;
    }
    else
    {
      v89 = *v8;
      if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_183;
      v106 = *(_QWORD *)(*v8 + 88) - (_QWORD)v12;
      v74 = sub_22077B0(v9);
      v9 = v106;
      v10 = (_DWORD *)v74;
      v73 = *(_BYTE **)(v89 + 88);
      v12 = *(_BYTE **)(v89 + 80);
      v27 = *v8;
      v75 = v73 - v12;
    }
    a1 = (unsigned __int64)v10 + v9;
    if ( v73 != v12 )
    {
      a1 = (unsigned __int64)v10;
      v107 = v27;
      memmove(v10, v12, v75);
      v27 = v107;
    }
    v88 = (char *)v10 + v75;
LABEL_32:
    v12 = *(_BYTE **)(v27 + 48);
    nb = v27;
    v94 = *(_DWORD *)(v27 + 40);
    v28 = *(_BYTE **)(v27 + 56);
    v29 = v28 - v12;
    if ( v28 == v12 )
    {
      v30 = 0;
    }
    else
    {
      if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_183;
      v30 = (void *)sub_22077B0(v29);
      v28 = *(_BYTE **)(nb + 56);
      v12 = *(_BYTE **)(nb + 48);
      v29 = v28 - v12;
    }
    if ( v28 != v12 )
    {
      nf = v29;
      memmove(v30, v12, v29);
      v29 = nf;
    }
    a1 = v109;
    if ( v94 == v109 && v88 - (char *)v10 == v29 )
    {
      if ( !v29 )
      {
        if ( !v30 )
          goto LABEL_118;
        goto LABEL_117;
      }
      a1 = (unsigned __int64)v10;
      if ( !memcmp(v10, v30, v29) )
      {
LABEL_117:
        j_j___libc_free_0((unsigned __int64)v30);
LABEL_118:
        v63 = sub_B46E30(a2);
        v66 = v63;
        if ( (unsigned __int64)v63 > *(_QWORD *)(a3 + 16) )
        {
          *(_QWORD *)(a3 + 8) = 0;
          sub_C8D290(a3, (const void *)(a3 + 24), v63, 1u, v64, v65);
          if ( v66 )
            memset(*(void **)a3, 1, v66);
        }
        else
        {
          v67 = *(_QWORD *)(a3 + 8);
          v68 = v67;
          if ( v66 <= v67 )
            v68 = v66;
          if ( v68 )
          {
            memset(*(void **)a3, 1, v68);
            v67 = *(_QWORD *)(a3 + 8);
          }
          if ( v66 > v67 )
          {
            v76 = (void *)(*(_QWORD *)a3 + v67);
            if ( v76 != (void *)(v66 + *(_QWORD *)a3) )
              memset(v76, 1, v66 - v67);
          }
        }
        *(_QWORD *)(a3 + 8) = v66;
        goto LABEL_99;
      }
    }
    v9 = *v8;
    v12 = *(_BYTE **)(*v8 + 80);
    nc = *v8;
    v95 = *(_DWORD *)(*v8 + 72);
    v31 = *(_BYTE **)(*v8 + 88);
    v32 = v31 - v12;
    if ( v31 == v12 )
    {
      v33 = 0;
    }
    else
    {
      if ( v32 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_183;
      v33 = (_DWORD *)sub_22077B0(v32);
      v31 = *(_BYTE **)(nc + 88);
      v12 = *(_BYTE **)(nc + 80);
      v32 = v31 - v12;
    }
    if ( v12 == v31 )
    {
      a1 = v109;
      if ( v95 != v109 )
        goto LABEL_107;
    }
    else
    {
      a1 = (unsigned __int64)v33;
      ng = v32;
      v34 = memmove(v33, v12, v32);
      v32 = ng;
      v33 = v34;
      v35 = 0;
      if ( v95 != v109 )
        goto LABEL_43;
    }
    if ( v32 == v88 - (char *)v10 )
    {
      if ( v32 )
      {
        v105 = v33;
        v62 = memcmp(v10, v33, v32);
        v33 = v105;
        v35 = v62 == 0;
        goto LABEL_43;
      }
      v35 = 1;
      goto LABEL_108;
    }
LABEL_107:
    v35 = 0;
LABEL_108:
    if ( !v33 )
    {
LABEL_44:
      if ( v30 )
      {
        a1 = (unsigned __int64)v30;
        v97 = v35;
        j_j___libc_free_0((unsigned __int64)v30);
        v35 = v97;
      }
      if ( v35 )
        goto LABEL_118;
      v9 = *v8;
      v36 = *(_BYTE **)(*v8 + 24);
      v12 = *(_BYTE **)(*v8 + 16);
      v98 = *v8;
      v37 = *(_DWORD *)(*v8 + 8);
      v38 = v36 - v12;
      if ( v36 == v12 )
      {
        v39 = 0;
      }
      else
      {
        if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_183;
        v39 = (void *)sub_22077B0(v38);
        v36 = *(_BYTE **)(v98 + 24);
        v12 = *(_BYTE **)(v98 + 16);
        v38 = v36 - v12;
      }
      if ( v36 == v12 )
      {
        if ( v37 == v109 )
        {
          v71 = v88 - (char *)v10;
          if ( v38 == v88 - (char *)v10 )
          {
LABEL_145:
            if ( v71 )
            {
              v111 = (unsigned __int64)v39;
              v72 = memcmp(v10, v39, v71);
              v39 = (void *)v111;
              if ( v72 )
              {
                j_j___libc_free_0(v111);
LABEL_148:
                j_j___libc_free_0((unsigned __int64)v10);
LABEL_54:
                v40 = (unsigned int)sub_B46E30(a2);
                if ( v40 <= *(_QWORD *)(a3 + 16) )
                {
                  v41 = *(_QWORD *)(a3 + 8);
                  v42 = v41;
                  if ( v40 <= v41 )
                    v42 = v40;
                  if ( !v42 )
                    goto LABEL_92;
                  v43 = *(void **)a3;
LABEL_91:
                  memset(v43, 1, v42);
                  v41 = *(_QWORD *)(a3 + 8);
LABEL_92:
                  if ( v40 > v41 )
                  {
                    v69 = (void *)(*(_QWORD *)a3 + v41);
                    if ( v69 != (void *)(v40 + *(_QWORD *)a3) )
                      memset(v69, 1, v40 - v41);
                  }
                  goto LABEL_93;
                }
                goto LABEL_158;
              }
            }
            else if ( !v39 )
            {
              goto LABEL_99;
            }
            j_j___libc_free_0((unsigned __int64)v39);
            goto LABEL_99;
          }
        }
        if ( !v39 )
        {
LABEL_53:
          if ( !v10 )
            goto LABEL_54;
          goto LABEL_148;
        }
      }
      else
      {
        v99 = v38;
        v39 = memmove(v39, v12, v38);
        if ( v37 == v109 )
        {
          v71 = v88 - (char *)v10;
          if ( v99 == v88 - (char *)v10 )
            goto LABEL_145;
        }
      }
      j_j___libc_free_0((unsigned __int64)v39);
      goto LABEL_53;
    }
LABEL_43:
    a1 = (unsigned __int64)v33;
    v96 = v35;
    j_j___libc_free_0((unsigned __int64)v33);
    v35 = v96;
    goto LABEL_44;
  }
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
  {
LABEL_27:
    **(_BYTE **)a3 = 1;
    return;
  }
  v9 = *(_QWORD *)(a2 - 96) & 0xFFFFFFFFFFFFFFF9LL;
  if ( a4 )
  {
    a1 = (unsigned __int64)&v112;
    sub_25A73A0((__int64)&v112, v8, v9);
    v10 = s1;
    v11 = *v8;
    v108 = v112;
    v87 = v114;
    goto LABEL_5;
  }
  v44 = *(unsigned int *)(a1 + 32);
  a1 = *(_QWORD *)(a1 + 16);
  if ( (_DWORD)v44 )
  {
    v45 = (v44 - 1) & (v9 ^ (v9 >> 9));
    v46 = (_DWORD *)(a1 + 40LL * v45);
    v47 = *(_QWORD *)v46;
    if ( *(_QWORD *)v46 == v9 )
    {
LABEL_61:
      if ( v46 != (_DWORD *)(a1 + 40 * v44) )
      {
        v12 = (_BYTE *)*((_QWORD *)v46 + 2);
        v108 = v46[2];
        v48 = (_BYTE *)*((_QWORD *)v46 + 3);
        v49 = v48 - v12;
        if ( v48 == v12 )
        {
          v51 = 0;
          v10 = 0;
        }
        else
        {
          if ( v49 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_183;
          v50 = sub_22077B0(v49);
          v12 = (_BYTE *)*((_QWORD *)v46 + 2);
          v10 = (_DWORD *)v50;
          v48 = (_BYTE *)*((_QWORD *)v46 + 3);
          v51 = v48 - v12;
        }
        a1 = (unsigned __int64)v10 + v49;
        if ( v48 != v12 )
        {
          a1 = (unsigned __int64)v10;
          memmove(v10, v12, v51);
        }
        v11 = *v8;
        v87 = (char *)v10 + v51;
        goto LABEL_5;
      }
    }
    else
    {
      v77 = 1;
      while ( v47 != -2 )
      {
        v45 = (v44 - 1) & (v77 + v45);
        v46 = (_DWORD *)(a1 + 40LL * v45);
        v47 = *(_QWORD *)v46;
        if ( *(_QWORD *)v46 == v9 )
          goto LABEL_61;
        ++v77;
      }
    }
  }
  v11 = *v8;
  v12 = *(_BYTE **)(*v8 + 80);
  v108 = *(_DWORD *)(*v8 + 72);
  v58 = *(_BYTE **)(*v8 + 88);
  v59 = v58 - v12;
  if ( v58 == v12 )
  {
    v60 = 0;
    v10 = 0;
  }
  else
  {
    v102 = *v8;
    if ( v59 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_183;
    v10 = (_DWORD *)sub_22077B0(v59);
    v58 = *(_BYTE **)(v102 + 88);
    v12 = *(_BYTE **)(v102 + 80);
    v11 = *v8;
    v60 = v58 - v12;
  }
  a1 = (unsigned __int64)v10 + v59;
  if ( v12 != v58 )
  {
    a1 = (unsigned __int64)v10;
    v103 = v11;
    memmove(v10, v12, v60);
    v11 = v103;
  }
  v87 = (char *)v10 + v60;
LABEL_5:
  v12 = *(_BYTE **)(v11 + 48);
  n = v11;
  v90 = *(_DWORD *)(v11 + 40);
  v13 = *(_BYTE **)(v11 + 56);
  v14 = v13 - v12;
  if ( v13 == v12 )
  {
    v15 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_183;
    v15 = (void *)sub_22077B0(v14);
    v13 = *(_BYTE **)(n + 56);
    v12 = *(_BYTE **)(n + 48);
    v14 = v13 - v12;
  }
  if ( v13 != v12 )
  {
    nd = v14;
    memmove(v15, v12, v14);
    v14 = nd;
  }
  a1 = v108;
  if ( v90 == v108 && v87 - (char *)v10 == v14 )
  {
    if ( !v14 )
    {
      if ( !v15 )
      {
LABEL_98:
        *(_BYTE *)(*(_QWORD *)a3 + 1LL) = 1;
        **(_BYTE **)a3 = 1;
        goto LABEL_99;
      }
LABEL_97:
      j_j___libc_free_0((unsigned __int64)v15);
      goto LABEL_98;
    }
    a1 = (unsigned __int64)v10;
    if ( !memcmp(v10, v15, v14) )
      goto LABEL_97;
  }
  v9 = *v8;
  v12 = *(_BYTE **)(*v8 + 80);
  na = *v8;
  v91 = *(_DWORD *)(*v8 + 72);
  v16 = *(_QWORD *)(*v8 + 88) - (_QWORD)v12;
  if ( v16 )
  {
    if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_183;
    v17 = sub_22077B0(v16);
    v9 = na;
    v18 = (_DWORD *)v17;
    v12 = *(_BYTE **)(na + 80);
    v16 = *(_QWORD *)(na + 88) - (_QWORD)v12;
    if ( *(_BYTE **)(na + 88) != v12 )
    {
LABEL_14:
      a1 = (unsigned __int64)v18;
      ne = v16;
      v19 = memmove(v18, v12, v16);
      v16 = ne;
      v18 = v19;
      v20 = 0;
      if ( v91 != v108 )
      {
LABEL_15:
        a1 = (unsigned __int64)v18;
        v92 = v20;
        j_j___libc_free_0((unsigned __int64)v18);
        v20 = v92;
        goto LABEL_16;
      }
      goto LABEL_102;
    }
  }
  else
  {
    v18 = 0;
    if ( *(_BYTE **)(*v8 + 88) != v12 )
      goto LABEL_14;
  }
  a1 = v108;
  if ( v91 != v108 )
  {
LABEL_70:
    v20 = 0;
    goto LABEL_71;
  }
LABEL_102:
  if ( v87 - (char *)v10 != v16 )
    goto LABEL_70;
  if ( v16 )
  {
    v104 = v18;
    v61 = memcmp(v10, v18, v16);
    v18 = v104;
    v20 = v61 == 0;
    goto LABEL_15;
  }
  v20 = 1;
LABEL_71:
  if ( v18 )
    goto LABEL_15;
LABEL_16:
  if ( v15 )
  {
    a1 = (unsigned __int64)v15;
    v93 = v20;
    j_j___libc_free_0((unsigned __int64)v15);
    v20 = v93;
  }
  if ( v20 )
    goto LABEL_98;
  v21 = *v8;
  v22 = *(_BYTE **)(v21 + 24);
  v12 = *(_BYTE **)(v21 + 16);
  v23 = *(_DWORD *)(v21 + 8);
  v24 = v22 - v12;
  if ( v22 != v12 )
  {
    if ( v24 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v25 = sub_22077B0(v24);
      v12 = *(_BYTE **)(v21 + 16);
      v26 = (void *)v25;
      v22 = *(_BYTE **)(v21 + 24);
      v24 = v22 - v12;
      goto LABEL_22;
    }
LABEL_183:
    sub_4261EA(a1, v12, v9);
  }
  v26 = 0;
LABEL_22:
  if ( v12 == v22 )
  {
    if ( v23 == v108 && v87 - (char *)v10 == v24 )
      goto LABEL_138;
    if ( !v26 )
    {
LABEL_25:
      if ( !v10 )
      {
LABEL_26:
        *(_BYTE *)(*(_QWORD *)a3 + 1LL) = 1;
        goto LABEL_27;
      }
LABEL_141:
      j_j___libc_free_0((unsigned __int64)v10);
      goto LABEL_26;
    }
LABEL_24:
    j_j___libc_free_0((unsigned __int64)v26);
    goto LABEL_25;
  }
  v26 = memmove(v26, v12, v24);
  if ( v23 != v108 || v24 != v87 - (char *)v10 )
    goto LABEL_24;
LABEL_138:
  if ( v24 )
  {
    v110 = (unsigned __int64)v26;
    v70 = memcmp(v10, v26, v24);
    v26 = (void *)v110;
    if ( v70 )
    {
      j_j___libc_free_0(v110);
      goto LABEL_141;
    }
  }
  else if ( !v26 )
  {
    goto LABEL_99;
  }
  j_j___libc_free_0((unsigned __int64)v26);
LABEL_99:
  if ( v10 )
    j_j___libc_free_0((unsigned __int64)v10);
}
