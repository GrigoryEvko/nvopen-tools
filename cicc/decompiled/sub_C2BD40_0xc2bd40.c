// Function: sub_C2BD40
// Address: 0xc2bd40
//
__int64 __fastcall sub_C2BD40(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rdx
  __int64 v5; // rcx
  char v6; // al
  char **v7; // rbx
  char *v8; // r14
  size_t v9; // r15
  char *v10; // r12
  int v11; // eax
  int v12; // r11d
  int v13; // r8d
  unsigned int i; // r10d
  size_t v15; // rax
  const void *v16; // rcx
  unsigned int v17; // r10d
  _QWORD *v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  __int64 result; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // r14
  __int64 v24; // rbx
  __int64 *v25; // r15
  __int64 *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rsi
  unsigned int v29; // r8d
  __int64 *v30; // rax
  __int64 v31; // r9
  bool v32; // al
  int v33; // eax
  char **v34; // rbx
  char **v35; // r15
  int v36; // eax
  size_t v37; // rdx
  __int64 v38; // rsi
  int v39; // eax
  unsigned int v40; // ecx
  __int64 v41; // rdi
  unsigned __int64 v42; // r13
  char *v43; // r12
  char *v44; // r14
  __int64 v45; // rax
  char *v46; // rdx
  size_t *v47; // rax
  size_t v48; // r10
  __int64 v49; // rax
  char *v50; // r13
  int v51; // r14d
  __int64 v52; // r12
  int v53; // r14d
  int v54; // eax
  int v55; // r9d
  unsigned int j; // ecx
  __int64 v57; // rax
  const void *v58; // rsi
  int v59; // eax
  bool v60; // al
  unsigned int v61; // ecx
  int v62; // r9d
  __int64 v63; // rax
  char *v64; // rdx
  char *v65; // rax
  const void *v66; // r8
  const void *v67; // r11
  char *v68; // r13
  size_t v69; // rdx
  const void *v70; // rdi
  const void *v71; // rsi
  __int64 v72; // rax
  int v73; // eax
  char *v74; // r9
  char *v75; // rcx
  char *v76; // r8
  size_t v77; // rdx
  const void *v78; // rdi
  const void *v79; // rsi
  int v80; // eax
  int v81; // eax
  int v82; // ecx
  __int64 *v83; // rax
  __int64 *v84; // r12
  __int64 *v85; // rbx
  __int64 v86; // rax
  int v87; // esi
  __int64 v88; // rdx
  __int64 *v89; // rcx
  __int64 v90; // rdi
  __int64 *v91; // rax
  __int64 *v92; // r15
  __int64 v93; // rsi
  __int64 *v94; // rbx
  __int64 v95; // rdx
  __int64 v96; // rsi
  unsigned int v97; // r9d
  __int64 *v98; // rax
  __int64 v99; // r8
  int v100; // eax
  int v101; // ecx
  int v102; // r11d
  __int64 *v103; // r10
  int v104; // edx
  char *v105; // [rsp+0h] [rbp-150h]
  const void *v106; // [rsp+8h] [rbp-148h]
  char *v107; // [rsp+8h] [rbp-148h]
  int v108; // [rsp+10h] [rbp-140h]
  const void *v109; // [rsp+10h] [rbp-140h]
  unsigned int v110; // [rsp+18h] [rbp-138h]
  size_t v111; // [rsp+18h] [rbp-138h]
  size_t v112; // [rsp+18h] [rbp-138h]
  size_t v113; // [rsp+18h] [rbp-138h]
  char **v114; // [rsp+20h] [rbp-130h]
  int v115; // [rsp+20h] [rbp-130h]
  const void *v116; // [rsp+20h] [rbp-130h]
  char *v117; // [rsp+20h] [rbp-130h]
  size_t n; // [rsp+28h] [rbp-128h]
  size_t na; // [rsp+28h] [rbp-128h]
  size_t nd; // [rsp+28h] [rbp-128h]
  size_t ne; // [rsp+28h] [rbp-128h]
  unsigned int nb; // [rsp+28h] [rbp-128h]
  size_t nf; // [rsp+28h] [rbp-128h]
  size_t nc; // [rsp+28h] [rbp-128h]
  int v125; // [rsp+30h] [rbp-120h]
  __int64 v126; // [rsp+30h] [rbp-120h]
  int v127; // [rsp+30h] [rbp-120h]
  char **v128; // [rsp+30h] [rbp-120h]
  __int64 v130; // [rsp+38h] [rbp-118h]
  __int64 v131; // [rsp+38h] [rbp-118h]
  __int64 v132; // [rsp+40h] [rbp-110h]
  __int64 v134; // [rsp+48h] [rbp-108h]
  __int64 v135; // [rsp+48h] [rbp-108h]
  __int64 v136[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v137; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v138; // [rsp+68h] [rbp-E8h]
  __int64 v139; // [rsp+70h] [rbp-E0h]
  __int64 v140; // [rsp+78h] [rbp-D8h]
  __int64 *v141; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v142; // [rsp+88h] [rbp-C8h]
  _QWORD *v143; // [rsp+90h] [rbp-C0h]
  _QWORD *v144; // [rsp+98h] [rbp-B8h]

  v132 = *(_QWORD *)(a1 + 208);
  if ( *(_QWORD *)(a1 + 88) )
  {
    v4 = *(_QWORD **)(a2 + 8);
    v5 = *(_QWORD *)a2;
    if ( *(_DWORD *)(a2 + 16) )
    {
      v144 = &v4[2 * *(unsigned int *)(a2 + 24)];
      v141 = (__int64 *)a2;
      v142 = v5;
      v143 = v4;
      sub_C25EF0((__int64)&v141);
      v22 = v143;
      v23 = v144;
      v24 = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
      while ( (_QWORD *)v24 != v22 )
      {
        sub_EF75F0(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL), *v22, v22[1]);
        do
          v22 += 2;
        while ( v22 != v23 && (*v22 == -1 || *v22 == -2) );
      }
    }
  }
  v6 = *(_BYTE *)(a1 + 204);
  if ( !*(_BYTE *)(a1 + 178) )
  {
    if ( v6 )
    {
      v18 = *(_QWORD **)(a2 + 8);
      v19 = *(_QWORD *)a2;
      v20 = &v18[2 * *(unsigned int *)(a2 + 24)];
      if ( *(_DWORD *)(a2 + 16) )
      {
        v141 = (__int64 *)a2;
        v144 = v20;
        v142 = v19;
        v143 = v18;
        sub_C25EF0((__int64)&v141);
        v25 = v144;
        v26 = v143;
        na = *(_QWORD *)(a2 + 8) + 16LL * *(unsigned int *)(a2 + 24);
        if ( v143 != (_QWORD *)na )
        {
          do
          {
            v126 = v26[1];
            v130 = *v26;
            sub_C7D030(&v141);
            sub_C7D280(&v141, v130, v126);
            sub_C7D290(&v141, &v137);
            v27 = *(unsigned int *)(a1 + 456);
            v28 = *(_QWORD *)(a1 + 440);
            if ( (_DWORD)v27 )
            {
              v29 = v137 & (v27 - 1);
              v30 = (__int64 *)(v28 + 16LL * v29);
              v31 = *v30;
              if ( v137 == *v30 )
              {
LABEL_27:
                if ( v30 != (__int64 *)(v28 + 16 * v27) )
                {
                  result = sub_C29BE0(a1, v30[1] + v132, a3);
                  if ( (_DWORD)result )
                    return result;
                }
              }
              else
              {
                v81 = 1;
                while ( v31 != -1 )
                {
                  v82 = v81 + 1;
                  v29 = (v27 - 1) & (v81 + v29);
                  v30 = (__int64 *)(v28 + 16LL * v29);
                  v31 = *v30;
                  if ( v137 == *v30 )
                    goto LABEL_27;
                  v81 = v82;
                }
              }
            }
            for ( v26 += 2; v25 != v26; v26 += 2 )
            {
              if ( (unsigned __int64)*v26 < 0xFFFFFFFFFFFFFFFELL )
                break;
            }
          }
          while ( v26 != (__int64 *)na );
        }
      }
    }
    else if ( *(_QWORD *)(a1 + 88) )
    {
      v7 = *(char ***)(a1 + 464);
      v114 = *(char ***)(a1 + 472);
      if ( v114 != v7 )
      {
        while ( 1 )
        {
          v8 = *v7;
          v9 = (size_t)v7[1];
          v10 = v7[5];
          if ( !*v7 )
            v9 = 0;
          v125 = *(_DWORD *)(a2 + 24);
          if ( v125 )
            break;
LABEL_37:
          if ( sub_EF7600(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL), v8, v9) )
          {
LABEL_38:
            result = sub_C29BE0(a1, (__int64)&v10[v132], a3);
            if ( (_DWORD)result )
              return result;
          }
          v7 += 6;
          if ( v114 == v7 )
            goto LABEL_16;
        }
        n = *(_QWORD *)(a2 + 8);
        v11 = sub_C94890(v8, v9);
        v12 = 1;
        v13 = v125 - 1;
        for ( i = (v125 - 1) & v11; ; i = v13 & v17 )
        {
          v15 = n + 16LL * i;
          v16 = *(const void **)v15;
          if ( *(_QWORD *)v15 == -1 )
          {
            v32 = v8 + 1 == 0;
          }
          else if ( v16 == (const void *)-2LL )
          {
            v32 = v8 + 2 == 0;
          }
          else
          {
            if ( v9 != *(_QWORD *)(v15 + 8) )
              goto LABEL_14;
            v108 = v12;
            v110 = i;
            v127 = v13;
            if ( !v9 )
              goto LABEL_38;
            v106 = *(const void **)v15;
            v33 = memcmp(v8, v16, v9);
            v16 = v106;
            v13 = v127;
            i = v110;
            v12 = v108;
            v32 = v33 == 0;
          }
          if ( v32 )
            goto LABEL_38;
          if ( v16 == (const void *)-1LL )
            goto LABEL_37;
LABEL_14:
          v17 = v12 + i;
          ++v12;
        }
      }
    }
    else if ( *(_DWORD *)(a2 + 16) )
    {
      v91 = *(__int64 **)(a2 + 8);
      v92 = &v91[2 * *(unsigned int *)(a2 + 24)];
      if ( v91 != v92 )
      {
        while ( 1 )
        {
          v93 = *v91;
          v94 = v91;
          if ( *v91 != -1 && v93 != -2 )
            break;
          v91 += 2;
          if ( v92 == v91 )
            goto LABEL_16;
        }
        if ( v91 != v92 )
        {
          do
          {
            v131 = v94[1];
            sub_C7D030(&v141);
            sub_C7D280(&v141, v93, v131);
            sub_C7D290(&v141, &v137);
            v95 = *(unsigned int *)(a1 + 456);
            v96 = *(_QWORD *)(a1 + 440);
            if ( (_DWORD)v95 )
            {
              v97 = v137 & (v95 - 1);
              v98 = (__int64 *)(v96 + 16LL * v97);
              v99 = *v98;
              if ( *v98 == v137 )
              {
LABEL_147:
                if ( v98 != (__int64 *)(v96 + 16 * v95) )
                {
                  result = sub_C29BE0(a1, v98[1] + v132, a3);
                  if ( (_DWORD)result )
                    return result;
                }
              }
              else
              {
                v100 = 1;
                while ( v99 != -1 )
                {
                  v101 = v100 + 1;
                  v97 = (v95 - 1) & (v100 + v97);
                  v98 = (__int64 *)(v96 + 16LL * v97);
                  v99 = *v98;
                  if ( v137 == *v98 )
                    goto LABEL_147;
                  v100 = v101;
                }
              }
            }
            v94 += 2;
            if ( v94 == v92 )
              break;
            while ( 1 )
            {
              v93 = *v94;
              if ( *v94 != -1 && v93 != -2 )
                break;
              v94 += 2;
              if ( v92 == v94 )
                goto LABEL_16;
            }
          }
          while ( v94 != v92 );
        }
      }
    }
LABEL_16:
    sub_C1AFD0();
    return 0;
  }
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  if ( v6 )
  {
    if ( *(_DWORD *)(a2 + 16) )
    {
      v83 = *(__int64 **)(a2 + 8);
      v84 = &v83[2 * *(unsigned int *)(a2 + 24)];
      if ( v83 != v84 )
      {
        while ( 1 )
        {
          v85 = v83;
          if ( *v83 != -1 && *v83 != -2 )
            break;
          v83 += 2;
          if ( v84 == v83 )
            goto LABEL_45;
        }
        if ( v84 != v83 )
        {
          while ( 1 )
          {
            v86 = sub_B2F650(*v85, v85[1]);
            v87 = v140;
            v136[0] = v86;
            if ( !(_DWORD)v140 )
              break;
            LODWORD(v88) = (v140 - 1) & (((0xBF58476D1CE4E5B9LL * v86) >> 31) ^ (484763065 * v86));
            v89 = (__int64 *)(v138 + 8LL * (unsigned int)v88);
            v90 = *v89;
            if ( v86 == *v89 )
              goto LABEL_128;
            v102 = 1;
            v103 = 0;
            while ( v90 != -1 )
            {
              if ( !v103 && v90 == -2 )
                v103 = v89;
              v88 = ((_DWORD)v140 - 1) & (unsigned int)(v88 + v102);
              v89 = (__int64 *)(v138 + 8 * v88);
              v90 = *v89;
              if ( v86 == *v89 )
                goto LABEL_128;
              ++v102;
            }
            if ( !v103 )
              v103 = v89;
            ++v137;
            v104 = v139 + 1;
            v141 = v103;
            if ( 4 * ((int)v139 + 1) >= (unsigned int)(3 * v140) )
              goto LABEL_170;
            if ( (int)v140 - HIDWORD(v139) - v104 <= (unsigned int)v140 >> 3 )
              goto LABEL_171;
LABEL_166:
            LODWORD(v139) = v104;
            if ( *v103 != -1 )
              --HIDWORD(v139);
            *v103 = v86;
LABEL_128:
            v85 += 2;
            if ( v85 != v84 )
            {
              while ( *v85 == -1 || *v85 == -2 )
              {
                v85 += 2;
                if ( v84 == v85 )
                  goto LABEL_45;
              }
              if ( v84 != v85 )
                continue;
            }
            goto LABEL_45;
          }
          ++v137;
          v141 = 0;
LABEL_170:
          v87 = 2 * v140;
LABEL_171:
          sub_A32210((__int64)&v137, v87);
          sub_A27FA0((__int64)&v137, v136, &v141);
          v86 = v136[0];
          v103 = v141;
          v104 = v139 + 1;
          goto LABEL_166;
        }
      }
    }
  }
LABEL_45:
  v34 = *(char ***)(a1 + 464);
  v35 = 0;
  v128 = *(char ***)(a1 + 472);
  if ( v128 == v34 )
  {
LABEL_96:
    sub_C7D6A0(v138, 8LL * (unsigned int)v140, 8);
    goto LABEL_16;
  }
  v134 = a1;
  while ( 1 )
  {
    v50 = *v34;
    v37 = (size_t)v34[1];
    if ( !*(_BYTE *)(v134 + 204) )
    {
      if ( !v50 )
        v37 = 0;
      goto LABEL_59;
    }
    if ( !v50 )
    {
      v36 = v140;
      v38 = v138;
      if ( !(_DWORD)v140 )
        goto LABEL_71;
LABEL_49:
      v39 = v36 - 1;
      v40 = v39 & (((0xBF58476D1CE4E5B9LL * v37) >> 31) ^ (484763065 * v37));
      v41 = *(_QWORD *)(v38 + 8LL * v40);
      if ( v41 == v37 )
        goto LABEL_50;
      v62 = 1;
      while ( v41 != -1 )
      {
        v40 = v39 & (v62 + v40);
        v41 = *(_QWORD *)(v38 + 8LL * v40);
        if ( v41 == v37 )
          goto LABEL_50;
        ++v62;
      }
      goto LABEL_70;
    }
    nd = (size_t)v34[1];
    sub_C7D030(&v141);
    sub_C7D280(&v141, v50, nd);
    sub_C7D290(&v141, v136);
    v36 = v140;
    v37 = v136[0];
    v38 = v138;
    if ( (_DWORD)v140 )
      goto LABEL_49;
LABEL_70:
    if ( *(_BYTE *)(v134 + 204) )
      goto LABEL_71;
    v37 = 0;
    v50 = 0;
LABEL_59:
    v51 = *(_DWORD *)(a2 + 24);
    if ( v51 )
      break;
LABEL_92:
    v72 = *(_QWORD *)(v134 + 88);
    if ( v72 && sub_EF7600(*(_QWORD *)(v72 + 8), v50, v37) )
      goto LABEL_50;
LABEL_71:
    if ( v34 == v35 )
      goto LABEL_54;
    if ( v35 )
    {
      v42 = (unsigned __int64)v35[3];
      if ( v42 <= (unsigned __int64)v34[3] )
      {
        v43 = v35[2];
        v44 = v34[2];
        v63 = 24 * v42 - 24;
        v64 = &v44[v63];
        v65 = &v43[v63];
        v66 = *(const void **)v65;
        v48 = *((_QWORD *)v65 + 1);
        v67 = *(const void **)v64;
        if ( v48 == *((_QWORD *)v64 + 1) )
          goto LABEL_75;
      }
    }
LABEL_55:
    v34 += 6;
    if ( v128 == v34 )
      goto LABEL_96;
  }
  v52 = *(_QWORD *)(a2 + 8);
  v53 = v51 - 1;
  ne = v37;
  v54 = sub_C94890(v50, v37);
  v37 = ne;
  v55 = 1;
  for ( j = v53 & v54; ; j = v53 & v61 )
  {
    v57 = v52 + 16LL * j;
    v58 = *(const void **)v57;
    if ( *(_QWORD *)v57 == -1 )
      break;
    if ( v58 == (const void *)-2LL )
    {
      v60 = v50 + 2 == 0;
    }
    else
    {
      if ( *(_QWORD *)(v57 + 8) != v37 )
        goto LABEL_67;
      v115 = v55;
      nb = j;
      if ( !v37 )
        goto LABEL_50;
      v111 = v37;
      v59 = memcmp(v50, v58, v37);
      v37 = v111;
      j = nb;
      v55 = v115;
      v60 = v59 == 0;
    }
    if ( v60 )
      goto LABEL_50;
LABEL_67:
    v61 = v55 + j;
    ++v55;
  }
  if ( v50 != (char *)-1LL )
    goto LABEL_92;
LABEL_50:
  if ( !v35 )
    goto LABEL_53;
  v42 = (unsigned __int64)v35[3];
  if ( v42 > (unsigned __int64)v34[3] )
    goto LABEL_53;
  v43 = v35[2];
  v44 = v34[2];
  v45 = 24 * v42 - 24;
  v46 = &v44[v45];
  v47 = (size_t *)&v43[v45];
  v48 = v47[1];
  if ( v48 != *((_QWORD *)v46 + 1) )
    goto LABEL_53;
  v66 = (const void *)*v47;
  v67 = *(const void **)v46;
  if ( *v47 != *(_QWORD *)v46 )
  {
    if ( !v67
      || !v66
      || (v112 = v47[1],
          v116 = *(const void **)v46,
          nf = *v47,
          v73 = memcmp((const void *)*v47, v67, v112),
          v66 = (const void *)nf,
          v67 = v116,
          v48 = v112,
          v73) )
    {
LABEL_53:
      v35 = v34;
      goto LABEL_54;
    }
  }
  if ( v43 != &v43[24 * v42 - 24] )
  {
    nc = (size_t)v66;
    v74 = v44;
    v75 = v43;
    v76 = &v43[24 * v42 - 24];
    while ( *((_DWORD *)v75 + 4) == *((_DWORD *)v74 + 4) )
    {
      if ( *((_DWORD *)v75 + 5) != *((_DWORD *)v74 + 5) )
        break;
      v77 = *((_QWORD *)v75 + 1);
      if ( v77 != *((_QWORD *)v74 + 1) )
        break;
      v78 = *(const void **)v75;
      v79 = *(const void **)v74;
      if ( *(_QWORD *)v75 != *(_QWORD *)v74 )
      {
        v105 = v75;
        v107 = v76;
        v109 = v67;
        v113 = v48;
        v117 = v74;
        if ( !v79 )
          break;
        if ( !v78 )
          break;
        v80 = memcmp(v78, v79, v77);
        v74 = v117;
        v48 = v113;
        v67 = v109;
        v76 = v107;
        v75 = v105;
        if ( v80 )
          break;
      }
      v75 += 24;
      v74 += 24;
      if ( v76 == v75 )
      {
        v66 = (const void *)nc;
        goto LABEL_112;
      }
    }
    goto LABEL_53;
  }
LABEL_112:
  if ( v34 != v35 )
  {
LABEL_75:
    if ( v67 == v66 || v67 && v66 && !memcmp(v66, v67, v48) )
    {
      v68 = &v43[24 * v42 - 24];
      if ( v68 == v43 )
        goto LABEL_54;
      while ( *((_DWORD *)v43 + 4) == *((_DWORD *)v44 + 4) )
      {
        if ( *((_DWORD *)v43 + 5) != *((_DWORD *)v44 + 5) )
          break;
        v69 = *((_QWORD *)v43 + 1);
        if ( v69 != *((_QWORD *)v44 + 1) )
          break;
        v70 = *(const void **)v43;
        v71 = *(const void **)v44;
        if ( *(_QWORD *)v43 != *(_QWORD *)v44 && (!v71 || !v70 || memcmp(v70, v71, v69)) )
          break;
        v43 += 24;
        v44 += 24;
        if ( v68 == v43 )
          goto LABEL_54;
      }
    }
    goto LABEL_55;
  }
LABEL_54:
  v49 = sub_C29DF0(v134, (__int64)&v34[5][v132]);
  if ( !(_DWORD)v49 )
    goto LABEL_55;
  v135 = v49;
  sub_C7D6A0(v138, 8LL * (unsigned int)v140, 8);
  return v135;
}
