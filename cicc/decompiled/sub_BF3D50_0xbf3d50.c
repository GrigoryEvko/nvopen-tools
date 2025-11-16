// Function: sub_BF3D50
// Address: 0xbf3d50
//
__int64 __fastcall sub_BF3D50(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  const char *v14; // rax
  __int64 v15; // r10
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rbx
  const char *v20; // rax
  __int64 v21; // r15
  _BYTE *v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rcx
  _BYTE *v26; // rax
  _BYTE *v27; // r15
  __int64 v28; // r8
  _BYTE *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned int v33; // r14d
  int v34; // r13d
  const char *v35; // r15
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r15
  const char *v39; // rax
  __int64 v40; // r13
  _BYTE *v41; // rax
  char v42; // al
  __int64 v43; // rdi
  _BYTE *v44; // rax
  __int64 v45; // rdi
  _BYTE *v46; // rax
  int v47; // edx
  __int64 *v48; // rcx
  __int64 v49; // rax
  __int64 *v50; // rbx
  __int64 *v51; // r14
  __int64 v52; // rax
  __int64 *v53; // rdx
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r15
  unsigned int v58; // r14d
  int v59; // r13d
  const char *v60; // rax
  unsigned __int8 v61; // dl
  const char **v62; // rbx
  __int64 v63; // rax
  __int64 v64; // r15
  unsigned int v65; // r14d
  int v66; // r13d
  const char *v67; // rax
  unsigned __int8 v68; // dl
  const char **v69; // rbx
  __int64 v70; // rax
  __int64 *v71; // rcx
  _QWORD *v72; // rdx
  __int64 *v73; // rsi
  __int64 v74; // r14
  _BYTE *v75; // r13
  __int64 v76; // r15
  _BYTE *v77; // rax
  __int64 v78; // rax
  int v79; // eax
  __int64 v80; // rdx
  _QWORD *v81; // rax
  _QWORD *j; // rdx
  char v84; // al
  __int64 v85; // r15
  unsigned __int64 v86; // rdx
  __int64 v87; // rax
  _BYTE *v88; // r15
  __int64 v89; // r8
  _BYTE *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 v94; // r13
  _BYTE *v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdi
  _BYTE *v98; // rax
  __int64 v99; // r13
  _BYTE *v100; // rax
  __int64 v101; // rsi
  __int64 v102; // rdi
  _BYTE *v103; // rax
  unsigned int v104; // ecx
  unsigned int v105; // eax
  _QWORD *v106; // rdi
  int v107; // ebx
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rax
  _QWORD *v110; // rax
  __int64 v111; // rdx
  _QWORD *k; // rdx
  __int64 v113; // rsi
  __int64 v114; // rax
  __int64 v115; // rbx
  int v116; // r13d
  unsigned int v117; // r15d
  __int64 *v118; // rax
  __int64 *v119; // rdx
  __int64 *v120; // rcx
  const char **v121; // rax
  __int64 v122; // rdx
  const char **v123; // r15
  const char *v124; // r14
  const char **v125; // r13
  const char **v126; // rax
  const char **v127; // rax
  unsigned int v128; // eax
  __int64 v129; // rdx
  __int64 v130; // r13
  _BYTE *v131; // rax
  __int64 v132; // rax
  char v133; // dl
  __int64 v134; // rcx
  _BYTE *v135; // rdx
  __int64 v136; // rax
  __int64 v137; // r13
  _BYTE *v138; // rax
  _BYTE *v139; // rdx
  __int64 v140; // rax
  char v141; // al
  _QWORD *v142; // rax
  __int64 v143; // [rsp+8h] [rbp-A8h]
  __int64 v144; // [rsp+8h] [rbp-A8h]
  __int64 v145; // [rsp+8h] [rbp-A8h]
  __int64 v146; // [rsp+8h] [rbp-A8h]
  __int64 v147; // [rsp+8h] [rbp-A8h]
  _BYTE *v148; // [rsp+8h] [rbp-A8h]
  const char *v149[4]; // [rsp+10h] [rbp-A0h] BYREF
  char v150; // [rsp+30h] [rbp-80h]
  char v151; // [rsp+31h] [rbp-7Fh]
  const char *v152; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v153; // [rsp+48h] [rbp-68h]
  __int64 v154; // [rsp+50h] [rbp-60h]
  int v155; // [rsp+58h] [rbp-58h]
  char v156; // [rsp+5Ch] [rbp-54h]
  _QWORD v157[10]; // [rsp+60h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 152) = 0;
  v4 = *(_QWORD *)(v3 + 32);
  for ( i = v3 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      if ( *(_DWORD *)(v4 - 20) == 146 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( i == v4 )
        goto LABEL_9;
    }
    v6 = *(unsigned int *)(a1 + 1240);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1244) )
    {
      a2 = a1 + 1248;
      sub_C8D5F0(a1 + 1232, a1 + 1248, v6 + 1, 8);
      v6 = *(unsigned int *)(a1 + 1240);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1232) + 8 * v6) = v4 - 56;
    ++*(_DWORD *)(a1 + 1240);
  }
LABEL_9:
  if ( *(_DWORD *)(a1 + 848) )
  {
    a2 = *(_QWORD *)(a1 + 840);
    v134 = a2 + 16LL * *(unsigned int *)(a1 + 856);
    if ( a2 != v134 )
    {
      while ( 1 )
      {
        v135 = *(_BYTE **)a2;
        v136 = a2;
        if ( *(_QWORD *)a2 != -4096 && v135 != (_BYTE *)-8192LL )
          break;
        a2 += 16;
        if ( v134 == a2 )
          goto LABEL_10;
      }
      while ( v134 != v136 )
      {
        a2 = *(unsigned int *)(v136 + 12);
        if ( *(_DWORD *)(v136 + 8) < (unsigned int)a2 )
        {
          v137 = *(_QWORD *)a1;
          v148 = v135;
          v152 = "all indices passed to llvm.localrecover must be less than the number of arguments passed to llvm.locale"
                 "scape in the parent function";
          LOWORD(v157[0]) = 259;
          if ( v137 )
          {
            a2 = v137;
            sub_CA0E80(&v152, v137);
            v138 = *(_BYTE **)(v137 + 32);
            v139 = v148;
            if ( (unsigned __int64)v138 >= *(_QWORD *)(v137 + 24) )
            {
              a2 = 10;
              sub_CB5D20(v137, 10);
              v140 = *(_QWORD *)a1;
              v139 = v148;
            }
            else
            {
              *(_QWORD *)(v137 + 32) = v138 + 1;
              *v138 = 10;
              v140 = *(_QWORD *)a1;
            }
            *(_BYTE *)(a1 + 152) = 1;
            if ( v140 && v139 )
            {
              a2 = (__int64)v139;
              sub_BDBD80(a1, v139);
            }
          }
          else
          {
            *(_BYTE *)(a1 + 152) = 1;
          }
          break;
        }
        v136 += 16;
        if ( v136 == v134 )
          break;
        while ( 1 )
        {
          v135 = *(_BYTE **)v136;
          if ( *(_QWORD *)v136 != -4096 && v135 != (_BYTE *)-8192LL )
            break;
          v136 += 16;
          if ( v134 == v136 )
            goto LABEL_10;
        }
      }
    }
  }
LABEL_10:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)(v7 + 16);
  v9 = v7 + 8;
  if ( v7 + 8 != v8 )
  {
    do
    {
      a2 = v8 - 56;
      if ( !v8 )
        a2 = 0;
      sub_BE9ED0(a1, a2);
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v9 != v8 );
    v7 = *(_QWORD *)(a1 + 8);
  }
  v10 = *(_QWORD *)(v7 + 48);
  v11 = v7 + 40;
  if ( v7 + 40 == v10 )
    goto LABEL_35;
  do
  {
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      v12 = *(_BYTE *)(v10 - 16) & 0xF;
      if ( ((v12 + 9) & 0xFu) > 1 && v12 > 5u )
      {
        BYTE1(v157[0]) = 1;
        v14 = "Alias should have private, internal, linkonce, weak, linkonce_odr, weak_odr, external, or available_externally linkage!";
        goto LABEL_29;
      }
      v13 = *(_QWORD *)(v10 - 80);
      if ( v13 )
        break;
      a2 = (__int64)&v152;
      v152 = "Aliasee cannot be NULL!";
      LOWORD(v157[0]) = 259;
      sub_BDBF70((__int64 *)a1, (__int64)&v152);
      if ( *(_QWORD *)a1 )
        goto LABEL_33;
LABEL_22:
      v10 = *(_QWORD *)(v10 + 8);
      if ( v11 == v10 )
        goto LABEL_34;
    }
    if ( *(_QWORD *)(v10 - 40) == *(_QWORD *)(v13 + 8) )
    {
      if ( *(_BYTE *)v13 == 5 || *(_BYTE *)v13 <= 3u )
      {
        v153 = v157;
        v154 = 0x100000004LL;
        v155 = 0;
        v156 = 1;
        v157[0] = v10 - 48;
        v152 = (const char *)1;
        sub_BDCEC0(a1, (__int64)&v152, (_BYTE *)(v10 - 48), v13);
        if ( !v156 )
          _libc_free(v153, &v152);
        a2 = v10 - 48;
        sub_BE9180(a1, v10 - 48);
        goto LABEL_22;
      }
      BYTE1(v157[0]) = 1;
      v14 = "Aliasee should be either GlobalValue or ConstantExpr";
    }
    else
    {
      BYTE1(v157[0]) = 1;
      v14 = "Alias and aliasee types should match!";
    }
LABEL_29:
    v15 = *(_QWORD *)a1;
    v152 = v14;
    LOBYTE(v157[0]) = 3;
    if ( !v15 )
    {
      *(_BYTE *)(a1 + 152) = 1;
      goto LABEL_22;
    }
    a2 = v15;
    v143 = v15;
    sub_CA0E80(&v152, v15);
    v16 = *(_BYTE **)(v143 + 32);
    if ( (unsigned __int64)v16 >= *(_QWORD *)(v143 + 24) )
    {
      a2 = 10;
      sub_CB5D20(v143, 10);
    }
    else
    {
      *(_QWORD *)(v143 + 32) = v16 + 1;
      *v16 = 10;
    }
    v17 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( !v17 )
      goto LABEL_22;
LABEL_33:
    a2 = v10 - 48;
    sub_BDBD80(a1, (_BYTE *)(v10 - 48));
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v11 != v10 );
LABEL_34:
  v7 = *(_QWORD *)(a1 + 8);
LABEL_35:
  v18 = *(_QWORD *)(v7 + 64);
  v19 = v7 + 56;
  if ( v7 + 56 == v18 )
    goto LABEL_60;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( !v18 )
        BUG();
      v24 = *(_BYTE *)(v18 - 24) & 0xF;
      if ( (unsigned __int8)v24 > 8u || (v25 = 445, !_bittest64(&v25, v24)) )
      {
        BYTE1(v157[0]) = 1;
        v20 = "IFunc should have private, internal, linkonce, weak, linkonce_odr, weak_odr, or external linkage!";
LABEL_38:
        v21 = *(_QWORD *)a1;
        v152 = v20;
        LOBYTE(v157[0]) = 3;
        if ( !v21 )
          goto LABEL_58;
        a2 = v21;
        sub_CA0E80(&v152, v21);
        v22 = *(_BYTE **)(v21 + 32);
        if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
        {
          a2 = 10;
          sub_CB5D20(v21, 10);
        }
        else
        {
          *(_QWORD *)(v21 + 32) = v22 + 1;
          *v22 = 10;
        }
        v23 = *(_QWORD *)a1;
LABEL_42:
        *(_BYTE *)(a1 + 152) = 1;
        if ( !v23 )
        {
LABEL_44:
          v18 = *(_QWORD *)(v18 + 8);
          if ( v19 == v18 )
            goto LABEL_59;
          continue;
        }
LABEL_43:
        a2 = v18 - 56;
        sub_BDBD80(a1, (_BYTE *)(v18 - 56));
        goto LABEL_44;
      }
      break;
    }
    v26 = sub_B30850(v18 - 56);
    v27 = v26;
    if ( !v26 )
    {
      BYTE1(v157[0]) = 1;
      v39 = "IFunc must have a Function resolver";
      goto LABEL_177;
    }
    if ( (v26[32] & 0xF) == 1 || sub_B2FC80((__int64)v26) )
    {
      BYTE1(v157[0]) = 1;
      v20 = "IFunc resolver must be a definition";
      goto LABEL_38;
    }
    if ( *(_BYTE *)(**(_QWORD **)(*((_QWORD *)v27 + 3) + 16LL) + 8LL) == 14 )
    {
      v38 = *(_QWORD *)(*(_QWORD *)(v18 - 88) + 8LL);
      a2 = *(_DWORD *)(*(_QWORD *)(v18 - 48) + 8LL) >> 8;
      if ( v38 == sub_BCE3C0(*(__int64 **)(a1 + 144), a2) )
        goto LABEL_44;
      BYTE1(v157[0]) = 1;
      v39 = "IFunc resolver has incorrect type";
LABEL_177:
      a2 = (__int64)&v152;
      v152 = v39;
      LOBYTE(v157[0]) = 3;
      sub_BDBF70((__int64 *)a1, (__int64)&v152);
      if ( !*(_QWORD *)a1 )
        goto LABEL_44;
      goto LABEL_43;
    }
    v28 = *(_QWORD *)a1;
    v152 = "IFunc resolver must return a pointer";
    LOWORD(v157[0]) = 259;
    if ( v28 )
    {
      a2 = v28;
      v144 = v28;
      sub_CA0E80(&v152, v28);
      v29 = *(_BYTE **)(v144 + 32);
      if ( (unsigned __int64)v29 >= *(_QWORD *)(v144 + 24) )
      {
        a2 = 10;
        sub_CB5D20(v144, 10);
      }
      else
      {
        *(_QWORD *)(v144 + 32) = v29 + 1;
        *v29 = 10;
      }
      v23 = *(_QWORD *)a1;
      goto LABEL_42;
    }
LABEL_58:
    *(_BYTE *)(a1 + 152) = 1;
    v18 = *(_QWORD *)(v18 + 8);
    if ( v19 != v18 )
      continue;
    break;
  }
LABEL_59:
  v7 = *(_QWORD *)(a1 + 8);
LABEL_60:
  v30 = *(_QWORD *)(v7 + 80);
  v145 = v7 + 72;
  if ( v7 + 72 == v30 )
    goto LABEL_89;
  while ( 2 )
  {
    v31 = sub_B91B20(v30);
    if ( v32 <= 8
      || *(_QWORD *)v31 != 0x6762642E6D766C6CLL
      || *(_BYTE *)(v31 + 8) != 46
      || (v92 = sub_B91B20(v30), v93 == 11)
      && *(_QWORD *)v92 == 0x6762642E6D766C6CLL
      && *(_WORD *)(v92 + 8) == 25390
      && *(_BYTE *)(v92 + 10) == 117 )
    {
      v33 = 0;
      v34 = sub_B91A00(v30);
      if ( !v34 )
        goto LABEL_87;
      while ( 2 )
      {
        a2 = v33;
        v35 = (const char *)sub_B91A10(v30, v33);
        v36 = sub_B91B20(v30);
        if ( v37 == 11
          && *(_QWORD *)v36 == 0x6762642E6D766C6CLL
          && *(_WORD *)(v36 + 8) == 25390
          && *(_BYTE *)(v36 + 10) == 117 )
        {
          if ( !v35 || *v35 != 17 )
          {
            v40 = *(_QWORD *)a1;
            v152 = "invalid compile unit";
            LOWORD(v157[0]) = 259;
            if ( v40 )
            {
              sub_CA0E80(&v152, v40);
              v41 = *(_BYTE **)(v40 + 32);
              if ( (unsigned __int64)v41 >= *(_QWORD *)(v40 + 24) )
              {
                sub_CB5D20(v40, 10);
              }
              else
              {
                *(_QWORD *)(v40 + 32) = v41 + 1;
                *v41 = 10;
              }
              a2 = *(_QWORD *)a1;
              v42 = *(_BYTE *)(a1 + 154);
              *(_BYTE *)(a1 + 153) = 1;
              *(_BYTE *)(a1 + 152) |= v42;
              if ( a2 )
              {
                if ( v30 )
                {
                  sub_A68F80(v30, a2, a1 + 16, 0);
                  v43 = *(_QWORD *)a1;
                  v44 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
                  if ( (unsigned __int64)v44 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
                  {
                    a2 = 10;
                    sub_CB5D20(v43, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v43 + 32) = v44 + 1;
                    *v44 = 10;
                  }
                }
                if ( v35 )
                {
                  a2 = *(_QWORD *)a1;
                  sub_A62C00(v35, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
                  v45 = *(_QWORD *)a1;
                  v46 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
                  if ( (unsigned __int64)v46 < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
                    goto LABEL_86;
LABEL_150:
                  a2 = 10;
                  sub_CB5D20(v45, 10);
                }
              }
            }
            else
            {
              v84 = *(_BYTE *)(a1 + 154);
              *(_BYTE *)(a1 + 153) = 1;
              *(_BYTE *)(a1 + 152) |= v84;
            }
            goto LABEL_87;
          }
LABEL_69:
          a2 = (__int64)v35;
          sub_BE3890(a1, (__int64)v35, 1u);
        }
        else if ( v35 )
        {
          goto LABEL_69;
        }
        if ( v34 == ++v33 )
          goto LABEL_87;
        continue;
      }
    }
    v152 = "unrecognized named metadata node in the llvm.dbg namespace";
    LOWORD(v157[0]) = 259;
    sub_BDD6D0((__int64 *)a1, (__int64)&v152);
    a2 = *(_QWORD *)a1;
    if ( v30 && a2 )
    {
      sub_A68F80(v30, a2, a1 + 16, 0);
      v45 = *(_QWORD *)a1;
      v46 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
      if ( (unsigned __int64)v46 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_150;
LABEL_86:
      *(_QWORD *)(v45 + 32) = v46 + 1;
      *v46 = 10;
    }
LABEL_87:
    v30 = *(_QWORD *)(v30 + 8);
    if ( v145 != v30 )
      continue;
    break;
  }
  v7 = *(_QWORD *)(a1 + 8);
LABEL_89:
  v47 = *(_DWORD *)(v7 + 136);
  if ( v47 )
  {
    v48 = *(__int64 **)(v7 + 128);
    v49 = *v48;
    v50 = v48;
    if ( *v48 != -8 )
      goto LABEL_92;
    do
    {
      do
      {
        v49 = v50[1];
        ++v50;
      }
      while ( v49 == -8 );
LABEL_92:
      ;
    }
    while ( !v49 );
    v51 = &v48[v47];
    while ( v51 != v50 )
    {
      while ( 1 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 128) + 52LL) == 1 )
        {
          v85 = *(_QWORD *)(a1 + 8);
          a2 = sub_AA8810((_QWORD *)(*v50 + 8));
          v87 = sub_BA8B30(v85, a2, v86);
          v88 = (_BYTE *)v87;
          if ( v87 )
          {
            if ( (*(_BYTE *)(v87 + 32) & 0xF) == 8 )
            {
              v89 = *(_QWORD *)a1;
              v152 = "comdat global value has private linkage";
              LOWORD(v157[0]) = 259;
              if ( v89 )
              {
                a2 = v89;
                v146 = v89;
                sub_CA0E80(&v152, v89);
                v90 = *(_BYTE **)(v146 + 32);
                if ( (unsigned __int64)v90 >= *(_QWORD *)(v146 + 24) )
                {
                  a2 = 10;
                  sub_CB5D20(v146, 10);
                }
                else
                {
                  *(_QWORD *)(v146 + 32) = v90 + 1;
                  *v90 = 10;
                }
                v91 = *(_QWORD *)a1;
                *(_BYTE *)(a1 + 152) = 1;
                if ( v91 )
                {
                  a2 = (__int64)v88;
                  sub_BDBD80(a1, v88);
                }
              }
              else
              {
                *(_BYTE *)(a1 + 152) = 1;
              }
            }
          }
        }
        v52 = v50[1];
        v53 = v50 + 1;
        if ( v52 == -8 || !v52 )
          break;
        ++v50;
        if ( v51 == v53 )
          goto LABEL_101;
      }
      v54 = v50 + 2;
      do
      {
        do
        {
          v55 = *v54;
          v50 = v54++;
        }
        while ( !v55 );
      }
      while ( v55 == -8 );
    }
  }
LABEL_101:
  sub_BF2BF0(a1, a2);
  v56 = sub_BA8DC0(*(_QWORD *)(a1 + 8), (__int64)"llvm.ident", 10);
  v57 = v56;
  if ( v56 )
  {
    v58 = 0;
    v59 = sub_B91A00(v56);
    if ( v59 )
    {
      while ( 1 )
      {
        v60 = (const char *)sub_B91A10(v57, v58);
        v149[0] = v60;
        v61 = *(v60 - 16);
        if ( (v61 & 2) != 0 )
        {
          if ( *((_DWORD *)v60 - 6) != 1 )
            goto LABEL_278;
          v62 = (const char **)*((_QWORD *)v60 - 4);
        }
        else
        {
          if ( ((*((_WORD *)v60 - 8) >> 6) & 0xF) != 1 )
          {
LABEL_278:
            v152 = "incorrect number of operands in llvm.ident metadata";
            LOWORD(v157[0]) = 259;
            sub_BE1BE0((_BYTE *)a1, (__int64)&v152, v149);
            goto LABEL_109;
          }
          v62 = (const char **)&v60[-16 - 8LL * ((v61 >> 2) & 0xF)];
        }
        if ( !*v62 || **v62 )
          break;
        if ( v59 == ++v58 )
          goto LABEL_109;
      }
      v99 = *(_QWORD *)a1;
      v152 = "invalid value for llvm.ident metadata entry operand(the operand should be a string)";
      LOWORD(v157[0]) = 259;
      if ( v99 )
      {
        sub_CA0E80(&v152, v99);
        v100 = *(_BYTE **)(v99 + 32);
        if ( (unsigned __int64)v100 >= *(_QWORD *)(v99 + 24) )
        {
          sub_CB5D20(v99, 10);
        }
        else
        {
          *(_QWORD *)(v99 + 32) = v100 + 1;
          *v100 = 10;
        }
        v101 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v101 && *v62 )
        {
          sub_A62C00(*v62, v101, a1 + 16, *(_QWORD *)(a1 + 8));
          v102 = *(_QWORD *)a1;
          v103 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v103 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            sub_CB5D20(v102, 10);
          }
          else
          {
            *(_QWORD *)(v102 + 32) = v103 + 1;
            *v103 = 10;
          }
        }
      }
      else
      {
        *(_BYTE *)(a1 + 152) = 1;
      }
    }
  }
LABEL_109:
  v63 = sub_BA8DC0(*(_QWORD *)(a1 + 8), (__int64)"llvm.commandline", 16);
  v64 = v63;
  if ( v63 )
  {
    v65 = 0;
    v66 = sub_B91A00(v63);
    if ( v66 )
    {
      while ( 1 )
      {
        v67 = (const char *)sub_B91A10(v64, v65);
        v149[0] = v67;
        v68 = *(v67 - 16);
        if ( (v68 & 2) != 0 )
        {
          if ( *((_DWORD *)v67 - 6) != 1 )
            goto LABEL_277;
          v69 = (const char **)*((_QWORD *)v67 - 4);
        }
        else
        {
          if ( ((*((_WORD *)v67 - 8) >> 6) & 0xF) != 1 )
          {
LABEL_277:
            v152 = "incorrect number of operands in llvm.commandline metadata";
            LOWORD(v157[0]) = 259;
            sub_BE1BE0((_BYTE *)a1, (__int64)&v152, v149);
            goto LABEL_117;
          }
          v69 = (const char **)&v67[-16 - 8LL * ((v68 >> 2) & 0xF)];
        }
        if ( !*v69 || **v69 )
          break;
        if ( v66 == ++v65 )
          goto LABEL_117;
      }
      v94 = *(_QWORD *)a1;
      v152 = "invalid value for llvm.commandline metadata entry operand(the operand should be a string)";
      LOWORD(v157[0]) = 259;
      if ( v94 )
      {
        sub_CA0E80(&v152, v94);
        v95 = *(_BYTE **)(v94 + 32);
        if ( (unsigned __int64)v95 >= *(_QWORD *)(v94 + 24) )
        {
          sub_CB5D20(v94, 10);
        }
        else
        {
          *(_QWORD *)(v94 + 32) = v95 + 1;
          *v95 = 10;
        }
        v96 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v96 && *v69 )
        {
          sub_A62C00(*v69, v96, a1 + 16, *(_QWORD *)(a1 + 8));
          v97 = *(_QWORD *)a1;
          v98 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( (unsigned __int64)v98 >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            sub_CB5D20(v97, 10);
          }
          else
          {
            *(_QWORD *)(v97 + 32) = v98 + 1;
            *v98 = 10;
          }
        }
      }
      else
      {
        *(_BYTE *)(a1 + 152) = 1;
      }
    }
  }
LABEL_117:
  if ( !(unsigned __int8)sub_B6F8F0(**(_QWORD **)(a1 + 8)) )
  {
    v113 = (__int64)"llvm.dbg.cu";
    v114 = sub_BA8DC0(*(_QWORD *)(a1 + 8), (__int64)"llvm.dbg.cu", 11);
    v152 = 0;
    v115 = v114;
    v154 = 2;
    v153 = v157;
    v155 = 0;
    v156 = 1;
    if ( v114 )
    {
      v116 = sub_B91A00(v114);
      if ( v116 )
      {
        v117 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v113 = sub_B91A10(v115, v117);
            if ( v156 )
              break;
LABEL_206:
            ++v117;
            sub_C8CC70(&v152, v113);
            if ( v116 == v117 )
              goto LABEL_200;
          }
          v118 = v153;
          v119 = &v153[HIDWORD(v154)];
          if ( v153 == v119 )
          {
LABEL_208:
            if ( HIDWORD(v154) >= (unsigned int)v154 )
              goto LABEL_206;
            ++v117;
            ++HIDWORD(v154);
            *v119 = v113;
            ++v152;
            if ( v116 == v117 )
              break;
          }
          else
          {
            while ( v113 != *v118 )
            {
              if ( v119 == ++v118 )
                goto LABEL_208;
            }
            if ( v116 == ++v117 )
              break;
          }
        }
      }
    }
LABEL_200:
    v120 = (__int64 *)*(unsigned __int8 *)(a1 + 796);
    v121 = *(const char ***)(a1 + 776);
    if ( (_BYTE)v120 )
      v122 = *(unsigned int *)(a1 + 788);
    else
      v122 = *(unsigned int *)(a1 + 784);
    v123 = &v121[v122];
    if ( v121 == v123 )
    {
LABEL_205:
      v147 = a1 + 768;
    }
    else
    {
      while ( 1 )
      {
        v124 = *v121;
        v125 = v121;
        if ( (unsigned __int64)*v121 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v123 == ++v121 )
          goto LABEL_205;
      }
      v147 = a1 + 768;
      if ( v123 != v121 )
      {
        if ( !v156 )
          goto LABEL_231;
LABEL_213:
        v126 = (const char **)v153;
        v120 = &v153[HIDWORD(v154)];
        if ( v153 == v120 )
        {
LABEL_232:
          v130 = *(_QWORD *)a1;
          v151 = 1;
          v149[0] = "DICompileUnit not listed in llvm.dbg.cu";
          v150 = 3;
          if ( v130 )
          {
            v113 = v130;
            sub_CA0E80(v149, v130);
            v131 = *(_BYTE **)(v130 + 32);
            if ( (unsigned __int64)v131 >= *(_QWORD *)(v130 + 24) )
            {
              v113 = 10;
              sub_CB5D20(v130, 10);
            }
            else
            {
              *(_QWORD *)(v130 + 32) = v131 + 1;
              *v131 = 10;
            }
            v132 = *(_QWORD *)a1;
            v133 = *(_BYTE *)(a1 + 154);
            *(_BYTE *)(a1 + 153) = 1;
            *(_BYTE *)(a1 + 152) |= v133;
            if ( v124 && v132 )
            {
              v113 = (__int64)v124;
              sub_BD9900((__int64 *)a1, v124);
            }
          }
          else
          {
            v141 = *(_BYTE *)(a1 + 154);
            *(_BYTE *)(a1 + 153) = 1;
            *(_BYTE *)(a1 + 152) |= v141;
          }
          goto LABEL_227;
        }
        while ( *v126 != v124 )
        {
          if ( v120 == (__int64 *)++v126 )
            goto LABEL_232;
        }
        while ( 1 )
        {
          v127 = v125 + 1;
          if ( v125 + 1 == v123 )
            break;
          while ( 1 )
          {
            v124 = *v127;
            v125 = v127;
            if ( (unsigned __int64)*v127 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v123 == ++v127 )
              goto LABEL_220;
          }
          if ( v127 == v123 )
            break;
          if ( v156 )
            goto LABEL_213;
LABEL_231:
          v113 = (__int64)v124;
          if ( !sub_C8CA60(&v152, v124, v122, v120) )
            goto LABEL_232;
        }
LABEL_220:
        LOBYTE(v120) = *(_BYTE *)(a1 + 796);
      }
    }
    ++*(_QWORD *)(a1 + 768);
    if ( !(_BYTE)v120 )
    {
      v128 = 4 * (*(_DWORD *)(a1 + 788) - *(_DWORD *)(a1 + 792));
      v129 = *(unsigned int *)(a1 + 784);
      if ( v128 < 0x20 )
        v128 = 32;
      if ( (unsigned int)v129 > v128 )
      {
        sub_C8C990(v147);
LABEL_227:
        if ( !v156 )
          _libc_free(v153, v113);
        goto LABEL_118;
      }
      v113 = 0xFFFFFFFFLL;
      memset(*(void **)(a1 + 776), -1, 8 * v129);
    }
    *(_QWORD *)(a1 + 788) = 0;
    goto LABEL_227;
  }
LABEL_118:
  v70 = *(unsigned int *)(a1 + 1240);
  if ( (_DWORD)v70 )
  {
    v71 = *(__int64 **)(a1 + 1232);
    v72 = v71 + 1;
    v73 = &v71[v70];
    v74 = *v71;
    if ( v71 + 1 != v73 )
    {
      while ( 1 )
      {
        v75 = (_BYTE *)*v72;
        if ( ((*(_WORD *)(v74 + 2) >> 4) & 0x3FF) != ((*(_WORD *)(*v72 + 2LL) >> 4) & 0x3FF) )
          break;
        if ( v73 == ++v72 )
          goto LABEL_128;
      }
      v76 = *(_QWORD *)a1;
      v152 = "All llvm.experimental.deoptimize declarations must have the same calling convention";
      LOWORD(v157[0]) = 259;
      if ( v76 )
      {
        sub_CA0E80(&v152, v76);
        v77 = *(_BYTE **)(v76 + 32);
        if ( (unsigned __int64)v77 >= *(_QWORD *)(v76 + 24) )
        {
          sub_CB5D20(v76, 10);
        }
        else
        {
          *(_QWORD *)(v76 + 32) = v77 + 1;
          *v77 = 10;
        }
        v78 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v78 )
        {
          sub_BDBD80(a1, (_BYTE *)v74);
          sub_BDBD80(a1, v75);
        }
      }
      else
      {
        *(_BYTE *)(a1 + 152) = 1;
      }
    }
  }
LABEL_128:
  v79 = *(_DWORD *)(a1 + 752);
  ++*(_QWORD *)(a1 + 736);
  if ( v79 )
  {
    v104 = 4 * v79;
    v80 = *(unsigned int *)(a1 + 760);
    if ( (unsigned int)(4 * v79) < 0x40 )
      v104 = 64;
    if ( v104 >= (unsigned int)v80 )
    {
LABEL_131:
      v81 = *(_QWORD **)(a1 + 744);
      for ( j = &v81[2 * v80]; j != v81; v81 += 2 )
        *v81 = -4096;
      *(_QWORD *)(a1 + 752) = 0;
      return *(unsigned __int8 *)(a1 + 152) ^ 1u;
    }
    v105 = v79 - 1;
    if ( v105 )
    {
      _BitScanReverse(&v105, v105);
      v106 = *(_QWORD **)(a1 + 744);
      v107 = 1 << (33 - (v105 ^ 0x1F));
      if ( v107 < 64 )
        v107 = 64;
      if ( v107 == (_DWORD)v80 )
      {
        *(_QWORD *)(a1 + 752) = 0;
        v142 = &v106[2 * (unsigned int)v107];
        do
        {
          if ( v106 )
            *v106 = -4096;
          v106 += 2;
        }
        while ( v142 != v106 );
        return *(unsigned __int8 *)(a1 + 152) ^ 1u;
      }
    }
    else
    {
      v106 = *(_QWORD **)(a1 + 744);
      v107 = 64;
    }
    sub_C7D6A0(v106, 16LL * *(unsigned int *)(a1 + 760), 8);
    v108 = ((((((((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
              | (4 * v107 / 3u + 1)
              | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 4)
            | (((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
            | (4 * v107 / 3u + 1)
            | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
            | (4 * v107 / 3u + 1)
            | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 4)
          | (((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
          | (4 * v107 / 3u + 1)
          | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 16;
    v109 = (v108
          | (((((((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
              | (4 * v107 / 3u + 1)
              | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 4)
            | (((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
            | (4 * v107 / 3u + 1)
            | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 8)
          | (((((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
            | (4 * v107 / 3u + 1)
            | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 4)
          | (((4 * v107 / 3u + 1) | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1)) >> 2)
          | (4 * v107 / 3u + 1)
          | ((unsigned __int64)(4 * v107 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 760) = v109;
    v110 = (_QWORD *)sub_C7D670(16 * v109, 8);
    v111 = *(unsigned int *)(a1 + 760);
    *(_QWORD *)(a1 + 752) = 0;
    *(_QWORD *)(a1 + 744) = v110;
    for ( k = &v110[2 * v111]; k != v110; v110 += 2 )
    {
      if ( v110 )
        *v110 = -4096;
    }
    return *(unsigned __int8 *)(a1 + 152) ^ 1u;
  }
  if ( *(_DWORD *)(a1 + 756) )
  {
    v80 = *(unsigned int *)(a1 + 760);
    if ( (unsigned int)v80 <= 0x40 )
      goto LABEL_131;
    sub_C7D6A0(*(_QWORD *)(a1 + 744), 16LL * *(unsigned int *)(a1 + 760), 8);
    *(_QWORD *)(a1 + 744) = 0;
    *(_QWORD *)(a1 + 752) = 0;
    *(_DWORD *)(a1 + 760) = 0;
  }
  return *(unsigned __int8 *)(a1 + 152) ^ 1u;
}
