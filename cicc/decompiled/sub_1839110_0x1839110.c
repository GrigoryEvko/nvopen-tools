// Function: sub_1839110
// Address: 0x1839110
//
__int64 __fastcall sub_1839110(__int64 a1, unsigned __int8 a2, _QWORD *a3, unsigned int a4)
{
  int *v4; // r13
  __int64 v5; // r12
  int *v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int8 v16; // dl
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // r13
  char *v22; // rbx
  __int64 v23; // r14
  char **v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rbx
  _QWORD *v29; // r15
  unsigned __int8 v30; // al
  _BYTE *v31; // r15
  unsigned int v32; // r12d
  __int64 v33; // rsi
  unsigned __int64 v34; // rdi
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rax
  _BYTE *v39; // rsi
  int *v40; // rsi
  __int64 v41; // r11
  int *v42; // r10
  char *v43; // rdi
  char *v44; // rdx
  char *v45; // rcx
  char *v46; // rax
  _BYTE *v47; // r8
  const void *v48; // rdi
  size_t v49; // rdx
  _BYTE *v50; // r9
  int *v51; // rsi
  int *v52; // r10
  char *v53; // rcx
  char *v54; // rax
  _QWORD *v55; // rdx
  _QWORD *v56; // rdx
  __int64 v57; // rax
  _QWORD *v58; // rax
  int v59; // eax
  unsigned __int64 v60; // rax
  __int64 v61; // r9
  __int64 v62; // rdx
  int *v63; // r14
  __int64 v64; // r13
  __int64 v65; // r12
  _QWORD *v66; // rbx
  __int64 v67; // rax
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // rax
  _BYTE *v71; // rsi
  _QWORD *v72; // r8
  __int64 v73; // rbx
  int v74; // r8d
  int v75; // r9d
  __int64 v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // r15
  __int64 *v79; // r10
  char **v80; // r9
  __int64 *v81; // r12
  __int64 *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 *v85; // rdx
  __int64 v86; // rdx
  char *v87; // rsi
  __int64 v88; // r12
  __int64 v89; // rbx
  __int64 v90; // rsi
  _QWORD *v91; // rax
  unsigned __int64 *v92; // rdi
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rsi
  unsigned __int64 v96; // r12
  __int64 v97; // rax
  __int64 v98; // rax
  char v99; // r8
  unsigned __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // r8
  unsigned __int64 v103; // r14
  __int64 v104; // rax
  __int64 v105; // r13
  __int64 v106; // rsi
  __int64 v107; // rcx
  char v108; // r9
  _QWORD *v109; // rbx
  unsigned __int64 v110; // r12
  __int64 v111; // rsi
  unsigned __int64 *v112; // rcx
  char v113; // al
  char v114; // r8
  bool v115; // al
  unsigned int v116; // eax
  char **v117; // [rsp+10h] [rbp-2F0h]
  _BYTE *v118; // [rsp+18h] [rbp-2E8h]
  __int64 v119; // [rsp+18h] [rbp-2E8h]
  __int64 *v120; // [rsp+18h] [rbp-2E8h]
  char *v121; // [rsp+18h] [rbp-2E8h]
  __int64 **v122; // [rsp+18h] [rbp-2E8h]
  __int64 v124; // [rsp+28h] [rbp-2D8h]
  unsigned __int64 v127; // [rsp+40h] [rbp-2C0h]
  _BYTE *v128; // [rsp+48h] [rbp-2B8h]
  int *v129; // [rsp+48h] [rbp-2B8h]
  __int64 **v130; // [rsp+48h] [rbp-2B8h]
  __int64 v131; // [rsp+58h] [rbp-2A8h] BYREF
  char *v132; // [rsp+60h] [rbp-2A0h] BYREF
  char *v133; // [rsp+68h] [rbp-298h]
  char *v134; // [rsp+70h] [rbp-290h]
  void *s2; // [rsp+80h] [rbp-280h] BYREF
  _BYTE *v136; // [rsp+88h] [rbp-278h]
  _BYTE *v137; // [rsp+90h] [rbp-270h]
  unsigned __int64 v138; // [rsp+A0h] [rbp-260h] BYREF
  unsigned __int64 *v139; // [rsp+A8h] [rbp-258h]
  __int64 v140; // [rsp+B0h] [rbp-250h]
  char *v141; // [rsp+B8h] [rbp-248h]
  __m128i v142[3]; // [rsp+C0h] [rbp-240h] BYREF
  char v143[8]; // [rsp+F0h] [rbp-210h] BYREF
  int v144; // [rsp+F8h] [rbp-208h] BYREF
  int *v145; // [rsp+100h] [rbp-200h]
  int *v146; // [rsp+108h] [rbp-1F8h]
  int *v147; // [rsp+110h] [rbp-1F0h]
  __int64 v148; // [rsp+118h] [rbp-1E8h]
  char v149[8]; // [rsp+120h] [rbp-1E0h] BYREF
  int v150; // [rsp+128h] [rbp-1D8h] BYREF
  int *v151; // [rsp+130h] [rbp-1D0h]
  int *v152; // [rsp+138h] [rbp-1C8h]
  int *v153; // [rsp+140h] [rbp-1C0h]
  __int64 v154; // [rsp+148h] [rbp-1B8h]
  unsigned __int64 v155; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v156; // [rsp+158h] [rbp-1A8h]
  __int64 v157; // [rsp+160h] [rbp-1A0h]
  __int64 v158; // [rsp+168h] [rbp-198h]
  __int64 v159; // [rsp+178h] [rbp-188h]
  __int64 v160; // [rsp+180h] [rbp-180h]
  __int64 v161; // [rsp+188h] [rbp-178h]
  __int64 **v162; // [rsp+190h] [rbp-170h] BYREF
  __int64 v163; // [rsp+198h] [rbp-168h]
  _BYTE v164[128]; // [rsp+1A0h] [rbp-160h] BYREF
  __int64 v165; // [rsp+220h] [rbp-E0h] BYREF
  _BYTE *v166; // [rsp+228h] [rbp-D8h]
  _BYTE *v167; // [rsp+230h] [rbp-D0h]
  __int64 v168; // [rsp+238h] [rbp-C8h]
  int v169; // [rsp+240h] [rbp-C0h]
  _BYTE v170[184]; // [rsp+248h] [rbp-B8h] BYREF

  if ( !*(_QWORD *)(a1 + 8) )
    return 1;
  v4 = &v144;
  v5 = a1;
  v6 = &v150;
  v144 = 0;
  v145 = 0;
  v146 = &v144;
  v147 = &v144;
  v148 = 0;
  v150 = 0;
  v151 = 0;
  v152 = &v150;
  v153 = &v150;
  v154 = 0;
  if ( !a2 )
  {
    v7 = *(_QWORD *)(a1 + 24);
    v8 = sub_1632FA0(*(_QWORD *)(v7 + 40));
    v9 = *(_QWORD *)(v7 + 8);
    v10 = v8;
    if ( v9 )
    {
      v11 = v9;
      v12 = 24LL * *(unsigned int *)(a1 + 32);
      while ( 1 )
      {
        v15 = (unsigned __int64)sub_1648700(v11);
        v16 = *(_BYTE *)(v15 + 16);
        if ( v16 <= 0x17u )
          break;
        if ( v16 == 78 )
        {
          v60 = v15 | 4;
        }
        else
        {
          v13 = 0;
          if ( v16 != 29 )
            goto LABEL_7;
          v60 = v15 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v13 = v60 & 0xFFFFFFFFFFFFFFF8LL;
        v14 = (v60 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v60 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
        if ( (v60 & 4) == 0 )
          goto LABEL_7;
LABEL_8:
        if ( !(unsigned __int8)sub_13F8680(*(_QWORD *)(v14 + v12), v10, 0, 0) )
        {
          v5 = a1;
          goto LABEL_15;
        }
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
        {
          v5 = a1;
          goto LABEL_13;
        }
      }
      v13 = 0;
LABEL_7:
      v14 = v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      goto LABEL_8;
    }
  }
LABEL_13:
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v17 = (_QWORD *)sub_22077B0(8);
  *v17 = 0;
  v165 = (__int64)v17;
  v167 = v17 + 1;
  v166 = v17 + 1;
  sub_18344A0((__int64)v143, &v165);
  if ( v165 )
    j_j___libc_free_0(v165, &v167[-v165]);
LABEL_15:
  v132 = 0;
  v18 = *(_QWORD *)(v5 + 24);
  v133 = 0;
  v134 = 0;
  v19 = *(_QWORD *)(v18 + 80);
  if ( !v19 )
    BUG();
  v20 = *(_QWORD *)(v19 + 24);
  if ( v20 == v19 + 16 )
    goto LABEL_29;
  v21 = v19 + 16;
  v22 = v143;
  v23 = v5;
  v24 = (char **)&v165;
  do
  {
    while ( 1 )
    {
      if ( !v20 )
        BUG();
      if ( *(_BYTE *)(v20 - 8) == 54 )
      {
        v26 = *(_QWORD *)(v20 - 48);
        if ( !v26 )
          BUG();
        if ( *(_BYTE *)(v26 + 16) == 56 )
        {
          v25 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
          if ( v23 == v25 && v25 )
          {
            v120 = *(__int64 **)(v20 - 48);
            sub_9C9810((__int64)&v132, (*(_DWORD *)(v26 + 20) & 0xFFFFFFFu) - 1);
            v79 = &v120[3 * (1LL - (*((_DWORD *)v120 + 5) & 0xFFFFFFF))];
            if ( v79 != v120 )
            {
              v80 = v24;
              v81 = v120;
              v121 = v22;
              v82 = v79;
              do
              {
                v83 = *v82;
                if ( *(_BYTE *)(*v82 + 16) != 13 )
                {
                  v32 = 0;
                  goto LABEL_38;
                }
                v84 = *(_DWORD *)(v83 + 32);
                v85 = *(__int64 **)(v83 + 24);
                if ( v84 <= 0x40 )
                  v86 = (__int64)((_QWORD)v85 << (64 - (unsigned __int8)v84)) >> (64 - (unsigned __int8)v84);
                else
                  v86 = *v85;
                v165 = v86;
                v87 = v133;
                if ( v133 == v134 )
                {
                  v117 = v80;
                  sub_A235E0((__int64)&v132, v133, v80);
                  v80 = v117;
                }
                else
                {
                  if ( v133 )
                  {
                    *(_QWORD *)v133 = v86;
                    v87 = v133;
                  }
                  v133 = v87 + 8;
                }
                v82 += 3;
              }
              while ( v81 != v82 );
              v22 = v121;
              v24 = v80;
            }
            sub_1838810(&v132, v22);
            if ( v132 != v133 )
              v133 = v132;
          }
          goto LABEL_20;
        }
        if ( v23 == v26 )
        {
          v165 = 0;
          v166 = 0;
          v167 = 0;
          v27 = (_QWORD *)sub_22077B0(8);
          *v27 = 0;
          v165 = (__int64)v27;
          v167 = v27 + 1;
          v166 = v27 + 1;
          sub_1838810(v24, v22);
          if ( v165 )
            break;
        }
      }
LABEL_20:
      v20 = *(_QWORD *)(v20 + 8);
      if ( v21 == v20 )
        goto LABEL_28;
    }
    j_j___libc_free_0(v165, &v167[-v165]);
    v20 = *(_QWORD *)(v20 + 8);
  }
  while ( v21 != v20 );
LABEL_28:
  v5 = v23;
  v4 = &v144;
  v6 = &v150;
LABEL_29:
  v28 = *(_QWORD *)(v5 + 8);
  s2 = 0;
  v162 = (__int64 **)v164;
  v163 = 0x1000000000LL;
  v136 = 0;
  v137 = 0;
  if ( !v28 )
  {
    v32 = 1;
    goto LABEL_36;
  }
  v29 = sub_1648700(v28);
  while ( 2 )
  {
    v30 = *((_BYTE *)v29 + 16);
    if ( v30 <= 0x17u )
    {
LABEL_34:
      v31 = s2;
      v32 = 0;
      v33 = v137 - (_BYTE *)s2;
      goto LABEL_35;
    }
    if ( v30 == 54 )
    {
      if ( sub_15F32D0((__int64)v29) || (*((_BYTE *)v29 + 18) & 1) != 0 )
        goto LABEL_34;
      v38 = (unsigned int)v163;
      if ( (unsigned int)v163 >= HIDWORD(v163) )
      {
        sub_16CD150((__int64)&v162, v164, 0, 8, v36, v37);
        v38 = (unsigned int)v163;
      }
      v162[v38] = v29;
      v39 = v136;
      LODWORD(v163) = v163 + 1;
      v165 = 0;
      if ( v136 == v137 )
      {
        sub_A235E0((__int64)&s2, v136, &v165);
      }
      else
      {
        if ( v136 )
        {
          *(_QWORD *)v136 = 0;
          v39 = v136;
        }
        v136 = v39 + 8;
      }
    }
    else
    {
      if ( v30 != 56 )
        goto LABEL_34;
      v61 = v29[1];
      if ( !v61 )
      {
        sub_15F20C0(v29);
        v116 = sub_1839110(v5, a2, a3, a4);
        v31 = s2;
        v32 = v116;
        v33 = v137 - (_BYTE *)s2;
        goto LABEL_35;
      }
      v62 = 1LL - (*((_DWORD *)v29 + 5) & 0xFFFFFFF);
      if ( &v29[3 * v62] == v29 )
        goto LABEL_124;
      v129 = v6;
      v63 = v4;
      v64 = v5;
      v65 = v28;
      v66 = &v29[3 * v62];
      do
      {
        v67 = *v66;
        if ( *(_BYTE *)(*v66 + 16LL) != 13 )
          goto LABEL_34;
        v68 = *(_DWORD *)(v67 + 32);
        v69 = *(__int64 **)(v67 + 24);
        if ( v68 > 0x40 )
          v70 = *v69;
        else
          v70 = (__int64)((_QWORD)v69 << (64 - (unsigned __int8)v68)) >> (64 - (unsigned __int8)v68);
        v165 = v70;
        v71 = v136;
        if ( v136 == v137 )
        {
          sub_A235E0((__int64)&s2, v136, &v165);
        }
        else
        {
          if ( v136 )
          {
            *(_QWORD *)v136 = v70;
            v71 = v136;
          }
          v136 = v71 + 8;
        }
        v66 += 3;
      }
      while ( v66 != v29 );
      v72 = v66;
      v28 = v65;
      v5 = v64;
      v4 = v63;
      v61 = v72[1];
      v6 = v129;
      if ( v61 )
      {
LABEL_124:
        v119 = v28;
        v73 = v61;
        do
        {
          v77 = sub_1648700(v73);
          v78 = v77;
          if ( *((_BYTE *)v77 + 16) != 54 || sub_15F32D0((__int64)v77) || (*((_BYTE *)v78 + 18) & 1) != 0 )
            goto LABEL_34;
          v76 = (unsigned int)v163;
          if ( (unsigned int)v163 >= HIDWORD(v163) )
          {
            sub_16CD150((__int64)&v162, v164, 0, 8, v74, v75);
            v76 = (unsigned int)v163;
          }
          v162[v76] = v78;
          LODWORD(v163) = v163 + 1;
          v73 = *(_QWORD *)(v73 + 8);
        }
        while ( v73 );
        v28 = v119;
      }
    }
    v40 = v145;
    v41 = (__int64)v146;
    v31 = s2;
    if ( v145 )
    {
      v42 = v4;
      while ( 1 )
      {
        v43 = (char *)*((_QWORD *)v40 + 5);
        v44 = (char *)*((_QWORD *)v40 + 4);
        v45 = (char *)s2 + v43 - v44;
        if ( v136 - (_BYTE *)s2 <= v43 - v44 )
          v45 = v136;
        if ( v45 == s2 )
        {
LABEL_92:
          if ( v43 == v44 )
          {
LABEL_93:
            v40 = (int *)*((_QWORD *)v40 + 3);
            goto LABEL_60;
          }
        }
        else
        {
          v46 = (char *)s2;
          while ( *(_QWORD *)v46 >= *(_QWORD *)v44 )
          {
            if ( *(_QWORD *)v46 > *(_QWORD *)v44 )
              goto LABEL_93;
            v46 += 8;
            v44 += 8;
            if ( v45 == v46 )
            {
              v43 = (char *)*((_QWORD *)v40 + 5);
              goto LABEL_92;
            }
          }
        }
        v42 = v40;
        v40 = (int *)*((_QWORD *)v40 + 2);
LABEL_60:
        if ( !v40 )
        {
          if ( v42 != v146 )
            goto LABEL_62;
          goto LABEL_63;
        }
      }
    }
    if ( v146 == v4 )
      goto LABEL_97;
    v42 = v4;
LABEL_62:
    v41 = sub_220EFE0(v42);
LABEL_63:
    if ( (int *)v41 == v4 )
      goto LABEL_97;
    v47 = v136;
    v48 = *(const void **)(v41 + 32);
    v49 = *(_QWORD *)(v41 + 40) - (_QWORD)v48;
    v50 = (_BYTE *)(v136 - v31);
    if ( v136 - v31 < v49 )
      goto LABEL_97;
    if ( v49 )
    {
      v118 = (_BYTE *)(v136 - v31);
      v128 = v136;
      v59 = memcmp(v48, v31, v49);
      v47 = v128;
      v50 = v118;
      if ( v59 )
      {
        v32 = 0;
        v33 = v137 - v31;
        goto LABEL_101;
      }
    }
    v51 = v151;
    if ( !v151 )
      goto LABEL_85;
    v52 = v6;
    while ( 2 )
    {
      v53 = (char *)*((_QWORD *)v51 + 5);
      v54 = (char *)*((_QWORD *)v51 + 4);
      if ( (__int64)v50 < v53 - v54 )
        v53 = &v50[(_QWORD)v54];
      v55 = v31;
      if ( v54 != v53 )
      {
        while ( *(_QWORD *)v54 >= *v55 )
        {
          if ( *(_QWORD *)v54 > *v55 )
            goto LABEL_95;
          v54 += 8;
          ++v55;
          if ( v53 == v54 )
            goto LABEL_94;
        }
        goto LABEL_75;
      }
LABEL_94:
      if ( v55 != (_QWORD *)v47 )
      {
LABEL_75:
        v51 = (int *)*((_QWORD *)v51 + 3);
        goto LABEL_76;
      }
LABEL_95:
      v52 = v51;
      v51 = (int *)*((_QWORD *)v51 + 2);
LABEL_76:
      if ( v51 )
        continue;
      break;
    }
    if ( v52 == v6 )
      goto LABEL_85;
    v56 = (_QWORD *)*((_QWORD *)v52 + 4);
    v57 = *((_QWORD *)v52 + 5) - (_QWORD)v56;
    if ( (__int64)v50 > v57 )
      v47 = &v31[v57];
    if ( v47 == v31 )
    {
LABEL_107:
      if ( *((_QWORD **)v52 + 5) != v56 )
      {
        if ( a4 )
          goto LABEL_86;
        goto LABEL_87;
      }
    }
    else
    {
      v58 = v31;
      while ( *v58 >= *v56 )
      {
        if ( *v58 > *v56 )
          goto LABEL_88;
        ++v58;
        ++v56;
        if ( v47 == (_BYTE *)v58 )
          goto LABEL_107;
      }
LABEL_85:
      if ( a4 )
      {
LABEL_86:
        if ( a4 != v154 )
          goto LABEL_87;
LABEL_97:
        v32 = 0;
        v33 = v137 - v31;
        goto LABEL_35;
      }
LABEL_87:
      sub_18344A0((__int64)v149, &s2);
    }
LABEL_88:
    v28 = *(_QWORD *)(v28 + 8);
    if ( v28 )
    {
      v29 = sub_1648700(v28);
      if ( s2 != v136 )
        v136 = s2;
      continue;
    }
    break;
  }
  if ( !(_DWORD)v163 )
  {
    v31 = s2;
    v32 = 1;
    v33 = v137 - (_BYTE *)s2;
    goto LABEL_35;
  }
  v165 = 0;
  v166 = v170;
  v167 = v170;
  v168 = 16;
  v130 = v162;
  v122 = &v162[(unsigned int)v163];
  v169 = 0;
  while ( 2 )
  {
    v88 = (__int64)*v130;
    v89 = (*v130)[5];
    sub_141EB40(v142, *v130);
    v90 = *(_QWORD *)(v89 + 48);
    if ( v90 )
      v90 -= 24;
    if ( (unsigned __int8)sub_134F310(a3, v90, v88, v142, 6u) )
      goto LABEL_209;
    do
    {
      v89 = *(_QWORD *)(v89 + 8);
      if ( !v89 )
        goto LABEL_214;
      v91 = sub_1648700(v89);
    }
    while ( (unsigned __int8)(*((_BYTE *)v91 + 16) - 25) > 9u );
LABEL_157:
    v92 = &v155;
    v131 = v91[5];
    sub_1838D70((__int64 *)&v155, &v131, (__int64)&v165);
    v94 = v157;
    v139 = 0;
    v95 = v156;
    v140 = 0;
    v138 = v155;
    v141 = 0;
    v96 = v157 - v156;
    if ( v157 == v156 )
    {
      v92 = 0;
      v96 = 0;
    }
    else
    {
      if ( v96 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_220;
      v97 = sub_22077B0(v157 - v156);
      v94 = v157;
      v95 = v156;
      v92 = (unsigned __int64 *)v97;
    }
    v139 = v92;
    v93 = (__int64)v92;
    v140 = (__int64)v92;
    v141 = (char *)v92 + v96;
    if ( v95 != v94 )
    {
      v98 = v95;
      do
      {
        if ( v93 )
        {
          *(_QWORD *)v93 = *(_QWORD *)v98;
          v99 = *(_BYTE *)(v98 + 16);
          *(_BYTE *)(v93 + 16) = v99;
          if ( v99 )
            *(_QWORD *)(v93 + 8) = *(_QWORD *)(v98 + 8);
        }
        v98 += 24;
        v93 += 24;
      }
      while ( v98 != v94 );
      v100 = v98 - 24 - v95;
      v95 = 0xAAAAAAAAAAAAAABLL;
      v93 = (__int64)&v92[(v100 >> 3) + 3];
    }
    v101 = v160;
    v102 = v159;
    v140 = v93;
    v103 = v160 - v159;
    if ( v160 == v159 )
    {
      v105 = 0;
    }
    else
    {
      if ( v103 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_220:
        sub_4261EA(v92, v95, v93);
      v104 = sub_22077B0(v160 - v159);
      v102 = v159;
      v93 = v140;
      v105 = v104;
      v92 = v139;
      v101 = v160;
    }
    if ( v101 == v102 )
    {
      v124 = v89;
      v110 = 0;
      v109 = a3;
      v127 = v103;
    }
    else
    {
      v106 = v105;
      v107 = v102;
      do
      {
        if ( v106 )
        {
          *(_QWORD *)v106 = *(_QWORD *)v107;
          v108 = *(_BYTE *)(v107 + 16);
          *(_BYTE *)(v106 + 16) = v108;
          if ( v108 )
            *(_QWORD *)(v106 + 8) = *(_QWORD *)(v107 + 8);
        }
        v107 += 24;
        v106 += 24;
      }
      while ( v101 != v107 );
      v124 = v89;
      v109 = a3;
      v127 = v103;
      v110 = 8 * ((unsigned __int64)(v101 - 24 - v102) >> 3) + 24;
    }
    while ( 2 )
    {
      if ( v110 != v93 - (_QWORD)v92 )
      {
LABEL_177:
        if ( (unsigned __int8)sub_134F4F0(v109, *(_QWORD *)(v93 - 24), v142) )
        {
          if ( v105 )
            j_j___libc_free_0(v105, v127);
          if ( v139 )
            j_j___libc_free_0(v139, v141 - (char *)v139);
          if ( v159 )
            j_j___libc_free_0(v159, v161 - v159);
          if ( v156 )
            j_j___libc_free_0(v156, v158 - v156);
LABEL_209:
          v32 = 0;
          goto LABEL_210;
        }
        sub_1838F70(&v138);
        v92 = v139;
        v93 = v140;
        continue;
      }
      break;
    }
    if ( (unsigned __int64 *)v93 != v92 )
    {
      v111 = v105;
      v112 = v92;
      while ( *v112 == *(_QWORD *)v111 )
      {
        v113 = *((_BYTE *)v112 + 16);
        v114 = *(_BYTE *)(v111 + 16);
        if ( v113 && v114 )
          v115 = v112[1] == *(_QWORD *)(v111 + 8);
        else
          v115 = v114 == v113;
        if ( !v115 )
          break;
        v112 += 3;
        v111 += 24;
        if ( (unsigned __int64 *)v93 == v112 )
          goto LABEL_188;
      }
      goto LABEL_177;
    }
LABEL_188:
    v89 = v124;
    if ( v105 )
    {
      j_j___libc_free_0(v105, v127);
      v92 = v139;
    }
    if ( v92 )
      j_j___libc_free_0(v92, v141 - (char *)v92);
    if ( v159 )
      j_j___libc_free_0(v159, v161 - v159);
    if ( v156 )
      j_j___libc_free_0(v156, v158 - v156);
    while ( 1 )
    {
      v89 = *(_QWORD *)(v89 + 8);
      if ( !v89 )
        break;
      v91 = sub_1648700(v89);
      if ( (unsigned __int8)(*((_BYTE *)v91 + 16) - 25) <= 9u )
        goto LABEL_157;
    }
LABEL_214:
    if ( v122 != ++v130 )
      continue;
    break;
  }
  v32 = 1;
LABEL_210:
  if ( v167 != v166 )
    _libc_free((unsigned __int64)v167);
  v31 = s2;
  v33 = v137 - (_BYTE *)s2;
LABEL_35:
  if ( !v31 )
  {
LABEL_36:
    v34 = (unsigned __int64)v162;
    if ( v162 != (__int64 **)v164 )
      goto LABEL_37;
    goto LABEL_38;
  }
LABEL_101:
  j_j___libc_free_0(v31, v33);
  v34 = (unsigned __int64)v162;
  if ( v162 != (__int64 **)v164 )
LABEL_37:
    _libc_free(v34);
LABEL_38:
  if ( v132 )
    j_j___libc_free_0(v132, v134 - v132);
  sub_1832E90(v151);
  sub_1832E90(v145);
  return v32;
}
