// Function: sub_34DDB40
// Address: 0x34ddb40
//
__int64 __fastcall sub_34DDB40(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 **a3,
        unsigned __int64 a4,
        unsigned int a5)
{
  __int64 v5; // r10
  __int64 v9; // r12
  int v10; // eax
  int v11; // ebx
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  int v20; // ebx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r11
  unsigned __int8 *v25; // rdx
  unsigned __int8 **v26; // r9
  int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // r11
  int v30; // r12d
  int v31; // eax
  __int64 v32; // r10
  __int64 v33; // r12
  unsigned __int8 v34; // al
  __int64 v35; // r14
  __int64 v36; // rax
  int v37; // r8d
  __int64 v38; // rdi
  __int64 v39; // rax
  int *v40; // rdi
  __int64 v41; // rdi
  bool v42; // al
  bool v43; // zf
  __int64 v44; // r11
  unsigned __int8 v45; // al
  __int64 v46; // rdi
  bool v47; // al
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rsi
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rdi
  unsigned __int64 v54; // rbx
  unsigned int v55; // edx
  __int64 v56; // rax
  char v57; // al
  char v58; // al
  __int64 *v59; // r9
  __int64 *v60; // r10
  __int64 v61; // rax
  unsigned __int64 v62; // rbx
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // r8
  unsigned __int64 v68; // rax
  __int64 v69; // rcx
  unsigned __int8 *v70; // rdi
  __int64 v71; // r14
  __int64 v72; // rax
  __int64 v73; // r8
  unsigned __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rsi
  __int64 *v80; // rcx
  unsigned __int8 *v81; // rbx
  int v82; // r14d
  __int64 v83; // rax
  int v84; // ecx
  bool v85; // cc
  __int64 v86; // rdx
  __int64 v87; // rbx
  __int64 v88; // r11
  __int64 v89; // rdi
  bool v90; // al
  __int64 v91; // rsi
  unsigned int v92; // esi
  bool v93; // al
  _BYTE *v94; // rdi
  bool v95; // al
  __int64 *v96; // rcx
  _BYTE *v97; // rdi
  bool v98; // al
  char v99; // al
  __int64 v100; // r9
  __int64 v101; // r11
  __int64 v102; // rax
  unsigned int v103; // ecx
  int *v104; // rax
  int *v105; // rsi
  int v106; // edx
  char v107; // al
  __int64 v108; // rax
  unsigned int v109; // ecx
  char v110; // al
  __int64 *v111; // rdi
  __int64 v112; // rax
  char v113; // al
  __int64 *v114; // rdi
  __int64 v115; // rax
  __int64 v116; // rbx
  unsigned __int64 v117; // rdx
  unsigned __int64 v118; // rax
  __int64 v119; // r8
  __int64 v120; // r9
  __int64 v121; // r11
  unsigned __int64 v122; // r14
  int *v123; // rax
  int *v124; // rdx
  int *v125; // rax
  int *v126; // rcx
  int v127; // edx
  signed __int64 v128; // rcx
  char v129; // al
  unsigned int v130; // edx
  char v131; // al
  unsigned int v132; // edx
  char v133; // al
  unsigned int v134; // edx
  char v135; // al
  unsigned int v136; // edx
  char v137; // al
  __int64 i; // rcx
  __int64 v139; // rdi
  unsigned int v140; // edx
  char v141; // al
  __int64 v142; // [rsp+8h] [rbp-188h]
  __int64 v143; // [rsp+10h] [rbp-180h]
  __int64 v144; // [rsp+18h] [rbp-178h]
  unsigned int v145; // [rsp+18h] [rbp-178h]
  __int64 v146; // [rsp+20h] [rbp-170h]
  __int64 v147; // [rsp+20h] [rbp-170h]
  size_t n; // [rsp+28h] [rbp-168h]
  size_t na; // [rsp+28h] [rbp-168h]
  __int64 nb; // [rsp+28h] [rbp-168h]
  __int64 v151; // [rsp+30h] [rbp-160h]
  __int64 v152; // [rsp+30h] [rbp-160h]
  __int64 v153; // [rsp+38h] [rbp-158h]
  __int64 v154; // [rsp+38h] [rbp-158h]
  __int64 v155; // [rsp+38h] [rbp-158h]
  __int64 v156; // [rsp+38h] [rbp-158h]
  __int64 v157; // [rsp+38h] [rbp-158h]
  __int64 v158; // [rsp+38h] [rbp-158h]
  __int64 v159; // [rsp+40h] [rbp-150h]
  __int64 v160; // [rsp+40h] [rbp-150h]
  __int64 v161; // [rsp+40h] [rbp-150h]
  unsigned int v162; // [rsp+40h] [rbp-150h]
  __int64 v163; // [rsp+40h] [rbp-150h]
  __int64 v164; // [rsp+40h] [rbp-150h]
  __int64 v165; // [rsp+40h] [rbp-150h]
  __int64 v166; // [rsp+40h] [rbp-150h]
  __int64 v167; // [rsp+50h] [rbp-140h]
  __int64 v168; // [rsp+50h] [rbp-140h]
  __int64 v169; // [rsp+50h] [rbp-140h]
  unsigned __int8 **v170; // [rsp+50h] [rbp-140h]
  __int64 v171; // [rsp+50h] [rbp-140h]
  __int64 v172; // [rsp+50h] [rbp-140h]
  __int64 *v173; // [rsp+50h] [rbp-140h]
  __int64 v174; // [rsp+50h] [rbp-140h]
  __int64 v175; // [rsp+50h] [rbp-140h]
  unsigned __int8 *v176; // [rsp+50h] [rbp-140h]
  __int64 v177; // [rsp+50h] [rbp-140h]
  unsigned __int64 v178; // [rsp+50h] [rbp-140h]
  unsigned __int8 *src; // [rsp+58h] [rbp-138h]
  unsigned __int8 *srch; // [rsp+58h] [rbp-138h]
  unsigned __int8 **srca; // [rsp+58h] [rbp-138h]
  void *srci; // [rsp+58h] [rbp-138h]
  void *srcj; // [rsp+58h] [rbp-138h]
  unsigned __int8 *srck; // [rsp+58h] [rbp-138h]
  _QWORD *srcb; // [rsp+58h] [rbp-138h]
  unsigned __int8 *srcc; // [rsp+58h] [rbp-138h]
  unsigned __int8 **srcl; // [rsp+58h] [rbp-138h]
  int *srcd; // [rsp+58h] [rbp-138h]
  unsigned __int8 *srce; // [rsp+58h] [rbp-138h]
  __int64 *srcm; // [rsp+58h] [rbp-138h]
  _QWORD *srcf; // [rsp+58h] [rbp-138h]
  void *srco; // [rsp+58h] [rbp-138h]
  void *srcg; // [rsp+58h] [rbp-138h]
  void *srcp; // [rsp+58h] [rbp-138h]
  void *srcn; // [rsp+58h] [rbp-138h]
  int v196; // [rsp+60h] [rbp-130h] BYREF
  int v197; // [rsp+64h] [rbp-12Ch] BYREF
  int v198; // [rsp+68h] [rbp-128h] BYREF
  unsigned int v199; // [rsp+6Ch] [rbp-124h] BYREF
  int *v200; // [rsp+70h] [rbp-120h] BYREF
  __int64 v201; // [rsp+78h] [rbp-118h]
  _BYTE v202[64]; // [rsp+80h] [rbp-110h] BYREF
  unsigned __int64 v203; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v204; // [rsp+C8h] [rbp-C8h]
  _BYTE *v205; // [rsp+D0h] [rbp-C0h] BYREF
  int *v206; // [rsp+D8h] [rbp-B8h]
  char v207; // [rsp+E8h] [rbp-A8h] BYREF
  char *v208; // [rsp+108h] [rbp-88h]
  char v209; // [rsp+118h] [rbp-78h] BYREF

  v5 = a1;
  v9 = (__int64)a2;
  v10 = *a2;
  if ( (unsigned __int8)v10 <= 0x1Cu )
  {
    if ( (_BYTE)v10 != 5 )
    {
      if ( a5 != 4 )
        return -(__int64)(a5 == 0) | 1;
      return 0;
    }
    v11 = *((unsigned __int16 *)a2 + 1);
    if ( a5 == 4 )
      return 0;
    v24 = *((_QWORD *)a2 + 1);
    v25 = 0;
    a2 = 0;
    goto LABEL_23;
  }
  v11 = (unsigned __int8)v10 - 29;
  if ( a5 == 4 )
    return 0;
  if ( (unsigned __int8)(v10 - 34) > 0x33u || (v13 = 0x8000000000041LL, !_bittest64(&v13, (unsigned int)(v10 - 34))) )
  {
    v24 = *((_QWORD *)a2 + 1);
    v25 = a2;
    a2 = 0;
LABEL_23:
    v26 = a3;
    switch ( v11 )
    {
      case 1:
      case 2:
      case 3:
      case 55:
        return v11 != 55 || a5 == 0;
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
        v167 = v24;
        src = v25;
        v27 = sub_DFB770(*a3);
        v28 = (__int64)src;
        v29 = v167;
        v30 = v27;
        v31 = 0;
        v32 = a1;
        if ( v11 != 12 )
        {
          v31 = sub_DFB770(a3[1]);
          v32 = a1;
          v29 = v167;
          v28 = (__int64)src;
        }
        return sub_34D2250(v32 - 8, v11, v29, a5, v30, v31, a3, a4, v28);
      case 31:
        if ( sub_B4D040(v9) )
          return 0;
        return -(__int64)(a5 == 0) | 1;
      case 32:
        if ( a5 == 1 )
          return 4;
        if ( a5 == 2 )
        {
          v76 = *(_QWORD *)(v9 + 16);
          if ( v76 )
          {
            if ( !*(_QWORD *)(v76 + 8) && (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 > 1 )
            {
              v77 = *(_QWORD *)(v76 + 24);
              if ( *(_BYTE *)v77 == 67 )
                v24 = *(_QWORD *)(v77 + 8);
            }
          }
        }
        v65 = a1 - 8;
        v66 = *(_QWORD *)(*(_QWORD *)(v9 - 32) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v66 + 8) - 17 <= 1 )
          v66 = **(_QWORD **)(v66 + 16);
        v67 = *(_DWORD *)(v66 + 8) >> 8;
        _BitScanReverse64(&v68, 1LL << (*(_WORD *)(v9 + 2) >> 1));
        v69 = (unsigned __int8)(63 - (v68 ^ 0x3F));
        BYTE1(v69) = 1;
        return sub_34D2F80(v65, 32, v24, v69, v67, a5);
      case 33:
        v70 = *a3;
        v174 = v5;
        v71 = *((_QWORD *)*a3 + 1);
        sub_DFB770(v70);
        v72 = *(_QWORD *)(*(_QWORD *)(v9 - 32) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v72 + 8) - 17 <= 1 )
          v72 = **(_QWORD **)(v72 + 16);
        v73 = *(_DWORD *)(v72 + 8) >> 8;
        _BitScanReverse64(&v74, 1LL << (*(_WORD *)(v9 + 2) >> 1));
        v75 = (unsigned __int8)(63 - (v74 ^ 0x3F));
        BYTE1(v75) = 1;
        return sub_34D2F80(v174 - 8, 33, v71, v75, v73, a5);
      case 34:
        srce = v25;
        v58 = sub_BD36B0(v9);
        v59 = (__int64 *)a3;
        v60 = (__int64 *)a1;
        if ( srce && v58 )
        {
          v61 = sub_B46690(*(_QWORD *)(*((_QWORD *)srce + 2) + 24LL));
          v59 = (__int64 *)a3;
          v60 = (__int64 *)a1;
          v62 = v61;
        }
        else
        {
          v62 = 0;
        }
        v63 = *v59;
        v173 = v60;
        srcm = v59;
        v64 = sub_BB5290(v9);
        return sub_34D1940(v173, v64, v63, (__int64)(srcm + 1), a4 - 1, v62);
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 50:
        v168 = v24;
        srch = v25;
        v33 = *((_QWORD *)*a3 + 1);
        v34 = sub_DFBCC0(v25);
        return sub_34D3270(a1 - 8, v11, v168, v33, v34, a5, srch);
      case 53:
      case 54:
        v169 = (__int64)v25;
        srca = a3;
        v35 = sub_DFB770(*a3);
        v36 = sub_DFB770(srca[1]);
        v37 = 42;
        v38 = a1 - 8;
        if ( v169 )
          v37 = *(_WORD *)(v169 + 2) & 0x3F;
        return sub_34D1290(v38, v11, *((__int64 **)*srca + 1), *(_QWORD *)(v9 + 8), v37, a5, v35, v36, v169);
      case 56:
        v39 = *(_QWORD *)(v9 - 32);
        if ( !v39 || *(_BYTE *)v39 || *(_QWORD *)(v39 + 24) != *(_QWORD *)(v9 + 80) )
          BUG();
        sub_DF86E0((__int64)&v203, *(_DWORD *)(v39 + 36), a2, 0, 1, 0, 0);
        result = sub_34D6FB0(a1 - 8, (__int64)&v203, a5);
        if ( v208 != &v209 )
        {
          srci = (void *)result;
          _libc_free((unsigned __int64)v208);
          result = (__int64)srci;
        }
        v40 = v206;
        if ( v206 != (int *)&v207 )
          goto LABEL_54;
        return result;
      case 57:
        if ( (unsigned __int8)v10 <= 0x1Cu )
          goto LABEL_70;
        v41 = v24;
        if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
          v41 = **(_QWORD **)(v24 + 16);
        v153 = v5;
        v159 = v24;
        srck = v25;
        v42 = sub_BCAC40(v41, 1);
        v25 = srck;
        v26 = a3;
        v43 = !v42;
        v44 = v159;
        v5 = v153;
        v45 = *(_BYTE *)v9;
        if ( v43 )
          goto LABEL_127;
        if ( v45 == 57 )
        {
          if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
            v80 = *(__int64 **)(v9 - 8);
          else
            v80 = (__int64 *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
          v50 = *v80;
          if ( !*v80 || (v81 = (unsigned __int8 *)v80[4]) == 0 )
          {
LABEL_128:
            v46 = *(_QWORD *)(v9 + 8);
            goto LABEL_64;
          }
        }
        else
        {
          if ( v45 != 86 )
            goto LABEL_127;
          v46 = *(_QWORD *)(v9 + 8);
          srcb = *(_QWORD **)(v9 - 96);
          if ( srcb[1] != v46 || **(_BYTE **)(v9 - 32) > 0x15u )
          {
LABEL_64:
            if ( (unsigned int)*(unsigned __int8 *)(v46 + 8) - 17 <= 1 )
              v46 = **(_QWORD **)(v46 + 16);
            v160 = v5;
            v170 = v26;
            srcc = v25;
            v154 = v44;
            v47 = sub_BCAC40(v46, 1);
            v25 = srcc;
            v26 = v170;
            v5 = v160;
            if ( !v47 )
              goto LABEL_70;
            v44 = v154;
            if ( *(_BYTE *)v9 == 58 )
            {
              if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
                v96 = *(__int64 **)(v9 - 8);
              else
                v96 = (__int64 *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
              v50 = *v96;
              if ( !*v96 )
                goto LABEL_70;
              v81 = (unsigned __int8 *)v96[4];
              if ( !v81 )
                goto LABEL_70;
            }
            else
            {
              if ( *(_BYTE *)v9 != 86 )
                goto LABEL_70;
              v50 = *(_QWORD *)(v9 - 96);
              if ( *(_QWORD *)(v50 + 8) != *(_QWORD *)(v9 + 8) )
                goto LABEL_70;
              v97 = *(_BYTE **)(v9 - 64);
              if ( *v97 > 0x15u )
                goto LABEL_70;
              v81 = *(unsigned __int8 **)(v9 - 32);
              v98 = sub_AD7A80(v97, v50, (__int64)srcc, v48, v49);
              v25 = srcc;
              v26 = v170;
              v5 = v160;
              if ( !v98 )
                goto LABEL_70;
              v44 = v154;
              if ( !v81 )
                goto LABEL_70;
            }
            goto LABEL_113;
          }
          v81 = *(unsigned __int8 **)(v9 - 64);
          v176 = v25;
          v93 = sub_AC30F0(*(_QWORD *)(v9 - 32));
          v25 = v176;
          v26 = a3;
          v44 = v159;
          v5 = v153;
          v50 = (__int64)srcb;
          if ( !v93 || !v81 )
          {
            v45 = *(_BYTE *)v9;
LABEL_127:
            if ( v45 <= 0x1Cu )
            {
LABEL_70:
              v161 = v5;
              v171 = (__int64)v25;
              srcl = v26;
              v51 = sub_DFB770(v26[1]);
              v52 = sub_DFB770(srcl[2]);
              return sub_34D1290(v161 - 8, 57, *(__int64 **)(v9 + 8), *((_QWORD *)*srcl + 1), 42, a5, v51, v52, v171);
            }
            goto LABEL_128;
          }
        }
LABEL_113:
        v156 = v5;
        v163 = v44;
        v175 = (__int64)v25;
        v82 = sub_DFB770((unsigned __int8 *)v50);
        v83 = sub_DFB770(v81);
        v84 = v83;
        v85 = *(_BYTE *)v9 <= 0x1Cu;
        v206 = (int *)v81;
        v203 = (unsigned __int64)&v205;
        v86 = v175;
        v87 = v156 - 8;
        v205 = (_BYTE *)v50;
        v88 = v163;
        v204 = 0x200000002LL;
        if ( v85 )
          goto LABEL_120;
        v89 = *(_QWORD *)(v9 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v89 + 8) - 17 <= 1 )
          v89 = **(_QWORD **)(v89 + 16);
        v157 = v83;
        v90 = sub_BCAC40(v89, 1);
        v86 = v175;
        v88 = v163;
        v84 = v157;
        if ( !v90 )
          goto LABEL_120;
        if ( *(_BYTE *)v9 == 58
          || *(_BYTE *)v9 == 86
          && (v91 = *(_QWORD *)(v9 + 8), *(_QWORD *)(*(_QWORD *)(v9 - 96) + 8LL) == v91)
          && (v94 = *(_BYTE **)(v9 - 64), *v94 <= 0x15u)
          && (v95 = sub_AD7A80(v94, v91, v175, v157, (__int64)&v205), v86 = v175, v88 = v163, v84 = v157, v95) )
        {
          v92 = 29;
        }
        else
        {
LABEL_120:
          v92 = 28;
        }
        result = sub_34D2250(v87, v92, v88, a5, v82, v84, &v205, 2u, v86);
        v40 = (int *)v203;
        if ( (_BYTE **)v203 == &v205 )
          return result;
        goto LABEL_54;
      case 61:
        if ( (_BYTE)v10 != 90 )
          return 1;
        v78 = a1 - 8;
        v79 = *((_QWORD *)*a3 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v79 + 8) - 17 <= 1 )
          v79 = **(_QWORD **)(v79 + 16);
        return (unsigned int)sub_34D06B0(v78, (__int64 *)v79);
      case 62:
        if ( (_BYTE)v10 != 91 )
          return 1;
        v53 = a1 - 8;
        if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
          v24 = **(_QWORD **)(v24 + 16);
        return (unsigned int)sub_34D06B0(v53, (__int64 *)v24);
      case 63:
        if ( (_BYTE)v10 != 92 )
          return 1;
        v155 = *((_QWORD *)*a3 + 1);
        srcd = *(int **)(v9 + 72);
        v162 = *(_DWORD *)(v9 + 80);
        v54 = v162;
        v172 = v162;
        v55 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL) + 32LL);
        if ( v55 != v162 )
        {
          if ( v55 >= v162 )
          {
            v56 = v24;
          }
          else
          {
            n = v24;
            if ( (unsigned __int8)sub_B4F540(v9) )
              return 0;
            v56 = *(_QWORD *)(v9 + 8);
            v26 = a3;
            v24 = n;
            v5 = a1;
          }
          v151 = v5 - 8;
          if ( *(_BYTE *)(v56 + 8) != 18 )
          {
            v146 = v24;
            na = (size_t)v26;
            v57 = sub_B4EFF0(
                    *(int **)(v9 + 72),
                    *(unsigned int *)(v9 + 80),
                    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL) + 32LL),
                    &v197);
            v26 = (unsigned __int8 **)na;
            v24 = v146;
            if ( v57 )
              return sub_34D5BE0(v151, 5, v155, srcd, v162, a5, v197, v146);
            if ( *(_BYTE *)(*(_QWORD *)(v9 + 8) + 8LL) != 18 )
            {
              v113 = sub_B4F0B0(
                       *(int **)(v9 + 72),
                       *(unsigned int *)(v9 + 80),
                       *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL) + 32LL),
                       &v196,
                       &v197);
              v26 = (unsigned __int8 **)na;
              v24 = v146;
              if ( v113 )
              {
                v114 = (__int64 *)v146;
                if ( (unsigned int)*(unsigned __int8 *)(v146 + 8) - 17 <= 1 )
                  v114 = **(__int64 ***)(v146 + 16);
                v115 = sub_BCDA70(v114, v196);
                return sub_34D5BE0(v151, 4, v146, srcd, v162, a5, v197, v115);
              }
            }
          }
          v144 = v24;
          v147 = (__int64)v26;
          v99 = sub_B4F660(v9, &v198, (int *)&v199);
          v100 = v147;
          v101 = v144;
          nb = 4LL * v162;
          if ( v99 )
          {
            LODWORD(v204) = v162;
            if ( v162 > 0x40 )
              sub_C43690((__int64)&v203, 0, 0);
            else
              v203 = 0;
            if ( srcd != &srcd[(unsigned __int64)nb / 4] )
            {
              for ( i = 0; ; ++i )
              {
                if ( srcd[i] != -1 )
                {
                  v139 = 1LL << i;
                  if ( (unsigned int)v204 <= 0x40 )
                    v203 |= v139;
                  else
                    *(_QWORD *)(v203 + 8LL * ((unsigned int)i >> 6)) |= v139;
                }
                if ( (unsigned __int64)(nb - 4) >> 2 == i )
                  break;
              }
            }
            v178 = sub_34D1730(v151, *(__int64 **)(v155 + 24), v198, v199, (__int64)&v203);
            sub_969240((__int64 *)&v203);
            return v178;
          }
          v145 = **(unsigned __int8 **)(v147 + 8) - 12;
          v196 = *(_DWORD *)(v155 + 32);
          v200 = (int *)v202;
          v201 = 0x1000000000LL;
          if ( (unsigned __int64)nb > 0x40 )
          {
            v142 = v101;
            sub_C8D5F0((__int64)&v200, v202, nb >> 2, 4u, (__int64)&v200, v147);
            v101 = v142;
          }
          else if ( !nb )
          {
            goto LABEL_147;
          }
          v143 = v101;
          memcpy(&v200[(unsigned int)v201], srcd, nb);
          v101 = v143;
LABEL_147:
          v102 = *(_QWORD *)(v9 - 64);
          v103 = *(_DWORD *)(v9 + 80);
          LODWORD(v201) = (nb >> 2) + v201;
          if ( *(_DWORD *)(*(_QWORD *)(v102 + 8) + 32LL) >= v103 )
          {
            v116 = v196 - (unsigned __int64)v162;
            v117 = v116 + (unsigned int)v201;
            if ( v117 > HIDWORD(v201) )
            {
              srcn = (void *)v101;
              sub_C8D5F0((__int64)&v200, v202, v117, 4u, (__int64)&v200, v100);
              v101 = (__int64)srcn;
            }
            if ( v116 )
            {
              srco = (void *)v101;
              memset(&v200[(unsigned int)v201], 255, 4 * v116);
              v101 = (__int64)srco;
            }
            LODWORD(v201) = v116 + v201;
            srcg = (void *)v101;
            v118 = sub_34D5BE0(v151, 7 - (unsigned int)(v145 >= 2), v155, v200, (unsigned int)v201, a5, 0, 0);
            v203 = (unsigned __int64)&v205;
            v121 = (__int64)srcg;
            v122 = v118;
            v204 = 0x1000000000LL;
            if ( v162 )
            {
              if ( v162 > 0x10uLL )
              {
                sub_C8D5F0((__int64)&v203, &v205, v162, 4u, v119, v120);
                v121 = (__int64)srcg;
              }
              v123 = (int *)(v203 + 4LL * (unsigned int)v204);
              v124 = (int *)(nb + v203);
              while ( v124 != v123 )
              {
                if ( v123 )
                  *v123 = 0;
                ++v123;
              }
              LODWORD(v204) = v162;
            }
            v125 = (int *)v203;
            v126 = (int *)(v203 + 4LL * (unsigned int)v204);
            v127 = 0;
            while ( v125 != v126 )
              *v125++ = v127++;
            v128 = sub_34D5BE0(v151, 5, v155, (int *)v203, (unsigned int)v204, a5, 0, v121);
            result = v128 + v122;
            if ( __OFADD__(v128, v122) )
            {
              result = 0x7FFFFFFFFFFFFFFFLL;
              if ( v128 <= 0 )
                result = 0x8000000000000000LL;
            }
            if ( (_BYTE **)v203 != &v205 )
            {
              srcp = (void *)result;
              _libc_free(v203);
              result = (__int64)srcp;
            }
          }
          else
          {
            v104 = v200;
            v105 = &v200[(unsigned int)v201];
            if ( v105 != v200 )
            {
              do
              {
                v106 = *v104;
                if ( *v104 >= v196 )
                  v106 = *v104 + v162 - v196;
                *v104++ = v106;
              }
              while ( v104 != v105 );
            }
            result = sub_34D5BE0(v151, 7 - (unsigned int)(v145 >= 2), v101, v200, (unsigned int)v201, a5, 0, 0);
          }
          v40 = v200;
          if ( v200 == (int *)v202 )
            return result;
LABEL_54:
          srcj = (void *)result;
          _libc_free((unsigned __int64)v40);
          return (__int64)srcj;
        }
        if ( *(_BYTE *)(v24 + 8) == 18 )
          goto LABEL_190;
        v164 = v24;
        v107 = sub_B4ED80(srcd, v172, v55);
        v24 = v164;
        v5 = a1;
        if ( v107 )
          return 0;
        v108 = *(_QWORD *)(v9 - 64);
        v109 = *(_DWORD *)(v9 + 80);
        v177 = a1 - 8;
        v162 = v109;
        if ( *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL) != v109 )
          goto LABEL_158;
        v172 = v109;
LABEL_190:
        v152 = v5;
        v158 = v24;
        v129 = sub_B4EDA0(*(int **)(v9 + 72), v172, v162);
        v24 = v158;
        v177 = v152 - 8;
        if ( v129 )
          return sub_34D5BE0(v177, 1, v158, srcd, v54, a5, 0, 0);
        v108 = *(_QWORD *)(v9 - 64);
        v130 = *(_DWORD *)(v9 + 80);
        if ( *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL) != v130 )
          goto LABEL_158;
        v131 = sub_B4EEA0(*(int **)(v9 + 72), v130, v130);
        v24 = v158;
        if ( v131 )
          return sub_34D5BE0(v177, 2, v158, srcd, v54, a5, 0, 0);
        v108 = *(_QWORD *)(v9 - 64);
        v132 = *(_DWORD *)(v9 + 80);
        if ( *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL) != v132 )
          goto LABEL_158;
        v133 = sub_B4EF10(*(_DWORD **)(v9 + 72), v132, v132);
        v24 = v158;
        if ( v133 )
          return sub_34D5BE0(v177, 3, v158, srcd, v54, a5, 0, 0);
        v108 = *(_QWORD *)(v9 - 64);
        v134 = *(_DWORD *)(v9 + 80);
        if ( *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL) != v134 )
          goto LABEL_158;
        v135 = sub_B4EE20(*(int **)(v9 + 72), v134, v134);
        v24 = v158;
        if ( v135 )
          return sub_34D5BE0(v177, 0, v158, srcd, v54, a5, 0, 0);
        v108 = *(_QWORD *)(v9 - 64);
        v136 = *(_DWORD *)(v9 + 80);
        if ( *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL) == v136 )
        {
          v137 = sub_B4ED30(*(int **)(v9 + 72), v136, v136);
          v24 = v158;
          if ( v137 )
            return sub_34D5BE0(v177, 7, v158, srcd, v54, a5, 0, 0);
          if ( *(_BYTE *)(*(_QWORD *)(v9 + 8) + 8LL) == 18 )
          {
LABEL_215:
            v140 = *(_DWORD *)(v9 + 80);
            if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v9 - 64) + 8LL) + 32LL) == v140 )
            {
              v166 = v24;
              v141 = sub_B4EF80(*(_QWORD *)(v9 + 72), v140, v140, &v197);
              v24 = v166;
              if ( v141 )
                return sub_34D5BE0(v177, 8, v166, srcd, v54, a5, v197, 0);
            }
            return sub_34D5BE0(v177, 6, v24, srcd, v54, a5, 0, 0);
          }
          v108 = *(_QWORD *)(v9 - 64);
        }
        else
        {
LABEL_158:
          if ( *(_BYTE *)(*(_QWORD *)(v9 + 8) + 8LL) == 18 )
            return sub_34D5BE0(v177, 6, v24, srcd, v54, a5, 0, 0);
        }
        v165 = v24;
        v110 = sub_B4F0B0(
                 *(int **)(v9 + 72),
                 *(unsigned int *)(v9 + 80),
                 *(_DWORD *)(*(_QWORD *)(v108 + 8) + 32LL),
                 &v196,
                 &v197);
        v24 = v165;
        if ( v110 )
        {
          v111 = (__int64 *)v165;
          if ( (unsigned int)*(unsigned __int8 *)(v165 + 8) - 17 <= 1 )
            v111 = **(__int64 ***)(v165 + 16);
          v112 = sub_BCDA70(v111, v196);
          return sub_34D5BE0(v177, 4, v165, srcd, v54, a5, v197, v112);
        }
        goto LABEL_215;
      case 64:
      case 65:
        result = 0;
        if ( v11 == 65 )
          return -(__int64)(a5 == 0) | 1;
        return result;
      case 67:
        return 0;
      default:
        return -(__int64)(a5 == 0) | 1;
    }
  }
  v14 = *((_QWORD *)a2 - 4);
  if ( (_BYTE)v10 == 85 )
  {
    if ( !v14 || *(_BYTE *)v14 )
    {
LABEL_12:
      if ( (unsigned __int8)v10 == 40 )
      {
        v16 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
      }
      else
      {
        v16 = 0;
        if ( (unsigned __int8)v10 != 85 )
        {
          v16 = 64;
          if ( (unsigned __int8)v10 != 34 )
            goto LABEL_225;
        }
      }
      if ( (a2[7] & 0x80u) != 0 )
      {
        v17 = sub_BD2BC0((__int64)a2);
        v19 = v17 + v18;
        if ( (a2[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v19 >> 4) )
            goto LABEL_225;
        }
        else if ( (unsigned int)((v19 - sub_BD2BC0((__int64)a2)) >> 4) )
        {
          if ( (a2[7] & 0x80u) != 0 )
          {
            v20 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
            if ( (a2[7] & 0x80u) == 0 )
              BUG();
            v21 = sub_BD2BC0((__int64)a2);
            v23 = 32LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20);
            return (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v16 - v23) >> 5) + 1;
          }
LABEL_225:
          BUG();
        }
      }
      v23 = 0;
      return (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v16 - v23) >> 5) + 1;
    }
    v15 = *(_QWORD *)(v14 + 24);
    if ( v15 != *((_QWORD *)a2 + 10) || (*(_BYTE *)(v14 + 33) & 0x20) == 0 )
      goto LABEL_11;
    v24 = *((_QWORD *)a2 + 1);
    v25 = a2;
    goto LABEL_23;
  }
  if ( !v14 || *(_BYTE *)v14 )
    goto LABEL_12;
  v15 = *(_QWORD *)(v14 + 24);
LABEL_11:
  if ( v15 != *((_QWORD *)a2 + 10) )
    goto LABEL_12;
  srcf = (_QWORD *)*((_QWORD *)a2 - 4);
  if ( (unsigned __int8)sub_DF7D80(a1, srcf) )
    return *(unsigned int *)(srcf[3] + 12LL);
  else
    return 1;
}
