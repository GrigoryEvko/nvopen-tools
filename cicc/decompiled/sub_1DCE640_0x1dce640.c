// Function: sub_1DCE640
// Address: 0x1dce640
//
__int64 __fastcall sub_1DCE640(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned int v4; // esi
  __int64 v5; // rcx
  __int64 *v6; // r9
  unsigned int v7; // ebx
  __int64 v8; // r8
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  _WORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  __int16 v15; // ax
  __int64 v16; // rdi
  __int64 v18; // rdx
  unsigned __int16 v19; // r12
  __int64 v20; // rdx
  _WORD *v21; // r15
  unsigned int v22; // r14d
  _DWORD *v23; // rcx
  _DWORD *v24; // rax
  __int16 v25; // ax
  unsigned int v26; // esi
  __int64 v27; // rdi
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // rcx
  unsigned int v31; // eax
  int *v32; // r13
  unsigned int v33; // edx
  int *v34; // rax
  _BOOL4 v35; // r8d
  __int64 v36; // rax
  __int64 v37; // r12
  char *v38; // r14
  unsigned int v39; // r15d
  int *i; // r13
  unsigned int v41; // edx
  int *v42; // rax
  _BOOL4 v43; // r12d
  __int64 v44; // rax
  bool v45; // zf
  unsigned int v46; // eax
  int *v47; // r8
  unsigned int v48; // r14d
  unsigned int v49; // edx
  int *v50; // rax
  _BOOL4 v51; // r13d
  __int64 v52; // rax
  unsigned int v53; // esi
  unsigned int v54; // r14d
  __int64 v55; // rcx
  __int64 *v56; // rax
  __int64 v57; // rdi
  unsigned int v58; // eax
  __int64 v59; // rax
  _WORD *v60; // rax
  __int16 *v61; // r14
  char *v62; // rax
  char *v63; // rdx
  __int16 v64; // ax
  __int64 v65; // rax
  int *v66; // rsi
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // rax
  _WORD *v75; // rax
  __int16 *v76; // r12
  __int16 *v77; // r13
  char *v78; // rsi
  char *v79; // rcx
  int v80; // r8d
  int v81; // edx
  char *v82; // rdi
  __int16 v83; // ax
  __int64 v84; // r12
  int *v85; // r14
  __int64 v86; // rdx
  bool v87; // al
  __int64 v88; // rsi
  __int64 v89; // rdi
  __int64 v90; // rcx
  __int64 v91; // rcx
  __int64 v92; // rdi
  __int64 v93; // rax
  unsigned int v94; // eax
  __int64 v95; // rdx
  __int64 v96; // rax
  unsigned __int16 v97; // cx
  __int16 v98; // dx
  int v99; // r11d
  int v100; // edx
  int v101; // r8d
  int v102; // esi
  unsigned int v103; // r12d
  __int64 *v104; // rcx
  __int64 v105; // rdi
  int v106; // r8d
  unsigned int v107; // ecx
  __int64 v108; // r11
  int v109; // edi
  __int64 *v110; // rsi
  int v111; // r10d
  int v112; // edx
  int v113; // r11d
  int v114; // r11d
  unsigned int v115; // ecx
  int v116; // edi
  __int64 *v117; // rsi
  int v118; // r11d
  int v119; // r11d
  unsigned int v120; // ecx
  int v121; // edi
  unsigned int v122; // eax
  __int64 v123; // rax
  __int64 v124; // rdx
  unsigned int v125; // eax
  __int64 v126; // rax
  int v127; // r8d
  int v128; // edx
  unsigned int v129; // ecx
  __int64 v130; // r11
  int v131; // ecx
  __int64 *v132; // rdx
  int v133; // edi
  int v134; // edi
  int v135; // esi
  unsigned int v136; // ebx
  __int64 *v137; // rcx
  int v138; // edi
  __int64 *v139; // rsi
  __int64 v141; // [rsp+20h] [rbp-130h]
  __int64 v142; // [rsp+28h] [rbp-128h]
  __int64 v143; // [rsp+30h] [rbp-120h]
  unsigned int v144; // [rsp+38h] [rbp-118h]
  unsigned int v145; // [rsp+3Ch] [rbp-114h]
  __int64 v146; // [rsp+40h] [rbp-110h]
  __int64 v147; // [rsp+48h] [rbp-108h]
  __int64 v148; // [rsp+50h] [rbp-100h]
  __int64 v149; // [rsp+58h] [rbp-F8h]
  _WORD *v150; // [rsp+60h] [rbp-F0h]
  unsigned __int16 v152; // [rsp+6Eh] [rbp-E2h]
  _BOOL4 v153; // [rsp+70h] [rbp-E0h]
  unsigned int v154; // [rsp+70h] [rbp-E0h]
  int *v155; // [rsp+70h] [rbp-E0h]
  int *v156; // [rsp+70h] [rbp-E0h]
  unsigned __int16 v157; // [rsp+70h] [rbp-E0h]
  __int16 *v158; // [rsp+78h] [rbp-D8h]
  __int16 *v159; // [rsp+78h] [rbp-D8h]
  unsigned __int16 v161; // [rsp+88h] [rbp-C8h]
  unsigned __int16 v162; // [rsp+88h] [rbp-C8h]
  __int64 v163; // [rsp+88h] [rbp-C8h]
  __int64 v164; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v165; // [rsp+98h] [rbp-B8h]
  __int64 v166; // [rsp+A0h] [rbp-B0h]
  __int64 v167; // [rsp+A8h] [rbp-A8h]
  __int64 v168; // [rsp+B0h] [rbp-A0h]
  void *dest; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v170; // [rsp+C8h] [rbp-88h]
  _BYTE v171[40]; // [rsp+D0h] [rbp-80h] BYREF
  int v172; // [rsp+F8h] [rbp-58h] BYREF
  int *v173; // [rsp+100h] [rbp-50h]
  int *v174; // [rsp+108h] [rbp-48h]
  int *v175; // [rsp+110h] [rbp-40h]
  __int64 v176; // [rsp+118h] [rbp-38h]

  v143 = 8LL * a2;
  v142 = a2;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + v143);
  v148 = *(_QWORD *)(*(_QWORD *)(a1 + 368) + v143);
  if ( !(v3 | v148) )
    return 0;
  v4 = *(_DWORD *)(a1 + 464);
  if ( !v3 )
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 368) + v143);
  v146 = a1 + 440;
  v147 = v3;
  v5 = v3;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 440);
    goto LABEL_231;
  }
  v6 = *(__int64 **)(a1 + 448);
  v7 = ((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9);
  v8 = (v4 - 1) & v7;
  v9 = &v6[2 * v8];
  v10 = *v9;
  if ( v5 != *v9 )
  {
    v131 = 1;
    v132 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v132 )
        v132 = v9;
      v8 = (v4 - 1) & (v131 + (_DWORD)v8);
      v9 = &v6[2 * (unsigned int)v8];
      v10 = *v9;
      if ( v147 == *v9 )
        goto LABEL_6;
      ++v131;
    }
    if ( v132 )
      v9 = v132;
    ++*(_QWORD *)(a1 + 440);
    v128 = *(_DWORD *)(a1 + 456) + 1;
    if ( 4 * v128 < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 460) - v128 > v4 >> 3 )
      {
LABEL_233:
        *(_DWORD *)(a1 + 456) = v128;
        if ( *v9 != -8 )
          --*(_DWORD *)(a1 + 460);
        *((_DWORD *)v9 + 2) = 0;
        v145 = 0;
        *v9 = v147;
        goto LABEL_7;
      }
      sub_1DC6D40(v146, v4);
      v133 = *(_DWORD *)(a1 + 464);
      if ( v133 )
      {
        v134 = v133 - 1;
        v6 = *(__int64 **)(a1 + 448);
        v135 = 1;
        v136 = v134 & v7;
        v128 = *(_DWORD *)(a1 + 456) + 1;
        v137 = 0;
        v9 = &v6[2 * v136];
        v8 = *v9;
        if ( v147 != *v9 )
        {
          while ( v8 != -8 )
          {
            if ( v8 == -16 && !v137 )
              v137 = v9;
            v136 = v134 & (v135 + v136);
            v9 = &v6[2 * v136];
            v8 = *v9;
            if ( v147 == *v9 )
              goto LABEL_233;
            ++v135;
          }
          if ( v137 )
            v9 = v137;
        }
        goto LABEL_233;
      }
LABEL_298:
      ++*(_DWORD *)(a1 + 456);
      BUG();
    }
LABEL_231:
    sub_1DC6D40(v146, 2 * v4);
    v127 = *(_DWORD *)(a1 + 464);
    if ( v127 )
    {
      v8 = (unsigned int)(v127 - 1);
      v6 = *(__int64 **)(a1 + 448);
      v128 = *(_DWORD *)(a1 + 456) + 1;
      v129 = v8 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
      v9 = &v6[2 * v129];
      v130 = *v9;
      if ( v147 != *v9 )
      {
        v138 = 1;
        v139 = 0;
        while ( v130 != -8 )
        {
          if ( !v139 && v130 == -16 )
            v139 = v9;
          v129 = v8 & (v138 + v129);
          v9 = &v6[2 * v129];
          v130 = *v9;
          if ( v147 == *v9 )
            goto LABEL_233;
          ++v138;
        }
        if ( v139 )
          v9 = v139;
      }
      goto LABEL_233;
    }
    goto LABEL_298;
  }
LABEL_6:
  v145 = *((_DWORD *)v9 + 2);
LABEL_7:
  v172 = 0;
  dest = v171;
  v170 = 0x800000000LL;
  v11 = *(_QWORD *)(a1 + 360);
  v173 = 0;
  v174 = &v172;
  v175 = &v172;
  v176 = 0;
  if ( !v11 )
    BUG();
  v12 = (_WORD *)(*(_QWORD *)(v11 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v11 + 8) + 24 * v142 + 4));
  v13 = *(_QWORD *)(a1 + 368);
  if ( !*v12 )
  {
    v141 = 0;
    goto LABEL_13;
  }
  v144 = 0;
  v161 = a2 + *v12;
  v158 = v12 + 1;
  v141 = 0;
  do
  {
    v14 = *(_QWORD *)(v13 + 8LL * v161);
    if ( v14 != 0 && v14 != v148 )
    {
      v53 = *(_DWORD *)(a1 + 464);
      if ( v53 )
      {
        v6 = *(__int64 **)(a1 + 448);
        v54 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
        LODWORD(v55) = (v53 - 1) & v54;
        v56 = &v6[2 * (unsigned int)v55];
        v57 = *v56;
        if ( v14 == *v56 )
        {
LABEL_89:
          v58 = *((_DWORD *)v56 + 2);
          if ( v58 > v144 )
          {
            v144 = v58;
            v141 = *(_QWORD *)(v13 + 8LL * v161);
          }
          goto LABEL_12;
        }
        v111 = 1;
        v8 = 0;
        while ( v57 != -8 )
        {
          if ( v57 == -16 && !v8 )
            v8 = (__int64)v56;
          v55 = (v53 - 1) & ((_DWORD)v55 + v111);
          v56 = &v6[2 * v55];
          v57 = *v56;
          if ( v14 == *v56 )
            goto LABEL_89;
          ++v111;
        }
        if ( v8 )
          v56 = (__int64 *)v8;
        ++*(_QWORD *)(a1 + 440);
        v112 = *(_DWORD *)(a1 + 456) + 1;
        if ( 4 * v112 < 3 * v53 )
        {
          if ( v53 - *(_DWORD *)(a1 + 460) - v112 > v53 >> 3 )
          {
LABEL_205:
            *(_DWORD *)(a1 + 456) = v112;
            if ( *v56 != -8 )
              --*(_DWORD *)(a1 + 460);
            *v56 = v14;
            *((_DWORD *)v56 + 2) = 0;
            v13 = *(_QWORD *)(a1 + 368);
            goto LABEL_12;
          }
          sub_1DC6D40(v146, v53);
          v118 = *(_DWORD *)(a1 + 464);
          if ( v118 )
          {
            v119 = v118 - 1;
            v6 = *(__int64 **)(a1 + 448);
            v117 = 0;
            v120 = v119 & v54;
            v112 = *(_DWORD *)(a1 + 456) + 1;
            v121 = 1;
            v56 = &v6[2 * (v119 & v54)];
            v8 = *v56;
            if ( v14 == *v56 )
              goto LABEL_205;
            while ( v8 != -8 )
            {
              if ( !v117 && v8 == -16 )
                v117 = v56;
              v120 = v119 & (v121 + v120);
              v56 = &v6[2 * v120];
              v8 = *v56;
              if ( v14 == *v56 )
                goto LABEL_205;
              ++v121;
            }
            goto LABEL_221;
          }
          goto LABEL_297;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 440);
      }
      sub_1DC6D40(v146, 2 * v53);
      v113 = *(_DWORD *)(a1 + 464);
      if ( v113 )
      {
        v114 = v113 - 1;
        v6 = *(__int64 **)(a1 + 448);
        v115 = v114 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v112 = *(_DWORD *)(a1 + 456) + 1;
        v56 = &v6[2 * v115];
        v8 = *v56;
        if ( v14 == *v56 )
          goto LABEL_205;
        v116 = 1;
        v117 = 0;
        while ( v8 != -8 )
        {
          if ( !v117 && v8 == -16 )
            v117 = v56;
          v115 = v114 & (v116 + v115);
          v56 = &v6[2 * v115];
          v8 = *v56;
          if ( v14 == *v56 )
            goto LABEL_205;
          ++v116;
        }
LABEL_221:
        if ( v117 )
          v56 = v117;
        goto LABEL_205;
      }
LABEL_297:
      ++*(_DWORD *)(a1 + 456);
      BUG();
    }
    v149 = *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * v161);
    if ( !v149 )
      goto LABEL_12;
    v18 = *(_QWORD *)(a1 + 360);
    if ( !v18 )
      BUG();
    v19 = v161;
    v20 = *(_QWORD *)(v18 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v18 + 8) + 24LL * v161 + 4);
LABEL_22:
    v21 = (_WORD *)v20;
    if ( !v20 )
      goto LABEL_31;
    do
    {
      v22 = v19;
      if ( v176 )
      {
        v32 = v173;
        if ( !v173 )
        {
          v32 = &v172;
          if ( v174 == &v172 )
          {
            v35 = 1;
            goto LABEL_44;
          }
LABEL_75:
          if ( (unsigned int)v19 <= *(_DWORD *)(sub_220EF80(v32) + 32) || !v32 )
            goto LABEL_29;
          v35 = 1;
          if ( v32 != &v172 )
            goto LABEL_78;
          goto LABEL_44;
        }
        while ( 1 )
        {
          v33 = v32[8];
          v34 = (int *)*((_QWORD *)v32 + 3);
          if ( v19 < v33 )
            v34 = (int *)*((_QWORD *)v32 + 2);
          if ( !v34 )
            break;
          v32 = v34;
        }
        if ( v19 < v33 )
        {
          if ( v174 != v32 )
            goto LABEL_75;
        }
        else if ( v19 <= v33 )
        {
          goto LABEL_29;
        }
        v35 = 1;
        if ( v32 != &v172 )
LABEL_78:
          v35 = v19 < (unsigned int)v32[8];
LABEL_44:
        v153 = v35;
        v36 = sub_22077B0(40);
        *(_DWORD *)(v36 + 32) = v19;
        sub_220F040(v153, v36, v32, &v172);
        ++v176;
        goto LABEL_29;
      }
      v23 = (char *)dest + 4 * (unsigned int)v170;
      if ( dest != v23 )
      {
        v24 = dest;
        while ( v19 != *v24 )
        {
          if ( v23 == ++v24 )
            goto LABEL_45;
        }
        if ( v23 != v24 )
          goto LABEL_29;
      }
LABEL_45:
      if ( (unsigned int)v170 <= 7uLL )
      {
        if ( (unsigned int)v170 >= HIDWORD(v170) )
        {
          sub_16CD150((__int64)&dest, v171, 0, 4, (int)v173, (int)v6);
          v23 = (char *)dest + 4 * (unsigned int)v170;
        }
        *v23 = v19;
        LODWORD(v170) = v170 + 1;
        goto LABEL_29;
      }
      v152 = v19;
      v37 = (__int64)v173;
      v154 = v22;
      v38 = (char *)dest + 4 * (unsigned int)v170 - 4;
      v150 = v21;
      if ( v173 )
      {
LABEL_47:
        v39 = *(_DWORD *)v38;
        for ( i = (int *)v37; ; i = v42 )
        {
          v41 = i[8];
          v42 = (int *)*((_QWORD *)i + 3);
          if ( v39 < v41 )
            v42 = (int *)*((_QWORD *)i + 2);
          if ( !v42 )
            break;
        }
        if ( v39 < v41 )
        {
          if ( v174 != i )
            goto LABEL_62;
        }
        else if ( v39 <= v41 )
        {
          goto LABEL_57;
        }
LABEL_54:
        v43 = 1;
        if ( i != &v172 )
          v43 = v39 < i[8];
LABEL_56:
        v44 = sub_22077B0(40);
        *(_DWORD *)(v44 + 32) = *(_DWORD *)v38;
        sub_220F040(v43, v44, i, &v172);
        ++v176;
        v37 = (__int64)v173;
        goto LABEL_57;
      }
      while ( 1 )
      {
        i = &v172;
        if ( v174 == &v172 )
        {
          v43 = 1;
          goto LABEL_56;
        }
        v39 = *(_DWORD *)v38;
LABEL_62:
        if ( v39 > *(_DWORD *)(sub_220EF80(i) + 32) )
          goto LABEL_54;
LABEL_57:
        v45 = (_DWORD)v170 == 1;
        v46 = v170 - 1;
        LODWORD(v170) = v170 - 1;
        if ( v45 )
          break;
        v38 = (char *)dest + 4 * v46 - 4;
        if ( v37 )
          goto LABEL_47;
      }
      v47 = (int *)v37;
      v48 = v154;
      v19 = v152;
      v21 = v150;
      if ( v47 )
      {
        while ( 1 )
        {
          v49 = v47[8];
          v50 = (int *)*((_QWORD *)v47 + 3);
          if ( v154 < v49 )
            v50 = (int *)*((_QWORD *)v47 + 2);
          if ( !v50 )
            break;
          v47 = v50;
        }
        if ( v154 < v49 )
        {
          if ( v47 == v174 )
            goto LABEL_72;
LABEL_83:
          v156 = v47;
          if ( v48 <= *(_DWORD *)(sub_220EF80(v47) + 32) )
            goto LABEL_29;
          v47 = v156;
          if ( !v156 )
            goto LABEL_29;
          v51 = 1;
          if ( v156 == &v172 )
            goto LABEL_73;
        }
        else
        {
          if ( v154 <= v49 )
            goto LABEL_29;
LABEL_72:
          v51 = 1;
          if ( v47 == &v172 )
            goto LABEL_73;
        }
        v51 = v48 < v47[8];
        goto LABEL_73;
      }
      v47 = &v172;
      if ( v174 != &v172 )
        goto LABEL_83;
      v51 = 1;
LABEL_73:
      v155 = v47;
      v52 = sub_22077B0(40);
      *(_DWORD *)(v52 + 32) = v48;
      sub_220F040(v51, v52, v155, &v172);
      ++v176;
LABEL_29:
      v25 = *v21;
      v20 = 0;
      ++v21;
      if ( !v25 )
        goto LABEL_22;
      v19 += v25;
    }
    while ( v21 );
LABEL_31:
    v26 = *(_DWORD *)(a1 + 464);
    if ( v26 )
    {
      v27 = *(_QWORD *)(a1 + 448);
      v8 = v26 - 1;
      v28 = v8 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( v149 == *v29 )
      {
LABEL_33:
        v31 = *((_DWORD *)v29 + 2);
        v13 = *(_QWORD *)(a1 + 368);
        if ( v31 > v145 )
        {
          v145 = v31;
          v147 = v149;
        }
        goto LABEL_12;
      }
      v99 = 1;
      v6 = 0;
      while ( v30 != -8 )
      {
        if ( !v6 && v30 == -16 )
          v6 = v29;
        v28 = v8 & (v99 + v28);
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( v149 == *v29 )
          goto LABEL_33;
        ++v99;
      }
      if ( v6 )
        v29 = v6;
      ++*(_QWORD *)(a1 + 440);
      v100 = *(_DWORD *)(a1 + 456) + 1;
      if ( 4 * v100 < 3 * v26 )
      {
        if ( v26 - *(_DWORD *)(a1 + 460) - v100 > v26 >> 3 )
          goto LABEL_182;
        sub_1DC6D40(v146, v26);
        v101 = *(_DWORD *)(a1 + 464);
        if ( v101 )
        {
          v8 = (unsigned int)(v101 - 1);
          v6 = *(__int64 **)(a1 + 448);
          v102 = 1;
          v103 = v8 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
          v100 = *(_DWORD *)(a1 + 456) + 1;
          v104 = 0;
          v29 = &v6[2 * v103];
          v105 = *v29;
          if ( v149 != *v29 )
          {
            while ( v105 != -8 )
            {
              if ( !v104 && v105 == -16 )
                v104 = v29;
              v103 = v8 & (v102 + v103);
              v29 = &v6[2 * v103];
              v105 = *v29;
              if ( v149 == *v29 )
                goto LABEL_182;
              ++v102;
            }
            if ( v104 )
              v29 = v104;
          }
          goto LABEL_182;
        }
LABEL_300:
        ++*(_DWORD *)(a1 + 456);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 440);
    }
    sub_1DC6D40(v146, 2 * v26);
    v106 = *(_DWORD *)(a1 + 464);
    if ( !v106 )
      goto LABEL_300;
    v8 = (unsigned int)(v106 - 1);
    v6 = *(__int64 **)(a1 + 448);
    v100 = *(_DWORD *)(a1 + 456) + 1;
    v107 = v8 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
    v29 = &v6[2 * v107];
    v108 = *v29;
    if ( v149 != *v29 )
    {
      v109 = 1;
      v110 = 0;
      while ( v108 != -8 )
      {
        if ( v108 == -16 && !v110 )
          v110 = v29;
        v107 = v8 & (v109 + v107);
        v29 = &v6[2 * v107];
        v108 = *v29;
        if ( v149 == *v29 )
          goto LABEL_182;
        ++v109;
      }
      if ( v110 )
        v29 = v110;
    }
LABEL_182:
    *(_DWORD *)(a1 + 456) = v100;
    if ( *v29 != -8 )
      --*(_DWORD *)(a1 + 460);
    *((_DWORD *)v29 + 2) = 0;
    *v29 = v149;
    v13 = *(_QWORD *)(a1 + 368);
LABEL_12:
    v15 = *v158;
    v161 += *v158++;
  }
  while ( v15 );
LABEL_13:
  v16 = *(_QWORD *)(v13 + 8 * v142);
  if ( !*(_QWORD *)(*(_QWORD *)(a1 + 392) + 8 * v142) )
  {
    sub_1E1B440(v16, a2, *(_QWORD *)(a1 + 360), 1);
    v59 = *(_QWORD *)(a1 + 360);
    if ( !v59 )
      BUG();
    v60 = (_WORD *)(*(_QWORD *)(v59 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v59 + 8) + 24 * v142 + 4));
    v61 = v60 + 1;
    v157 = *v60 + a2;
    if ( !*v60 )
      v61 = 0;
    while ( 1 )
    {
      v159 = v61;
      while ( 1 )
      {
        if ( !v159 )
          goto LABEL_17;
        if ( v176 )
        {
          v65 = (__int64)v173;
          if ( v173 )
          {
            v66 = &v172;
            do
            {
              while ( 1 )
              {
                v67 = *(_QWORD *)(v65 + 16);
                v68 = *(_QWORD *)(v65 + 24);
                if ( (unsigned int)v157 <= *(_DWORD *)(v65 + 32) )
                  break;
                v65 = *(_QWORD *)(v65 + 24);
                if ( !v68 )
                  goto LABEL_118;
              }
              v66 = (int *)v65;
              v65 = *(_QWORD *)(v65 + 16);
            }
            while ( v67 );
LABEL_118:
            if ( v66 != &v172 && v157 >= (unsigned int)v66[8] )
            {
LABEL_120:
              v69 = *(_QWORD *)(a1 + 368);
              v70 = *(_QWORD *)(v69 + v143);
              if ( v70 != *(_QWORD *)(v69 + 8LL * v157) )
                goto LABEL_121;
              v163 = *(_QWORD *)(v69 + v143);
              v94 = sub_1E16810(v70, v157, 0, 0, 0);
              if ( v94 == -1 || !(*(_QWORD *)(v163 + 32) + 40LL * v94) )
              {
                v70 = *(_QWORD *)(*(_QWORD *)(a1 + 368) + v143);
LABEL_121:
                v165 = v157;
                v164 = 805306368;
                v166 = 0;
                v167 = 0;
                v168 = 0;
                sub_1E1AFD0(v70, &v164);
              }
              v71 = sub_1DCDE50(a1, v157);
              if ( v71 )
              {
                sub_1E1AFE0(v71, v157, *(_QWORD *)(a1 + 360), 1, v72, v73);
              }
              else
              {
                sub_1E1AFE0(v147, v157, *(_QWORD *)(a1 + 360), 1, v72, v73);
                v95 = *(_QWORD *)(a1 + 360);
                if ( !v95 )
                  BUG();
                v96 = *(_QWORD *)(v95 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v95 + 8) + 24LL * v157 + 4);
                v97 = v157;
                while ( v96 )
                {
                  while ( 1 )
                  {
                    v96 += 2;
                    *(_QWORD *)(*(_QWORD *)(a1 + 392) + 8LL * v97) = v147;
                    v98 = *(_WORD *)(v96 - 2);
                    if ( !v98 )
                      break;
                    v97 += v98;
                    if ( !v96 )
                      goto LABEL_124;
                  }
                  v96 = 0;
                }
              }
LABEL_124:
              v74 = *(_QWORD *)(a1 + 360);
              if ( !v74 )
                BUG();
              v75 = (_WORD *)(*(_QWORD *)(v74 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v74 + 8) + 24LL * v157 + 4));
              v76 = v75 + 1;
              if ( !*v75 )
                v76 = 0;
              v162 = *v75 + v157;
              v77 = v76;
              while ( v77 )
              {
                if ( !v176 )
                {
                  v78 = (char *)dest;
                  v79 = (char *)dest + 4 * (unsigned int)v170;
                  v80 = v170;
                  if ( dest != v79 )
                  {
                    while ( 1 )
                    {
                      v81 = *(_DWORD *)v78;
                      v82 = v78;
                      v78 += 4;
                      if ( v162 == v81 )
                        break;
                      if ( v79 == v78 )
                        goto LABEL_137;
                    }
                    if ( v79 != v78 )
                    {
                      memmove(v82, v78, v79 - v78);
                      v80 = v170;
                    }
                    LODWORD(v170) = v80 - 1;
                  }
                  goto LABEL_137;
                }
                if ( v173 )
                {
                  v84 = (__int64)v173;
                  v85 = &v172;
                  while ( 1 )
                  {
                    while ( (unsigned int)v162 > *(_DWORD *)(v84 + 32) )
                    {
                      v84 = *(_QWORD *)(v84 + 24);
                      if ( !v84 )
                        goto LABEL_147;
                    }
                    v86 = *(_QWORD *)(v84 + 16);
                    if ( (unsigned int)v162 >= *(_DWORD *)(v84 + 32) )
                      break;
                    v85 = (int *)v84;
                    v84 = *(_QWORD *)(v84 + 16);
                    if ( !v86 )
                    {
LABEL_147:
                      v87 = v85 == &v172;
                      goto LABEL_148;
                    }
                  }
                  v88 = *(_QWORD *)(v84 + 24);
                  if ( v88 )
                  {
                    do
                    {
                      while ( 1 )
                      {
                        v89 = *(_QWORD *)(v88 + 16);
                        v90 = *(_QWORD *)(v88 + 24);
                        if ( (unsigned int)v162 < *(_DWORD *)(v88 + 32) )
                          break;
                        v88 = *(_QWORD *)(v88 + 24);
                        if ( !v90 )
                          goto LABEL_156;
                      }
                      v85 = (int *)v88;
                      v88 = *(_QWORD *)(v88 + 16);
                    }
                    while ( v89 );
                  }
LABEL_156:
                  while ( v86 )
                  {
                    while ( 1 )
                    {
                      v91 = *(_QWORD *)(v86 + 24);
                      if ( (unsigned int)v162 <= *(_DWORD *)(v86 + 32) )
                        break;
                      v86 = *(_QWORD *)(v86 + 24);
                      if ( !v91 )
                        goto LABEL_159;
                    }
                    v84 = v86;
                    v86 = *(_QWORD *)(v86 + 16);
                  }
LABEL_159:
                  if ( v174 != (int *)v84 || v85 != &v172 )
                  {
                    for ( ; v85 != (int *)v84; --v176 )
                    {
                      v92 = v84;
                      v84 = sub_220EF30(v84);
                      v93 = sub_220F330(v92, &v172);
                      j_j___libc_free_0(v93, 40);
                    }
                    goto LABEL_137;
                  }
                }
                else
                {
                  v87 = 1;
                  v85 = &v172;
LABEL_148:
                  if ( v174 != v85 || !v87 )
                    goto LABEL_137;
                }
                sub_1DCADB0((__int64)v173);
                v173 = 0;
                v174 = &v172;
                v175 = &v172;
                v176 = 0;
LABEL_137:
                v83 = *v77++;
                if ( v83 )
                  v162 += v83;
                else
                  v77 = 0;
                continue;
              }
            }
          }
        }
        else
        {
          v62 = (char *)dest;
          v63 = (char *)dest + 4 * (unsigned int)v170;
          if ( dest != v63 )
          {
            while ( v157 != *(_DWORD *)v62 )
            {
              v62 += 4;
              if ( v63 == v62 )
                goto LABEL_111;
            }
            if ( v62 != v63 )
              goto LABEL_120;
          }
        }
LABEL_111:
        v64 = *v159++;
        if ( !v64 )
          break;
        v157 += v64;
      }
      v61 = 0;
    }
  }
  if ( v147 == v16 && a3 != v147 )
  {
    if ( v141 )
    {
      v164 = 1610612736;
      v166 = 0;
      v165 = a2;
      v167 = 0;
      v168 = 0;
      sub_1E1AFD0(v141, &v164);
    }
    else
    {
      v122 = sub_1E16810(v147, a2, 0, 0, *(_QWORD *)(a1 + 360));
      if ( v122 == -1 )
        BUG();
      v123 = *(_QWORD *)(v147 + 32) + 40LL * v122;
      v124 = *(_QWORD *)(a1 + 360);
      if ( (*(_BYTE *)(v123 + 4) & 4) == 0 || a2 == *(_DWORD *)(v123 + 8) )
      {
        sub_1E1B440(v147, a2, v124, 1);
      }
      else
      {
        sub_1E1B440(v147, a2, v124, 1);
        v125 = sub_1E16810(v147, a2, 0, 0, 0);
        if ( v125 != -1 )
        {
          v126 = *(_QWORD *)(v147 + 32) + 40LL * v125;
          if ( v126 )
            *(_BYTE *)(v126 + 4) |= 4u;
        }
      }
    }
  }
  else
  {
    sub_1E1AFE0(v147, a2, *(_QWORD *)(a1 + 360), 1, v8, v6);
  }
LABEL_17:
  sub_1DCADB0((__int64)v173);
  if ( dest != v171 )
    _libc_free((unsigned __int64)dest);
  return 1;
}
