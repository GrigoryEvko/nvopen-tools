// Function: sub_DA8A30
// Address: 0xda8a30
//
__int64 __fastcall sub_DA8A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r9
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // ebx
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  int v15; // r10d
  _QWORD *v16; // r11
  int v17; // eax
  unsigned int v18; // ebx
  unsigned __int64 v19; // r12
  _QWORD *v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // r11
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // r15
  int v30; // r11d
  __int64 *v31; // rax
  unsigned int v32; // edi
  __int64 *v33; // rdx
  __int64 v34; // r8
  _QWORD *v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // r12
  unsigned int v38; // ecx
  int v39; // edx
  __int64 v40; // r9
  __int64 v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // r11
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rax
  int v48; // ecx
  __int64 v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // r10
  _BYTE *v52; // rbx
  __int64 v53; // rdx
  int v54; // esi
  unsigned int v55; // ecx
  int v56; // edx
  _QWORD *v57; // rax
  __int64 v58; // rdi
  _QWORD *v59; // rax
  __int64 v60; // rsi
  __int64 v61; // r9
  int v62; // r11d
  __int64 *v63; // rdx
  unsigned int v64; // ecx
  __int64 *v65; // rax
  __int64 v66; // rdi
  _BYTE *v67; // rax
  bool v68; // r15
  __int64 v69; // rdi
  int v70; // eax
  int v71; // eax
  unsigned int v72; // eax
  __int64 *v73; // rax
  __int64 *v74; // r13
  __int64 v75; // r12
  __int64 *v76; // rbx
  __int64 v77; // rsi
  __int64 *v78; // rax
  __int64 *v79; // rbx
  __int64 *v80; // r13
  __int64 v81; // r14
  int v82; // r11d
  _QWORD *v83; // rdx
  unsigned int v84; // r8d
  _QWORD *v85; // rax
  __int64 v86; // rdi
  unsigned __int8 *v87; // rdi
  unsigned __int8 **v88; // r9
  __int64 v89; // r12
  unsigned int v90; // eax
  int v91; // edi
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rax
  unsigned __int8 v95; // al
  _QWORD *v96; // r9
  int v97; // r10d
  unsigned int v98; // ecx
  __int64 v99; // rsi
  __int64 v100; // rdx
  __int64 *v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // rax
  int v105; // r11d
  unsigned int v106; // ecx
  _QWORD *v107; // rdx
  __int64 v108; // rdi
  unsigned int v109; // ecx
  int v110; // eax
  __int64 v111; // rdi
  __int64 v112; // r14
  __int64 *v113; // rdx
  _QWORD *v114; // rsi
  int v115; // r9d
  unsigned int v116; // ecx
  __int64 v117; // rdi
  unsigned int v118; // ecx
  __int64 v119; // rdi
  int v120; // edi
  __int64 *v121; // rsi
  unsigned int v122; // ecx
  __int64 v123; // r9
  int v124; // r10d
  int v125; // edi
  int v126; // edi
  __int64 v127; // r9
  unsigned int v128; // edx
  __int64 v129; // r8
  __int64 *v130; // r10
  int v131; // esi
  __int64 *v132; // rcx
  int v133; // eax
  int v134; // esi
  int v135; // esi
  __int64 v136; // rdi
  unsigned int v137; // ebx
  __int64 v138; // rcx
  __int64 *v139; // r9
  int v140; // edx
  __int64 *v141; // r8
  int v142; // edi
  int v143; // r9d
  __int64 v144; // [rsp+0h] [rbp-170h]
  unsigned __int8 **v145; // [rsp+8h] [rbp-168h]
  __int64 *v146; // [rsp+8h] [rbp-168h]
  _BYTE *v147; // [rsp+10h] [rbp-160h]
  _BYTE *v149; // [rsp+20h] [rbp-150h]
  __int64 *v150; // [rsp+28h] [rbp-148h]
  int v151; // [rsp+30h] [rbp-140h]
  unsigned int v152; // [rsp+34h] [rbp-13Ch]
  __int64 v154; // [rsp+40h] [rbp-130h]
  __int64 v155; // [rsp+40h] [rbp-130h]
  __int64 v156; // [rsp+40h] [rbp-130h]
  __int64 v157; // [rsp+40h] [rbp-130h]
  __int64 v158; // [rsp+40h] [rbp-130h]
  __int64 v159; // [rsp+40h] [rbp-130h]
  __int64 v160; // [rsp+40h] [rbp-130h]
  int v161; // [rsp+48h] [rbp-128h]
  __int64 v162; // [rsp+50h] [rbp-120h]
  __int64 v164; // [rsp+58h] [rbp-118h]
  __int64 v165; // [rsp+58h] [rbp-118h]
  __int64 v166; // [rsp+58h] [rbp-118h]
  __int64 v167; // [rsp+68h] [rbp-108h] BYREF
  __int64 v168; // [rsp+70h] [rbp-100h] BYREF
  __int64 *v169; // [rsp+78h] [rbp-F8h]
  __int64 v170; // [rsp+80h] [rbp-F0h]
  unsigned int v171; // [rsp+88h] [rbp-E8h]
  __int64 v172; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v173; // [rsp+98h] [rbp-D8h]
  __int64 v174; // [rsp+A0h] [rbp-D0h]
  unsigned int v175; // [rsp+A8h] [rbp-C8h]
  __int64 *v176; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-B8h]
  _BYTE v178[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v5 = a1 + 744;
  v7 = *(_DWORD *)(a1 + 768);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 752);
    v9 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v10 = (v7 - 1) & v9;
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      return v11[1];
    v15 = 1;
    v16 = 0;
    while ( v12 != -4096 )
    {
      if ( !v16 && v12 == -8192 )
        v16 = v11;
      v10 = (v7 - 1) & (v15 + v10);
      v11 = (_QWORD *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == a2 )
        return v11[1];
      ++v15;
    }
    if ( v16 )
      v11 = v16;
    v150 = v11;
    ++*(_QWORD *)(a1 + 744);
    v17 = *(_DWORD *)(a1 + 760) + 1;
    if ( 4 * v17 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 764) - v17 > v7 >> 3 )
        goto LABEL_11;
      sub_DA8850(v5, v7);
      v134 = *(_DWORD *)(a1 + 768);
      if ( v134 )
      {
        v135 = v134 - 1;
        v136 = *(_QWORD *)(a1 + 752);
        v137 = v135 & v9;
        v150 = (__int64 *)(v136 + 16LL * v137);
        v138 = *v150;
        v17 = *(_DWORD *)(a1 + 760) + 1;
        if ( *v150 != a2 )
        {
          v139 = (__int64 *)(v136 + 16LL * v137);
          v140 = 1;
          v141 = 0;
          while ( v138 != -4096 )
          {
            if ( v138 == -8192 && !v141 )
              v141 = v139;
            v137 = v135 & (v140 + v137);
            v139 = (__int64 *)(v136 + 16LL * v137);
            v138 = *v139;
            if ( *v139 == a2 )
            {
              v150 = (__int64 *)(v136 + 16LL * v137);
              goto LABEL_11;
            }
            ++v140;
          }
          if ( !v141 )
            v141 = v139;
          v150 = v141;
        }
        goto LABEL_11;
      }
LABEL_287:
      ++*(_DWORD *)(a1 + 760);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 744);
  }
  sub_DA8850(v5, 2 * v7);
  v125 = *(_DWORD *)(a1 + 768);
  if ( !v125 )
    goto LABEL_287;
  v126 = v125 - 1;
  v127 = *(_QWORD *)(a1 + 752);
  v128 = v126 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v150 = (__int64 *)(v127 + 16LL * v128);
  v129 = *v150;
  v17 = *(_DWORD *)(a1 + 760) + 1;
  if ( *v150 != a2 )
  {
    v130 = (__int64 *)(v127 + 16LL * (v126 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v131 = 1;
    v132 = 0;
    while ( v129 != -4096 )
    {
      if ( !v132 && v129 == -8192 )
        v132 = v130;
      v128 = v126 & (v131 + v128);
      v130 = (__int64 *)(v127 + 16LL * v128);
      v129 = *v130;
      if ( *v130 == a2 )
      {
        v150 = (__int64 *)(v127 + 16LL * v128);
        goto LABEL_11;
      }
      ++v131;
    }
    if ( !v132 )
      v132 = v130;
    v150 = v132;
  }
LABEL_11:
  *(_DWORD *)(a1 + 760) = v17;
  if ( *v150 != -4096 )
    --*(_DWORD *)(a1 + 764);
  v150[1] = 0;
  *v150 = a2;
  v18 = *(_DWORD *)(a3 + 8);
  v19 = (unsigned int)dword_4F89AE8;
  if ( v18 > 0x40 )
  {
    if ( v18 - (unsigned int)sub_C444A0(a3) > 0x40 )
      return 0;
    v20 = **(_QWORD ***)a3;
  }
  else
  {
    v20 = *(_QWORD **)a3;
  }
  if ( v19 < (unsigned __int64)v20 )
    return 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v21 = *(__int64 **)(a4 + 32);
  v171 = 0;
  v22 = *v21;
  v162 = *v21;
  v23 = sub_D47930(a4);
  v24 = v23;
  if ( !v23 )
    goto LABEL_38;
  v154 = v23;
  v25 = sub_AA5930(v22);
  v26 = v154;
  v27 = v25;
  v29 = v28;
  if ( v28 != v25 )
  {
    while ( 1 )
    {
      v37 = sub_D90990(v27, v154);
      if ( !v37 )
        goto LABEL_23;
      if ( !v171 )
        break;
      v30 = 1;
      v31 = 0;
      v32 = (v171 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v33 = &v169[2 * v32];
      v34 = *v33;
      if ( v27 != *v33 )
      {
        while ( v34 != -4096 )
        {
          if ( !v31 && v34 == -8192 )
            v31 = v33;
          v32 = (v171 - 1) & (v30 + v32);
          v33 = &v169[2 * v32];
          v34 = *v33;
          if ( v27 == *v33 )
            goto LABEL_21;
          ++v30;
        }
        if ( !v31 )
          v31 = v33;
        ++v168;
        v39 = v170 + 1;
        if ( 4 * ((int)v170 + 1) < 3 * v171 )
        {
          if ( v171 - HIDWORD(v170) - v39 <= v171 >> 3 )
          {
            sub_DA7360((__int64)&v168, v171);
            if ( !v171 )
            {
LABEL_288:
              LODWORD(v170) = v170 + 1;
              BUG();
            }
            v120 = 1;
            v121 = 0;
            v122 = (v171 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v39 = v170 + 1;
            v31 = &v169[2 * v122];
            v123 = *v31;
            if ( v27 != *v31 )
            {
              while ( v123 != -4096 )
              {
                if ( !v121 && v123 == -8192 )
                  v121 = v31;
                v122 = (v171 - 1) & (v120 + v122);
                v31 = &v169[2 * v122];
                v123 = *v31;
                if ( v27 == *v31 )
                  goto LABEL_33;
                ++v120;
              }
LABEL_196:
              if ( v121 )
                v31 = v121;
            }
          }
LABEL_33:
          LODWORD(v170) = v39;
          if ( *v31 != -4096 )
            --HIDWORD(v170);
          *v31 = v27;
          v35 = v31 + 1;
          *v35 = 0;
          goto LABEL_22;
        }
LABEL_31:
        sub_DA7360((__int64)&v168, 2 * v171);
        if ( !v171 )
          goto LABEL_288;
        v38 = (v171 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v39 = v170 + 1;
        v31 = &v169[2 * v38];
        v40 = *v31;
        if ( v27 != *v31 )
        {
          v142 = 1;
          v121 = 0;
          while ( v40 != -4096 )
          {
            if ( !v121 && v40 == -8192 )
              v121 = v31;
            v38 = (v171 - 1) & (v142 + v38);
            v31 = &v169[2 * v38];
            v40 = *v31;
            if ( v27 == *v31 )
              goto LABEL_33;
            ++v142;
          }
          goto LABEL_196;
        }
        goto LABEL_33;
      }
LABEL_21:
      v35 = v33 + 1;
LABEL_22:
      *v35 = v37;
LABEL_23:
      if ( !v27 )
        BUG();
      v36 = *(_QWORD *)(v27 + 32);
      if ( !v36 )
        BUG();
      v27 = 0;
      if ( *(_BYTE *)(v36 - 24) == 84 )
        v27 = v36 - 24;
      if ( v29 == v27 )
      {
        v26 = v154;
        goto LABEL_42;
      }
    }
    ++v168;
    goto LABEL_31;
  }
LABEL_42:
  if ( !v171 )
  {
LABEL_172:
    v24 = 0;
    v150[1] = 0;
    goto LABEL_38;
  }
  v152 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v44 = (v171 - 1) & v152;
  v45 = v169[2 * v44];
  if ( v45 != a2 )
  {
    v133 = 1;
    while ( v45 != -4096 )
    {
      LODWORD(v44) = (v171 - 1) & (v133 + v44);
      v45 = v169[2 * (unsigned int)v44];
      if ( v45 == a2 )
        goto LABEL_44;
      ++v133;
    }
    goto LABEL_172;
  }
LABEL_44:
  v46 = *(_QWORD *)(a2 - 8);
  v47 = 0x1FFFFFFFE0LL;
  v48 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v48 )
  {
    v49 = 0;
    do
    {
      if ( v26 == *(_QWORD *)(v46 + 32LL * *(unsigned int *)(a2 + 72) + 8 * v49) )
      {
        v47 = 32 * v49;
        goto LABEL_49;
      }
      ++v49;
    }
    while ( v48 != (_DWORD)v49 );
    v47 = 0x1FFFFFFFE0LL;
  }
LABEL_49:
  v149 = *(_BYTE **)(v46 + v47);
  v50 = *(_QWORD **)a3;
  if ( *(_DWORD *)(a3 + 8) > 0x40u )
    v50 = (_QWORD *)*v50;
  v151 = (int)v50;
  v51 = v26;
  v161 = 0;
  v147 = *(_BYTE **)(a1 + 8);
  if ( !(_DWORD)v50 )
  {
LABEL_66:
    v176 = (__int64 *)a2;
    v24 = *sub_DA7A60((__int64)&v168, (__int64 *)&v176);
    v150[1] = v24;
    goto LABEL_38;
  }
  while ( 2 )
  {
    v52 = v149;
    v53 = 1;
    v172 = 0;
    v173 = 0;
    v174 = 0;
    v175 = 0;
    if ( *v149 <= 0x15u )
      goto LABEL_53;
    v158 = v51;
    if ( *v149 <= 0x1Cu )
    {
      v41 = 0;
      v42 = 0;
      v43 = 0;
      goto LABEL_37;
    }
    v104 = sub_DA7540((__int64)v149, a4, (__int64)&v168, v147, *(__int64 **)(a1 + 24));
    v51 = v158;
    v52 = (_BYTE *)v104;
    if ( !v104 )
    {
      v41 = v173;
      v43 = 0;
      v42 = 16LL * v175;
      goto LABEL_37;
    }
    if ( !v175 )
    {
      v53 = v172 + 1;
LABEL_53:
      v172 = v53;
      v54 = 0;
LABEL_54:
      v155 = v51;
      sub_DA7360((__int64)&v172, v54);
      if ( !v175 )
        goto LABEL_283;
      v51 = v155;
      v55 = (v175 - 1) & v152;
      v56 = v174 + 1;
      v57 = (_QWORD *)(v173 + 16LL * v55);
      v58 = *v57;
      if ( *v57 == a2 )
        goto LABEL_56;
      v143 = 1;
      v114 = 0;
      while ( v58 != -4096 )
      {
        if ( !v114 && v58 == -8192 )
          v114 = v57;
        v55 = (v175 - 1) & (v143 + v55);
        v57 = (_QWORD *)(v173 + 16LL * v55);
        v58 = *v57;
        if ( *v57 == a2 )
          goto LABEL_56;
        ++v143;
      }
LABEL_155:
      if ( v114 )
        v57 = v114;
      goto LABEL_56;
    }
    v105 = 1;
    v57 = 0;
    v106 = (v175 - 1) & v152;
    v107 = (_QWORD *)(v173 + 16LL * v106);
    v108 = *v107;
    if ( *v107 == a2 )
    {
LABEL_127:
      v107[1] = v52;
      v60 = v171;
      if ( v171 )
        goto LABEL_59;
LABEL_128:
      ++v168;
      goto LABEL_129;
    }
    while ( v108 != -4096 )
    {
      if ( !v57 && v108 == -8192 )
        v57 = v107;
      v106 = (v175 - 1) & (v105 + v106);
      v107 = (_QWORD *)(v173 + 16LL * v106);
      v108 = *v107;
      if ( *v107 == a2 )
        goto LABEL_127;
      ++v105;
    }
    if ( !v57 )
      v57 = v107;
    ++v172;
    v56 = v174 + 1;
    if ( 4 * ((int)v174 + 1) >= 3 * v175 )
    {
      v54 = 2 * v175;
      goto LABEL_54;
    }
    if ( v175 - HIDWORD(v174) - v56 <= v175 >> 3 )
    {
      sub_DA7360((__int64)&v172, v175);
      if ( !v175 )
      {
LABEL_283:
        LODWORD(v174) = v174 + 1;
        BUG();
      }
      v114 = 0;
      v51 = v158;
      v115 = 1;
      v116 = (v175 - 1) & v152;
      v56 = v174 + 1;
      v57 = (_QWORD *)(v173 + 16LL * v116);
      v117 = *v57;
      if ( *v57 != a2 )
      {
        while ( v117 != -4096 )
        {
          if ( v117 == -8192 && !v114 )
            v114 = v57;
          v116 = (v175 - 1) & (v115 + v116);
          v57 = (_QWORD *)(v173 + 16LL * v116);
          v117 = *v57;
          if ( *v57 == a2 )
            goto LABEL_56;
          ++v115;
        }
        goto LABEL_155;
      }
    }
LABEL_56:
    LODWORD(v174) = v56;
    if ( *v57 != -4096 )
      --HIDWORD(v174);
    v57[1] = 0;
    v59 = v57 + 1;
    *(v59 - 1) = a2;
    *v59 = v52;
    v60 = v171;
    if ( !v171 )
      goto LABEL_128;
LABEL_59:
    v61 = (unsigned int)(v60 - 1);
    v62 = 1;
    v63 = 0;
    v64 = v61 & v152;
    v65 = &v169[2 * ((unsigned int)v61 & v152)];
    v66 = *v65;
    if ( *v65 != a2 )
    {
      while ( v66 != -4096 )
      {
        if ( v66 == -8192 && !v63 )
          v63 = v65;
        v64 = v61 & (v62 + v64);
        v65 = &v169[2 * v64];
        v66 = *v65;
        if ( *v65 == a2 )
          goto LABEL_60;
        ++v62;
      }
      if ( !v63 )
        v63 = v65;
      ++v168;
      v110 = v170 + 1;
      if ( 4 * ((int)v170 + 1) < (unsigned int)(3 * v60) )
      {
        if ( (int)v60 - HIDWORD(v170) - v110 > (unsigned int)v60 >> 3 )
          goto LABEL_168;
        v160 = v51;
        sub_DA7360((__int64)&v168, v60);
        if ( !v171 )
        {
LABEL_285:
          LODWORD(v170) = v170 + 1;
          BUG();
        }
        v60 = 0;
        v51 = v160;
        v61 = 1;
        v118 = (v171 - 1) & v152;
        v110 = v170 + 1;
        v63 = &v169[2 * v118];
        v119 = *v63;
        if ( *v63 == a2 )
          goto LABEL_168;
        while ( v119 != -4096 )
        {
          if ( v119 == -8192 && !v60 )
            v60 = (__int64)v63;
          v118 = (v171 - 1) & (v61 + v118);
          v63 = &v169[2 * v118];
          v119 = *v63;
          if ( *v63 == a2 )
            goto LABEL_168;
          v61 = (unsigned int)(v61 + 1);
        }
        goto LABEL_180;
      }
LABEL_129:
      v60 = (unsigned int)(2 * v60);
      v159 = v51;
      sub_DA7360((__int64)&v168, v60);
      if ( !v171 )
        goto LABEL_285;
      v51 = v159;
      v109 = (v171 - 1) & v152;
      v110 = v170 + 1;
      v63 = &v169[2 * v109];
      v111 = *v63;
      if ( *v63 == a2 )
        goto LABEL_168;
      v61 = 1;
      v60 = 0;
      while ( v111 != -4096 )
      {
        if ( v111 == -8192 && !v60 )
          v60 = (__int64)v63;
        v109 = (v171 - 1) & (v61 + v109);
        v63 = &v169[2 * v109];
        v111 = *v63;
        if ( *v63 == a2 )
          goto LABEL_168;
        v61 = (unsigned int)(v61 + 1);
      }
LABEL_180:
      if ( v60 )
        v63 = (__int64 *)v60;
LABEL_168:
      LODWORD(v170) = v110;
      if ( *v63 != -4096 )
        --HIDWORD(v170);
      v63[1] = 0;
      *v63 = a2;
      v67 = 0;
      goto LABEL_61;
    }
LABEL_60:
    v67 = (_BYTE *)v65[1];
LABEL_61:
    v68 = v52 == v67;
    v176 = (__int64 *)v178;
    v177 = 0x800000000LL;
    if ( !(_DWORD)v170 )
      goto LABEL_62;
    v73 = v169;
    v74 = &v169[2 * v171];
    if ( v169 == v74 )
      goto LABEL_62;
    while ( 1 )
    {
      v75 = *v73;
      v76 = v73;
      if ( *v73 != -4096 && *v73 != -8192 )
        break;
      v73 += 2;
      if ( v74 == v73 )
        goto LABEL_62;
    }
    if ( v74 == v73 )
      goto LABEL_62;
    v77 = 0;
    v78 = (__int64 *)&v176;
    do
    {
      if ( *(_BYTE *)v75 == 84 && v75 != a2 && v162 == *(_QWORD *)(v75 + 40) )
      {
        v100 = (unsigned int)v77;
        if ( HIDWORD(v177) <= (unsigned __int64)(unsigned int)v77 )
        {
          v112 = v76[1];
          if ( HIDWORD(v177) < (unsigned __int64)(unsigned int)v77 + 1 )
          {
            v144 = v51;
            v146 = v78;
            sub_C8D5F0((__int64)v78, v178, (unsigned int)v77 + 1LL, 0x10u, (unsigned int)v77 + 1LL, v61);
            v100 = (unsigned int)v177;
            v51 = v144;
            v78 = v146;
          }
          v113 = &v176[2 * v100];
          *v113 = v75;
          v113[1] = v112;
          v77 = (unsigned int)(v177 + 1);
          LODWORD(v177) = v177 + 1;
        }
        else
        {
          v101 = &v176[2 * (unsigned int)v77];
          if ( v101 )
          {
            v102 = v76[1];
            *v101 = v75;
            v101[1] = v102;
            LODWORD(v77) = v177;
          }
          v77 = (unsigned int)(v77 + 1);
          LODWORD(v177) = v77;
        }
      }
      v76 += 2;
      if ( v76 == v74 )
        break;
      while ( 1 )
      {
        v75 = *v76;
        if ( *v76 != -8192 && v75 != -4096 )
          break;
        v76 += 2;
        if ( v74 == v76 )
          goto LABEL_80;
      }
    }
    while ( v74 != v76 );
LABEL_80:
    v79 = v176;
    v60 = 16 * v77;
    v80 = (__int64 *)((char *)v176 + v60);
    if ( v176 == (__int64 *)((char *)v176 + v60) )
      goto LABEL_62;
    v81 = v51;
    while ( 2 )
    {
      v60 = v175;
      v89 = *v79;
      if ( !v175 )
      {
        ++v172;
        goto LABEL_89;
      }
      v82 = 1;
      v83 = 0;
      v84 = (v175 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v85 = (_QWORD *)(v173 + 16LL * v84);
      v86 = *v85;
      if ( v89 == *v85 )
        goto LABEL_83;
      while ( 1 )
      {
        if ( v86 == -4096 )
        {
          if ( !v83 )
            v83 = v85;
          ++v172;
          v91 = v174 + 1;
          if ( 4 * ((int)v174 + 1) >= 3 * v175 )
          {
LABEL_89:
            sub_DA7360((__int64)&v172, 2 * v175);
            if ( v175 )
            {
              v90 = (v175 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
              v91 = v174 + 1;
              v83 = (_QWORD *)(v173 + 16LL * v90);
              v92 = *v83;
              if ( v89 != *v83 )
              {
                v124 = 1;
                v96 = 0;
                while ( v92 != -4096 )
                {
                  if ( v92 == -8192 && !v96 )
                    v96 = v83;
                  v90 = (v175 - 1) & (v124 + v90);
                  v83 = (_QWORD *)(v173 + 16LL * v90);
                  v92 = *v83;
                  if ( v89 == *v83 )
                    goto LABEL_91;
                  ++v124;
                }
LABEL_110:
                if ( v96 )
                  v83 = v96;
              }
LABEL_91:
              LODWORD(v174) = v91;
              if ( *v83 != -4096 )
                --HIDWORD(v174);
              *v83 = v89;
              v88 = (unsigned __int8 **)(v83 + 1);
              v83[1] = 0;
LABEL_94:
              v60 = *(_QWORD *)(v89 - 8);
              v93 = 0x1FFFFFFFE0LL;
              if ( (*(_DWORD *)(v89 + 4) & 0x7FFFFFF) != 0 )
              {
                v94 = 0;
                do
                {
                  if ( v81 == *(_QWORD *)(v60 + 32LL * *(unsigned int *)(v89 + 72) + 8 * v94) )
                  {
                    v93 = 32 * v94;
                    goto LABEL_99;
                  }
                  ++v94;
                }
                while ( (*(_DWORD *)(v89 + 4) & 0x7FFFFFF) != (_DWORD)v94 );
                v87 = *(unsigned __int8 **)(v60 + 0x1FFFFFFFE0LL);
                v95 = *v87;
                if ( *v87 <= 0x15u )
                  goto LABEL_100;
              }
              else
              {
LABEL_99:
                v87 = *(unsigned __int8 **)(v60 + v93);
                v95 = *v87;
                if ( *v87 <= 0x15u )
                {
LABEL_100:
                  *v88 = v87;
                  goto LABEL_84;
                }
              }
              if ( v95 <= 0x1Cu )
              {
                v87 = 0;
              }
              else
              {
                v145 = v88;
                v60 = a4;
                v103 = sub_DA7540((__int64)v87, a4, (__int64)&v168, v147, *(__int64 **)(a1 + 24));
                v88 = v145;
                v87 = (unsigned __int8 *)v103;
              }
              goto LABEL_100;
            }
          }
          else
          {
            if ( v175 - HIDWORD(v174) - v91 > v175 >> 3 )
              goto LABEL_91;
            sub_DA7360((__int64)&v172, v175);
            if ( v175 )
            {
              v96 = 0;
              v97 = 1;
              v98 = (v175 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
              v91 = v174 + 1;
              v83 = (_QWORD *)(v173 + 16LL * v98);
              v99 = *v83;
              if ( v89 != *v83 )
              {
                while ( v99 != -4096 )
                {
                  if ( !v96 && v99 == -8192 )
                    v96 = v83;
                  v98 = (v175 - 1) & (v97 + v98);
                  v83 = (_QWORD *)(v173 + 16LL * v98);
                  v99 = *v83;
                  if ( v89 == *v83 )
                    goto LABEL_91;
                  ++v97;
                }
                goto LABEL_110;
              }
              goto LABEL_91;
            }
          }
          LODWORD(v174) = v174 + 1;
          BUG();
        }
        if ( v86 != -8192 || v83 )
          v85 = v83;
        v84 = (v175 - 1) & (v82 + v84);
        v86 = *(_QWORD *)(v173 + 16LL * v84);
        if ( v89 == v86 )
          break;
        ++v82;
        v83 = v85;
        v85 = (_QWORD *)(v173 + 16LL * v84);
      }
      v85 = (_QWORD *)(v173 + 16LL * v84);
LABEL_83:
      v87 = (unsigned __int8 *)v85[1];
      v88 = (unsigned __int8 **)(v85 + 1);
      if ( !v87 )
        goto LABEL_94;
LABEL_84:
      if ( (unsigned __int8 *)v79[1] != v87 )
        v68 = 0;
      v79 += 2;
      if ( v80 != v79 )
        continue;
      break;
    }
    v51 = v81;
LABEL_62:
    if ( !v68 )
    {
      v69 = (__int64)v169;
      ++v168;
      v169 = (__int64 *)v173;
      v70 = v170;
      LODWORD(v170) = v174;
      LODWORD(v174) = v70;
      v71 = HIDWORD(v170);
      HIDWORD(v170) = HIDWORD(v174);
      HIDWORD(v174) = v71;
      v72 = v171;
      ++v172;
      v173 = v69;
      v171 = v175;
      v175 = v72;
      if ( v176 != (__int64 *)v178 )
      {
        v156 = v51;
        _libc_free(v176, v60);
        v72 = v175;
        v69 = v173;
        v51 = v156;
      }
      v157 = v51;
      sub_C7D6A0(v69, 16LL * v72, 8);
      ++v161;
      v51 = v157;
      if ( v151 == v161 )
        goto LABEL_66;
      continue;
    }
    break;
  }
  v167 = a2;
  v43 = *sub_DA7A60((__int64)&v168, &v167);
  v150[1] = v43;
  if ( v176 != (__int64 *)v178 )
  {
    v166 = v43;
    _libc_free(v176, &v167);
    v43 = v166;
  }
  v41 = v173;
  v42 = 16LL * v175;
LABEL_37:
  v164 = v43;
  sub_C7D6A0(v41, v42, 8);
  v24 = v164;
LABEL_38:
  v165 = v24;
  sub_C7D6A0((__int64)v169, 16LL * v171, 8);
  return v165;
}
