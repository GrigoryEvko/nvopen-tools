// Function: sub_3144670
// Address: 0x3144670
//
__int64 __fastcall sub_3144670(__int64 *a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  char v9; // r13
  __int64 *v10; // r15
  __int64 v11; // r12
  __int64 i; // r12
  unsigned __int8 *v13; // r14
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r9
  __int64 v18; // r8
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  unsigned __int64 v21; // r14
  __int64 v22; // rbx
  _BYTE *v23; // r12
  __int64 v24; // r8
  int v25; // r10d
  _QWORD *v26; // r9
  unsigned int v27; // ecx
  _QWORD *v28; // rdx
  _BYTE *v29; // rax
  int v30; // edx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // r13
  unsigned int v35; // eax
  __int64 *v36; // rcx
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r14
  __int16 v42; // dx
  __int64 v43; // r12
  __int64 ***v44; // r15
  unsigned int v45; // edx
  __int64 ****v46; // rcx
  __int64 ***v47; // r8
  __int64 *v48; // rbx
  __int64 v49; // rdi
  __int64 v51; // rbx
  char v52; // al
  unsigned __int8 *v53; // rax
  unsigned __int8 *v54; // r15
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  __int64 v59; // rdx
  unsigned __int64 v60; // r12
  __int64 v61; // rdx
  _QWORD *v62; // r15
  _QWORD *v63; // rbx
  __int64 v64; // r13
  __int64 *v65; // r14
  __int64 v66; // rsi
  unsigned __int8 *v67; // rsi
  unsigned __int64 *v68; // rbx
  __int64 v69; // r9
  int v70; // r11d
  _QWORD *v71; // r10
  __int64 v72; // r8
  unsigned int v73; // eax
  _QWORD *v74; // rdi
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rdi
  int v78; // edx
  unsigned __int64 v79; // r15
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rdx
  int v85; // r11d
  __int64 v86; // rax
  __int64 v87; // rdi
  int v88; // ecx
  int v89; // r9d
  _QWORD *v90; // rax
  __int64 v91; // rbx
  __int64 *v92; // rsi
  __int64 v93; // r15
  __int64 *v94; // r12
  __int64 v95; // r13
  __int64 v96; // rax
  __int64 v97; // r14
  __int64 v98; // rcx
  _QWORD *v99; // rax
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rax
  unsigned __int64 v103; // rdx
  __int64 v104; // rbx
  __int64 *v105; // rax
  __int64 v106; // r9
  __int64 *v107; // r13
  int v108; // r14d
  __int64 v109; // r12
  __int64 v110; // r15
  _QWORD *v111; // rax
  __int64 v112; // rdi
  unsigned __int16 v113; // ax
  __int64 v114; // rdx
  unsigned __int64 v115; // r8
  unsigned int v116; // eax
  __int64 v117; // rsi
  int v118; // r10d
  unsigned int v119; // r15d
  _QWORD *v120; // rdi
  __int64 v121; // rcx
  int v122; // ecx
  int v123; // r8d
  __int64 v124; // rdx
  _QWORD *v125; // r11
  int v126; // eax
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  __int64 j; // rbx
  unsigned __int8 *v130; // r12
  int v131; // edx
  unsigned int v132; // edx
  __int64 v133; // rcx
  _QWORD *v134; // rdi
  int v135; // edi
  _QWORD *v136; // rsi
  unsigned int v137; // r12d
  __int64 v138; // rdx
  int v139; // r11d
  __int64 *v141; // [rsp+8h] [rbp-1E8h]
  __int64 v142; // [rsp+10h] [rbp-1E0h]
  unsigned __int64 v144; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v145; // [rsp+28h] [rbp-1C8h]
  __int64 v146; // [rsp+50h] [rbp-1A0h]
  unsigned __int64 v147; // [rsp+58h] [rbp-198h]
  __int64 v148; // [rsp+60h] [rbp-190h]
  unsigned __int64 v149; // [rsp+60h] [rbp-190h]
  __int64 *v150; // [rsp+70h] [rbp-180h]
  __int64 v151; // [rsp+70h] [rbp-180h]
  __int64 v152; // [rsp+78h] [rbp-178h]
  __int64 *v153; // [rsp+78h] [rbp-178h]
  unsigned __int8 v154; // [rsp+80h] [rbp-170h]
  __int64 v155; // [rsp+80h] [rbp-170h]
  __int64 v156; // [rsp+80h] [rbp-170h]
  __int64 v157; // [rsp+88h] [rbp-168h]
  __int64 v158; // [rsp+90h] [rbp-160h]
  unsigned __int64 v159; // [rsp+98h] [rbp-158h]
  unsigned __int8 v161; // [rsp+A0h] [rbp-150h]
  unsigned __int64 v162; // [rsp+A0h] [rbp-150h]
  __int64 *v163; // [rsp+A8h] [rbp-148h]
  char v164; // [rsp+A8h] [rbp-148h]
  unsigned __int64 v165; // [rsp+A8h] [rbp-148h]
  int v166; // [rsp+A8h] [rbp-148h]
  int v167; // [rsp+B4h] [rbp-13Ch] BYREF
  __int64 v168; // [rsp+B8h] [rbp-138h] BYREF
  __int64 v169[4]; // [rsp+C0h] [rbp-130h] BYREF
  __int16 v170; // [rsp+E0h] [rbp-110h]
  __int64 v171; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v172; // [rsp+F8h] [rbp-F8h]
  __int64 v173; // [rsp+100h] [rbp-F0h]
  __int64 v174; // [rsp+108h] [rbp-E8h]
  __int64 *v175; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v176; // [rsp+118h] [rbp-D8h]
  __int64 v177; // [rsp+120h] [rbp-D0h] BYREF
  __int64 v178; // [rsp+128h] [rbp-C8h]
  __int64 v179; // [rsp+130h] [rbp-C0h]
  __int64 v180; // [rsp+138h] [rbp-B8h]
  unsigned __int64 *v181; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v182; // [rsp+148h] [rbp-A8h]
  _BYTE *v183; // [rsp+150h] [rbp-A0h] BYREF
  __int64 v184; // [rsp+158h] [rbp-98h]
  _BYTE v185[32]; // [rsp+160h] [rbp-90h] BYREF
  _BYTE *v186; // [rsp+180h] [rbp-70h] BYREF
  __int64 v187; // [rsp+188h] [rbp-68h]
  _BYTE v188[96]; // [rsp+190h] [rbp-60h] BYREF

  v6 = a3;
  v7 = &a1[a2];
  v186 = v188;
  v187 = 0x600000000LL;
  v141 = v7;
  if ( a1 == v7 )
  {
    LODWORD(v8) = 0;
  }
  else
  {
    v8 = 0;
    v9 = a5;
    v10 = a1;
    do
    {
      v11 = *v10;
      if ( v9 )
      {
        if ( v8 + 1 > (unsigned __int64)HIDWORD(v187) )
        {
          sub_C8D5F0((__int64)&v186, v188, v8 + 1, 8u, a5, a6);
          v8 = (unsigned int)v187;
        }
        *(_QWORD *)&v186[8 * v8] = v11;
        v8 = (unsigned int)(v187 + 1);
        LODWORD(v187) = v187 + 1;
      }
      else
      {
        for ( i = *(_QWORD *)(v11 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v13 = *(unsigned __int8 **)(i + 24);
          v14 = *v13;
          if ( (_BYTE)v14 == 5 || (unsigned int)(v14 - 9) <= 2 )
          {
            if ( v8 + 1 > (unsigned __int64)HIDWORD(v187) )
            {
              sub_C8D5F0((__int64)&v186, v188, v8 + 1, 8u, a5, a6);
              v8 = (unsigned int)v187;
            }
            *(_QWORD *)&v186[8 * v8] = v13;
            v8 = (unsigned int)(v187 + 1);
            LODWORD(v187) = v187 + 1;
          }
        }
      }
      ++v10;
    }
    while ( v7 != v10 );
    v6 = a3;
  }
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v176 = 0;
  v175 = &v177;
  while ( 1 )
  {
    v15 = 8LL * (unsigned int)v8 - 8;
    if ( !(_DWORD)v8 )
      break;
    while ( 1 )
    {
      LODWORD(v8) = v8 - 1;
      v16 = *(_QWORD *)&v186[v15];
      LODWORD(v187) = v8;
      if ( !(_DWORD)v174 )
      {
        ++v171;
        goto LABEL_201;
      }
      v17 = v172;
      v18 = ((_DWORD)v174 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v19 = (_QWORD *)(v172 + 8 * v18);
      v20 = *v19;
      if ( v16 != *v19 )
        break;
LABEL_21:
      v15 -= 8;
      if ( !(_DWORD)v8 )
        goto LABEL_22;
    }
    v166 = 1;
    v125 = 0;
    while ( v20 != -4096 )
    {
      if ( v125 || v20 != -8192 )
        v19 = v125;
      v18 = ((_DWORD)v174 - 1) & (unsigned int)(v166 + v18);
      v20 = *(_QWORD *)(v172 + 8LL * (unsigned int)v18);
      if ( v16 == v20 )
        goto LABEL_21;
      ++v166;
      v125 = v19;
      v19 = (_QWORD *)(v172 + 8LL * (unsigned int)v18);
    }
    if ( !v125 )
      v125 = v19;
    ++v171;
    v126 = v173 + 1;
    if ( 4 * ((int)v173 + 1) < (unsigned int)(3 * v174) )
    {
      if ( (int)v174 - (v126 + HIDWORD(v173)) > (unsigned int)v174 >> 3 )
        goto LABEL_188;
      sub_2631D80((__int64)&v171, v174);
      if ( (_DWORD)v174 )
      {
        v18 = v172;
        v135 = 1;
        v136 = 0;
        v137 = (v174 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v125 = (_QWORD *)(v172 + 8LL * v137);
        v138 = *v125;
        v126 = v173 + 1;
        if ( v16 != *v125 )
        {
          while ( v138 != -4096 )
          {
            if ( !v136 && v138 == -8192 )
              v136 = v125;
            v17 = (unsigned int)(v135 + 1);
            v137 = (v174 - 1) & (v135 + v137);
            v125 = (_QWORD *)(v172 + 8LL * v137);
            v138 = *v125;
            if ( v16 == *v125 )
              goto LABEL_188;
            ++v135;
          }
          if ( v136 )
            v125 = v136;
        }
        goto LABEL_188;
      }
LABEL_262:
      LODWORD(v173) = v173 + 1;
      BUG();
    }
LABEL_201:
    sub_2631D80((__int64)&v171, 2 * v174);
    if ( !(_DWORD)v174 )
      goto LABEL_262;
    v18 = v172;
    v132 = (v174 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v125 = (_QWORD *)(v172 + 8LL * v132);
    v133 = *v125;
    v126 = v173 + 1;
    if ( v16 != *v125 )
    {
      v17 = 1;
      v134 = 0;
      while ( v133 != -4096 )
      {
        if ( !v134 && v133 == -8192 )
          v134 = v125;
        v132 = (v174 - 1) & (v17 + v132);
        v125 = (_QWORD *)(v172 + 8LL * v132);
        v133 = *v125;
        if ( v16 == *v125 )
          goto LABEL_188;
        v17 = (unsigned int)(v17 + 1);
      }
      if ( v134 )
        v125 = v134;
    }
LABEL_188:
    LODWORD(v173) = v126;
    if ( *v125 != -4096 )
      --HIDWORD(v173);
    *v125 = v16;
    v127 = (unsigned int)v176;
    v128 = (unsigned int)v176 + 1LL;
    if ( v128 > HIDWORD(v176) )
    {
      sub_C8D5F0((__int64)&v175, &v177, v128, 8u, v18, v17);
      v127 = (unsigned int)v176;
    }
    v175[v127] = v16;
    v8 = (unsigned int)v187;
    LODWORD(v176) = v176 + 1;
    for ( j = *(_QWORD *)(v16 + 16); j; j = *(_QWORD *)(j + 8) )
    {
      v130 = *(unsigned __int8 **)(j + 24);
      v131 = *v130;
      if ( (_BYTE)v131 == 5 || (unsigned int)(v131 - 9) <= 2 )
      {
        if ( v8 + 1 > (unsigned __int64)HIDWORD(v187) )
        {
          sub_C8D5F0((__int64)&v186, v188, v8 + 1, 8u, v18, v17);
          v8 = (unsigned int)v187;
        }
        *(_QWORD *)&v186[8 * v8] = v130;
        v8 = (unsigned int)(v187 + 1);
        LODWORD(v187) = v187 + 1;
      }
    }
  }
LABEL_22:
  v21 = (unsigned __int64)v175;
  v181 = (unsigned __int64 *)&v183;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v182 = 0;
  v163 = &v175[(unsigned int)v176];
  if ( v163 == v175 )
    goto LABEL_223;
  do
  {
    v22 = *(_QWORD *)(*(_QWORD *)v21 + 16LL);
    if ( v22 )
    {
      while ( 1 )
      {
        v23 = *(_BYTE **)(v22 + 24);
        if ( *v23 <= 0x1Cu || v6 && v6 != sub_B43CB0(*(_QWORD *)(v22 + 24)) )
          goto LABEL_26;
        if ( !(_DWORD)v180 )
          break;
        v24 = (unsigned int)(v180 - 1);
        v25 = 1;
        v26 = 0;
        v27 = v24 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v28 = (_QWORD *)(v178 + 8LL * v27);
        v29 = (_BYTE *)*v28;
        if ( v23 == (_BYTE *)*v28 )
        {
LABEL_26:
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            goto LABEL_41;
        }
        else
        {
          while ( v29 != (_BYTE *)-4096LL )
          {
            if ( v29 != (_BYTE *)-8192LL || v26 )
              v28 = v26;
            v27 = v24 & (v25 + v27);
            v29 = *(_BYTE **)(v178 + 8LL * v27);
            if ( v23 == v29 )
              goto LABEL_26;
            ++v25;
            v26 = v28;
            v28 = (_QWORD *)(v178 + 8LL * v27);
          }
          if ( !v26 )
            v26 = v28;
          ++v177;
          v30 = v179 + 1;
          if ( 4 * ((int)v179 + 1) < (unsigned int)(3 * v180) )
          {
            if ( (int)v180 - HIDWORD(v179) - v30 <= (unsigned int)v180 >> 3 )
            {
              sub_CF4090((__int64)&v177, v180);
              if ( !(_DWORD)v180 )
              {
LABEL_261:
                LODWORD(v179) = v179 + 1;
                BUG();
              }
              v24 = 1;
              v119 = (v180 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v26 = (_QWORD *)(v178 + 8LL * v119);
              v30 = v179 + 1;
              v120 = 0;
              v121 = *v26;
              if ( v23 != (_BYTE *)*v26 )
              {
                while ( v121 != -4096 )
                {
                  if ( !v120 && v121 == -8192 )
                    v120 = v26;
                  v119 = (v180 - 1) & (v24 + v119);
                  v26 = (_QWORD *)(v178 + 8LL * v119);
                  v121 = *v26;
                  if ( v23 == (_BYTE *)*v26 )
                    goto LABEL_36;
                  v24 = (unsigned int)(v24 + 1);
                }
                if ( v120 )
                  v26 = v120;
              }
            }
            goto LABEL_36;
          }
LABEL_162:
          sub_CF4090((__int64)&v177, 2 * v180);
          if ( !(_DWORD)v180 )
            goto LABEL_261;
          v116 = (v180 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v26 = (_QWORD *)(v178 + 8LL * v116);
          v117 = *v26;
          v30 = v179 + 1;
          if ( v23 != (_BYTE *)*v26 )
          {
            v118 = 1;
            v24 = 0;
            while ( v117 != -4096 )
            {
              if ( v117 == -8192 && !v24 )
                v24 = (__int64)v26;
              v116 = (v180 - 1) & (v118 + v116);
              v26 = (_QWORD *)(v178 + 8LL * v116);
              v117 = *v26;
              if ( v23 == (_BYTE *)*v26 )
                goto LABEL_36;
              ++v118;
            }
            if ( v24 )
              v26 = (_QWORD *)v24;
          }
LABEL_36:
          LODWORD(v179) = v30;
          if ( *v26 != -4096 )
            --HIDWORD(v179);
          *v26 = v23;
          v31 = (unsigned int)v182;
          v32 = (unsigned int)v182 + 1LL;
          if ( v32 > HIDWORD(v182) )
          {
            sub_C8D5F0((__int64)&v181, &v183, v32, 8u, v24, (__int64)v26);
            v31 = (unsigned int)v182;
          }
          v181[v31] = (unsigned __int64)v23;
          LODWORD(v182) = v182 + 1;
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            goto LABEL_41;
        }
      }
      ++v177;
      goto LABEL_162;
    }
LABEL_41:
    v21 += 8LL;
  }
  while ( v163 != (__int64 *)v21 );
  v33 = (unsigned int)v182;
  if ( (_DWORD)v182 )
  {
    v154 = 0;
    while ( 1 )
    {
      v34 = v181[v33 - 1];
      if ( (_DWORD)v180 )
      {
        v35 = (v180 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v36 = (__int64 *)(v178 + 8LL * v35);
        v37 = *v36;
        if ( v34 == *v36 )
        {
LABEL_46:
          *v36 = -8192;
          LODWORD(v179) = v179 - 1;
          ++HIDWORD(v179);
        }
        else
        {
          v122 = 1;
          while ( v37 != -4096 )
          {
            v123 = v122 + 1;
            v35 = (v180 - 1) & (v122 + v35);
            v36 = (__int64 *)(v178 + 8LL * v35);
            v37 = *v36;
            if ( v34 == *v36 )
              goto LABEL_46;
            v122 = v123;
          }
        }
      }
      v38 = *(_QWORD *)(v34 + 48);
      LODWORD(v182) = v182 - 1;
      v168 = v38;
      if ( v38 )
        sub_B96E90((__int64)&v168, v38, 1);
      v39 = 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v34 + 7) & 0x40) != 0 )
      {
        v40 = *(_QWORD *)(v34 - 8);
        v159 = v40 + v39;
      }
      else
      {
        v159 = v34;
        v40 = v34 - v39;
      }
      if ( v40 != v159 )
        break;
LABEL_63:
      if ( v168 )
        sub_B91220((__int64)&v168, v168);
      v33 = (unsigned int)v182;
      if ( !(_DWORD)v182 )
        goto LABEL_66;
    }
    v41 = v40;
    v158 = v34 + 24;
    while ( 1 )
    {
      if ( *(_BYTE *)v34 == 84 )
      {
        v43 = sub_AA5190(*(_QWORD *)(*(_QWORD *)(v34 - 8)
                                   + 32LL * *(unsigned int *)(v34 + 72)
                                   + 8LL * (unsigned int)((__int64)(v41 - *(_QWORD *)(v34 - 8)) >> 5)));
        if ( v43 )
        {
          v161 = v42;
          v164 = HIBYTE(v42);
          goto LABEL_55;
        }
      }
      else
      {
        v43 = v158;
      }
      v164 = 0;
      v161 = 0;
LABEL_55:
      v44 = *(__int64 ****)v41;
      if ( **(_BYTE **)v41 <= 0x15u && (_DWORD)v174 )
      {
        v45 = (v174 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v46 = (__int64 ****)(v172 + 8LL * v45);
        v47 = *v46;
        if ( *v46 != v44 )
        {
          v88 = 1;
          while ( v47 != (__int64 ***)-4096LL )
          {
            v89 = v88 + 1;
            v45 = (v174 - 1) & (v88 + v45);
            v46 = (__int64 ****)(v172 + 8LL * v45);
            v47 = *v46;
            if ( *v46 == v44 )
              goto LABEL_58;
            v88 = v89;
          }
          goto LABEL_59;
        }
LABEL_58:
        if ( v46 != (__int64 ****)(v172 + 8LL * (unsigned int)v174) )
        {
          v51 = v161;
          BYTE1(v51) = v164;
          v183 = v185;
          v184 = 0x400000000LL;
          v52 = *(_BYTE *)v44;
          if ( *(_BYTE *)v44 == 5 )
          {
            v53 = sub_AC5700((__int64)v44);
            v54 = v53;
            if ( !v43 )
              BUG();
            sub_B44150(v53, *(_QWORD *)(v43 + 16), (unsigned __int64 *)v43, v51);
            v57 = (unsigned int)v184;
            v58 = (unsigned int)v184 + 1LL;
            if ( v58 > HIDWORD(v184) )
            {
              sub_C8D5F0((__int64)&v183, v185, v58, 8u, v55, v56);
              v57 = (unsigned int)v184;
            }
            *(_QWORD *)&v183[8 * v57] = v54;
            v59 = (unsigned int)(v184 + 1);
            LODWORD(v184) = v184 + 1;
            goto LABEL_82;
          }
          if ( (unsigned __int8)(v52 - 9) > 1u )
          {
            if ( v52 != 11 )
              BUG();
            v150 = (__int64 *)v44;
            v90 = (_QWORD *)sub_BD5C60((__int64)v44);
            v148 = sub_BCB2D0(v90);
            v91 = sub_ACADE0(v44[1]);
            if ( (*((_BYTE *)v44 + 7) & 0x40) != 0 )
            {
              v92 = (__int64 *)*(v44 - 1);
              v150 = &v92[4 * (*((_DWORD *)v44 + 1) & 0x7FFFFFF)];
            }
            else
            {
              v92 = (__int64 *)&v44[-4 * (*((_DWORD *)v44 + 1) & 0x7FFFFFF)];
            }
            v93 = 0;
            if ( v92 != v150 )
            {
              v145 = v34;
              v144 = v41;
              v146 = v43;
              v94 = v92;
              do
              {
                v95 = v91;
                v170 = 257;
                v96 = sub_AD64C0(v148, v93, 0);
                v97 = *v94;
                v152 = v96;
                v98 = v161;
                BYTE1(v98) = v164;
                v157 = v98;
                v99 = sub_BD2C40(72, 3u);
                v91 = (__int64)v99;
                if ( v99 )
                  sub_B4DFA0((__int64)v99, v95, v97, v152, (__int64)v169, v101, v146, v157);
                v102 = (unsigned int)v184;
                v103 = (unsigned int)v184 + 1LL;
                if ( v103 > HIDWORD(v184) )
                {
                  sub_C8D5F0((__int64)&v183, v185, v103, 8u, v100, v101);
                  v102 = (unsigned int)v184;
                }
                v94 += 4;
                ++v93;
                *(_QWORD *)&v183[8 * v102] = v91;
                v59 = (unsigned int)(v184 + 1);
                LODWORD(v184) = v184 + 1;
              }
              while ( v94 != v150 );
              v34 = v145;
              v41 = v144;
              goto LABEL_82;
            }
LABEL_181:
            v59 = (unsigned int)v184;
            goto LABEL_82;
          }
          v104 = sub_ACADE0(v44[1]);
          if ( (*((_BYTE *)v44 + 7) & 0x40) != 0 )
          {
            v105 = (__int64 *)*(v44 - 1);
            v153 = &v105[4 * (*((_DWORD *)v44 + 1) & 0x7FFFFFF)];
            if ( v105 == v153 )
              goto LABEL_181;
          }
          else
          {
            v153 = (__int64 *)v44;
            v124 = 32LL * (*((_DWORD *)v44 + 1) & 0x7FFFFFF);
            v105 = (__int64 *)&v44[v124 / 0xFFFFFFFFFFFFFFF8LL];
            if ( &v44[v124 / 0xFFFFFFFFFFFFFFF8LL] == v44 )
              goto LABEL_181;
          }
          v151 = v43;
          v106 = v142;
          v149 = v34;
          v107 = v105;
          v147 = v41;
          v108 = 0;
          do
          {
            v155 = v106;
            v170 = 257;
            v109 = v104;
            v167 = v108;
            v110 = *v107;
            v111 = sub_BD2C40(104, unk_3F148BC);
            v106 = v155;
            v104 = (__int64)v111;
            if ( v111 )
            {
              v112 = (__int64)v111;
              LOBYTE(v113) = v161;
              HIBYTE(v113) = v164;
              sub_B44260(v112, *(_QWORD *)(v109 + 8), 65, 2u, v151, v113);
              *(_QWORD *)(v104 + 72) = v104 + 88;
              *(_QWORD *)(v104 + 80) = 0x400000000LL;
              sub_B4FD20(v104, v109, v110, &v167, 1, (__int64)v169);
            }
            v114 = (unsigned int)v184;
            v115 = (unsigned int)v184 + 1LL;
            if ( v115 > HIDWORD(v184) )
            {
              v156 = v106;
              sub_C8D5F0((__int64)&v183, v185, (unsigned int)v184 + 1LL, 8u, v115, v106);
              v114 = (unsigned int)v184;
              v106 = v156;
            }
            ++v108;
            v107 += 4;
            *(_QWORD *)&v183[8 * v114] = v104;
            v59 = (unsigned int)(v184 + 1);
            LODWORD(v184) = v184 + 1;
          }
          while ( v107 != v153 );
          v142 = v106;
          v34 = v149;
          v41 = v147;
LABEL_82:
          v60 = (unsigned __int64)v183;
          v61 = 8 * v59;
          v62 = &v183[v61];
          if ( &v183[v61] == v183 )
          {
LABEL_108:
            v82 = *(_QWORD *)(v60 + v61 - 8);
            if ( *(_QWORD *)v41 )
            {
              v83 = *(_QWORD *)(v41 + 8);
              **(_QWORD **)(v41 + 16) = v83;
              if ( v83 )
                *(_QWORD *)(v83 + 16) = *(_QWORD *)(v41 + 16);
            }
            *(_QWORD *)v41 = v82;
            if ( v82 )
            {
              v84 = *(_QWORD *)(v82 + 16);
              *(_QWORD *)(v41 + 8) = v84;
              if ( v84 )
                *(_QWORD *)(v84 + 16) = v41 + 8;
              *(_QWORD *)(v41 + 16) = v82 + 16;
              *(_QWORD *)(v82 + 16) = v41;
            }
            if ( v183 != v185 )
              _libc_free((unsigned __int64)v183);
            v154 = 1;
            goto LABEL_59;
          }
          v63 = v183;
          v165 = v34;
          v162 = v41;
          while ( 2 )
          {
            while ( 2 )
            {
              v64 = *v63;
              v65 = (__int64 *)(*v63 + 48LL);
              v169[0] = v168;
              if ( v168 )
              {
                sub_B96E90((__int64)v169, v168, 1);
                if ( v65 == v169 )
                {
                  if ( v169[0] )
                    sub_B91220((__int64)v169, v169[0]);
                }
                else
                {
                  v66 = *(_QWORD *)(v64 + 48);
                  if ( v66 )
                    goto LABEL_91;
LABEL_92:
                  v67 = (unsigned __int8 *)v169[0];
                  *(_QWORD *)(v64 + 48) = v169[0];
                  if ( v67 )
                  {
                    ++v63;
                    sub_B976B0((__int64)v169, v67, (__int64)v65);
                    if ( v62 != v63 )
                      continue;
LABEL_94:
                    v68 = (unsigned __int64 *)v183;
                    v34 = v165;
                    v41 = v162;
                    v61 = 8LL * (unsigned int)v184;
                    v60 = (unsigned __int64)&v183[v61];
                    if ( v183 == &v183[v61] )
                      goto LABEL_108;
                    while ( 2 )
                    {
                      while ( 2 )
                      {
                        if ( !(_DWORD)v180 )
                        {
                          ++v177;
                          goto LABEL_100;
                        }
                        v69 = (unsigned int)(v180 - 1);
                        v70 = 1;
                        v71 = 0;
                        v72 = v178;
                        v73 = v69 & (((unsigned int)*v68 >> 9) ^ ((unsigned int)*v68 >> 4));
                        v74 = (_QWORD *)(v178 + 8LL * v73);
                        v75 = *v74;
                        if ( *v74 == *v68 )
                        {
LABEL_97:
                          if ( ++v68 == (unsigned __int64 *)v60 )
                            goto LABEL_107;
                          continue;
                        }
                        break;
                      }
                      while ( v75 != -4096 )
                      {
                        if ( v71 || v75 != -8192 )
                          v74 = v71;
                        v73 = v69 & (v70 + v73);
                        v75 = *(_QWORD *)(v178 + 8LL * v73);
                        if ( *v68 == v75 )
                          goto LABEL_97;
                        ++v70;
                        v71 = v74;
                        v74 = (_QWORD *)(v178 + 8LL * v73);
                      }
                      if ( !v71 )
                        v71 = v74;
                      ++v177;
                      v78 = v179 + 1;
                      if ( 4 * ((int)v179 + 1) >= (unsigned int)(3 * v180) )
                      {
LABEL_100:
                        sub_CF4090((__int64)&v177, 2 * v180);
                        if ( !(_DWORD)v180 )
                          goto LABEL_263;
                        v72 = v178;
                        LODWORD(v76) = (v180 - 1) & (((unsigned int)*v68 >> 9) ^ ((unsigned int)*v68 >> 4));
                        v71 = (_QWORD *)(v178 + 8LL * (unsigned int)v76);
                        v77 = *v71;
                        v78 = v179 + 1;
                        if ( *v71 != *v68 )
                        {
                          v139 = 1;
                          v69 = 0;
                          while ( v77 != -4096 )
                          {
                            if ( !v69 && v77 == -8192 )
                              v69 = (__int64)v71;
                            v76 = ((_DWORD)v180 - 1) & (unsigned int)(v76 + v139);
                            v71 = (_QWORD *)(v178 + 8 * v76);
                            v77 = *v71;
                            if ( *v68 == *v71 )
                              goto LABEL_102;
                            ++v139;
                          }
                          goto LABEL_132;
                        }
                      }
                      else if ( (int)v180 - HIDWORD(v179) - v78 <= (unsigned int)v180 >> 3 )
                      {
                        sub_CF4090((__int64)&v177, v180);
                        if ( !(_DWORD)v180 )
                        {
LABEL_263:
                          LODWORD(v179) = v179 + 1;
                          BUG();
                        }
                        v72 = v178;
                        v69 = 0;
                        v85 = 1;
                        LODWORD(v86) = (v180 - 1) & (((unsigned int)*v68 >> 9) ^ ((unsigned int)*v68 >> 4));
                        v71 = (_QWORD *)(v178 + 8LL * (unsigned int)v86);
                        v87 = *v71;
                        v78 = v179 + 1;
                        if ( *v68 != *v71 )
                        {
                          while ( v87 != -4096 )
                          {
                            if ( v87 == -8192 && !v69 )
                              v69 = (__int64)v71;
                            v86 = ((_DWORD)v180 - 1) & (unsigned int)(v86 + v85);
                            v71 = (_QWORD *)(v178 + 8 * v86);
                            v87 = *v71;
                            if ( *v68 == *v71 )
                              goto LABEL_102;
                            ++v85;
                          }
LABEL_132:
                          if ( v69 )
                            v71 = (_QWORD *)v69;
                        }
                      }
LABEL_102:
                      LODWORD(v179) = v78;
                      if ( *v71 != -4096 )
                        --HIDWORD(v179);
                      v79 = *v68;
                      *v71 = *v68;
                      v80 = (unsigned int)v182;
                      v81 = (unsigned int)v182 + 1LL;
                      if ( v81 > HIDWORD(v182) )
                      {
                        sub_C8D5F0((__int64)&v181, &v183, v81, 8u, v72, v69);
                        v80 = (unsigned int)v182;
                      }
                      ++v68;
                      v181[v80] = v79;
                      LODWORD(v182) = v182 + 1;
                      if ( v68 == (unsigned __int64 *)v60 )
                      {
LABEL_107:
                        v60 = (unsigned __int64)v183;
                        v61 = 8LL * (unsigned int)v184;
                        goto LABEL_108;
                      }
                      continue;
                    }
                  }
                }
              }
              else if ( v65 != v169 )
              {
                v66 = *(_QWORD *)(v64 + 48);
                if ( v66 )
                {
LABEL_91:
                  sub_B91220((__int64)v65, v66);
                  goto LABEL_92;
                }
              }
              break;
            }
            if ( v62 == ++v63 )
              goto LABEL_94;
            continue;
          }
        }
      }
LABEL_59:
      v41 += 32LL;
      if ( v159 == v41 )
        goto LABEL_63;
    }
  }
LABEL_223:
  v154 = 0;
LABEL_66:
  if ( a4 && a1 != v141 )
  {
    v48 = a1;
    do
    {
      v49 = *v48++;
      sub_AD0030(v49);
    }
    while ( v141 != v48 );
  }
  if ( v181 != (unsigned __int64 *)&v183 )
    _libc_free((unsigned __int64)v181);
  sub_C7D6A0(v178, 8LL * (unsigned int)v180, 8);
  if ( v175 != &v177 )
    _libc_free((unsigned __int64)v175);
  sub_C7D6A0(v172, 8LL * (unsigned int)v174, 8);
  if ( v186 != v188 )
    _libc_free((unsigned __int64)v186);
  return v154;
}
