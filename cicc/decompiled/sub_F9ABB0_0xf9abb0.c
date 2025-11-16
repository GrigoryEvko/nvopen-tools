// Function: sub_F9ABB0
// Address: 0xf9abb0
//
__int64 __fastcall sub_F9ABB0(__int64 a1, __int64 a2, __int64 a3, __int64 **a4)
{
  __int64 v4; // r14
  __int64 *v5; // r15
  __int64 v6; // r13
  bool v7; // zf
  __int64 *v8; // rax
  __int64 *v9; // r13
  unsigned int v10; // r13d
  __int64 v12; // r13
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 *v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r12
  __int64 **v23; // r13
  __int64 v24; // rbx
  __int64 *v25; // r12
  __int64 *v26; // rax
  __int64 *v27; // r12
  __int64 **v28; // r13
  __int64 v29; // rbx
  __int64 *v30; // r12
  __int64 *v31; // rax
  __int64 *v32; // r12
  __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 *v35; // rax
  __int64 *v36; // rdx
  __int64 *v37; // r12
  __int64 *v38; // rax
  __int64 *v39; // rbx
  __int64 *v40; // r13
  __int64 *v41; // r12
  __int64 v42; // r14
  __int64 v43; // r15
  __int64 v44; // rax
  __int64 v45; // r11
  unsigned __int16 v46; // ax
  unsigned __int16 v47; // ax
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r11
  __int64 v51; // r15
  __int64 v52; // r14
  __int64 *v53; // r13
  __int64 *v54; // rax
  __int64 *v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r14
  __int64 v58; // r15
  __int64 v59; // r12
  __int64 v60; // r14
  __int64 v61; // r15
  __int64 v62; // r12
  __int64 v63; // r14
  __int64 v64; // r12
  char v65; // al
  __int64 v66; // r11
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r8
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int16 v73; // dx
  char v74; // cl
  char v75; // dl
  __int64 v76; // rax
  __int64 v77; // r11
  __int64 v78; // rdi
  __int64 *v79; // rax
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // r11
  __int64 v84; // rsi
  __int64 *v85; // rax
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // r11
  __int64 v90; // rsi
  __int64 v91; // r11
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // r11
  unsigned __int64 v96; // rax
  __int64 v97; // rax
  char v98; // al
  __int16 v99; // cx
  _QWORD *v100; // rax
  __int64 v101; // r11
  _QWORD *v102; // r9
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  unsigned __int64 v106; // rcx
  unsigned __int64 v107; // rax
  unsigned __int8 v108; // dl
  __int64 *v109; // rax
  __int64 *v110; // r13
  __int64 *v111; // r14
  __int64 v112; // r12
  __int64 *v113; // r15
  __int64 *v114; // rbx
  __int64 v115; // rdx
  unsigned int v116; // esi
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // [rsp+0h] [rbp-280h]
  __int64 v120; // [rsp+8h] [rbp-278h]
  char v121; // [rsp+8h] [rbp-278h]
  __int64 v122; // [rsp+8h] [rbp-278h]
  __int64 v123; // [rsp+8h] [rbp-278h]
  __int64 v124; // [rsp+10h] [rbp-270h]
  __int64 v125; // [rsp+10h] [rbp-270h]
  __int64 v126; // [rsp+10h] [rbp-270h]
  __int64 v127; // [rsp+18h] [rbp-268h]
  __int64 v128; // [rsp+18h] [rbp-268h]
  __int64 v129; // [rsp+18h] [rbp-268h]
  __int64 v130; // [rsp+18h] [rbp-268h]
  __int64 v131; // [rsp+20h] [rbp-260h]
  __int64 v132; // [rsp+20h] [rbp-260h]
  __int64 v133; // [rsp+20h] [rbp-260h]
  __int64 v134; // [rsp+20h] [rbp-260h]
  __int64 v135; // [rsp+20h] [rbp-260h]
  __int64 v136; // [rsp+20h] [rbp-260h]
  __int64 *v137; // [rsp+20h] [rbp-260h]
  __int64 v138; // [rsp+20h] [rbp-260h]
  __int64 v139; // [rsp+20h] [rbp-260h]
  __int64 v140; // [rsp+20h] [rbp-260h]
  __int64 v141; // [rsp+20h] [rbp-260h]
  __int64 v142; // [rsp+20h] [rbp-260h]
  __int64 v143; // [rsp+28h] [rbp-258h]
  __int64 v144; // [rsp+28h] [rbp-258h]
  __int64 v145; // [rsp+28h] [rbp-258h]
  __int64 *v146; // [rsp+28h] [rbp-258h]
  __int64 v147; // [rsp+28h] [rbp-258h]
  __int64 v148; // [rsp+28h] [rbp-258h]
  __int64 v149; // [rsp+28h] [rbp-258h]
  __int64 v150; // [rsp+28h] [rbp-258h]
  __int64 v151; // [rsp+30h] [rbp-250h]
  __int64 *v152; // [rsp+30h] [rbp-250h]
  __int64 v153; // [rsp+30h] [rbp-250h]
  __int64 v154; // [rsp+30h] [rbp-250h]
  __int64 v155; // [rsp+30h] [rbp-250h]
  __int64 v156; // [rsp+30h] [rbp-250h]
  __int64 v157; // [rsp+30h] [rbp-250h]
  _QWORD *v158; // [rsp+30h] [rbp-250h]
  __int64 v159; // [rsp+30h] [rbp-250h]
  __int64 *v160; // [rsp+30h] [rbp-250h]
  __int64 v161; // [rsp+38h] [rbp-248h]
  unsigned __int8 v162; // [rsp+50h] [rbp-230h]
  __int64 v163; // [rsp+50h] [rbp-230h]
  __int64 v164; // [rsp+50h] [rbp-230h]
  __int64 v165; // [rsp+50h] [rbp-230h]
  __int16 v166; // [rsp+5Ah] [rbp-226h]
  char v167; // [rsp+5Ch] [rbp-224h]
  unsigned __int8 v168; // [rsp+5Dh] [rbp-223h]
  char v169; // [rsp+5Eh] [rbp-222h]
  char v170; // [rsp+5Fh] [rbp-221h]
  __int64 v173; // [rsp+70h] [rbp-210h]
  __int64 v174; // [rsp+70h] [rbp-210h]
  __int64 v175; // [rsp+78h] [rbp-208h]
  __int64 *v176; // [rsp+80h] [rbp-200h]
  __int64 v177; // [rsp+88h] [rbp-1F8h]
  __int64 **v178; // [rsp+A8h] [rbp-1D8h] BYREF
  _QWORD v179[2]; // [rsp+B0h] [rbp-1D0h] BYREF
  __int64 v180[4]; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v181[4]; // [rsp+E0h] [rbp-1A0h] BYREF
  __int16 v182; // [rsp+100h] [rbp-180h]
  __int64 v183[4]; // [rsp+110h] [rbp-170h] BYREF
  __int16 v184; // [rsp+130h] [rbp-150h]
  __int64 v185; // [rsp+140h] [rbp-140h] BYREF
  __int64 *v186; // [rsp+148h] [rbp-138h]
  __int64 v187; // [rsp+150h] [rbp-130h]
  int v188; // [rsp+158h] [rbp-128h]
  char v189; // [rsp+15Ch] [rbp-124h]
  char v190; // [rsp+160h] [rbp-120h] BYREF
  __int64 v191; // [rsp+180h] [rbp-100h] BYREF
  __int64 *v192; // [rsp+188h] [rbp-F8h]
  __int64 v193; // [rsp+190h] [rbp-F0h]
  int v194; // [rsp+198h] [rbp-E8h]
  char v195; // [rsp+19Ch] [rbp-E4h]
  char v196; // [rsp+1A0h] [rbp-E0h] BYREF
  __int64 *v197; // [rsp+1C0h] [rbp-C0h] BYREF
  __int64 v198; // [rsp+1C8h] [rbp-B8h]
  _BYTE v199[32]; // [rsp+1D0h] [rbp-B0h] BYREF
  __int64 v200; // [rsp+1F0h] [rbp-90h]
  __int64 *v201; // [rsp+1F8h] [rbp-88h]
  __int64 v202; // [rsp+200h] [rbp-80h]
  __int64 v203; // [rsp+208h] [rbp-78h]
  void **v204; // [rsp+210h] [rbp-70h]
  void **v205; // [rsp+218h] [rbp-68h]
  __int64 v206; // [rsp+220h] [rbp-60h]
  int v207; // [rsp+228h] [rbp-58h]
  __int16 v208; // [rsp+22Ch] [rbp-54h]
  char v209; // [rsp+22Eh] [rbp-52h]
  __int64 v210; // [rsp+230h] [rbp-50h]
  __int64 v211; // [rsp+238h] [rbp-48h]
  void *v212; // [rsp+240h] [rbp-40h] BYREF
  void *v213; // [rsp+248h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(__int64 **)(a2 - 32);
  v176 = *(__int64 **)(a1 - 32);
  v177 = *(_QWORD *)(a1 - 64);
  v6 = sub_AA56F0(v4);
  v7 = v4 == sub_AA56F0((__int64)v5);
  v8 = (__int64 *)v6;
  if ( v7 )
    v8 = (__int64 *)v4;
  v175 = (__int64)v8;
  if ( !v8 )
    return 0;
  v9 = *(__int64 **)(a2 + 40);
  if ( (__int64 *)v177 == v9 )
  {
    if ( (__int64 *)v4 != v8 )
    {
      v169 = 1;
      v177 = (__int64)v176;
LABEL_13:
      v176 = 0;
LABEL_7:
      v170 = 0;
      if ( v5 != v8 )
        goto LABEL_8;
      goto LABEL_15;
    }
    v169 = 1;
    v4 = (__int64)v5;
    v170 = 1;
    v177 = (__int64)v176;
    goto LABEL_70;
  }
  if ( (__int64 *)v4 != v8 )
  {
    v169 = 0;
    if ( v176 != v9 )
      goto LABEL_7;
    goto LABEL_13;
  }
  v169 = 0;
  v4 = (__int64)v5;
  v170 = 1;
  if ( v176 == v9 )
LABEL_70:
    v176 = 0;
LABEL_15:
  v5 = 0;
LABEL_8:
  v173 = *(_QWORD *)(a1 + 40);
  if ( v173 != sub_AA54C0(v177) )
    return 0;
  if ( v9 != (__int64 *)sub_AA56F0(v177) )
    return 0;
  v12 = *(_QWORD *)(a2 + 40);
  if ( v12 != sub_AA54C0(v4) )
    return 0;
  if ( v175 != sub_AA56F0(v4) )
    return 0;
  if ( v176 )
  {
    v13 = *(_QWORD *)(a1 + 40);
    v14 = *(_QWORD *)(a2 + 40);
    if ( v13 != sub_AA54C0((__int64)v176) || v14 != sub_AA56F0((__int64)v176) )
      return 0;
  }
  if ( v5 )
  {
    v15 = *(_QWORD *)(a2 + 40);
    if ( v15 != sub_AA54C0((__int64)v5) || v175 != sub_AA56F0((__int64)v5) )
      return 0;
  }
  v16 = *(_QWORD *)(a2 + 40);
  v17 = 2;
  v168 = sub_BD3610(v16, 2);
  if ( !v168 )
    return 0;
  v22 = v176;
  v185 = 0;
  v186 = (__int64 *)&v190;
  v192 = (__int64 *)&v196;
  v187 = 4;
  v198 = v177;
  v23 = &v197;
  v188 = 0;
  v189 = 1;
  v191 = 0;
  v193 = 4;
  v194 = 0;
  v195 = 1;
  v197 = v176;
  while ( 1 )
  {
    if ( v22 )
    {
      v24 = v22[7];
      v25 = v22 + 6;
      if ( (__int64 *)v24 != v25 )
      {
        do
        {
          if ( !v24 )
            BUG();
          if ( *(_BYTE *)(v24 - 24) == 62 )
          {
            v17 = *(_QWORD *)(v24 - 56);
            if ( !v189 )
              goto LABEL_71;
            v26 = v186;
            v19 = HIDWORD(v187);
            v18 = &v186[HIDWORD(v187)];
            if ( v186 != v18 )
            {
              while ( v17 != *v26 )
              {
                if ( v18 == ++v26 )
                  goto LABEL_73;
              }
              goto LABEL_36;
            }
LABEL_73:
            if ( HIDWORD(v187) < (unsigned int)v187 )
            {
              v19 = (unsigned int)++HIDWORD(v187);
              *v18 = v17;
              ++v185;
            }
            else
            {
LABEL_71:
              sub_C8CC70((__int64)&v185, v17, (__int64)v18, v19, v20, v21);
            }
          }
LABEL_36:
          v24 = *(_QWORD *)(v24 + 8);
        }
        while ( v25 != (__int64 *)v24 );
      }
    }
    if ( v199 == (_BYTE *)++v23 )
      break;
    v22 = *v23;
  }
  v197 = v5;
  v27 = v5;
  v28 = &v197;
  v198 = v4;
  while ( 1 )
  {
    if ( v27 )
    {
      v29 = v27[7];
      v30 = v27 + 6;
      if ( (__int64 *)v29 != v30 )
      {
        do
        {
          if ( !v29 )
            BUG();
          if ( *(_BYTE *)(v29 - 24) == 62 )
          {
            v17 = *(_QWORD *)(v29 - 56);
            if ( !v195 )
              goto LABEL_72;
            v31 = v192;
            v19 = HIDWORD(v193);
            v18 = &v192[HIDWORD(v193)];
            if ( v192 != v18 )
            {
              while ( v17 != *v31 )
              {
                if ( v18 == ++v31 )
                  goto LABEL_75;
              }
              goto LABEL_48;
            }
LABEL_75:
            if ( HIDWORD(v193) < (unsigned int)v193 )
            {
              v19 = (unsigned int)++HIDWORD(v193);
              *v18 = v17;
              ++v191;
            }
            else
            {
LABEL_72:
              sub_C8CC70((__int64)&v191, v17, (__int64)v18, v19, v20, v21);
            }
          }
LABEL_48:
          v29 = *(_QWORD *)(v29 + 8);
        }
        while ( v30 != (__int64 *)v29 );
      }
    }
    if ( ++v28 == (__int64 **)v199 )
      break;
    v27 = *v28;
  }
  v32 = v186;
  if ( !v189 )
  {
    v33 = (unsigned int)v187;
    v34 = &v186[(unsigned int)v187];
    if ( v186 == v34 )
      goto LABEL_97;
    while ( 1 )
    {
      v17 = *v32;
      if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v195 )
        {
          v35 = v192;
          v36 = &v192[HIDWORD(v193)];
          if ( v192 != v36 )
          {
            while ( v17 != *v35 )
            {
              if ( v36 == ++v35 )
                goto LABEL_78;
            }
            goto LABEL_58;
          }
LABEL_78:
          *v32 = -2;
          ++v188;
          ++v185;
          goto LABEL_58;
        }
        if ( !sub_C8CA60((__int64)&v191, v17) )
          goto LABEL_78;
      }
LABEL_58:
      if ( ++v32 == v34 )
        goto LABEL_59;
    }
  }
  v53 = &v186[HIDWORD(v187)];
  if ( v53 == v186 )
  {
    v34 = &v186[HIDWORD(v187)];
LABEL_60:
    v37 = &v34[HIDWORD(v187)];
    goto LABEL_61;
  }
  do
  {
    v17 = *v32;
    if ( v195 )
    {
      v54 = v192;
      v55 = &v192[HIDWORD(v193)];
      if ( v192 != v55 )
      {
        while ( v17 != *v54 )
        {
          if ( v55 == ++v54 )
            goto LABEL_116;
        }
LABEL_112:
        ++v32;
        continue;
      }
    }
    else if ( sub_C8CA60((__int64)&v191, v17) )
    {
      goto LABEL_112;
    }
LABEL_116:
    v56 = *--v53;
    *v32 = v56;
    --HIDWORD(v187);
    ++v185;
  }
  while ( v32 != v53 );
LABEL_59:
  v34 = v186;
  if ( v189 )
    goto LABEL_60;
  v33 = (unsigned int)v187;
LABEL_97:
  v37 = &v34[v33];
LABEL_61:
  v38 = v34;
  if ( v34 == v37 )
    goto LABEL_64;
  while ( 1 )
  {
    v39 = v38;
    if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v37 == ++v38 )
      goto LABEL_64;
  }
  if ( v37 == v38 )
  {
LABEL_64:
    v10 = 0;
    goto LABEL_65;
  }
  v174 = (__int64)v5;
  v161 = v4 + 48;
  v162 = 0;
  v40 = v37;
  v41 = (__int64 *)v4;
  v42 = *v38;
  while ( 2 )
  {
    v17 = (__int64)v41;
    v43 = sub_F8E330((__int64)v176, v177);
    v44 = sub_F8E330(v174, (__int64)v41);
    v45 = v44;
    if ( !v43 )
      goto LABEL_90;
    if ( !v44 )
      goto LABEL_90;
    v46 = *(_WORD *)(v44 + 2);
    if ( ((v46 >> 7) & 6) != 0 )
      goto LABEL_90;
    if ( (v46 & 1) != 0 )
      goto LABEL_90;
    v47 = *(_WORD *)(v43 + 2);
    if ( ((v47 >> 7) & 6) != 0
      || (v47 & 1) != 0
      || *(_QWORD *)(*(_QWORD *)(v43 - 64) + 8LL) != *(_QWORD *)(*(_QWORD *)(v45 - 64) + 8LL) )
    {
      goto LABEL_90;
    }
    v143 = v45;
    v49 = sub_AA54C0((__int64)v41);
    v50 = v143;
    v151 = v49 + 48;
    if ( *(_QWORD *)(v49 + 56) != v49 + 48 )
    {
      v131 = v43;
      v51 = *(_QWORD *)(v49 + 56);
      v144 = v42;
      v127 = v50;
      do
      {
        v52 = 0;
        if ( v51 )
          v52 = v51 - 24;
        if ( (unsigned __int8)sub_B46420(v52) || (unsigned __int8)sub_B46490(v52) )
          goto LABEL_90;
        v51 = *(_QWORD *)(v51 + 8);
      }
      while ( v151 != v51 );
      v42 = v144;
      v43 = v131;
      v50 = v127;
    }
    if ( v41[7] != v161 )
    {
      v152 = v41;
      v145 = v42;
      v57 = v50;
      v132 = v43;
      v58 = v41[7];
      while ( v58 )
      {
        v59 = v58 - 24;
        if ( v57 != v58 - 24 )
          goto LABEL_122;
LABEL_125:
        v58 = *(_QWORD *)(v58 + 8);
        if ( v161 == v58 )
        {
          v50 = v57;
          v41 = v152;
          v43 = v132;
          v42 = v145;
          goto LABEL_127;
        }
      }
      v59 = 0;
LABEL_122:
      if ( (unsigned __int8)sub_B46420(v59) || (unsigned __int8)sub_B46490(v59) )
      {
        v41 = v152;
        goto LABEL_90;
      }
      goto LABEL_125;
    }
LABEL_127:
    if ( v174 && *(_QWORD *)(v174 + 56) != v174 + 48 )
    {
      v133 = v42;
      v60 = v50;
      v128 = v43;
      v61 = *(_QWORD *)(v174 + 56);
      v146 = v41;
      while ( v61 )
      {
        v62 = v61 - 24;
        if ( v60 != v61 - 24 )
          goto LABEL_134;
LABEL_131:
        v61 = *(_QWORD *)(v61 + 8);
        if ( v174 + 48 == v61 )
        {
          v50 = v60;
          v41 = v146;
          v43 = v128;
          v42 = v133;
          goto LABEL_139;
        }
      }
      v62 = 0;
LABEL_134:
      if ( (unsigned __int8)sub_B46420(v62) || (unsigned __int8)sub_B46490(v62) )
      {
LABEL_135:
        v41 = v146;
        goto LABEL_90;
      }
      goto LABEL_131;
    }
LABEL_139:
    v153 = *(_QWORD *)(v43 + 40) + 48LL;
    if ( v43 + 24 != v153 )
    {
      v134 = v42;
      v63 = v43 + 24;
      v146 = v41;
      v129 = v50;
      while ( v63 )
      {
        v64 = v63 - 24;
        if ( v43 != v63 - 24 )
          goto LABEL_145;
LABEL_142:
        v63 = *(_QWORD *)(v63 + 8);
        if ( v153 == v63 )
        {
          v41 = v146;
          v42 = v134;
          v50 = v129;
          goto LABEL_150;
        }
      }
      v64 = 0;
LABEL_145:
      if ( (unsigned __int8)sub_B46420(v64) || (unsigned __int8)sub_B46490(v64) )
        goto LABEL_135;
      goto LABEL_142;
    }
LABEL_150:
    v179[0] = v43;
    v179[1] = v50;
    v178 = a4;
    if ( (_BYTE)qword_4F8CB68
      || (v17 = (__int64)v176, v135 = v50, (unsigned __int8)sub_F92670(&v178, (__int64)v176, v179, 2))
      && (v17 = v177, (unsigned __int8)sub_F92670(&v178, v177, v179, 2))
      && (v17 = v174, (unsigned __int8)sub_F92670(&v178, v174, v179, 2))
      && (v17 = (__int64)v41, v65 = sub_F92670(&v178, (__int64)v41, v179, 2), v50 = v135, v65) )
    {
      v154 = v50;
      v197 = *(__int64 **)(v175 + 16);
      sub_D4B000((__int64 *)&v197);
      v197 = (__int64 *)v197[1];
      sub_D4B000((__int64 *)&v197);
      v197 = (__int64 *)v197[1];
      sub_D4B000((__int64 *)&v197);
      v66 = v154;
      if ( v197 )
      {
        v67 = v174;
        if ( !v174 )
        {
          v67 = sub_AA54C0((__int64)v41);
          v66 = v154;
        }
        v17 = (__int64)&v197;
        v155 = v66;
        v197 = v41;
        v198 = v67;
        v68 = sub_F41DE0(v175, &v197, 2, "condstore.split", a3, 0, 0, 0);
        v66 = v155;
        v69 = v68;
        if ( !v68 )
          goto LABEL_90;
      }
      else
      {
        v69 = v175;
      }
      v136 = v69;
      v163 = v66;
      v70 = sub_AA54C0(v177);
      v156 = *(_QWORD *)(sub_986580(v70) - 96);
      v71 = sub_AA54C0((__int64)v41);
      v147 = *(_QWORD *)(sub_986580(v71) - 96);
      v72 = sub_F92230(*(_QWORD *)(v43 - 64), *(_QWORD *)(v43 + 40), 0);
      v120 = v163;
      v130 = sub_F92230(*(_QWORD *)(v163 - 64), *(_QWORD *)(v163 + 40), v72);
      v164 = v136;
      v137 = (__int64 *)sub_AA5190(v136);
      v124 = (__int64)v137;
      if ( v137 )
      {
        v74 = v73;
        v75 = HIBYTE(v73);
      }
      else
      {
        v75 = 0;
        v74 = 0;
      }
      v167 = v75;
      v119 = v120;
      v121 = v74;
      v76 = sub_AA48A0(v164);
      v209 = 7;
      v203 = v76;
      v204 = &v212;
      v205 = &v213;
      v197 = (__int64 *)v199;
      v77 = v119;
      v198 = 0x200000000LL;
      v212 = &unk_49DA100;
      v206 = 0;
      v207 = 0;
      v213 = &unk_49DA0B0;
      LOBYTE(v76) = v121;
      v208 = 512;
      BYTE1(v76) = v167;
      v210 = 0;
      LOWORD(v202) = v76;
      v211 = 0;
      v200 = v164;
      v201 = v137;
      if ( v137 == (__int64 *)(v164 + 48) )
        goto LABEL_169;
      if ( v137 )
        v78 = (__int64)(v137 - 3);
      else
        v78 = 0;
      v79 = (__int64 *)sub_B46C60(v78);
      v83 = v119;
      v84 = *v79;
      v183[0] = v84;
      if ( v84 )
      {
        sub_B96E90((__int64)v183, v84, 1);
        v83 = v119;
      }
      v122 = v83;
      sub_F80810((__int64)&v197, 0, v183[0], v80, v81, v82);
      v77 = v122;
      if ( v183[0] )
      {
        sub_B91220((__int64)v183, v183[0]);
        v77 = v122;
      }
      if ( v137 )
LABEL_169:
        v124 = (__int64)(v137 - 3);
      v138 = v77;
      v85 = (__int64 *)sub_B46C60(v124);
      v89 = v138;
      v90 = *v85;
      v183[0] = *v85;
      if ( v183[0] )
      {
        sub_B96E90((__int64)v183, v90, 1);
        v89 = v138;
      }
      v139 = v89;
      sub_F80810((__int64)&v197, 0, v183[0], v86, v87, v88);
      sub_9C6650(v183);
      v91 = v139;
      if ( *(__int64 **)(v43 + 40) != v176 )
      {
        v184 = 257;
        v92 = sub_A82B60((unsigned int **)&v197, v156, (__int64)v183);
        v91 = v139;
        v156 = v92;
      }
      if ( v174 != *(_QWORD *)(v91 + 40) )
      {
        v140 = v91;
        v184 = 257;
        v93 = sub_A82B60((unsigned int **)&v197, v147, (__int64)v183);
        v91 = v140;
        v147 = v93;
      }
      if ( v169 )
      {
        v142 = v91;
        v184 = 257;
        v118 = sub_A82B60((unsigned int **)&v197, v156, (__int64)v183);
        v91 = v142;
        v156 = v118;
      }
      if ( v170 )
      {
        v141 = v91;
        v184 = 257;
        v117 = sub_A82B60((unsigned int **)&v197, v147, (__int64)v183);
        v91 = v141;
        v147 = v117;
      }
      v125 = v91;
      v182 = 257;
      v94 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v204 + 2))(v204, 29, v156, v147);
      v95 = v125;
      if ( !v94 )
      {
        v184 = 257;
        v159 = sub_B504D0(29, v156, v147, (__int64)v183, 0, 0);
        (*((void (__fastcall **)(void **, __int64, __int64 *, __int64 *, __int64))*v205 + 2))(
          v205,
          v159,
          v181,
          v201,
          v202);
        v94 = v159;
        v95 = v125;
        if ( v197 != &v197[2 * (unsigned int)v198] )
        {
          v160 = v40;
          v110 = &v197[2 * (unsigned int)v198];
          v150 = v42;
          v111 = v41;
          v112 = v94;
          v126 = v43;
          v113 = v39;
          v114 = v197;
          do
          {
            v115 = v114[1];
            v116 = *(_DWORD *)v114;
            v114 += 2;
            v123 = v95;
            sub_B99FD0(v112, v116, v115);
            v95 = v123;
          }
          while ( v110 != v114 );
          v39 = v113;
          v94 = v112;
          v41 = v111;
          v40 = v160;
          v43 = v126;
          v42 = v150;
        }
      }
      v157 = v95;
      v96 = sub_F38250(v94, v201, (unsigned __int16)v202, 0, 0, a3, 0, 0);
      sub_D5F1F0((__int64)&v197, v96);
      v97 = sub_AA4E30(v200);
      v98 = sub_AE5020(v97, *(_QWORD *)(v130 + 8));
      HIBYTE(v99) = HIBYTE(v166);
      LOBYTE(v99) = v98;
      v166 = v99;
      v184 = 257;
      v100 = sub_BD2C40(80, unk_3F10A10);
      v101 = v157;
      v102 = v100;
      if ( v100 )
      {
        v148 = v157;
        v158 = v100;
        sub_B4D3C0((__int64)v100, v130, v42, 0, v166, (__int64)v100, 0, 0);
        v101 = v148;
        v102 = v158;
      }
      v149 = v101;
      v165 = (__int64)v102;
      (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64 *, __int64))*v205 + 2))(
        v205,
        v102,
        v183,
        v201,
        v202);
      sub_94AAF0((unsigned int **)&v197, v165);
      sub_B91FC0(v181, v43);
      sub_B91FC0(v183, v149);
      sub_E01E30(v180, v181, v183, v103, v104, v105);
      sub_B9A100(v165, v180);
      _BitScanReverse64(&v106, 1LL << (*(_WORD *)(v149 + 2) >> 1));
      v17 = 63 - ((unsigned int)v106 ^ 0x3F);
      LOBYTE(v181[0]) = 63 - (v106 ^ 0x3F);
      _BitScanReverse64(&v107, 1LL << (*(_WORD *)(v43 + 2) >> 1));
      v108 = 63 - (v107 ^ 0x3F);
      v109 = v181;
      if ( LOBYTE(v181[0]) >= v108 )
        v109 = v183;
      LOBYTE(v183[0]) = v108;
      *(_WORD *)(v165 + 2) = (2 * *(unsigned __int8 *)v109) | *(_WORD *)(v165 + 2) & 0xFF81;
      sub_B43D60((_QWORD *)v149);
      sub_B43D60((_QWORD *)v43);
      sub_F94A20(&v197, v17);
      v162 = v168;
    }
LABEL_90:
    v48 = v39 + 1;
    if ( v39 + 1 != v40 )
    {
      while ( 1 )
      {
        v42 = *v48;
        v39 = v48;
        if ( (unsigned __int64)*v48 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v40 == ++v48 )
          goto LABEL_93;
      }
      if ( v48 != v40 )
        continue;
    }
    break;
  }
LABEL_93:
  v10 = v162;
LABEL_65:
  if ( !v195 )
    _libc_free(v192, v17);
  if ( !v189 )
    _libc_free(v186, v17);
  return v10;
}
