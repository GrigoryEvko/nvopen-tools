// Function: sub_F27020
// Address: 0xf27020
//
unsigned __int8 *__fastcall sub_F27020(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rbx
  char v9; // dl
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v12; // rbx
  __int64 *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // r8
  __int64 *v23; // rbx
  __int64 *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rbx
  unsigned __int64 v30; // rax
  int v31; // edx
  unsigned __int64 v32; // rax
  __int64 v33; // rsi
  unsigned __int8 *v34; // rax
  __int64 v35; // r8
  unsigned __int8 v36; // al
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int16 v39; // ax
  __int64 v40; // rax
  const char *v41; // rdi
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int8 *result; // rax
  __int64 v45; // rsi
  __int64 v46; // r8
  char v47; // al
  __int64 v48; // rdi
  unsigned __int64 v49; // rax
  int v50; // edx
  __int64 v51; // rax
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rcx
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // rax
  unsigned __int8 *v67; // rdi
  __int64 v68; // rdx
  _BYTE *v69; // rax
  __int64 *v70; // rax
  __int64 v71; // r14
  __int64 v72; // r15
  __int64 v73; // r13
  unsigned int *v74; // rbx
  __int64 *v75; // r9
  int v76; // esi
  unsigned int v77; // edx
  __int64 *v78; // rax
  const char *v79; // r11
  unsigned __int8 **v80; // r12
  __int64 v81; // rdi
  __int64 v82; // r14
  const char *v83; // r15
  __int64 v84; // rax
  __int64 v85; // r10
  __int64 v86; // rcx
  __int64 v87; // r12
  __int64 v88; // r14
  __int64 v89; // r13
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rax
  __int64 v94; // rax
  unsigned __int64 v95; // rax
  unsigned __int64 v96; // rsi
  __int64 v97; // rax
  __int64 *v98; // r10
  int v99; // edx
  unsigned int v100; // esi
  __int64 *v101; // rax
  const char *v102; // rdi
  int v103; // r9d
  __int64 *v104; // r8
  unsigned int v105; // eax
  unsigned int v106; // edx
  unsigned int v107; // edi
  unsigned int v108; // esi
  const char **v109; // rax
  int v110; // ebx
  __int64 v111; // r12
  __int64 v112; // rax
  int v113; // ebx
  _QWORD *v114; // r10
  __int64 v115; // rdi
  const char *v116; // rsi
  __int64 v117; // r10
  const char **v118; // rbx
  __int64 v119; // r15
  __int64 v120; // r12
  __int64 v121; // rbx
  int v122; // eax
  unsigned int v123; // edx
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // rdx
  __int64 v127; // rsi
  __int64 v128; // r13
  int v129; // eax
  __int64 v130; // r13
  __int64 v131; // r10
  __int64 v132; // rbx
  __int64 v133; // r13
  __int64 v134; // rax
  __int64 *v135; // r12
  int v136; // eax
  int v137; // ecx
  __int64 v138; // rsi
  unsigned __int8 *v139; // rsi
  __int64 v140; // [rsp+8h] [rbp-1F8h]
  __int64 v141; // [rsp+20h] [rbp-1E0h]
  __int64 v142; // [rsp+30h] [rbp-1D0h]
  __int64 v143; // [rsp+38h] [rbp-1C8h]
  char v144; // [rsp+43h] [rbp-1BDh]
  __int64 v145; // [rsp+48h] [rbp-1B8h]
  __int64 v146; // [rsp+50h] [rbp-1B0h]
  __int16 v147; // [rsp+58h] [rbp-1A8h]
  __int64 v148; // [rsp+60h] [rbp-1A0h]
  __int64 v149; // [rsp+60h] [rbp-1A0h]
  __int64 v150; // [rsp+68h] [rbp-198h]
  unsigned __int8 **v151; // [rsp+68h] [rbp-198h]
  __int64 v152; // [rsp+70h] [rbp-190h]
  __int64 v153; // [rsp+78h] [rbp-188h]
  __int64 v154; // [rsp+78h] [rbp-188h]
  __int64 v155; // [rsp+80h] [rbp-180h]
  __int64 v156; // [rsp+80h] [rbp-180h]
  __int64 v157; // [rsp+80h] [rbp-180h]
  __int64 v158; // [rsp+88h] [rbp-178h]
  unsigned int v159; // [rsp+90h] [rbp-170h]
  __int16 v160; // [rsp+96h] [rbp-16Ah]
  __int64 v161; // [rsp+98h] [rbp-168h]
  __int64 v162; // [rsp+98h] [rbp-168h]
  __int64 i; // [rsp+A0h] [rbp-160h]
  __int64 v164; // [rsp+A0h] [rbp-160h]
  _QWORD *v165; // [rsp+A0h] [rbp-160h]
  __int64 v166; // [rsp+A0h] [rbp-160h]
  __int64 v167; // [rsp+A0h] [rbp-160h]
  __int64 v168; // [rsp+A0h] [rbp-160h]
  __int64 v169; // [rsp+A0h] [rbp-160h]
  __int64 v170; // [rsp+A8h] [rbp-158h]
  unsigned __int8 *v171; // [rsp+A8h] [rbp-158h]
  unsigned __int8 *v172; // [rsp+A8h] [rbp-158h]
  __int64 v173; // [rsp+A8h] [rbp-158h]
  unsigned int *v174; // [rsp+A8h] [rbp-158h]
  unsigned __int8 *v175; // [rsp+A8h] [rbp-158h]
  unsigned __int8 *v176; // [rsp+A8h] [rbp-158h]
  const char **v177; // [rsp+B8h] [rbp-148h] BYREF
  _BYTE *v178; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-138h]
  _BYTE v180[48]; // [rsp+D0h] [rbp-130h] BYREF
  unsigned int *v181; // [rsp+100h] [rbp-100h] BYREF
  __int64 v182; // [rsp+108h] [rbp-F8h]
  _BYTE v183[48]; // [rsp+110h] [rbp-F0h] BYREF
  unsigned __int8 **v184; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v185; // [rsp+148h] [rbp-B8h]
  _BYTE v186[16]; // [rsp+150h] [rbp-B0h] BYREF
  __int16 v187; // [rsp+160h] [rbp-A0h]
  __int64 v188; // [rsp+180h] [rbp-80h] BYREF
  __int64 v189; // [rsp+188h] [rbp-78h]
  __int64 *v190; // [rsp+190h] [rbp-70h] BYREF
  __int64 v191; // [rsp+198h] [rbp-68h]
  __int64 v192; // [rsp+1A0h] [rbp-60h]
  unsigned __int64 v193; // [rsp+1A8h] [rbp-58h]
  __int64 v194; // [rsp+1B0h] [rbp-50h]
  __int64 v195; // [rsp+1B8h] [rbp-48h]
  __int16 v196; // [rsp+1C0h] [rbp-40h]
  char v197; // [rsp+1D0h] [rbp-30h] BYREF

  HIBYTE(v160) = a4;
  v159 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  if ( !v159 )
    return 0;
  v6 = *(_QWORD *)(a3 + 16);
  if ( v6 )
  {
    if ( *(_QWORD *)(v6 + 8) )
    {
      if ( a4 )
      {
        v160 = 0;
      }
      else
      {
        do
        {
          v45 = *(_QWORD *)(v6 + 24);
          if ( a2 != v45 && !sub_B46220(a2, v45) )
            return 0;
          v6 = *(_QWORD *)(v6 + 8);
        }
        while ( v6 );
        LOBYTE(v160) = 1;
      }
    }
    else
    {
      v160 = 256;
    }
  }
  else
  {
    v160 = a4 ^ 1;
  }
  v9 = *(_BYTE *)(a2 + 7);
  v10 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v11 = 4 * v10;
  if ( (v9 & 0x40) != 0 )
  {
    v12 = *(__int64 **)(a2 - 8);
    v13 = &v12[v11];
  }
  else
  {
    v12 = (__int64 *)(a2 - v11 * 8);
    v13 = (__int64 *)a2;
  }
  if ( v12 != v13 )
  {
    do
    {
      v14 = *v12;
      if ( a3 != *v12 && *(_BYTE *)v14 > 0x1Cu )
      {
        v15 = *(_QWORD *)(a3 + 40);
        if ( (*(_BYTE *)v14 != 84 || *(_QWORD *)(v14 + 40) != v15)
          && !(unsigned __int8)sub_B19D00(*(_QWORD *)(a1 + 80), v14, v15) )
        {
          return 0;
        }
      }
      v12 += 4;
    }
    while ( v13 != v12 );
    v9 = *(_BYTE *)(a2 + 7);
    v10 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  v16 = a3;
  v161 = 0;
  v178 = v180;
  v179 = 0x600000000LL;
  v144 = 0;
  v181 = (unsigned int *)v183;
  v182 = 0xC00000000LL;
  v17 = v10;
  for ( i = a2; ; v17 = *(_DWORD *)(i + 4) & 0x7FFFFFF )
  {
    v19 = *(_QWORD *)(v16 - 8);
    v20 = 32 * v17;
    v170 = *(_QWORD *)(v19 + 32 * v161);
    v21 = *(_QWORD *)(v19 + 32LL * *(unsigned int *)(v16 + 72) + 8 * v161);
    v143 = *(_QWORD *)(a1 + 88);
    v152 = *(_QWORD *)(a1 + 96);
    v150 = *(_QWORD *)(a1 + 104);
    v153 = *(_QWORD *)(a1 + 112);
    v146 = *(_QWORD *)(a1 + 120);
    v155 = *(_QWORD *)(a1 + 128);
    v148 = *(_QWORD *)(a1 + 144);
    v145 = *(_QWORD *)(a1 + 152);
    v147 = *(_WORD *)(a1 + 160);
    v184 = (unsigned __int8 **)v186;
    v185 = 0x600000000LL;
    if ( (v9 & 0x40) != 0 )
    {
      v22 = *(_QWORD *)(i - 8);
      v23 = (__int64 *)(v22 + v20);
    }
    else
    {
      v22 = i - v20;
      v23 = (__int64 *)i;
    }
    v24 = (__int64 *)v22;
    if ( (__int64 *)v22 != v23 )
    {
      do
      {
        while ( v16 == *v24 )
        {
          v27 = (unsigned int)v185;
          v28 = (unsigned int)v185 + 1LL;
          if ( v28 > HIDWORD(v185) )
          {
            sub_C8D5F0((__int64)&v184, v186, v28, 8u, v22, (__int64)a6);
            v27 = (unsigned int)v185;
          }
          v24 += 4;
          v184[v27] = (unsigned __int8 *)v170;
          LODWORD(v185) = v185 + 1;
          if ( v23 == v24 )
            goto LABEL_26;
        }
        v25 = sub_BD5BF0(*v24, *(_QWORD *)(v16 + 40), v21);
        v26 = (unsigned int)v185;
        a6 = (__int64 *)((unsigned int)v185 + 1LL);
        if ( (unsigned __int64)a6 > HIDWORD(v185) )
        {
          v142 = v25;
          sub_C8D5F0((__int64)&v184, v186, (unsigned int)v185 + 1LL, 8u, v22, (__int64)a6);
          v26 = (unsigned int)v185;
          v25 = v142;
        }
        v24 += 4;
        v184[v26] = (unsigned __int8 *)v25;
        LODWORD(v185) = v185 + 1;
      }
      while ( v23 != v24 );
    }
LABEL_26:
    v29 = v21 + 48;
    v30 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v21 + 48 == v30 )
    {
      v32 = 0;
    }
    else
    {
      if ( !v30 )
        BUG();
      v31 = *(unsigned __int8 *)(v30 - 24);
      v32 = v30 - 24;
      if ( (unsigned int)(v31 - 30) >= 0xB )
        v32 = 0;
    }
    v193 = v32;
    v33 = (__int64)v184;
    v188 = v152;
    v189 = v150;
    v190 = (__int64 *)v153;
    v191 = v146;
    v192 = v155;
    v194 = v148;
    v195 = v145;
    v196 = v147;
    v34 = (unsigned __int8 *)sub_1020E00(i, v184, (unsigned int)v185, &v188);
    v35 = (__int64)v34;
    if ( v34 && (unsigned __int8 *)v16 != v34 )
    {
      v36 = *v34;
      if ( v36 > 0x15u )
        goto LABEL_103;
      if ( v36 != 5 )
      {
        v156 = v35;
        if ( !(unsigned __int8)sub_AD6CA0(v35) )
        {
          v35 = v156;
LABEL_103:
          v41 = (const char *)v184;
          if ( v184 == (unsigned __int8 **)v186 )
          {
LABEL_46:
            v42 = (unsigned int)v179;
            v43 = (unsigned int)v179 + 1LL;
            if ( v43 > HIDWORD(v179) )
            {
              v173 = v35;
              sub_C8D5F0((__int64)&v178, v180, v43, 8u, v35, (__int64)a6);
              v42 = (unsigned int)v179;
              v35 = v173;
            }
            *(_QWORD *)&v178[8 * v42] = v35;
            LODWORD(v179) = v179 + 1;
            goto LABEL_49;
          }
LABEL_44:
          v157 = v35;
          _libc_free(v41, v33);
          v35 = v157;
          goto LABEL_45;
        }
      }
    }
    v37 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29 == v37 )
      goto LABEL_230;
    if ( !v37 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30 > 0xA )
LABEL_230:
      BUG();
    if ( *(_BYTE *)(v37 - 24) != 31
      || *(_BYTE *)i != 82
      || (*(_DWORD *)(v37 - 20) & 0x7FFFFFF) != 3
      || (v38 = *(_QWORD *)(v37 - 56), v38 == *(_QWORD *)(v37 - 88))
      || (v33 = ((unsigned __int64)((*(_BYTE *)(i + 1) & 2) != 0) << 32)
              | v141 & 0xFFFFFF0000000000LL
              | *(_WORD *)(i + 2) & 0x3FLL,
          v141 = v33,
          v39 = sub_9A13D0(*(_QWORD *)(v37 - 120), v33, (__int64)*v184, v184[1], v143, *(_QWORD *)(v16 + 40) == v38, 0),
          !HIBYTE(v39)) )
    {
      if ( v184 != (unsigned __int8 **)v186 )
        _libc_free(v184, v33);
      goto LABEL_63;
    }
    v33 = (unsigned __int8)v39;
    v40 = sub_AD64A0(*(_QWORD *)(i + 8), v39);
    v41 = (const char *)v184;
    v35 = v40;
    if ( v184 != (unsigned __int8 **)v186 )
      goto LABEL_44;
LABEL_45:
    if ( v35 )
      goto LABEL_46;
LABEL_63:
    if ( !(unsigned __int8)sub_BD36B0(v170) )
      goto LABEL_80;
    v47 = *(_BYTE *)v170;
    if ( *(_BYTE *)v170 == 85 )
    {
      v64 = *(_QWORD *)(v170 - 32);
      if ( !v64 )
        goto LABEL_80;
      if ( *(_BYTE *)v64 )
        goto LABEL_80;
      if ( *(_QWORD *)(v64 + 24) != *(_QWORD *)(v170 + 80) )
        goto LABEL_80;
      if ( (*(_BYTE *)(v64 + 33) & 0x20) == 0 )
        goto LABEL_80;
      v65 = *(_DWORD *)(v64 + 36);
      if ( v65 != 313 && v65 != 362 )
        goto LABEL_80;
      if ( *(_BYTE *)i != 82 )
        goto LABEL_80;
      v66 = *(_QWORD *)(i - 64);
      if ( v16 != v66 || !v66 )
        goto LABEL_80;
      v67 = *(unsigned __int8 **)(i - 32);
      v68 = *v67;
      if ( (_BYTE)v68 == 17 )
        goto LABEL_74;
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v67 + 1) + 8LL) - 17 > 1 || (unsigned __int8)v68 > 0x15u )
        goto LABEL_80;
      v33 = 0;
      v69 = sub_AD7630((__int64)v67, 0, v68);
      if ( v69 && *v69 == 17 )
        goto LABEL_74;
      v47 = *(_BYTE *)v170;
    }
    if ( v47 == 68 )
    {
      v48 = *(_QWORD *)(*(_QWORD *)(v170 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v48 + 8) - 17 <= 1 )
        v48 = **(_QWORD **)(v48 + 16);
      v33 = 1;
      if ( sub_BCAC40(v48, 1) )
      {
        v188 = 32;
        if ( *(_BYTE *)i == 82 )
        {
          v49 = sub_B53900(i);
          v33 = v188;
          v184 = (unsigned __int8 **)sub_B53630(v49, v188);
          LODWORD(v185) = v50;
          if ( (_BYTE)v50 )
          {
            v51 = *(_QWORD *)(i - 64);
            if ( v16 == v51 && v51 && (unsigned __int8)sub_F08D10(*(_QWORD *)(i - 32)) )
            {
LABEL_74:
              v52 = (unsigned int)v182;
              v53 = (unsigned int)v182 + 1LL;
              if ( v53 > HIDWORD(v182) )
              {
                sub_C8D5F0((__int64)&v181, v183, v53, 4u, v46, (__int64)a6);
                v52 = (unsigned int)v182;
              }
              v181[v52] = v161;
              v54 = (unsigned int)v179;
              LODWORD(v182) = v182 + 1;
              v55 = (unsigned int)v179 + 1LL;
              if ( v55 > HIDWORD(v179) )
              {
                sub_C8D5F0((__int64)&v178, v180, v55, 8u, v46, (__int64)a6);
                v54 = (unsigned int)v179;
              }
              *(_QWORD *)&v178[8 * v54] = 0;
              LODWORD(v179) = v179 + 1;
              goto LABEL_49;
            }
          }
        }
      }
    }
LABEL_80:
    if ( (HIBYTE(v160) | (unsigned __int8)v160) != 1 || v144 )
      goto LABEL_97;
    v56 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29 == v56 )
      goto LABEL_232;
    if ( !v56 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v56 - 24) - 30 > 0xA )
LABEL_232:
      BUG();
    if ( *(_BYTE *)(v56 - 24) != 31 )
      goto LABEL_97;
    if ( (*(_DWORD *)(v56 - 20) & 0x7FFFFFF) != 1 )
      goto LABEL_97;
    v57 = *(_QWORD *)(a1 + 80);
    v58 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
    if ( (unsigned int)v58 >= *(_DWORD *)(v57 + 32) || !*(_QWORD *)(*(_QWORD *)(v57 + 24) + 8 * v58) )
      goto LABEL_97;
    v59 = (unsigned int)v179;
    v60 = (unsigned int)v179 + 1LL;
    if ( v60 > HIDWORD(v179) )
    {
      v33 = (__int64)v180;
      sub_C8D5F0((__int64)&v178, v180, v60, 8u, v46, (__int64)a6);
      v59 = (unsigned int)v179;
    }
    *(_QWORD *)&v178[8 * v59] = 0;
    v61 = (unsigned int)v182;
    v62 = HIDWORD(v182);
    LODWORD(v179) = v179 + 1;
    v63 = (unsigned int)v182 + 1LL;
    if ( v63 > HIDWORD(v182) )
    {
      v33 = (__int64)v183;
      sub_C8D5F0((__int64)&v181, v183, v63, 4u, v46, (__int64)a6);
      v61 = (unsigned int)v182;
    }
    if ( (v181[v61] = v161, LODWORD(v182) = v182 + 1, *(_BYTE *)v170 == 34) && v21 == *(_QWORD *)(v170 + 40)
      || (v33 = v21, sub_F1FA60(a1, v21, *(__int64 **)(v16 + 40), v62, v46, a6)) )
    {
LABEL_97:
      result = 0;
      goto LABEL_98;
    }
    v144 = 1;
LABEL_49:
    if ( v159 == (_DWORD)++v161 )
      break;
    v9 = *(_BYTE *)(i + 7);
  }
  v70 = (__int64 *)&v190;
  v71 = a1;
  v72 = i;
  v188 = 0;
  v73 = v16;
  v189 = 1;
  do
  {
    *v70 = -4096;
    v70 += 2;
  }
  while ( v70 != (__int64 *)&v197 );
  v174 = &v181[(unsigned int)v182];
  if ( v174 == v181 )
    goto LABEL_176;
  v154 = v71;
  v74 = v181;
  while ( 2 )
  {
    v81 = *(_QWORD *)(v73 - 8);
    v82 = *v74;
    v83 = *(const char **)(v81 + 32LL * *(unsigned int *)(v73 + 72) + 8 * v82);
    if ( (v189 & 1) != 0 )
    {
      v75 = (__int64 *)&v190;
      v76 = 3;
      goto LABEL_127;
    }
    v75 = v190;
    if ( (_DWORD)v191 )
    {
      v76 = v191 - 1;
LABEL_127:
      v77 = v76 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
      v78 = &v75[2 * v77];
      v79 = (const char *)*v78;
      if ( v83 == (const char *)*v78 )
      {
LABEL_128:
        v80 = (unsigned __int8 **)v78[1];
        if ( v80 )
          goto LABEL_129;
      }
      else
      {
        v136 = 1;
        while ( v79 != (const char *)-4096LL )
        {
          v137 = v136 + 1;
          v77 = v76 & (v136 + v77);
          v78 = &v75[2 * v77];
          v79 = (const char *)*v78;
          if ( v83 == (const char *)*v78 )
            goto LABEL_128;
          v136 = v137;
        }
      }
    }
    v162 = *(_QWORD *)(v81 + 32 * v82);
    v84 = sub_B47F80((_BYTE *)i);
    v80 = (unsigned __int8 **)v84;
    if ( (*(_BYTE *)(v84 + 7) & 0x40) != 0 )
    {
      v85 = *(_QWORD *)(v84 - 8);
      v86 = v85 + 32LL * (*(_DWORD *)(v84 + 4) & 0x7FFFFFF);
    }
    else
    {
      v86 = v84;
      v85 = v84 - 32LL * (*(_DWORD *)(v84 + 4) & 0x7FFFFFF);
    }
    if ( v85 != v86 )
    {
      v151 = (unsigned __int8 **)v84;
      v87 = v86;
      v149 = v82;
      v88 = v73;
      v89 = v85;
      do
      {
        if ( v88 == *(_QWORD *)v89 )
        {
          v93 = *(_QWORD *)(v89 + 8);
          **(_QWORD **)(v89 + 16) = v93;
          if ( v93 )
            *(_QWORD *)(v93 + 16) = *(_QWORD *)(v89 + 16);
          *(_QWORD *)v89 = v162;
          if ( v162 )
          {
            v94 = *(_QWORD *)(v162 + 16);
            *(_QWORD *)(v89 + 8) = v94;
            if ( v94 )
              *(_QWORD *)(v94 + 16) = v89 + 8;
            *(_QWORD *)(v89 + 16) = v162 + 16;
            *(_QWORD *)(v162 + 16) = v89;
          }
        }
        else
        {
          v90 = sub_BD5BF0(*(_QWORD *)v89, *(_QWORD *)(v88 + 40), (__int64)v83);
          if ( *(_QWORD *)v89 )
          {
            v91 = *(_QWORD *)(v89 + 8);
            **(_QWORD **)(v89 + 16) = v91;
            if ( v91 )
              *(_QWORD *)(v91 + 16) = *(_QWORD *)(v89 + 16);
          }
          *(_QWORD *)v89 = v90;
          if ( v90 )
          {
            v92 = *(_QWORD *)(v90 + 16);
            *(_QWORD *)(v89 + 8) = v92;
            if ( v92 )
              *(_QWORD *)(v92 + 16) = v89 + 8;
            *(_QWORD *)(v89 + 16) = v90 + 16;
            *(_QWORD *)(v90 + 16) = v89;
          }
        }
        v89 += 32;
      }
      while ( v87 != v89 );
      v73 = v88;
      v80 = v151;
      v82 = v149;
    }
    v95 = *((_QWORD *)v83 + 6) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (const char *)v95 == v83 + 48 )
    {
      v96 = 0;
    }
    else
    {
      if ( !v95 )
        BUG();
      v96 = v95 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v95 - 24) - 30 >= 0xB )
        v96 = 0;
    }
    v97 = v140;
    LOWORD(v97) = 0;
    sub_B44220(v80, v96 + 24, v97);
    v184 = v80;
    sub_F200C0(*(_QWORD *)(v154 + 40) + 2096LL, (__int64 *)&v184);
    v184 = (unsigned __int8 **)v83;
    v185 = (__int64)v80;
    if ( (v189 & 1) != 0 )
    {
      v98 = (__int64 *)&v190;
      v99 = 3;
      goto LABEL_159;
    }
    v108 = v191;
    v98 = v190;
    v99 = v191 - 1;
    if ( !(_DWORD)v191 )
    {
      v105 = v189;
      ++v188;
      v177 = 0;
      v106 = ((unsigned int)v189 >> 1) + 1;
      goto LABEL_173;
    }
LABEL_159:
    v100 = v99 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
    v101 = &v98[2 * v100];
    v102 = (const char *)*v101;
    if ( v83 != (const char *)*v101 )
    {
      v103 = 1;
      v104 = 0;
      while ( v102 != (const char *)-4096LL )
      {
        if ( v102 == (const char *)-8192LL && !v104 )
          v104 = v101;
        v100 = v99 & (v103 + v100);
        v101 = &v98[2 * v100];
        v102 = (const char *)*v101;
        if ( v83 == (const char *)*v101 )
          goto LABEL_129;
        ++v103;
      }
      if ( v104 )
        v101 = v104;
      ++v188;
      v177 = (const char **)v101;
      v105 = v189;
      v106 = ((unsigned int)v189 >> 1) + 1;
      if ( (v189 & 1) != 0 )
      {
        v107 = 12;
        v108 = 4;
      }
      else
      {
        v108 = v191;
LABEL_173:
        v107 = 3 * v108;
      }
      if ( 4 * v106 >= v107 )
      {
        v108 *= 2;
      }
      else if ( v108 - HIDWORD(v189) - v106 > v108 >> 3 )
      {
        goto LABEL_168;
      }
      sub_F19200((__int64)&v188, v108);
      sub_F17E00((__int64)&v188, (__int64 *)&v184, &v177);
      v83 = (const char *)v184;
      v105 = v189;
LABEL_168:
      LODWORD(v189) = (2 * (v105 >> 1) + 2) | v105 & 1;
      v109 = v177;
      if ( *v177 != (const char *)-4096LL )
        --HIDWORD(v189);
      *v177 = v83;
      v109[1] = (const char *)v185;
    }
LABEL_129:
    ++v74;
    *(_QWORD *)&v178[8 * v82] = v80;
    if ( v174 != v74 )
      continue;
    break;
  }
  v71 = v154;
  v72 = i;
LABEL_176:
  v110 = *(_DWORD *)(v73 + 4);
  v111 = *(_QWORD *)(v72 + 8);
  v187 = 257;
  v112 = sub_BD2DA0(80);
  v113 = v110 & 0x7FFFFFF;
  v114 = (_QWORD *)v112;
  if ( v112 )
  {
    v175 = (unsigned __int8 *)v112;
    v164 = v112;
    sub_B44260(v112, v111, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v164 + 72) = v113;
    sub_BD6B50((unsigned __int8 *)v164, (const char **)&v184);
    sub_BD2A10(v164, *(_DWORD *)(v164 + 72), 1);
    v114 = (_QWORD *)v164;
  }
  else
  {
    v175 = 0;
  }
  v165 = v114;
  sub_B44220(v114, v73 + 24, 0);
  v115 = *(_QWORD *)(v71 + 40) + 2096LL;
  v184 = (unsigned __int8 **)v165;
  sub_F200C0(v115, (__int64 *)&v184);
  sub_BD6B90(v175, (unsigned __int8 *)v73);
  v116 = *(const char **)(v73 + 48);
  v117 = (__int64)v165;
  v184 = (unsigned __int8 **)v116;
  v118 = (const char **)(v165 + 6);
  if ( !v116 )
  {
    if ( v118 == (const char **)&v184 )
      goto LABEL_182;
    v138 = v165[6];
    if ( !v138 )
      goto LABEL_182;
LABEL_211:
    v168 = v117;
    sub_B91220((__int64)v118, v138);
    v117 = v168;
    goto LABEL_212;
  }
  sub_B96E90((__int64)&v184, (__int64)v116, 1);
  v117 = (__int64)v165;
  if ( v118 == (const char **)&v184 )
  {
    if ( v184 )
    {
      sub_B91220((__int64)&v184, (__int64)v184);
      v117 = (__int64)v165;
    }
    goto LABEL_182;
  }
  v138 = v165[6];
  if ( v138 )
    goto LABEL_211;
LABEL_212:
  v139 = (unsigned __int8 *)v184;
  *(_QWORD *)(v117 + 48) = v184;
  if ( v139 )
  {
    v169 = v117;
    sub_B976B0((__int64)&v184, v139, (__int64)v118);
    v117 = v169;
  }
LABEL_182:
  v158 = v72;
  v119 = v73;
  v120 = 0;
  v121 = v117;
  do
  {
    v127 = *(_QWORD *)(*(_QWORD *)(v119 - 8) + 32LL * *(unsigned int *)(v119 + 72) + v120);
    v128 = *(_QWORD *)&v178[v120];
    v129 = *(_DWORD *)(v121 + 4) & 0x7FFFFFF;
    if ( v129 == *(_DWORD *)(v121 + 72) )
    {
      v166 = *(_QWORD *)(*(_QWORD *)(v119 - 8) + 32LL * *(unsigned int *)(v119 + 72) + v120);
      sub_B48D90(v121);
      v127 = v166;
      v129 = *(_DWORD *)(v121 + 4) & 0x7FFFFFF;
    }
    v122 = (v129 + 1) & 0x7FFFFFF;
    v123 = v122 | *(_DWORD *)(v121 + 4) & 0xF8000000;
    v124 = *(_QWORD *)(v121 - 8) + 32LL * (unsigned int)(v122 - 1);
    *(_DWORD *)(v121 + 4) = v123;
    if ( *(_QWORD *)v124 )
    {
      v125 = *(_QWORD *)(v124 + 8);
      **(_QWORD **)(v124 + 16) = v125;
      if ( v125 )
        *(_QWORD *)(v125 + 16) = *(_QWORD *)(v124 + 16);
    }
    *(_QWORD *)v124 = v128;
    if ( v128 )
    {
      v126 = *(_QWORD *)(v128 + 16);
      *(_QWORD *)(v124 + 8) = v126;
      if ( v126 )
        *(_QWORD *)(v126 + 16) = v124 + 8;
      *(_QWORD *)(v124 + 16) = v128 + 16;
      *(_QWORD *)(v128 + 16) = v124;
    }
    v120 += 8;
    *(_QWORD *)(*(_QWORD *)(v121 - 8)
              + 32LL * *(unsigned int *)(v121 + 72)
              + 8LL * ((*(_DWORD *)(v121 + 4) & 0x7FFFFFFu) - 1)) = v127;
  }
  while ( v120 != 8LL * v159 );
  v130 = v119;
  v131 = v121;
  if ( (_BYTE)v160 )
  {
    v132 = *(_QWORD *)(v119 + 16);
    if ( v132 )
    {
      v133 = v131;
      do
      {
        v134 = v132;
        v132 = *(_QWORD *)(v132 + 8);
        v135 = *(__int64 **)(v134 + 24);
        if ( (__int64 *)v158 != v135 )
        {
          sub_F162A0(v71, *(_QWORD *)(v134 + 24), v133);
          sub_F207A0(v71, v135);
        }
      }
      while ( v132 );
      v131 = v133;
      v130 = v119;
    }
LABEL_200:
    v167 = v131;
    sub_F55740(v130, v175, v130, *(_QWORD *)(v71 + 80));
    v131 = v167;
  }
  else if ( HIBYTE(v160) )
  {
    goto LABEL_200;
  }
  v33 = v158;
  result = sub_F162A0(v71, v158, v131);
  if ( (v189 & 1) == 0 )
  {
    v176 = result;
    v33 = 16LL * (unsigned int)v191;
    sub_C7D6A0((__int64)v190, v33, 8);
    result = v176;
  }
LABEL_98:
  if ( v181 != (unsigned int *)v183 )
  {
    v171 = result;
    _libc_free(v181, v33);
    result = v171;
  }
  if ( v178 != v180 )
  {
    v172 = result;
    _libc_free(v178, v33);
    return v172;
  }
  return result;
}
