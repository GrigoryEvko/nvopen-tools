// Function: sub_12F5F30
// Address: 0x12f5f30
//
__int64 __fastcall sub_12F5F30(int a1, __int64 a2, _QWORD *a3, _BYTE *a4, __int64 a5, __int64 a6, _QWORD *a7)
{
  char **v7; // r12
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  bool v12; // cf
  bool v13; // zf
  int v14; // ebx
  __int64 v15; // rcx
  const char *v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rsi
  char v19; // al
  bool v20; // cf
  bool v21; // zf
  __int64 v22; // rcx
  const char *v23; // rdi
  char v24; // al
  bool v25; // cf
  bool v26; // zf
  __int64 v27; // rcx
  const char *v28; // rdi
  char v29; // al
  bool v30; // cf
  bool v31; // zf
  __int64 v32; // rcx
  const char *v33; // rdi
  char v34; // al
  bool v35; // cf
  bool v36; // zf
  __int64 v37; // rcx
  const char *v38; // rdi
  char v39; // al
  bool v40; // cf
  bool v41; // zf
  __int64 v42; // rcx
  const char *v43; // rdi
  char v44; // al
  bool v45; // cf
  bool v46; // zf
  __int64 v47; // rcx
  const char *v48; // rdi
  char v49; // al
  bool v50; // cf
  bool v51; // zf
  __int64 v52; // rcx
  const char *p_src; // rdi
  char v54; // al
  bool v55; // cf
  bool v56; // zf
  bool v57; // al
  bool v58; // cf
  bool v59; // zf
  char v60; // al
  bool v61; // cf
  bool v62; // zf
  char v63; // al
  bool v64; // cf
  bool v65; // zf
  char v66; // al
  bool v67; // cf
  bool v68; // zf
  char v69; // al
  bool v70; // cf
  bool v71; // zf
  _BYTE *v72; // rax
  __int64 v73; // rcx
  size_t v74; // r8
  char *v75; // rdx
  char v76; // al
  __int64 v77; // rax
  __m128i si128; // xmm0
  _QWORD *v79; // r14
  char v80; // r15
  char v81; // al
  _QWORD *v82; // r13
  char *v83; // rdx
  _BYTE *v84; // r9
  size_t v85; // r8
  char *v86; // rax
  _QWORD *v87; // rsi
  _BYTE *v88; // rax
  size_t v89; // r8
  char *v90; // rdx
  __int64 v91; // rax
  _BYTE *v92; // rax
  size_t v93; // r8
  char *v94; // rdx
  __int64 v95; // rax
  char *v96; // rax
  char *v97; // rdi
  _BYTE *v98; // rax
  size_t v99; // r8
  char *v100; // rdx
  __int64 v101; // rax
  char *v102; // rax
  char *v103; // rdi
  _BYTE *v104; // rax
  size_t v105; // r8
  char *v106; // rdx
  __int64 v107; // rax
  _BYTE *v109; // rax
  size_t v110; // r8
  char *v111; // rdx
  __int64 v112; // rax
  char *v113; // rax
  char *v114; // rdi
  _BYTE *v115; // rax
  size_t v116; // r8
  char *v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rax
  _BYTE *v120; // rdi
  _QWORD *v121; // rsi
  __int64 v122; // rdx
  __int64 v123; // rax
  __int64 v124; // rsi
  __int64 v125; // rsi
  __int64 v126; // rdx
  __int64 v127; // rsi
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  __int64 v132; // rax
  void *v133; // r14
  size_t v134; // r13
  char *v135; // rax
  __int64 v136; // rdx
  __int64 v137; // rcx
  __int64 v138; // r8
  __int64 v139; // rax
  char *v140; // r13
  size_t v141; // rdx
  char *v142; // rax
  char *v143; // rdi
  char *v144; // rax
  char *v145; // rdi
  __int64 v146; // r15
  unsigned int v147; // eax
  __int64 v148; // r13
  __int64 i; // r12
  _BYTE *v150; // rsi
  __int64 v151; // rax
  _BYTE *v152; // r14
  char v153; // al
  _QWORD *v154; // rdx
  __int64 v155; // rcx
  __int64 v156; // r8
  _BYTE *v157; // r10
  size_t v158; // r9
  _QWORD *v159; // rax
  _QWORD *v160; // rsi
  char *v161; // rax
  char *v162; // rdi
  __int64 v163; // rax
  __int64 v164; // r13
  int v165; // ebx
  __int64 v166; // rax
  __int64 v167; // r12
  __int64 v168; // rax
  _QWORD *j; // rbx
  char *v170; // rax
  char *v171; // rdi
  char *v172; // rdi
  __int64 v173; // rax
  int v174; // eax
  _QWORD *v175; // rdx
  __int64 v176; // rcx
  __int64 v177; // r8
  _BYTE *v178; // r10
  size_t v179; // r9
  _QWORD *v180; // rax
  _QWORD *v181; // r9
  char v182; // al
  size_t v183; // rax
  __int64 v184; // rax
  _QWORD *v185; // rdi
  __int64 v186; // rdx
  const char *v187; // rsi
  _QWORD *v188; // rdx
  __int64 v189; // rcx
  __int64 v190; // r8
  _BYTE *v191; // r10
  size_t v192; // r9
  _QWORD *v193; // rax
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // rax
  __int64 v197; // rax
  _QWORD *v198; // rdi
  __int64 v199; // rax
  _QWORD *v200; // rdi
  __int64 v201; // rax
  _QWORD *v202; // rdi
  __int64 v203; // [rsp-10h] [rbp-350h]
  char v205; // [rsp+10h] [rbp-330h]
  size_t v206; // [rsp+10h] [rbp-330h]
  size_t v207; // [rsp+10h] [rbp-330h]
  size_t v208; // [rsp+10h] [rbp-330h]
  char v209; // [rsp+18h] [rbp-328h]
  _BYTE *v210; // [rsp+18h] [rbp-328h]
  _BYTE *v211; // [rsp+18h] [rbp-328h]
  _BYTE *v212; // [rsp+18h] [rbp-328h]
  size_t n; // [rsp+20h] [rbp-320h]
  unsigned __int8 v217; // [rsp+54h] [rbp-2ECh]
  char v218; // [rsp+55h] [rbp-2EBh]
  char v219; // [rsp+56h] [rbp-2EAh]
  char v220; // [rsp+57h] [rbp-2E9h]
  __int128 v221; // [rsp+60h] [rbp-2E0h]
  _BYTE *v222; // [rsp+60h] [rbp-2E0h]
  _BYTE *v223; // [rsp+68h] [rbp-2D8h]
  size_t v224; // [rsp+70h] [rbp-2D0h]
  size_t v225; // [rsp+70h] [rbp-2D0h]
  size_t v226; // [rsp+78h] [rbp-2C8h]
  size_t v227; // [rsp+78h] [rbp-2C8h]
  size_t v228; // [rsp+80h] [rbp-2C0h]
  size_t v229; // [rsp+80h] [rbp-2C0h]
  size_t v230; // [rsp+88h] [rbp-2B8h]
  size_t v231; // [rsp+88h] [rbp-2B8h]
  __int64 v232; // [rsp+90h] [rbp-2B0h]
  _QWORD *v234; // [rsp+98h] [rbp-2A8h]
  _BYTE *v235; // [rsp+98h] [rbp-2A8h]
  __int64 v236; // [rsp+A8h] [rbp-298h] BYREF
  unsigned int v237; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v238; // [rsp+B8h] [rbp-288h]
  _BYTE v239[32]; // [rsp+C0h] [rbp-280h] BYREF
  void *src; // [rsp+E0h] [rbp-260h] BYREF
  size_t v241; // [rsp+E8h] [rbp-258h]
  _QWORD v242[2]; // [rsp+F0h] [rbp-250h] BYREF
  char *s; // [rsp+100h] [rbp-240h] BYREF
  size_t v244; // [rsp+108h] [rbp-238h]
  _QWORD v245[2]; // [rsp+110h] [rbp-230h] BYREF
  _QWORD v246[2]; // [rsp+120h] [rbp-220h] BYREF
  __int64 v247; // [rsp+130h] [rbp-210h] BYREF
  char *v248; // [rsp+140h] [rbp-200h] BYREF
  _BYTE *v249; // [rsp+148h] [rbp-1F8h]
  _QWORD v250[8]; // [rsp+150h] [rbp-1F0h] BYREF
  _QWORD v251[16]; // [rsp+190h] [rbp-1B0h] BYREF
  _DWORD v252[58]; // [rsp+210h] [rbp-130h] BYREF
  bool v253; // [rsp+2F8h] [rbp-48h]

  src = v242;
  v241 = 0;
  LOBYTE(v242[0]) = 0;
  sub_1611EE0(v239);
  memset(v251, 0, 0x78u);
  sub_1C13890(v252);
  v252[1] = 2;
  if ( a1 <= 0 )
  {
    v220 = 0;
    v217 = 0;
    v218 = 0;
    v232 = 0;
    v226 = 0;
    v228 = 0;
    v224 = 0;
    v221 = 0u;
    goto LABEL_194;
  }
  v221 = 0u;
  v12 = 0;
  v13 = 1;
  v14 = 0;
  v224 = 0;
  v228 = 0;
  v226 = 0;
  v232 = 0;
  v217 = 0;
  v218 = 0;
  v220 = 0;
  v219 = 0;
  v209 = 0;
  do
  {
    v15 = 14;
    v16 = "-arch=compute_";
    v17 = *(_QWORD *)(a2 + 8LL * v14);
    v10 = 8LL * v14;
    v18 = v17;
    do
    {
      if ( !v15 )
        break;
      v12 = *(_BYTE *)v18 < *v16;
      v13 = *(_BYTE *)v18++ == *v16++;
      --v15;
    }
    while ( v13 );
    v19 = (!v12 && !v13) - v12;
    v20 = 0;
    v21 = v19 == 0;
    if ( !v19 )
    {
      v7 = (char **)v250;
      v248 = (char *)v250;
      v72 = (_BYTE *)strlen((const char *)(v17 + 6));
      v246[0] = v72;
      v74 = (size_t)v72;
      if ( (unsigned __int64)v72 > 0xF )
      {
        n = (size_t)v72;
        v96 = (char *)sub_22409D0(&v248, v246, 0);
        v74 = n;
        v248 = v96;
        v97 = v96;
        v250[0] = v246[0];
      }
      else
      {
        if ( v72 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 6);
          v75 = (char *)v250;
          goto LABEL_65;
        }
        if ( !v72 )
        {
          v75 = (char *)v250;
          goto LABEL_65;
        }
        v97 = (char *)v250;
      }
      v18 = v17 + 6;
      memcpy(v97, (const void *)(v17 + 6), v74);
      v72 = (_BYTE *)v246[0];
      v75 = v248;
LABEL_65:
      v249 = v72;
      v72[(_QWORD)v75] = 0;
      v252[0] = 10 * sub_1CFBEC0(v248, v18, v75, v73, v74);
      v76 = sub_1CFBF00(v248);
      p_src = v248;
      v253 = v76 != 0;
      if ( v248 == (char *)v250 )
        goto LABEL_67;
LABEL_66:
      v18 = v250[0] + 1LL;
      j_j___libc_free_0(p_src, v250[0] + 1LL);
      goto LABEL_67;
    }
    v22 = 13;
    v18 = *(_QWORD *)(a2 + 8LL * v14);
    v23 = "-host-ref-ek=";
    do
    {
      if ( !v22 )
        break;
      v20 = *(_BYTE *)v18 < *v23;
      v21 = *(_BYTE *)v18++ == *v23++;
      --v22;
    }
    while ( v21 );
    v24 = (!v20 && !v21) - v20;
    v25 = 0;
    v26 = v24 == 0;
    if ( !v24 )
    {
      v7 = (char **)v250;
      v248 = (char *)v250;
      v88 = (_BYTE *)strlen((const char *)(v17 + 13));
      v246[0] = v88;
      v89 = (size_t)v88;
      if ( (unsigned __int64)v88 > 0xF )
      {
        v230 = (size_t)v88;
        v102 = (char *)sub_22409D0(&v248, v246, 0);
        v89 = v230;
        v248 = v102;
        v103 = v102;
        v250[0] = v246[0];
      }
      else
      {
        if ( v88 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
          v90 = (char *)v250;
          goto LABEL_93;
        }
        if ( !v88 )
        {
          v90 = (char *)v250;
          goto LABEL_93;
        }
        v103 = (char *)v250;
      }
      v18 = v17 + 13;
      memcpy(v103, (const void *)(v17 + 13), v89);
      v88 = (_BYTE *)v246[0];
      v90 = v248;
LABEL_93:
      v249 = v88;
      v88[(_QWORD)v90] = 0;
      v91 = sub_1682150(v248);
      p_src = v248;
      v232 = v91;
      if ( v248 != (char *)v250 )
        goto LABEL_66;
      goto LABEL_67;
    }
    v27 = 13;
    v18 = *(_QWORD *)(a2 + 8LL * v14);
    v28 = "-host-ref-ik=";
    do
    {
      if ( !v27 )
        break;
      v25 = *(_BYTE *)v18 < *v28;
      v26 = *(_BYTE *)v18++ == *v28++;
      --v27;
    }
    while ( v26 );
    v29 = (!v25 && !v26) - v25;
    v30 = 0;
    v31 = v29 == 0;
    if ( !v29 )
    {
      v7 = (char **)v250;
      v248 = (char *)v250;
      v92 = (_BYTE *)strlen((const char *)(v17 + 13));
      v246[0] = v92;
      v93 = (size_t)v92;
      if ( (unsigned __int64)v92 > 0xF )
      {
        v227 = (size_t)v92;
        v113 = (char *)sub_22409D0(&v248, v246, 0);
        v93 = v227;
        v248 = v113;
        v114 = v113;
        v250[0] = v246[0];
      }
      else
      {
        if ( v92 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
          v94 = (char *)v250;
          goto LABEL_98;
        }
        if ( !v92 )
        {
          v94 = (char *)v250;
          goto LABEL_98;
        }
        v114 = (char *)v250;
      }
      v18 = v17 + 13;
      memcpy(v114, (const void *)(v17 + 13), v93);
      v92 = (_BYTE *)v246[0];
      v94 = v248;
LABEL_98:
      v249 = v92;
      v92[(_QWORD)v94] = 0;
      v95 = sub_1682150(v248);
      p_src = v248;
      v226 = v95;
      if ( v248 != (char *)v250 )
        goto LABEL_66;
      goto LABEL_67;
    }
    v32 = 13;
    v18 = *(_QWORD *)(a2 + 8LL * v14);
    v33 = "-host-ref-ec=";
    do
    {
      if ( !v32 )
        break;
      v30 = *(_BYTE *)v18 < *v33;
      v31 = *(_BYTE *)v18++ == *v33++;
      --v32;
    }
    while ( v31 );
    v34 = (!v30 && !v31) - v30;
    v35 = 0;
    v36 = v34 == 0;
    if ( !v34 )
    {
      v7 = (char **)v250;
      v248 = (char *)v250;
      v98 = (_BYTE *)strlen((const char *)(v17 + 13));
      v246[0] = v98;
      v99 = (size_t)v98;
      if ( (unsigned __int64)v98 > 0xF )
      {
        v229 = (size_t)v98;
        v142 = (char *)sub_22409D0(&v248, v246, 0);
        v99 = v229;
        v248 = v142;
        v143 = v142;
        v250[0] = v246[0];
      }
      else
      {
        if ( v98 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
          v100 = (char *)v250;
          goto LABEL_107;
        }
        if ( !v98 )
        {
          v100 = (char *)v250;
          goto LABEL_107;
        }
        v143 = (char *)v250;
      }
      v18 = v17 + 13;
      memcpy(v143, (const void *)(v17 + 13), v99);
      v98 = (_BYTE *)v246[0];
      v100 = v248;
LABEL_107:
      v249 = v98;
      v98[(_QWORD)v100] = 0;
      v101 = sub_1682150(v248);
      p_src = v248;
      v228 = v101;
      if ( v248 != (char *)v250 )
        goto LABEL_66;
      goto LABEL_67;
    }
    v37 = 13;
    v18 = *(_QWORD *)(a2 + 8LL * v14);
    v38 = "-host-ref-ic=";
    do
    {
      if ( !v37 )
        break;
      v35 = *(_BYTE *)v18 < *v38;
      v36 = *(_BYTE *)v18++ == *v38++;
      --v37;
    }
    while ( v36 );
    v39 = (!v35 && !v36) - v35;
    v40 = 0;
    v41 = v39 == 0;
    if ( !v39 )
    {
      v7 = (char **)v250;
      v248 = (char *)v250;
      v104 = (_BYTE *)strlen((const char *)(v17 + 13));
      v246[0] = v104;
      v105 = (size_t)v104;
      if ( (unsigned __int64)v104 > 0xF )
      {
        v225 = (size_t)v104;
        v144 = (char *)sub_22409D0(&v248, v246, 0);
        v105 = v225;
        v248 = v144;
        v145 = v144;
        v250[0] = v246[0];
      }
      else
      {
        if ( v104 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
          v106 = (char *)v250;
          goto LABEL_116;
        }
        if ( !v104 )
        {
          v106 = (char *)v250;
          goto LABEL_116;
        }
        v145 = (char *)v250;
      }
      v18 = v17 + 13;
      memcpy(v145, (const void *)(v17 + 13), v105);
      v104 = (_BYTE *)v246[0];
      v106 = v248;
LABEL_116:
      v249 = v104;
      v104[(_QWORD)v106] = 0;
      v107 = sub_1682150(v248);
      p_src = v248;
      v224 = v107;
      if ( v248 != (char *)v250 )
        goto LABEL_66;
      goto LABEL_67;
    }
    v42 = 13;
    v43 = "-host-ref-eg=";
    v18 = *(_QWORD *)(a2 + 8LL * v14);
    do
    {
      if ( !v42 )
        break;
      v40 = *(_BYTE *)v18 < *v43;
      v41 = *(_BYTE *)v18++ == *v43++;
      --v42;
    }
    while ( v41 );
    v44 = (!v40 && !v41) - v40;
    v45 = 0;
    v46 = v44 == 0;
    if ( v44 )
    {
      v47 = 13;
      v48 = "-host-ref-ig=";
      v18 = *(_QWORD *)(a2 + 8LL * v14);
      do
      {
        if ( !v47 )
          break;
        v45 = *(_BYTE *)v18 < *v48;
        v46 = *(_BYTE *)v18++ == *v48++;
        --v47;
      }
      while ( v46 );
      v49 = (!v45 && !v46) - v45;
      v50 = 0;
      v51 = v49 == 0;
      if ( v49 )
      {
        v52 = 22;
        p_src = "-has-global-host-info";
        v18 = *(_QWORD *)(a2 + 8LL * v14);
        do
        {
          if ( !v52 )
            break;
          v50 = *(_BYTE *)v18 < *p_src;
          v51 = *(_BYTE *)v18++ == *p_src++;
          --v52;
        }
        while ( v51 );
        v54 = (!v50 && !v51) - v50;
        v55 = 0;
        v56 = v54 == 0;
        if ( v54 )
        {
          v52 = 27;
          p_src = "-optimize-unused-variables";
          v18 = *(_QWORD *)(a2 + 8LL * v14);
          do
          {
            if ( !v52 )
              break;
            v55 = *(_BYTE *)v18 < *p_src;
            v56 = *(_BYTE *)v18++ == *p_src++;
            --v52;
          }
          while ( v56 );
          if ( (!v55 && !v56) == v55 )
          {
            v220 = 1;
          }
          else
          {
            v57 = strcmp(*(const char **)(a2 + 8LL * v14), "-olto") != 0;
            v58 = 0;
            v59 = !v57;
            if ( v57 )
            {
              v52 = 11;
              p_src = "--device-c";
              v18 = *(_QWORD *)(a2 + 8LL * v14);
              do
              {
                if ( !v52 )
                  break;
                v58 = *(_BYTE *)v18 < *p_src;
                v59 = *(_BYTE *)v18++ == *p_src++;
                --v52;
              }
              while ( v59 );
              v60 = (!v58 && !v59) - v58;
              v61 = 0;
              v62 = v60 == 0;
              if ( v60 )
              {
                v52 = 17;
                p_src = "--force-device-c";
                v18 = *(_QWORD *)(a2 + 8LL * v14);
                do
                {
                  if ( !v52 )
                    break;
                  v61 = *(_BYTE *)v18 < *p_src;
                  v62 = *(_BYTE *)v18++ == *p_src++;
                  --v52;
                }
                while ( v62 );
                v63 = (!v61 && !v62) - v61;
                v64 = 0;
                v65 = v63 == 0;
                if ( v63 )
                {
                  v52 = 9;
                  p_src = "-gen-lto";
                  v18 = *(_QWORD *)(a2 + 8LL * v14);
                  do
                  {
                    if ( !v52 )
                      break;
                    v64 = *(_BYTE *)v18 < *p_src;
                    v65 = *(_BYTE *)v18++ == *p_src++;
                    --v52;
                  }
                  while ( v65 );
                  v66 = (!v64 && !v65) - v64;
                  v67 = 0;
                  v68 = v66 == 0;
                  if ( v66 )
                  {
                    v52 = 10;
                    p_src = "-link-lto";
                    v18 = *(_QWORD *)(a2 + 8LL * v14);
                    do
                    {
                      if ( !v52 )
                        break;
                      v67 = *(_BYTE *)v18 < *p_src;
                      v68 = *(_BYTE *)v18++ == *p_src++;
                      --v52;
                    }
                    while ( v68 );
                    v69 = (!v67 && !v68) - v67;
                    v70 = 0;
                    v71 = v69 == 0;
                    if ( v69 )
                    {
                      v52 = 8;
                      p_src = "--trace";
                      v18 = *(_QWORD *)(a2 + 8LL * v14);
                      v9 = v217;
                      do
                      {
                        if ( !v52 )
                          break;
                        v70 = *(_BYTE *)v18 < *p_src;
                        v71 = *(_BYTE *)v18++ == *p_src++;
                        --v52;
                      }
                      while ( v71 );
                      if ( (!v70 && !v71) == v70 )
                        v9 = 1;
                      v217 = v9;
                    }
                    else
                    {
                      v205 = 0;
                    }
                  }
                  else
                  {
                    v205 = 1;
                  }
                }
                else
                {
                  v219 = 1;
                }
              }
              else
              {
                v209 = 1;
              }
            }
            else
            {
              v7 = *(char ***)(a2 + v10 + 8);
              ++v14;
              v183 = strlen((const char *)v7);
              p_src = (const char *)&src;
              v18 = 0;
              sub_2241130(&src, 0, v241, v7, v183);
            }
          }
        }
        else
        {
          v218 = 1;
        }
        goto LABEL_67;
      }
      v7 = (char **)v250;
      v248 = (char *)v250;
      v109 = (_BYTE *)strlen((const char *)(v17 + 13));
      v246[0] = v109;
      v110 = (size_t)v109;
      if ( (unsigned __int64)v109 > 0xF )
      {
        v222 = v109;
        v161 = (char *)sub_22409D0(&v248, v246, 0);
        v110 = (size_t)v222;
        v248 = v161;
        v162 = v161;
        v250[0] = v246[0];
      }
      else
      {
        if ( v109 == (_BYTE *)1 )
        {
          LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
          v111 = (char *)v250;
          goto LABEL_143;
        }
        if ( !v109 )
        {
          v111 = (char *)v250;
LABEL_143:
          v249 = v109;
          v109[(_QWORD)v111] = 0;
          v112 = sub_1682150(v248);
          p_src = v248;
          *(_QWORD *)&v221 = v112;
          if ( v248 != (char *)v250 )
            goto LABEL_66;
          goto LABEL_67;
        }
        v162 = (char *)v250;
      }
      v18 = v17 + 13;
      memcpy(v162, (const void *)(v17 + 13), v110);
      v109 = (_BYTE *)v246[0];
      v111 = v248;
      goto LABEL_143;
    }
    v7 = (char **)v250;
    v248 = (char *)v250;
    v115 = (_BYTE *)strlen((const char *)(v17 + 13));
    v246[0] = v115;
    v116 = (size_t)v115;
    if ( (unsigned __int64)v115 > 0xF )
    {
      v223 = v115;
      v170 = (char *)sub_22409D0(&v248, v246, 0);
      v116 = (size_t)v223;
      v248 = v170;
      v171 = v170;
      v250[0] = v246[0];
    }
    else
    {
      if ( v115 == (_BYTE *)1 )
      {
        LOBYTE(v250[0]) = *(_BYTE *)(v17 + 13);
        v117 = (char *)v250;
        goto LABEL_150;
      }
      if ( !v115 )
      {
        v117 = (char *)v250;
        goto LABEL_150;
      }
      v171 = (char *)v250;
    }
    v18 = v17 + 13;
    memcpy(v171, (const void *)(v17 + 13), v116);
    v115 = (_BYTE *)v246[0];
    v117 = v248;
LABEL_150:
    v249 = v115;
    v115[(_QWORD)v117] = 0;
    v118 = sub_1682150(v248);
    p_src = v248;
    *((_QWORD *)&v221 + 1) = v118;
    if ( v248 != (char *)v250 )
      goto LABEL_66;
LABEL_67:
    v12 = a1 < (unsigned int)++v14;
    v13 = a1 == v14;
  }
  while ( a1 > v14 );
  if ( v220 )
  {
    if ( (unsigned __int64)v221 | *((_QWORD *)&v221 + 1) | v224 | v228 )
    {
      v7 = &v248;
      v246[0] = 113;
      v248 = (char *)v250;
      v77 = sub_22409D0(&v248, v246, 0);
      v248 = (char *)v77;
      v18 = 1;
      v250[0] = v246[0];
      *(__m128i *)v77 = _mm_load_si128((const __m128i *)&xmmword_4284F10);
      si128 = _mm_load_si128((const __m128i *)&xmmword_4284F20);
      *(_BYTE *)(v77 + 112) = 115;
      *(__m128i *)(v77 + 16) = si128;
      *(__m128i *)(v77 + 32) = _mm_load_si128((const __m128i *)&xmmword_4284F30);
      *(__m128i *)(v77 + 48) = _mm_load_si128((const __m128i *)&xmmword_4284F40);
      *(__m128i *)(v77 + 64) = _mm_load_si128((const __m128i *)&xmmword_4284F50);
      *(__m128i *)(v77 + 80) = _mm_load_si128((const __m128i *)&xmmword_4284F60);
      *(__m128i *)(v77 + 96) = _mm_load_si128((const __m128i *)&xmmword_4284F70);
      v249 = (_BYTE *)v246[0];
      v248[v246[0]] = 0;
      sub_1C3EFD0(&v248, 1);
      p_src = v248;
      if ( v248 != (char *)v250 )
      {
        v18 = v250[0] + 1LL;
        j_j___libc_free_0(v248, v250[0] + 1LL);
      }
      v220 = 0;
    }
    else
    {
      v228 = 0;
      v221 = 0u;
      v224 = 0;
    }
  }
  if ( !v205 )
  {
    if ( v209 )
    {
      *a4 = 1;
      v79 = (_QWORD *)a3[4];
      if ( v79 == a3 + 3 )
        goto LABEL_123;
      v80 = 0;
      while ( 2 )
      {
        v82 = v79 - 7;
        if ( !v79 )
          v82 = 0;
        if ( (unsigned __int8)sub_15E4F60(v82)
          && (*((_BYTE *)v82 + 33) & 0x20) == 0
          && (unsigned __int8)sub_1648D00(v82, 1) )
        {
          v84 = (_BYTE *)sub_1649960(v82);
          v85 = (size_t)v83;
          if ( v84 )
          {
            v248 = v83;
            v86 = v83;
            v251[0] = &v251[2];
            if ( (unsigned __int64)v83 > 0xF )
            {
              v231 = (size_t)v83;
              v235 = v84;
              v184 = sub_22409D0(v251, &v248, 0);
              v84 = v235;
              v85 = v231;
              v251[0] = v184;
              v185 = (_QWORD *)v184;
              v251[2] = v248;
            }
            else
            {
              if ( v83 == (char *)1 )
              {
                LOBYTE(v251[2]) = *v84;
                v83 = (char *)&v251[2];
                goto LABEL_89;
              }
              if ( !v83 )
              {
                v83 = (char *)&v251[2];
                goto LABEL_89;
              }
              v185 = &v251[2];
            }
            memcpy(v185, v84, v85);
            v86 = v248;
            v83 = (char *)v251[0];
LABEL_89:
            v251[1] = v86;
            v86[(_QWORD)v83] = 0;
            v87 = (_QWORD *)v251[0];
          }
          else
          {
            v251[1] = 0;
            v251[0] = &v251[2];
            v87 = &v251[2];
            LOBYTE(v251[2]) = 0;
          }
          if ( !(unsigned __int8)sub_1681F50(0, v87, v83, &v251[2], v85) )
          {
            if ( (_QWORD *)v251[0] != &v251[2] )
              j_j___libc_free_0(v251[0], v251[2] + 1LL);
            *a4 = 0;
            if ( v80 )
              goto LABEL_124;
LABEL_123:
            *a4 = 0;
LABEL_124:
            if ( v232 )
              sub_1688090(v232, sub_1683C50);
            if ( v226 )
              sub_1688090(v226, sub_1683C50);
            if ( v228 )
              sub_1688090(v228, sub_1683C50);
            if ( v224 )
              sub_1688090(v224, sub_1683C50);
            if ( *((_QWORD *)&v221 + 1) )
              sub_1688090(*((_QWORD *)&v221 + 1), sub_1683C50);
            if ( (_QWORD)v221 )
              sub_1688090(v221, sub_1683C50);
            LODWORD(v7) = 1;
            sub_1C3E9C0(a6);
            goto LABEL_137;
          }
          if ( (_QWORD *)v251[0] != &v251[2] )
            j_j___libc_free_0(v251[0], v251[2] + 1LL);
        }
        v81 = sub_1C2F070(v82);
        v79 = (_QWORD *)v79[1];
        if ( v81 )
          v80 = v81;
        if ( v79 == a3 + 3 )
        {
          if ( !v80 )
            goto LABEL_123;
          if ( *a4 == 1 && !v219 )
            goto LABEL_195;
          goto LABEL_124;
        }
        continue;
      }
    }
    if ( v219 )
      goto LABEL_123;
LABEL_194:
    *a4 = 1;
LABEL_195:
    v234 = (_QWORD *)sub_16321C0(a3, "llvm.used", 9, 0, v10);
    if ( (v226 | v232 | v228 | *((_QWORD *)&v221 + 1) | (unsigned __int64)v221 | v224 || v220) && v234 )
    {
      v248 = 0;
      v249 = 0;
      v250[0] = 0;
      v146 = *(v234 - 3);
      v147 = *(_DWORD *)(v146 + 20) & 0xFFFFFFF;
      if ( v147 )
      {
        v148 = v147 - 1;
        for ( i = 0; ; ++i )
        {
          v152 = (_BYTE *)sub_1649F00(*(_QWORD *)(v146 + 24 * (i - v147)));
          v153 = v152[16];
          if ( v153 )
          {
            if ( v153 != 3 )
              goto LABEL_201;
            v174 = *(_DWORD *)(*(_QWORD *)v152 + 8LL) >> 8;
            if ( v174 != 4 )
            {
              if ( v174 != 1 )
                goto LABEL_201;
              v178 = (_BYTE *)sub_1649960(v152);
              v179 = (size_t)v175;
              if ( !v178 )
              {
                LOBYTE(v251[2]) = 0;
                v251[0] = &v251[2];
                v251[1] = 0;
                goto LABEL_267;
              }
              v246[0] = v175;
              v180 = v175;
              v251[0] = &v251[2];
              if ( (unsigned __int64)v175 > 0xF )
              {
                v208 = (size_t)v175;
                v212 = v178;
                v201 = sub_22409D0(v251, v246, 0);
                v178 = v212;
                v179 = v208;
                v251[0] = v201;
                v202 = (_QWORD *)v201;
                v251[2] = v246[0];
              }
              else
              {
                if ( v175 == (_QWORD *)1 )
                {
                  LOBYTE(v251[2]) = *v178;
                  v175 = &v251[2];
                  goto LABEL_266;
                }
                if ( !v175 )
                {
                  v175 = &v251[2];
LABEL_266:
                  v251[1] = v180;
                  *((_BYTE *)v180 + (_QWORD)v175) = 0;
LABEL_267:
                  if ( !v220 )
                  {
                    v181 = (_QWORD *)v251[0];
                    if ( v221 == 0 && !v218 )
                      goto LABEL_272;
                    if ( (v152[32] & 0xFu) - 7 > 1 )
                    {
                      if ( (unsigned __int8)sub_16820A0(*((_QWORD *)&v221 + 1), v251[0], v175, v176, v177, v251[0]) )
                      {
                        v181 = (_QWORD *)v251[0];
LABEL_272:
                        if ( v181 != &v251[2] )
                          j_j___libc_free_0(v181, v251[2] + 1LL);
LABEL_201:
                        v150 = v249;
                        v151 = *(_QWORD *)(v146 + 24 * (i - (*(_DWORD *)(v146 + 20) & 0xFFFFFFF)));
                        v251[0] = v151;
                        if ( v249 == (_BYTE *)v250[0] )
                        {
                          sub_12F5DA0((__int64)&v248, v249, v251);
                        }
                        else
                        {
                          if ( v249 )
                          {
                            *(_QWORD *)v249 = v151;
                            v150 = v249;
                          }
                          v249 = v150 + 8;
                        }
                        goto LABEL_205;
                      }
                    }
                    else
                    {
                      v182 = sub_16820A0(v221, v251[0], v175, v176, v177, v251[0]);
                      v181 = (_QWORD *)v251[0];
                      if ( v182 )
                        goto LABEL_272;
                    }
LABEL_291:
                    if ( v217 )
                    {
                      v194 = sub_223E4D0(&unk_4FD4D00, "no reference to variable ");
                      v195 = sub_223E0D0(v194, v251[0], v251[1]);
                      sub_223E4D0(v195, "\n");
                    }
                    goto LABEL_281;
                  }
LABEL_296:
                  if ( !v217 )
                    goto LABEL_281;
                  v186 = 25;
                  v187 = "no reference to variable ";
LABEL_298:
                  sub_223E0D0(&unk_4FD4D00, v187, v186);
                  v196 = sub_223E0D0(&unk_4FD4D00, v251[0], v251[1]);
                  sub_223E0D0(v196, "\n", 1);
                  goto LABEL_281;
                }
                v202 = &v251[2];
              }
              memcpy(v202, v178, v179);
              v180 = (_QWORD *)v246[0];
              v175 = (_QWORD *)v251[0];
              goto LABEL_266;
            }
            v191 = (_BYTE *)sub_1649960(v152);
            v192 = (size_t)v188;
            if ( v191 )
            {
              v246[0] = v188;
              v193 = v188;
              v251[0] = &v251[2];
              if ( (unsigned __int64)v188 > 0xF )
              {
                v207 = (size_t)v188;
                v211 = v191;
                v199 = sub_22409D0(v251, v246, 0);
                v191 = v211;
                v192 = v207;
                v251[0] = v199;
                v200 = (_QWORD *)v199;
                v251[2] = v246[0];
              }
              else
              {
                if ( v188 == (_QWORD *)1 )
                {
                  LOBYTE(v251[2]) = *v191;
                  v188 = &v251[2];
                  goto LABEL_287;
                }
                if ( !v188 )
                {
                  v188 = &v251[2];
                  goto LABEL_287;
                }
                v200 = &v251[2];
              }
              memcpy(v200, v191, v192);
              v193 = (_QWORD *)v246[0];
              v188 = (_QWORD *)v251[0];
LABEL_287:
              v251[1] = v193;
              *((_BYTE *)v193 + (_QWORD)v188) = 0;
            }
            else
            {
              LOBYTE(v251[2]) = 0;
              v251[0] = &v251[2];
              v251[1] = 0;
            }
            if ( !v220 )
            {
              if ( (v152[32] & 0xFu) - 7 > 1 )
              {
                if ( (unsigned __int8)sub_16820A0(v228, v251[0], v188, v189, v190, v192) )
                {
LABEL_217:
                  if ( (_QWORD *)v251[0] != &v251[2] )
                    j_j___libc_free_0(v251[0], v251[2] + 1LL);
                  goto LABEL_201;
                }
              }
              else if ( (unsigned __int8)sub_16820A0(v224, v251[0], v188, v189, v190, v192) )
              {
                goto LABEL_217;
              }
              goto LABEL_291;
            }
            goto LABEL_296;
          }
          if ( !(unsigned __int8)sub_1C2F070(v152) || !(v226 | v232) )
            goto LABEL_201;
          v157 = (_BYTE *)sub_1649960(v152);
          v158 = (size_t)v154;
          if ( v157 )
            break;
          LOBYTE(v251[2]) = 0;
          v251[0] = &v251[2];
          v160 = &v251[2];
          v251[1] = 0;
LABEL_215:
          if ( (v152[32] & 0xF) == 7 )
          {
            if ( (unsigned __int8)sub_16820A0(v226, v160, v154, v155, v156, v158) )
              goto LABEL_217;
          }
          else if ( (unsigned __int8)sub_16820A0(v232, v160, v154, v155, v156, v158) )
          {
            goto LABEL_217;
          }
          v186 = 23;
          v187 = "no reference to kernel ";
          if ( v217 )
            goto LABEL_298;
LABEL_281:
          if ( (_QWORD *)v251[0] != &v251[2] )
            j_j___libc_free_0(v251[0], v251[2] + 1LL);
LABEL_205:
          if ( i == v148 )
            goto LABEL_221;
          v147 = *(_DWORD *)(v146 + 20) & 0xFFFFFFF;
        }
        v246[0] = v154;
        v159 = v154;
        v251[0] = &v251[2];
        if ( (unsigned __int64)v154 > 0xF )
        {
          v206 = (size_t)v154;
          v210 = v157;
          v197 = sub_22409D0(v251, v246, 0);
          v157 = v210;
          v158 = v206;
          v251[0] = v197;
          v198 = (_QWORD *)v197;
          v251[2] = v246[0];
        }
        else
        {
          if ( v154 == (_QWORD *)1 )
          {
            LOBYTE(v251[2]) = *v157;
            v154 = &v251[2];
LABEL_214:
            v251[1] = v159;
            *((_BYTE *)v159 + (_QWORD)v154) = 0;
            v160 = (_QWORD *)v251[0];
            goto LABEL_215;
          }
          if ( !v154 )
          {
            v154 = &v251[2];
            goto LABEL_214;
          }
          v198 = &v251[2];
        }
        memcpy(v198, v157, v158);
        v159 = (_QWORD *)v246[0];
        v154 = (_QWORD *)v251[0];
        goto LABEL_214;
      }
LABEL_221:
      sub_15E55B0(v234);
      v163 = sub_16471D0(*a3, 0);
      v164 = sub_1645D80(v163, (v249 - v248) >> 3);
      v165 = sub_159DFD0(v164, v248, (v249 - v248) >> 3);
      LOWORD(v251[2]) = 259;
      v251[0] = "llvm.used";
      v166 = sub_1648A60(88, 1);
      v167 = v166;
      if ( v166 )
        sub_15E51E0(v166, (_DWORD)a3, v164, 0, 6, v165, (__int64)v251, 0, 0, 0, 0);
      sub_15E5D20(v167, "llvm.metadata", 13);
      if ( v248 )
        j_j___libc_free_0(v248, v250[0] - (_QWORD)v248);
    }
    v168 = sub_1870320();
    sub_1619140(v239, v168, 1);
    sub_1619BD0(v239, a3);
    for ( j = (_QWORD *)a3[2]; a3 + 1 != j; j = (_QWORD *)j[1] )
    {
      if ( !j )
        BUG();
      if ( (*(_BYTE *)(j - 3) & 0xF) == 0
        && *(_DWORD *)(*(j - 7) + 8LL) >> 8 == 3
        && *(_BYTE *)(*(j - 4) + 8LL) == 14
        && (unsigned int)(1 << (*((_DWORD *)j - 6) >> 15)) <= 0x1F )
      {
        sub_15E4CC0(j - 7, 16);
      }
    }
    goto LABEL_124;
  }
  v119 = sub_1857160(p_src, v18, v9, v52, v10, v11);
  sub_1619140(v239, v119, 1);
  v251[2] = 0;
  v120 = v239;
  v121 = (_QWORD *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v251);
  sub_1619140(v239, v121, 1);
  v122 = v203;
  if ( v251[2] )
  {
    v121 = v251;
    v120 = v251;
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v251[2])(v251, v251, 3);
  }
  v123 = sub_1B26330(v120, v121, v122);
  sub_1619140(v239, v123, 1);
  v124 = sub_18DEFF0();
  sub_1619140(v239, v124, 1);
  v125 = sub_18F5480(v239, v124);
  sub_1619140(v239, v125, 1);
  v127 = sub_18B1DE0(v239, v125, v126);
  sub_1619140(v239, v127, 1);
  v132 = sub_1857160(v239, v127, v128, v129, v130, v131);
  sub_1619140(v239, v132, 1);
  sub_160FB70(v239, *a7, a7[1]);
  sub_1619BD0(v239, a3);
  v133 = src;
  v134 = v241;
  s = (char *)v245;
  LOBYTE(v7) = src == 0 && (char *)src + v241 != 0;
  if ( (_BYTE)v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v251[0] = v241;
  if ( v241 > 0xF )
  {
    s = (char *)sub_22409D0(&s, v251, 0);
    v172 = s;
    v245[0] = v251[0];
    goto LABEL_241;
  }
  if ( v241 != 1 )
  {
    if ( !v241 )
    {
      v135 = (char *)v245;
      goto LABEL_158;
    }
    v172 = (char *)v245;
LABEL_241:
    memcpy(v172, v133, v134);
    v134 = v251[0];
    v135 = s;
    goto LABEL_158;
  }
  LOBYTE(v245[0]) = *(_BYTE *)src;
  v135 = (char *)v245;
LABEL_158:
  v244 = v134;
  v135[v134] = 0;
  sub_1C13840(&v248, sub_12F5D90, sub_12F5D80, 0, 0);
  sub_1C17AF0(&v236, a3, v252, &v248, 1);
  if ( v244 )
  {
    v237 = 0;
    v139 = sub_2241E40(&v236, a3, v136, v137, v138);
    v140 = s;
    v141 = 0;
    v238 = v139;
    if ( s )
      v141 = strlen(s);
    sub_16E8AF0(v251, v140, v141, &v237, 0);
    if ( v237 )
    {
      sub_223E0D0(qword_4FD4BE0, "IO error: ", 10);
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*(_QWORD *)v238 + 32LL))(v246, v238, v237);
      v173 = sub_223E0D0(qword_4FD4BE0, v246[0], v246[1]);
      sub_223E0D0(v173, "\n", 1);
      if ( (__int64 *)v246[0] != &v247 )
        j_j___libc_free_0(v246[0], v247 + 1);
    }
    else
    {
      LODWORD(v7) = 1;
      sub_1C23B90(v236, v251, 1, 1);
      sub_16E8B10(v251);
    }
    sub_16E7C30(v251);
  }
  else
  {
    LODWORD(v251[4]) = 1;
    v251[0] = &unk_49EFBE0;
    v251[5] = a5;
    memset(&v251[1], 0, 24);
    sub_1C23B90(v236, v251, 1, 1);
    if ( v251[3] != v251[1] )
      sub_16E7BA0(v251);
    LODWORD(v7) = 1;
    sub_16E7BC0(v251);
  }
  if ( v236 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v236 + 56LL))(v236);
  if ( s != (char *)v245 )
    j_j___libc_free_0(s, v245[0] + 1LL);
  if ( (_BYTE)v7 )
    goto LABEL_124;
  if ( v232 )
    sub_1688090(v232, sub_1683C50);
  if ( v226 )
    sub_1688090(v226, sub_1683C50);
  if ( v228 )
    sub_1688090(v228, sub_1683C50);
  if ( v224 )
    sub_1688090(v224, sub_1683C50);
  if ( *((_QWORD *)&v221 + 1) )
    sub_1688090(*((_QWORD *)&v221 + 1), sub_1683C50);
  if ( (_QWORD)v221 )
    sub_1688090(v221, sub_1683C50);
LABEL_137:
  sub_160FE50(v239);
  if ( src != v242 )
    j_j___libc_free_0(src, v242[0] + 1LL);
  return (unsigned int)v7;
}
