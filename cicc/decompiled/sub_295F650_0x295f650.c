// Function: sub_295F650
// Address: 0x295f650
//
__int64 __fastcall sub_295F650(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 *a13,
        __int64 a14)
{
  __int64 v16; // rdx
  __int64 v17; // r15
  unsigned __int64 v18; // rdx
  __int64 *v19; // r12
  __int64 *v20; // rbx
  __int64 v21; // rdi
  int v22; // esi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r10
  __int64 v26; // rdx
  __int64 v27; // r8
  char v28; // cl
  unsigned int v29; // esi
  __int64 v30; // rsi
  __int64 v31; // rcx
  int v32; // edi
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rdx
  char v37; // si
  __int64 v38; // rax
  __int64 v39; // rdi
  const char *v40; // rax
  void *v41; // rdx
  char v42; // cl
  __int64 *v43; // rsi
  int v44; // eax
  int v45; // eax
  unsigned int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  int v50; // eax
  int v51; // eax
  unsigned int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // r13
  __int64 v57; // r12
  __int16 v58; // dx
  __int64 v59; // r14
  __int64 v60; // rax
  char v61; // dl
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // r10
  __int64 v65; // rbx
  _QWORD *v66; // r11
  const char *v67; // rsi
  void **v68; // r14
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  unsigned __int8 *v71; // r14
  _QWORD *v72; // r10
  _QWORD *v73; // rdx
  _QWORD *v74; // r13
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // r8
  __int64 *v81; // r9
  __int64 v82; // rdi
  __int64 v83; // rax
  __int64 v84; // rcx
  __int64 v85; // r14
  __int64 v86; // r15
  char v87; // si
  __int64 v88; // rdi
  int v89; // r8d
  unsigned int v90; // edx
  __int64 *v91; // rax
  __int64 v92; // r9
  __int64 v93; // rdx
  unsigned __int64 v94; // rax
  unsigned int v95; // r12d
  __int64 v96; // r14
  __int64 v97; // r15
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rbx
  __int64 v101; // r15
  __int64 v102; // rax
  unsigned int v103; // esi
  __int64 v104; // rax
  __int64 v105; // rbx
  unsigned __int64 v106; // rax
  __int64 v107; // r14
  unsigned int k; // r12d
  __int64 v109; // rax
  _QWORD *v110; // r12
  char *v111; // r14
  _QWORD *v112; // rax
  _QWORD *v113; // r15
  __int64 v114; // rsi
  void **v115; // r8
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // r12
  __int64 v119; // r13
  char v120; // r14
  int v121; // r15d
  __int64 v122; // rbx
  __int64 v123; // rax
  unsigned __int64 v124; // rax
  __int64 v125; // r12
  int v126; // eax
  char v127; // r15
  unsigned int v128; // r13d
  __int64 *v129; // rdx
  __int64 v130; // rcx
  __int64 v131; // r8
  __int64 v132; // r9
  __int64 v133; // r14
  _QWORD *v134; // rax
  unsigned int v135; // eax
  char v137; // dl
  __int64 v138; // rdx
  __int64 *v139; // rdx
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  int v144; // eax
  int v145; // eax
  int v146; // eax
  int v147; // r9d
  int v148; // ecx
  int v149; // r9d
  __int64 v151; // [rsp+10h] [rbp-180h]
  __int64 v152; // [rsp+20h] [rbp-170h]
  __int64 v153; // [rsp+28h] [rbp-168h]
  __int64 *v155; // [rsp+38h] [rbp-158h]
  unsigned __int64 v156; // [rsp+40h] [rbp-150h]
  __int64 v157; // [rsp+48h] [rbp-148h]
  unsigned __int8 *v159; // [rsp+58h] [rbp-138h]
  __int64 *i; // [rsp+60h] [rbp-130h]
  __int64 v161; // [rsp+68h] [rbp-128h]
  __int64 v162; // [rsp+70h] [rbp-120h]
  _QWORD *v163; // [rsp+70h] [rbp-120h]
  __int64 *v164; // [rsp+70h] [rbp-120h]
  __int64 *v165; // [rsp+70h] [rbp-120h]
  __int64 *v166; // [rsp+70h] [rbp-120h]
  __int64 *v167; // [rsp+78h] [rbp-118h]
  int v168; // [rsp+78h] [rbp-118h]
  __int64 v169; // [rsp+78h] [rbp-118h]
  __int64 v170; // [rsp+80h] [rbp-110h]
  __int64 v171; // [rsp+80h] [rbp-110h]
  __int64 v172; // [rsp+80h] [rbp-110h]
  __int64 v173; // [rsp+80h] [rbp-110h]
  const char *v174; // [rsp+80h] [rbp-110h]
  __int64 *v175; // [rsp+80h] [rbp-110h]
  __int64 v176; // [rsp+88h] [rbp-108h]
  __int64 j; // [rsp+88h] [rbp-108h]
  unsigned __int16 v178; // [rsp+88h] [rbp-108h]
  void **v179; // [rsp+88h] [rbp-108h]
  __int64 v180; // [rsp+90h] [rbp-100h]
  __int64 v181; // [rsp+90h] [rbp-100h]
  __int64 *v182; // [rsp+90h] [rbp-100h]
  int v183; // [rsp+90h] [rbp-100h]
  __int64 v184; // [rsp+90h] [rbp-100h]
  __int64 v185; // [rsp+90h] [rbp-100h]
  int v186; // [rsp+90h] [rbp-100h]
  __int64 v187; // [rsp+98h] [rbp-F8h] BYREF
  _QWORD v188[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v189[4]; // [rsp+C0h] [rbp-D0h] BYREF
  char v190; // [rsp+E0h] [rbp-B0h]
  char v191; // [rsp+E1h] [rbp-AFh]
  __int64 *v192; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v193; // [rsp+F8h] [rbp-98h]
  _BYTE v194[32]; // [rsp+100h] [rbp-90h] BYREF
  const char *v195; // [rsp+120h] [rbp-70h] BYREF
  void *s; // [rsp+128h] [rbp-68h]
  __int128 v197; // [rsp+130h] [rbp-60h]
  __int16 v198; // [rsp+140h] [rbp-50h] BYREF
  __int64 v199; // [rsp+1A0h] [rbp+10h]
  __int64 v200; // [rsp+1A8h] [rbp+18h]

  v16 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
  v192 = (__int64 *)v194;
  v17 = a8;
  v18 = a4 + (unsigned int)(v16 >> 3);
  v187 = a2;
  v193 = 0x400000000LL;
  if ( v18 > 4 )
    sub_C8D5F0((__int64)&v192, v194, v18, 8u, a5, a6);
  v188[2] = &v192;
  v188[1] = &v187;
  v188[0] = a8;
  v151 = sub_295F380((__int64)v188, v187);
  v19 = *(__int64 **)(a1 + 40);
  v20 = *(__int64 **)(a1 + 32);
  if ( v20 != v19 )
  {
    while ( 1 )
    {
      v27 = *v20;
      v28 = *(_BYTE *)(a7 + 8) & 1;
      if ( v28 )
      {
        v21 = a7 + 16;
        v22 = 15;
      }
      else
      {
        v29 = *(_DWORD *)(a7 + 24);
        v21 = *(_QWORD *)(a7 + 16);
        if ( !v29 )
          goto LABEL_197;
        v22 = v29 - 1;
      }
      v23 = v22 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( v27 != *v24 )
        break;
LABEL_7:
      v26 = 256;
      if ( !v28 )
        v26 = 16LL * *(unsigned int *)(a7 + 24);
      if ( v24 == (__int64 *)(v21 + v26) || a6 == v24[1] )
      {
        v30 = *v20++;
        sub_295F380((__int64)v188, v30);
        if ( v19 == v20 )
        {
LABEL_16:
          v17 = a8;
          goto LABEL_17;
        }
      }
      else if ( v19 == ++v20 )
      {
        goto LABEL_16;
      }
    }
    v145 = 1;
    while ( v25 != -4096 )
    {
      v147 = v145 + 1;
      v23 = v22 & (v145 + v23);
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( v27 == *v24 )
        goto LABEL_7;
      v145 = v147;
    }
    if ( v28 )
    {
      v143 = 256;
    }
    else
    {
      v29 = *(_DWORD *)(a7 + 24);
LABEL_197:
      v143 = 16LL * v29;
    }
    v24 = (__int64 *)(v21 + v143);
    goto LABEL_7;
  }
LABEL_17:
  v155 = &a3[a4];
  if ( a3 != v155 )
  {
    for ( i = a3; v155 != i; ++i )
    {
      v170 = *i;
      v37 = *(_BYTE *)(a7 + 8) & 1;
      if ( v37 )
      {
        v31 = a7 + 16;
        v32 = 15;
      }
      else
      {
        v31 = *(_QWORD *)(a7 + 16);
        v38 = *(unsigned int *)(a7 + 24);
        if ( !(_DWORD)v38 )
          goto LABEL_191;
        v32 = v38 - 1;
      }
      v33 = v32 & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
      v34 = (__int64 *)(v31 + 16LL * v33);
      v35 = *v34;
      if ( v170 == *v34 )
        goto LABEL_21;
      v144 = 1;
      while ( v35 != -4096 )
      {
        v149 = v144 + 1;
        v33 = v32 & (v144 + v33);
        v34 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v34;
        if ( v170 == *v34 )
          goto LABEL_21;
        v144 = v149;
      }
      if ( v37 )
      {
        v141 = 256;
        goto LABEL_192;
      }
      v38 = *(unsigned int *)(a7 + 24);
LABEL_191:
      v141 = 16 * v38;
LABEL_192:
      v34 = (__int64 *)(v31 + v141);
LABEL_21:
      v36 = 256;
      if ( !v37 )
        v36 = 16LL * *(unsigned int *)(a7 + 24);
      if ( v34 == (__int64 *)(v31 + v36) || a6 == v34[1] )
      {
        v39 = v153;
        v198 = 257;
        LOWORD(v39) = 1;
        v153 = v39;
        v159 = (unsigned __int8 *)sub_F36960(v170, *(__int64 **)(v170 + 56), v39, a11, a12, a13, (void **)&v195, 0);
        sub_BD6B90(v159, (unsigned __int8 *)v170);
        v191 = 1;
        v189[0] = (__int64)".split";
        v190 = 3;
        v40 = sub_BD5D20((__int64)v159);
        v42 = v190;
        if ( v190 )
        {
          if ( v190 == 1 )
          {
            v195 = v40;
            s = v41;
            v198 = 261;
          }
          else
          {
            if ( v191 == 1 )
            {
              v43 = (__int64 *)v189[0];
              v152 = v189[1];
            }
            else
            {
              v43 = v189;
              v42 = 2;
            }
            v195 = v40;
            s = v41;
            *(_QWORD *)&v197 = v43;
            *((_QWORD *)&v197 + 1) = v152;
            LOBYTE(v198) = 5;
            HIBYTE(v198) = v42;
          }
        }
        else
        {
          v198 = 256;
        }
        sub_BD6B50((unsigned __int8 *)v170, &v195);
        v157 = sub_295F380((__int64)v188, v170);
        v176 = *(_QWORD *)(v157 + 56);
        v156 = *(_QWORD *)(v170 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        v180 = *(_QWORD *)(v170 + 56);
        if ( v180 != v156 )
        {
          while ( 1 )
          {
            v56 = v176 - 24;
            v57 = v180 - 24;
            if ( !v176 )
              v56 = 0;
            if ( !v180 )
              v57 = 0;
            if ( a14 && *(_BYTE *)v57 == 84 )
              sub_DACA20(a14, a1, v57);
            v59 = sub_AA5190((__int64)v159);
            if ( v59 )
            {
              LOBYTE(v60) = v58;
              v61 = HIBYTE(v58);
            }
            else
            {
              v61 = 0;
              LOBYTE(v60) = 0;
            }
            v198 = 259;
            v60 = (unsigned __int8)v60;
            BYTE1(v60) = v61;
            v161 = v60;
            v195 = ".us-phi";
            v162 = *(_QWORD *)(v57 + 8);
            v62 = sub_BD2DA0(80);
            v63 = v162;
            v64 = v161;
            v65 = v62;
            if ( v62 )
            {
              v163 = (_QWORD *)v62;
              sub_B44260(v62, v63, 55, 0x8000000u, 0, 0);
              *(_DWORD *)(v65 + 72) = 2;
              sub_BD6B50((unsigned __int8 *)v65, &v195);
              sub_BD2A10(v65, *(_DWORD *)(v65 + 72), 1);
              v64 = v161;
              v66 = v163;
            }
            else
            {
              v66 = 0;
            }
            sub_B44220(v66, v59, v64);
            if ( !v59 )
              BUG();
            v67 = *(const char **)(v59 + 24);
            v68 = (void **)(v65 + 48);
            v195 = v67;
            if ( v67 )
              break;
            if ( v68 != (void **)&v195 )
            {
              v69 = *(_QWORD *)(v65 + 48);
              if ( v69 )
                goto LABEL_72;
            }
LABEL_40:
            sub_BD84D0(v57, v65);
            v44 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
            if ( v44 == *(_DWORD *)(v65 + 72) )
            {
              sub_B48D90(v65);
              v44 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
            }
            v45 = (v44 + 1) & 0x7FFFFFF;
            v46 = v45 | *(_DWORD *)(v65 + 4) & 0xF8000000;
            v47 = *(_QWORD *)(v65 - 8) + 32LL * (unsigned int)(v45 - 1);
            *(_DWORD *)(v65 + 4) = v46;
            if ( *(_QWORD *)v47 )
            {
              v48 = *(_QWORD *)(v47 + 8);
              **(_QWORD **)(v47 + 16) = v48;
              if ( v48 )
                *(_QWORD *)(v48 + 16) = *(_QWORD *)(v47 + 16);
            }
            *(_QWORD *)v47 = v57;
            v49 = *(_QWORD *)(v57 + 16);
            *(_QWORD *)(v47 + 8) = v49;
            if ( v49 )
              *(_QWORD *)(v49 + 16) = v47 + 8;
            *(_QWORD *)(v47 + 16) = v57 + 16;
            *(_QWORD *)(v57 + 16) = v47;
            *(_QWORD *)(*(_QWORD *)(v65 - 8)
                      + 32LL * *(unsigned int *)(v65 + 72)
                      + 8LL * ((*(_DWORD *)(v65 + 4) & 0x7FFFFFFu) - 1)) = v170;
            v50 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
            if ( v50 == *(_DWORD *)(v65 + 72) )
            {
              sub_B48D90(v65);
              v50 = *(_DWORD *)(v65 + 4) & 0x7FFFFFF;
            }
            v51 = (v50 + 1) & 0x7FFFFFF;
            v52 = v51 | *(_DWORD *)(v65 + 4) & 0xF8000000;
            v53 = *(_QWORD *)(v65 - 8) + 32LL * (unsigned int)(v51 - 1);
            *(_DWORD *)(v65 + 4) = v52;
            if ( *(_QWORD *)v53 )
            {
              v54 = *(_QWORD *)(v53 + 8);
              **(_QWORD **)(v53 + 16) = v54;
              if ( v54 )
                *(_QWORD *)(v54 + 16) = *(_QWORD *)(v53 + 16);
            }
            *(_QWORD *)v53 = v56;
            if ( v56 )
            {
              v55 = *(_QWORD *)(v56 + 16);
              *(_QWORD *)(v53 + 8) = v55;
              if ( v55 )
                *(_QWORD *)(v55 + 16) = v53 + 8;
              *(_QWORD *)(v53 + 16) = v56 + 16;
              *(_QWORD *)(v56 + 16) = v53;
            }
            *(_QWORD *)(*(_QWORD *)(v65 - 8)
                      + 32LL * *(unsigned int *)(v65 + 72)
                      + 8LL * ((*(_DWORD *)(v65 + 4) & 0x7FFFFFFu) - 1)) = v157;
            v180 = *(_QWORD *)(v180 + 8);
            v176 = *(_QWORD *)(v176 + 8);
            if ( v180 == v156 )
              goto LABEL_25;
          }
          sub_B96E90((__int64)&v195, (__int64)v67, 1);
          if ( v68 == (void **)&v195 )
          {
            if ( v195 )
              sub_B91220((__int64)&v195, (__int64)v195);
            goto LABEL_40;
          }
          v69 = *(_QWORD *)(v65 + 48);
          if ( v69 )
LABEL_72:
            sub_B91220(v65 + 48, v69);
          v70 = (unsigned __int8 *)v195;
          *(_QWORD *)(v65 + 48) = v195;
          if ( v70 )
            sub_B976B0((__int64)&v195, v70, v65 + 48);
          goto LABEL_40;
        }
      }
LABEL_25:
      ;
    }
  }
  v171 = *(_QWORD *)(*(_QWORD *)(v151 + 72) + 40LL);
  v164 = &v192[(unsigned int)v193];
  if ( v192 != v164 )
  {
    v167 = v192;
    do
    {
      v71 = *(unsigned __int8 **)(*v167 + 56);
      for ( j = *v167 + 48; (unsigned __int8 *)j != v71; v71 = (unsigned __int8 *)*((_QWORD *)v71 + 1) )
      {
        while ( 1 )
        {
          if ( !v71 )
            BUG();
          v82 = *((_QWORD *)v71 + 5);
          if ( v82 )
          {
            v72 = (_QWORD *)sub_B14240(v82);
            v74 = v73;
          }
          else
          {
            v74 = &qword_4F81430[1];
            v72 = &qword_4F81430[1];
          }
          v181 = (__int64)v72;
          sub_FC75A0((__int64 *)&v195, v17, 3, 0, 0, 0);
          sub_FCD310((__int64 *)&v195, v171, v181, (__int64)v74);
          sub_FC7680((__int64 *)&v195, v171);
          sub_FC75A0((__int64 *)&v195, v17, 3, 0, 0, 0);
          sub_FCD280((__int64 *)&v195, v71 - 24, v75, v76, v77, v78);
          sub_FC7680((__int64 *)&v195, (__int64)(v71 - 24));
          if ( *(v71 - 24) == 85 )
          {
            v83 = *((_QWORD *)v71 - 7);
            if ( v83 )
            {
              if ( !*(_BYTE *)v83 )
              {
                v84 = *((_QWORD *)v71 + 7);
                if ( *(_QWORD *)(v83 + 24) == v84 && (*(_BYTE *)(v83 + 33) & 0x20) != 0 && *(_DWORD *)(v83 + 36) == 11 )
                  break;
              }
            }
          }
          v71 = (unsigned __int8 *)*((_QWORD *)v71 + 1);
          if ( (unsigned __int8 *)j == v71 )
            goto LABEL_94;
        }
        sub_CFEAE0(a10, (__int64)(v71 - 24), v79, v84, v80, v81);
      }
LABEL_94:
      ++v167;
    }
    while ( v164 != v167 );
  }
  v165 = *(__int64 **)(a1 + 40);
  if ( *(__int64 **)(a1 + 32) == v165 )
    goto LABEL_124;
  v182 = *(__int64 **)(a1 + 32);
  v85 = a7;
  v200 = v17;
  do
  {
    v86 = *v182;
    v87 = *(_BYTE *)(v85 + 8) & 1;
    if ( v87 )
    {
      v88 = v85 + 16;
      v89 = 15;
    }
    else
    {
      v140 = *(unsigned int *)(v85 + 24);
      v88 = *(_QWORD *)(v85 + 16);
      if ( !(_DWORD)v140 )
        goto LABEL_194;
      v89 = v140 - 1;
    }
    v90 = v89 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
    v91 = (__int64 *)(v88 + 16LL * v90);
    v92 = *v91;
    if ( v86 == *v91 )
      goto LABEL_100;
    v146 = 1;
    while ( v92 != -4096 )
    {
      v148 = v146 + 1;
      v90 = v89 & (v146 + v90);
      v91 = (__int64 *)(v88 + 16LL * v90);
      v92 = *v91;
      if ( v86 == *v91 )
        goto LABEL_100;
      v146 = v148;
    }
    if ( v87 )
    {
      v142 = 256;
      goto LABEL_195;
    }
    v140 = *(unsigned int *)(v85 + 24);
LABEL_194:
    v142 = 16 * v140;
LABEL_195:
    v91 = (__int64 *)(v88 + v142);
LABEL_100:
    v93 = 256;
    if ( !v87 )
      v93 = 16LL * *(unsigned int *)(v85 + 24);
    if ( v91 != (__int64 *)(v88 + v93) && a6 != v91[1] )
    {
      v94 = sub_986580(*v182);
      v172 = v94;
      if ( v94 )
      {
        v168 = sub_B46E30(v94);
        if ( v168 )
        {
          v199 = v85;
          v95 = 0;
          v96 = v86;
          do
          {
            v189[0] = sub_B46EC0(v172, v95);
            sub_D696B0((unsigned __int64 *)&v195, v200, v189);
            v97 = v197;
            if ( (_QWORD)v197 )
            {
              if ( (_QWORD)v197 != -4096 && (_QWORD)v197 != -8192 )
                sub_BD60C0(&v195);
              v98 = sub_AA5930(v97);
              v100 = v99;
              v101 = v98;
              while ( v100 != v101 )
              {
                if ( (*(_DWORD *)(v101 + 4) & 0x7FFFFFF) != 0 )
                {
                  v102 = 0;
                  while ( 1 )
                  {
                    v103 = v102;
                    if ( v96 == *(_QWORD *)(*(_QWORD *)(v101 - 8) + 32LL * *(unsigned int *)(v101 + 72) + 8 * v102) )
                      break;
                    if ( (*(_DWORD *)(v101 + 4) & 0x7FFFFFF) == (_DWORD)++v102 )
                      goto LABEL_179;
                  }
                }
                else
                {
LABEL_179:
                  v103 = -1;
                }
                sub_B48BF0(v101, v103, 0);
                v104 = *(_QWORD *)(v101 + 32);
                if ( !v104 )
                  BUG();
                v101 = 0;
                if ( *(_BYTE *)(v104 - 24) == 84 )
                  v101 = v104 - 24;
              }
            }
            ++v95;
          }
          while ( v168 != v95 );
          v85 = v199;
        }
      }
    }
    ++v182;
  }
  while ( v165 != v182 );
  v17 = v200;
LABEL_124:
  v189[0] = a5;
  sub_D696B0((unsigned __int64 *)&v195, v17, v189);
  v105 = v197;
  sub_D68D70(&v195);
  v106 = sub_986580(a5);
  v107 = v106;
  if ( v106 )
  {
    v183 = sub_B46E30(v106);
    if ( v183 )
    {
      for ( k = 0; k != v183; ++k )
      {
        v109 = sub_B46EC0(v107, k);
        if ( v109 != a6 )
        {
          v189[0] = v109;
          sub_D696B0((unsigned __int64 *)&v195, v17, v189);
          if ( (_QWORD)v197 )
          {
            v173 = v197;
            sub_D68D70(&v195);
            sub_AA5980(v173, v105, 1u);
          }
          else
          {
            sub_D68D70(&v195);
          }
        }
      }
    }
  }
  v189[0] = a6;
  sub_D696B0((unsigned __int64 *)&v195, v17, v189);
  v184 = v197;
  sub_D68D70(&v195);
  v110 = (_QWORD *)sub_986580(v105);
  if ( *(_BYTE *)v110 == 31 )
  {
    v111 = (char *)*(v110 - 12);
  }
  else
  {
    v111 = 0;
    if ( *(_BYTE *)v110 == 32 )
      v111 = *(char **)*(v110 - 1);
  }
  sub_B43C20((__int64)&v195, v105);
  v174 = v195;
  v178 = (unsigned __int16)s;
  v112 = sub_BD2C40(72, 1u);
  v113 = v112;
  if ( v112 )
    sub_B4C8F0((__int64)v112, v184, 1u, (__int64)v174, v178);
  v114 = v110[6];
  v115 = (void **)(v113 + 6);
  v195 = (const char *)v114;
  if ( !v114 )
  {
    if ( v115 == (void **)&v195 )
      goto LABEL_139;
    v114 = v113[6];
    if ( !v114 )
      goto LABEL_139;
    goto LABEL_214;
  }
  sub_B96E90((__int64)&v195, v114, 1);
  v115 = (void **)(v113 + 6);
  if ( v113 + 6 == &v195 )
  {
    v114 = (__int64)v195;
    if ( v195 )
      sub_B91220((__int64)&v195, (__int64)v195);
    goto LABEL_139;
  }
  v114 = v113[6];
  if ( v114 )
  {
LABEL_214:
    v179 = v115;
    sub_B91220((__int64)v115, v114);
    v115 = v179;
  }
  v114 = (__int64)v195;
  v113[6] = v195;
  if ( v114 )
    sub_B976B0((__int64)&v195, (unsigned __int8 *)v114, (__int64)v115);
LABEL_139:
  sub_B43D60(v110);
  if ( v111 )
  {
    v114 = 0;
    *(_QWORD *)&v197 = 0;
    sub_F5CAB0(v111, 0, a13, (__int64)&v195);
    if ( (_QWORD)v197 )
    {
      v114 = (__int64)&v195;
      ((void (__fastcall *)(const char **, const char **, __int64))v197)(&v195, &v195, 3);
    }
  }
  v116 = sub_AA5930(v184);
  v185 = v117;
  if ( v116 != v117 )
  {
    v118 = v105;
    v119 = v116;
    while ( 1 )
    {
      v120 = 0;
      v121 = (*(_DWORD *)(v119 + 4) & 0x7FFFFFF) - 1;
      v122 = 8LL * v121;
      if ( (*(_DWORD *)(v119 + 4) & 0x7FFFFFF) != 0 )
        break;
LABEL_150:
      v123 = *(_QWORD *)(v119 + 32);
      if ( !v123 )
        BUG();
      v119 = 0;
      if ( *(_BYTE *)(v123 - 24) == 84 )
        v119 = v123 - 24;
      if ( v185 == v119 )
        goto LABEL_154;
    }
    while ( 1 )
    {
LABEL_147:
      if ( v118 != *(_QWORD *)(*(_QWORD *)(v119 - 8) + 32LL * *(unsigned int *)(v119 + 72) + v122) )
        goto LABEL_146;
      if ( !v120 )
        break;
      v114 = (unsigned int)v121--;
      sub_B48BF0(v119, v114, 0);
      v122 -= 8;
      if ( v121 == -1 )
        goto LABEL_150;
    }
    v120 = 1;
LABEL_146:
    --v121;
    v122 -= 8;
    if ( v121 == -1 )
      goto LABEL_150;
    goto LABEL_147;
  }
LABEL_154:
  v195 = 0;
  s = &v198;
  *(_QWORD *)&v197 = 4;
  DWORD2(v197) = 0;
  BYTE12(v197) = 1;
  v166 = &v192[(unsigned int)v193];
  if ( v192 == v166 )
    goto LABEL_175;
  v175 = v192;
  while ( 2 )
  {
    v169 = *v175;
    v124 = sub_986580(*v175);
    v125 = v124;
    if ( v124 )
    {
      v126 = sub_B46E30(v124);
      v127 = BYTE12(v197);
      v186 = v126;
      if ( v126 )
      {
        v128 = 0;
        while ( 1 )
        {
          v133 = sub_B46EC0(v125, v128);
          if ( v127 )
          {
            v134 = s;
            v114 = DWORD1(v197);
            v129 = (__int64 *)((char *)s + 8 * DWORD1(v197));
            if ( s != v129 )
            {
              while ( v133 != *v134 )
              {
                if ( v129 == ++v134 )
                  goto LABEL_185;
              }
              goto LABEL_164;
            }
LABEL_185:
            if ( DWORD1(v197) < (unsigned int)v197 )
            {
              v114 = (unsigned int)++DWORD1(v197);
              *v129 = v133;
              ++v195;
              goto LABEL_181;
            }
          }
          v114 = v133;
          sub_C8CC70((__int64)&v195, v133, (__int64)v129, v130, v131, v132);
          v127 = BYTE12(v197);
          if ( v137 )
          {
LABEL_181:
            v138 = *(unsigned int *)(a9 + 8);
            if ( v138 + 1 > (unsigned __int64)*(unsigned int *)(a9 + 12) )
            {
              v114 = a9 + 16;
              sub_C8D5F0(a9, (const void *)(a9 + 16), v138 + 1, 0x10u, v131, v138 + 1);
              v138 = *(unsigned int *)(a9 + 8);
            }
            v139 = (__int64 *)(*(_QWORD *)a9 + 16 * v138);
            ++v128;
            v139[1] = v133 & 0xFFFFFFFFFFFFFFFBLL;
            *v139 = v169;
            v127 = BYTE12(v197);
            ++*(_DWORD *)(a9 + 8);
            if ( v128 == v186 )
              break;
          }
          else
          {
LABEL_164:
            if ( ++v128 == v186 )
              break;
          }
        }
      }
    }
    else
    {
      v127 = BYTE12(v197);
    }
    ++v195;
    if ( v127 )
    {
LABEL_170:
      *(_QWORD *)((char *)&v197 + 4) = 0;
    }
    else
    {
      v135 = 4 * (DWORD1(v197) - DWORD2(v197));
      if ( v135 < 0x20 )
        v135 = 32;
      if ( (unsigned int)v197 <= v135 )
      {
        v114 = 0xFFFFFFFFLL;
        memset(s, -1, 8LL * (unsigned int)v197);
        goto LABEL_170;
      }
      sub_C8C990((__int64)&v195, v114);
    }
    if ( v166 != ++v175 )
      continue;
    break;
  }
  if ( !BYTE12(v197) )
    _libc_free((unsigned __int64)s);
  v166 = v192;
LABEL_175:
  if ( v166 != (__int64 *)v194 )
    _libc_free((unsigned __int64)v166);
  return v151;
}
