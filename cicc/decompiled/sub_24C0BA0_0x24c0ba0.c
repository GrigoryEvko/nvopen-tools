// Function: sub_24C0BA0
// Address: 0x24c0ba0
//
__int64 __fastcall sub_24C0BA0(__int64 a1, char *a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  __int64 v8; // r14
  __int64 *v9; // r15
  __int64 v10; // r9
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r15
  char *v13; // r14
  char *v14; // rdi
  char v15; // dl
  _BYTE *v16; // rsi
  char v17; // cl
  __int64 v18; // rdx
  char v19; // al
  void *v20; // rax
  unsigned __int64 v21; // rcx
  _BYTE *v22; // rsi
  unsigned __int64 v23; // rax
  __int64 *v24; // rax
  char *v25; // rcx
  __int64 v26; // rbx
  char *v27; // r12
  char *v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // r9
  _QWORD *v31; // r14
  __int64 v32; // rax
  unsigned __int8 *v33; // r13
  unsigned __int8 v34; // bl
  unsigned __int8 *v35; // r12
  unsigned __int8 *v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 *v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 *v42; // rbx
  unsigned __int64 *v43; // r12
  unsigned __int64 v44; // rdi
  __int64 *v45; // r12
  char *v46; // rax
  size_t v47; // rdx
  size_t v48; // r13
  char *v49; // r12
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // rsi
  char *v56; // r12
  char *v57; // rax
  size_t v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  char *v62; // r12
  char *v63; // rax
  size_t v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  char v68; // al
  char v69; // al
  char v70; // al
  char v71; // al
  _BYTE *v72; // r12
  const char *v73; // rcx
  __int64 v74; // r15
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  char *v77; // rax
  size_t v78; // rdx
  _QWORD *v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // rax
  char *v82; // rax
  size_t v83; // rdx
  void *v84; // rdx
  size_t v85; // rax
  int v86; // r14d
  const char *v87; // r14
  __int64 v88; // rax
  int v89; // r15d
  _BYTE *v90; // r13
  __int64 v91; // rdx
  unsigned __int8 *v92; // rax
  __int64 v93; // r12
  const char *v94; // rax
  unsigned __int64 v95; // rdx
  const char *v96; // rax
  unsigned __int64 v97; // rdx
  __int64 v98; // rax
  __int64 *v99; // rsi
  __int64 v100; // rbx
  char *j; // rax
  __int64 v102; // rdx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rdx
  __int64 v106; // rcx
  unsigned __int64 v107; // rsi
  char *v108; // rax
  const char **v109; // rsi
  __m128i *v110; // rdi
  __m128i v111; // xmm0
  __int64 v112; // rdx
  size_t v113; // rax
  __int64 v114; // rcx
  unsigned __int64 *v115; // rbx
  __int64 v116; // rax
  unsigned __int64 v117; // rdx
  char v118; // r8
  unsigned __int64 v119; // rax
  __int64 *v120; // r13
  __int64 *v121; // rbx
  unsigned __int64 k; // rax
  __int64 v123; // rdi
  unsigned int v124; // ecx
  __int64 v125; // rsi
  __int64 *v126; // rbx
  __int64 *v127; // r12
  __int64 v128; // rsi
  __int64 v129; // rdi
  unsigned __int64 v130; // r12
  size_t v132; // rdx
  char v134; // [rsp+68h] [rbp-428h]
  __int64 v135; // [rsp+68h] [rbp-428h]
  __int64 v136; // [rsp+78h] [rbp-418h]
  __int64 *v137; // [rsp+78h] [rbp-418h]
  char *v138; // [rsp+80h] [rbp-410h]
  _QWORD *v139; // [rsp+88h] [rbp-408h]
  int v140; // [rsp+88h] [rbp-408h]
  _QWORD *v141; // [rsp+90h] [rbp-400h]
  int v142; // [rsp+90h] [rbp-400h]
  int v143; // [rsp+98h] [rbp-3F8h]
  int v144; // [rsp+98h] [rbp-3F8h]
  _BYTE *v145; // [rsp+98h] [rbp-3F8h]
  char *v146; // [rsp+A0h] [rbp-3F0h]
  __int64 v147; // [rsp+A8h] [rbp-3E8h]
  _QWORD *v148; // [rsp+B8h] [rbp-3D8h]
  _QWORD *v149; // [rsp+C8h] [rbp-3C8h]
  unsigned __int64 v150; // [rsp+D0h] [rbp-3C0h]
  __int64 *v151; // [rsp+E0h] [rbp-3B0h]
  __int64 *v152; // [rsp+E0h] [rbp-3B0h]
  __int64 *v153; // [rsp+E8h] [rbp-3A8h]
  unsigned __int64 v154; // [rsp+E8h] [rbp-3A8h]
  __int64 v155[4]; // [rsp+F0h] [rbp-3A0h] BYREF
  _QWORD v156[4]; // [rsp+110h] [rbp-380h] BYREF
  __int64 *v157; // [rsp+130h] [rbp-360h] BYREF
  __int64 v158; // [rsp+138h] [rbp-358h]
  __int64 v159; // [rsp+140h] [rbp-350h] BYREF
  _QWORD *v160; // [rsp+150h] [rbp-340h] BYREF
  __int64 v161; // [rsp+158h] [rbp-338h]
  _QWORD v162[2]; // [rsp+160h] [rbp-330h] BYREF
  __int64 *v163; // [rsp+170h] [rbp-320h] BYREF
  __int64 v164; // [rsp+178h] [rbp-318h]
  __int64 v165; // [rsp+180h] [rbp-310h] BYREF
  __int64 *v166; // [rsp+190h] [rbp-300h] BYREF
  __int64 v167; // [rsp+198h] [rbp-2F8h]
  _QWORD v168[2]; // [rsp+1A0h] [rbp-2F0h] BYREF
  __int64 v169; // [rsp+1B0h] [rbp-2E0h] BYREF
  __int64 v170; // [rsp+1B8h] [rbp-2D8h]
  __int64 v171; // [rsp+1C0h] [rbp-2D0h]
  __int64 v172; // [rsp+1C8h] [rbp-2C8h]
  __int64 *v173; // [rsp+1D0h] [rbp-2C0h]
  __int64 v174; // [rsp+1D8h] [rbp-2B8h]
  __int64 *v175; // [rsp+1E0h] [rbp-2B0h] BYREF
  __int64 v176; // [rsp+1E8h] [rbp-2A8h]
  _QWORD v177[6]; // [rsp+1F0h] [rbp-2A0h] BYREF
  const char *v178; // [rsp+220h] [rbp-270h] BYREF
  __int64 v179; // [rsp+228h] [rbp-268h]
  const char *v180; // [rsp+230h] [rbp-260h] BYREF
  __int64 v181; // [rsp+238h] [rbp-258h]
  _WORD v182[24]; // [rsp+240h] [rbp-250h] BYREF
  void *s2; // [rsp+270h] [rbp-220h] BYREF
  size_t n; // [rsp+278h] [rbp-218h]
  unsigned __int64 v185[2]; // [rsp+280h] [rbp-210h] BYREF
  _WORD v186[32]; // [rsp+290h] [rbp-200h] BYREF
  char *v187; // [rsp+2D0h] [rbp-1C0h] BYREF
  char *v188; // [rsp+2D8h] [rbp-1B8h]
  char *i; // [rsp+2E0h] [rbp-1B0h]
  _QWORD *v190; // [rsp+2E8h] [rbp-1A8h] BYREF
  _QWORD v191[4]; // [rsp+2F8h] [rbp-198h] BYREF
  __int64 v192; // [rsp+318h] [rbp-178h]
  __int64 v193[2]; // [rsp+320h] [rbp-170h] BYREF
  _QWORD v194[2]; // [rsp+330h] [rbp-160h] BYREF
  _BYTE *v195; // [rsp+340h] [rbp-150h]
  __int64 v196; // [rsp+348h] [rbp-148h]
  _BYTE v197[32]; // [rsp+350h] [rbp-140h] BYREF
  __int64 v198; // [rsp+370h] [rbp-120h]
  __int64 v199; // [rsp+378h] [rbp-118h]
  __int16 v200; // [rsp+380h] [rbp-110h]
  __int64 *v201; // [rsp+388h] [rbp-108h]
  void **v202; // [rsp+390h] [rbp-100h]
  void **v203; // [rsp+398h] [rbp-F8h]
  __int64 v204; // [rsp+3A0h] [rbp-F0h]
  int v205; // [rsp+3A8h] [rbp-E8h]
  __int16 v206; // [rsp+3ACh] [rbp-E4h]
  char v207; // [rsp+3AEh] [rbp-E2h]
  __int64 v208; // [rsp+3B0h] [rbp-E0h]
  __int64 v209; // [rsp+3B8h] [rbp-D8h]
  void *v210; // [rsp+3C0h] [rbp-D0h] BYREF
  void *v211; // [rsp+3C8h] [rbp-C8h] BYREF
  _QWORD v212[2]; // [rsp+3D0h] [rbp-C0h] BYREF
  __int64 *v213; // [rsp+3E0h] [rbp-B0h]
  __int64 v214; // [rsp+3E8h] [rbp-A8h]
  _BYTE v215[32]; // [rsp+3F0h] [rbp-A0h] BYREF
  __int64 *v216; // [rsp+410h] [rbp-80h]
  __int64 v217; // [rsp+418h] [rbp-78h]
  _QWORD v218[2]; // [rsp+420h] [rbp-70h] BYREF
  _QWORD v219[2]; // [rsp+430h] [rbp-60h] BYREF
  __int64 v220; // [rsp+440h] [rbp-50h]
  __int64 v221; // [rsp+448h] [rbp-48h]
  __int64 v222; // [rsp+450h] [rbp-40h]

  if ( *((_QWORD *)a2 + 2) )
  {
    sub_CA41E0(&v178);
    v5 = *((_QWORD *)a2 + 2);
    v6 = *((_QWORD *)a2 + 1);
    v187 = 0;
    v188 = 0;
    i = 0;
    v7 = 32 * v5;
    v153 = (__int64 *)v178;
    v8 = v6 + v7;
    if ( v7 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v9 = 0;
    if ( v7 )
    {
      v150 = v7;
      v187 = (char *)sub_22077B0(v7);
      v9 = (__int64 *)v187;
      for ( i = &v187[v150]; v8 != v6; v9 += 4 )
      {
        if ( v9 )
        {
          *v9 = (__int64)(v9 + 2);
          sub_24C0030(v9, *(_BYTE **)v6, *(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
        }
        v6 += 32;
      }
    }
    v188 = (char *)v9;
    sub_23CA6B0(&s2, &v187, v153);
    v11 = (unsigned __int64 *)v188;
    v12 = (unsigned __int64 *)v187;
    v13 = (char *)s2;
    if ( v188 != v187 )
    {
      do
      {
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          j_j___libc_free_0(*v12);
        v12 += 4;
      }
      while ( v11 != v12 );
      v12 = (unsigned __int64 *)v187;
    }
    if ( v12 )
      j_j___libc_free_0((unsigned __int64)v12);
    v14 = (char *)v178;
    if ( v178 && !_InterlockedSub((volatile signed __int32 *)v178 + 2, 1u) )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v14 + 8LL))(v14);
    if ( sub_23C76F0((__int64)v13, "metadata", 8u, "src", 3u, v10, *(char **)(a3 + 200), *(_QWORD *)(a3 + 208), 0, 0) )
    {
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 8) = a1 + 32;
      *(_QWORD *)(a1 + 56) = a1 + 80;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
      if ( v13 )
      {
        sub_23C6FB0((__int64)v13);
        j_j___libc_free_0((unsigned __int64)v13);
      }
      return a1;
    }
  }
  else
  {
    v13 = 0;
  }
  v15 = byte_4FECC88 | a2[1];
  v187 = (char *)a3;
  v16 = *(_BYTE **)(a3 + 232);
  v17 = *a2;
  BYTE1(v188) = v15;
  v18 = *(_QWORD *)(a3 + 240);
  v19 = a2[2];
  i = v13;
  BYTE2(v188) = qword_4FECBA8 | v19;
  LOBYTE(v188) = byte_4FECD68 | v17;
  v190 = v191;
  sub_24C0030((__int64 *)&v190, v16, (__int64)&v16[v18]);
  v191[2] = *(_QWORD *)(a3 + 264);
  v191[3] = *(_QWORD *)(a3 + 272);
  v192 = *(_QWORD *)(a3 + 280);
  v20 = (void *)sub_BAA610((__int64)v187);
  v21 = 2;
  s2 = v20;
  if ( BYTE4(v20) )
    v21 = (unsigned int)((_DWORD)v20 - 3) < 2 ? 65538LL : 2LL;
  v22 = (char *)v185 + 5;
  do
  {
    *--v22 = v21 % 0xA + 48;
    v23 = v21;
    v21 /= 0xAu;
  }
  while ( v23 > 9 );
  v193[0] = (__int64)v194;
  sub_24C0030(v193, v22, (__int64)v185 + 5);
  v24 = *(__int64 **)a3;
  v204 = 0;
  v201 = v24;
  v202 = &v210;
  v203 = &v211;
  v206 = 512;
  v200 = 0;
  v195 = v197;
  v196 = 0x200000000LL;
  v210 = &unk_49DA100;
  v205 = 0;
  v207 = 7;
  v211 = &unk_49DA0B0;
  v213 = (__int64 *)v215;
  v214 = 0x400000000LL;
  v216 = v218;
  v208 = 0;
  v209 = 0;
  v198 = 0;
  v199 = 0;
  v212[0] = 0;
  v212[1] = 0;
  v217 = 0;
  v218[0] = 0;
  v218[1] = 1;
  v219[0] = v212;
  v173 = (__int64 *)&v175;
  v219[1] = 0;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  v174 = 0;
  v25 = (char *)*((_QWORD *)v187 + 4);
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v146 = v25;
  v138 = v187 + 24;
  if ( v25 == v187 + 24 )
  {
LABEL_173:
    sub_C7D6A0(v170, 8LL * (unsigned int)v172, 8);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_174;
  }
  do
  {
    if ( !v146 )
LABEL_216:
      BUG();
    v136 = (__int64)(v146 - 56);
    v139 = v146 + 16;
    if ( v146 + 16 != (char *)(*((_QWORD *)v146 + 2) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v26 = (__int64)(v146 - 56);
      if ( !(unsigned __int8)sub_B2D610((__int64)(v146 - 56), 20) )
      {
        v134 = sub_B2D610(v26, 10);
        if ( !v134 )
        {
          v27 = i;
          if ( !i
            || (v28 = (char *)sub_BD5D20(v26), !sub_23C76F0(
                                                  (__int64)v27,
                                                  "metadata",
                                                  8u,
                                                  "fun",
                                                  3u,
                                                  v30,
                                                  v28,
                                                  v29,
                                                  0,
                                                  0)) )
          {
            if ( (*(v146 - 24) & 0xF) != 1 )
            {
              v163 = (__int64 *)sub_B2BE50(v136);
              if ( *(_WORD *)((char *)&v188 + 1) )
              {
                v147 = 0;
                v141 = (_QWORD *)*((_QWORD *)v146 + 3);
                if ( v139 != v141 )
                {
                  while ( 1 )
                  {
                    if ( !v141 )
                      BUG();
                    v31 = (_QWORD *)v141[4];
                    v148 = v141 + 3;
                    if ( v31 != v141 + 3 )
                      break;
LABEL_64:
                    v141 = (_QWORD *)v141[1];
                    if ( v139 == v141 )
                      goto LABEL_65;
                  }
                  while ( 1 )
                  {
                    v32 = 0;
                    if ( v31 )
                      v32 = (__int64)(v31 - 3);
                    v33 = (unsigned __int8 *)v32;
                    v166 = v168;
                    v167 = 0x100000000LL;
                    if ( !BYTE2(v188) || (v147 & 2) != 0 )
                      goto LABEL_42;
                    v34 = *(_BYTE *)v32;
                    if ( *(_BYTE *)v32 == 60 )
                    {
                      if ( !(unsigned __int8)sub_24C09A0(v32) )
                        goto LABEL_119;
                      goto LABEL_118;
                    }
                    if ( v34 == 85 )
                    {
                      if ( (*(_WORD *)(v32 + 2) & 3u) - 1 <= 1 )
                      {
                        if ( (unsigned __int8)sub_24C0860(v32) )
                        {
LABEL_119:
                          if ( !BYTE1(v188) )
                            goto LABEL_120;
LABEL_43:
                          v34 = *v33;
                          if ( (*v33 == 62 || v34 == 61) && (v35 = (unsigned __int8 *)*((_QWORD *)v33 - 4)) != 0 )
                          {
                            if ( ((unsigned __int8)sub_B46420((__int64)v33) || (unsigned __int8)sub_B46490((__int64)v33))
                              && (*sub_98ACB0(v35, 6u) != 60 || (unsigned __int8)sub_D13FA0((__int64)v35, 1, 0)) )
                            {
                              v36 = sub_BD4CB0(
                                      v35,
                                      (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_182,
                                      (__int64)&s2);
                              if ( *v36 != 3 || (v36[80] & 1) == 0 )
                              {
                                if ( !sub_B46500(v33) )
                                  goto LABEL_110;
                                v34 = *v33;
LABEL_107:
                                if ( v34 != 61 && v34 != 62 && v34 != 64 && v34 != 65 && v34 != 66 )
                                  goto LABEL_216;
                                if ( v33[72] )
                                  goto LABEL_154;
                                if ( v35 )
                                {
LABEL_110:
                                  v92 = sub_BD4CB0(
                                          v35,
                                          (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_182,
                                          (__int64)&s2);
                                  v93 = (__int64)v92;
                                  if ( *v92 == 3 )
                                  {
                                    if ( (v92[35] & 4) == 0 )
                                      goto LABEL_112;
                                    sub_ED12E0((__int64)&s2, 1, *((_DWORD *)v187 + 71), 0);
                                    v113 = 0;
                                    v114 = 0;
                                    if ( (*(_BYTE *)(v93 + 35) & 4) != 0 )
                                    {
                                      v114 = sub_B31D10(v93, 1, v112);
                                      v113 = v132;
                                    }
                                    v115 = (unsigned __int64 *)s2;
                                    if ( n <= v113 && (!n || !memcmp((const void *)(v114 + v113 - n), s2, n)) )
                                    {
                                      if ( v115 != v185 )
                                        j_j___libc_free_0((unsigned __int64)v115);
                                      goto LABEL_154;
                                    }
                                    if ( v115 != v185 )
                                      j_j___libc_free_0((unsigned __int64)v115);
LABEL_112:
                                    if ( (v94 = sub_BD5D20(v93), v95 > 0xA)
                                      && *(_QWORD *)v94 == 0x675F6D766C6C5F5FLL
                                      && *((_WORD *)v94 + 4) == 28515
                                      && v94[10] == 118
                                      || (v96 = sub_BD5D20(v93), v97 > 0xA)
                                      && *(_QWORD *)v96 == 0x675F6D766C6C5F5FLL
                                      && *((_WORD *)v96 + 4) == 25699
                                      && v96[10] == 97 )
                                    {
LABEL_154:
                                      v116 = (unsigned int)v167;
                                      v117 = (unsigned int)v167 + 1LL;
                                      if ( v117 > HIDWORD(v167) )
                                      {
                                        sub_C8D5F0((__int64)&v166, v168, v117, 8u, v37, v38);
                                        v116 = (unsigned int)v167;
                                      }
                                      v166[v116] = (__int64)&qword_4FECF80;
                                      v98 = (unsigned int)(v167 + 1);
                                      LODWORD(v167) = v167 + 1;
LABEL_157:
                                      v147 |= 1uLL;
                                      v45 = v166;
                                      v134 = 1;
LABEL_121:
                                      if ( (_DWORD)v98 )
                                      {
                                        v151 = &v45[v98];
                                        do
                                        {
                                          v99 = v45++;
                                          sub_24C05E0((__int64)&v169, v99);
                                        }
                                        while ( v151 != v45 );
                                        v100 = (__int64)v166;
                                        s2 = v185;
                                        n = 0x100000000LL;
                                        v152 = &v166[(unsigned int)v167];
                                        if ( v166 == v152 )
                                        {
                                          v39 = v185;
                                          v40 = 0;
                                        }
                                        else
                                        {
                                          for ( j = sub_24C03C0(
                                                      (__int64)&v187,
                                                      *(_QWORD *)(*v166 + 16),
                                                      *(_QWORD *)(*v166 + 24));
                                                ;
                                                j = sub_24C03C0(
                                                      (__int64)&v187,
                                                      *(_QWORD *)(*(_QWORD *)v100 + 16LL),
                                                      *(_QWORD *)(*(_QWORD *)v100 + 24LL)) )
                                          {
                                            v176 = 0x600000000LL;
                                            v178 = j;
                                            v175 = v177;
                                            v106 = (unsigned int)n;
                                            v179 = v102;
                                            v107 = (unsigned int)n + 1LL;
                                            v180 = (const char *)v182;
                                            v105 = (unsigned int)n;
                                            v181 = 0x600000000LL;
                                            if ( v107 > HIDWORD(n) )
                                            {
                                              if ( s2 > &v178
                                                || (v145 = s2,
                                                    &v178 >= (const char **)((char *)s2 + 80 * (unsigned int)n)) )
                                              {
                                                sub_24C0A80(
                                                  (__int64)&s2,
                                                  v107,
                                                  (__int64)s2,
                                                  (unsigned int)n,
                                                  v103,
                                                  v104);
                                                v106 = (unsigned int)n;
                                                v108 = (char *)s2;
                                                v109 = &v178;
                                                v105 = (unsigned int)n;
                                              }
                                              else
                                              {
                                                sub_24C0A80(
                                                  (__int64)&s2,
                                                  v107,
                                                  (__int64)s2,
                                                  (unsigned int)n,
                                                  v103,
                                                  v104);
                                                v108 = (char *)s2;
                                                v106 = (unsigned int)n;
                                                v109 = (const char **)((char *)s2 + (char *)&v178 - v145);
                                                v105 = (unsigned int)n;
                                              }
                                            }
                                            else
                                            {
                                              v108 = (char *)s2;
                                              v109 = &v178;
                                            }
                                            v110 = (__m128i *)&v108[80 * v106];
                                            if ( v110 )
                                            {
                                              v111 = _mm_loadu_si128((const __m128i *)v109);
                                              v110[1].m128i_i64[1] = 0x600000000LL;
                                              v110[1].m128i_i64[0] = (__int64)v110[2].m128i_i64;
                                              *v110 = v111;
                                              if ( *((_DWORD *)v109 + 6) )
                                                sub_24C00E0(
                                                  (__int64)v110[1].m128i_i64,
                                                  (char **)v109 + 2,
                                                  v105,
                                                  v106,
                                                  v103,
                                                  v104);
                                              LODWORD(v105) = n;
                                            }
                                            LODWORD(n) = v105 + 1;
                                            if ( v180 != (const char *)v182 )
                                              _libc_free((unsigned __int64)v180);
                                            v100 += 8;
                                            if ( v152 == (__int64 *)v100 )
                                              break;
                                          }
                                          v39 = (unsigned __int64 *)s2;
                                          v40 = (unsigned int)n;
                                        }
                                        v41 = sub_B8CB30(&v163, (__int64)v39, v40);
                                        sub_B99FD0((__int64)v33, 0x25u, v41);
                                        v42 = (unsigned __int64 *)s2;
                                        v43 = (unsigned __int64 *)((char *)s2 + 80 * (unsigned int)n);
                                        if ( s2 != v43 )
                                        {
                                          do
                                          {
                                            v43 -= 10;
                                            v44 = v43[2];
                                            if ( (unsigned __int64 *)v44 != v43 + 4 )
                                              _libc_free(v44);
                                          }
                                          while ( v42 != v43 );
                                          v43 = (unsigned __int64 *)s2;
                                        }
                                        if ( v43 != v185 )
                                          _libc_free((unsigned __int64)v43);
                                        v45 = v166;
                                      }
                                      if ( v45 != v168 )
                                        _libc_free((unsigned __int64)v45);
                                      goto LABEL_63;
                                    }
                                  }
                                }
LABEL_116:
                                v98 = (unsigned int)v167;
                                goto LABEL_157;
                              }
                            }
                          }
                          else
                          {
LABEL_104:
                            if ( (unsigned __int8)sub_B46420((__int64)v33) || (unsigned __int8)sub_B46490((__int64)v33) )
                            {
                              v35 = 0;
                              if ( sub_B46500(v33) )
                                goto LABEL_107;
                              goto LABEL_116;
                            }
                          }
LABEL_120:
                          v98 = (unsigned int)v167;
                          v45 = v166;
                          goto LABEL_121;
                        }
LABEL_118:
                        v147 |= 2uLL;
                        goto LABEL_119;
                      }
                      if ( BYTE1(v188) )
                        goto LABEL_104;
                      v31 = (_QWORD *)v31[1];
                      if ( v148 == v31 )
                        goto LABEL_64;
                    }
                    else
                    {
LABEL_42:
                      if ( BYTE1(v188) )
                        goto LABEL_43;
LABEL_63:
                      v31 = (_QWORD *)v31[1];
                      if ( v148 == v31 )
                        goto LABEL_64;
                    }
                  }
                }
LABEL_65:
                if ( !(_BYTE)qword_4FECE48 )
                {
LABEL_66:
                  if ( *(_DWORD *)(*((_QWORD *)v146 - 4) + 8LL) >> 8 )
                  {
                    v147 &= ~2uLL;
                  }
                  else if ( (v147 & 2) != 0 )
                  {
LABEL_68:
                    v175 = &qword_4FECFA0;
                    sub_24C05E0((__int64)&v169, (__int64 *)&v175);
                    v46 = sub_24C03C0((__int64)&v187, v175[2], v175[3]);
                    v48 = v47;
                    v49 = v46;
                    v50 = sub_BCB2E0(v201);
                    v51 = sub_ACD640(v50, v147, 0);
                    s2 = v49;
                    v179 = 0x600000001LL;
                    n = v48;
                    v180 = (const char *)v51;
                    v185[1] = 0x600000000LL;
                    v178 = (const char *)&v180;
                    v185[0] = (unsigned __int64)v186;
                    sub_24C0240((__int64)v185, (__int64)&v178, v52, 0x600000000LL, v51, v53);
                    v54 = sub_B8CB30(&v163, (__int64)&s2, 1);
                    sub_B99110(v136, 37, v54);
                    if ( (_WORD *)v185[0] != v186 )
                      _libc_free(v185[0]);
                    if ( v178 != (const char *)&v180 )
                      _libc_free((unsigned __int64)v178);
                    goto LABEL_72;
                  }
                  if ( !(_BYTE)v188 && (!v147 || !v134) )
                    goto LABEL_72;
                  goto LABEL_68;
                }
              }
              else
              {
                v134 = 0;
                v147 = 0;
                if ( !(_BYTE)qword_4FECE48 )
                  goto LABEL_66;
              }
              v118 = sub_B2D620(v136, "no_sanitize_thread", 0x12u);
              v119 = v147 & 0xFFFFFFFFFFFFFFFELL;
              if ( !v118 )
                v119 = v147;
              v147 = v119;
              goto LABEL_66;
            }
          }
        }
      }
    }
LABEL_72:
    v146 = (char *)*((_QWORD *)v146 + 1);
  }
  while ( v138 != v146 );
  if ( !(_DWORD)v174 )
  {
    if ( v173 != (__int64 *)&v175 )
      _libc_free((unsigned __int64)v173);
    goto LABEL_173;
  }
  v149 = (_QWORD *)sub_BCE3C0(v201, 0);
  v155[1] = (__int64)v149;
  v155[0] = sub_BCB2D0(v201);
  v155[2] = (__int64)v149;
  v55 = 2;
  s2 = (void *)sub_BAA610((__int64)v187);
  if ( BYTE4(s2) )
    v55 = (unsigned int)((_DWORD)s2 - 3) < 2 ? 65538LL : 2LL;
  v135 = sub_ACD640(v155[0], v55, 0);
  v137 = &v173[(unsigned int)v174];
  if ( v173 != v137 )
  {
    v154 = (unsigned __int64)v173;
    do
    {
      v74 = *(_QWORD *)v154;
      v75 = *(_QWORD *)(*(_QWORD *)v154 + 16LL);
      v156[0] = v135;
      v76 = *(_QWORD *)(v74 + 24);
      v182[0] = 1283;
      v186[0] = 1026;
      v178 = "__start_";
      s2 = &v178;
      v185[0] = (unsigned __int64)v193;
      v180 = (const char *)v75;
      v181 = v76;
      v77 = sub_C94C70((__int64)v219, (__int64)&s2);
      n = v78;
      v186[0] = 261;
      s2 = v77;
      v79 = sub_24C0320((__int64 *)&v187, (__int64)&s2, v149);
      v80 = *(_QWORD *)(v74 + 16);
      v156[1] = v79;
      v81 = *(_QWORD *)(v74 + 24);
      v182[0] = 1283;
      v178 = "__stop_";
      v186[0] = 1026;
      s2 = &v178;
      v185[0] = (unsigned __int64)v193;
      v180 = (const char *)v80;
      v181 = v81;
      v82 = sub_C94C70((__int64)v219, (__int64)&s2);
      n = v83;
      v186[0] = 261;
      s2 = v82;
      v156[2] = sub_24C0320((__int64 *)&v187, (__int64)&s2, v149);
      v84 = *(void **)v74;
      v85 = *(_QWORD *)(v74 + 8);
      v186[0] = 1029;
      s2 = v84;
      n = v85;
      v185[0] = (unsigned __int64)v193;
      sub_CA0F50((__int64 *)&v157, &s2);
      v86 = (unsigned __int8)qword_4FECF28;
      v186[0] = 773;
      s2 = *(void **)v74;
      n = *(_QWORD *)(v74 + 8);
      v185[0] = (unsigned __int64)"_add";
      sub_CA0F50((__int64 *)&v175, &s2);
      v143 = (int)v175;
      v142 = v176;
      v166 = v168;
      sub_24C0030((__int64 *)&v166, v157, (__int64)v157 + v158);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v167) <= 0xB )
        goto LABEL_215;
      sub_2241490((unsigned __int64 *)&v166, ".module_ctor", 0xCu);
      sub_2A41510(
        (unsigned int)&v178,
        (_DWORD)v187,
        (_DWORD)v166,
        v167,
        v143,
        v142,
        (__int64)v155,
        3,
        (__int64)v156,
        3,
        0,
        0,
        v86);
      v87 = v178;
      if ( v166 != v168 )
        j_j___libc_free_0((unsigned __int64)v166);
      if ( v175 != v177 )
        j_j___libc_free_0((unsigned __int64)v175);
      v144 = (unsigned __int8)qword_4FECF28;
      v182[0] = 773;
      v178 = *(const char **)v74;
      v88 = *(_QWORD *)(v74 + 8);
      v180 = "_del";
      v179 = v88;
      sub_CA0F50((__int64 *)&v163, (void **)&v178);
      v89 = v164;
      v140 = (int)v163;
      v160 = v162;
      sub_24C0030((__int64 *)&v160, v157, (__int64)v157 + v158);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v161) <= 0xB )
LABEL_215:
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v160, ".module_dtor", 0xCu);
      sub_2A41510(
        (unsigned int)&s2,
        (_DWORD)v187,
        (_DWORD)v160,
        v161,
        v140,
        v89,
        (__int64)v155,
        3,
        (__int64)v156,
        3,
        0,
        0,
        v144);
      v90 = s2;
      if ( v160 != v162 )
        j_j___libc_free_0((unsigned __int64)v160);
      if ( v163 != &v165 )
        j_j___libc_free_0((unsigned __int64)v163);
      if ( HIDWORD(v192) > 8 || (v91 = 292, v72 = 0, v73 = 0, !_bittest64(&v91, HIDWORD(v192))) )
      {
        v56 = v187;
        v57 = (char *)sub_BD5D20((__int64)v87);
        v59 = sub_BAA410((__int64)v56, v57, v58);
        sub_B2F990((__int64)v87, v59, v60, v61);
        v62 = v187;
        v63 = (char *)sub_BD5D20((__int64)v90);
        v65 = sub_BAA410((__int64)v62, v63, v64);
        sub_B2F990((__int64)v90, v65, v66, v67);
        v68 = v87[32];
        *((_BYTE *)v87 + 32) = v68 & 0xF0;
        if ( (v68 & 0x30) != 0 )
          *((_BYTE *)v87 + 33) |= 0x40u;
        v69 = v90[32];
        v90[32] = v69 & 0xF0;
        if ( (v69 & 0x30) != 0 )
          v90[33] |= 0x40u;
        v70 = v87[32] & 0xCF | 0x10;
        *((_BYTE *)v87 + 32) = v70;
        if ( (v70 & 0xF) != 9 )
          *((_BYTE *)v87 + 33) |= 0x40u;
        v71 = v90[32] & 0xCF | 0x10;
        v90[32] = v71;
        if ( (v71 & 0xF) != 9 )
          v90[33] |= 0x40u;
        v72 = v90;
        v73 = v87;
      }
      sub_2A3ED40(v187, v87, 2, v73);
      sub_2A3ED60(v187, v90, 2, v72);
      if ( v157 != &v159 )
        j_j___libc_free_0((unsigned __int64)v157);
      v154 += 8LL;
    }
    while ( v137 != (__int64 *)v154 );
    v137 = v173;
  }
  if ( v137 != (__int64 *)&v175 )
    _libc_free((unsigned __int64)v137);
  sub_C7D6A0(v170, 8LL * (unsigned int)v172, 8);
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
LABEL_174:
  sub_C7D6A0(v220, 16LL * (unsigned int)v222, 8);
  v120 = v213;
  v121 = &v213[(unsigned int)v214];
  if ( v213 != v121 )
  {
    for ( k = (unsigned __int64)v213; ; k = (unsigned __int64)v213 )
    {
      v123 = *v120;
      v124 = (unsigned int)((__int64)((__int64)v120 - k) >> 3) >> 7;
      v125 = 4096LL << v124;
      if ( v124 >= 0x1E )
        v125 = 0x40000000000LL;
      ++v120;
      sub_C7D6A0(v123, v125, 16);
      if ( v121 == v120 )
        break;
    }
  }
  v126 = v216;
  v127 = &v216[2 * (unsigned int)v217];
  if ( v216 != v127 )
  {
    do
    {
      v128 = v126[1];
      v129 = *v126;
      v126 += 2;
      sub_C7D6A0(v129, v128, 16);
    }
    while ( v127 != v126 );
    v127 = v216;
  }
  if ( v127 != v218 )
    _libc_free((unsigned __int64)v127);
  if ( v213 != (__int64 *)v215 )
    _libc_free((unsigned __int64)v213);
  nullsub_61();
  v210 = &unk_49DA100;
  nullsub_63();
  if ( v195 != v197 )
    _libc_free((unsigned __int64)v195);
  if ( (_QWORD *)v193[0] != v194 )
    j_j___libc_free_0(v193[0]);
  if ( v190 != v191 )
    j_j___libc_free_0((unsigned __int64)v190);
  v130 = (unsigned __int64)i;
  if ( i )
  {
    sub_23C6FB0((__int64)i);
    j_j___libc_free_0(v130);
  }
  return a1;
}
