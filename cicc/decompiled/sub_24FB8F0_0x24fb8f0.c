// Function: sub_24FB8F0
// Address: 0x24fb8f0
//
__int64 __fastcall sub_24FB8F0(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, unsigned __int8 *),
        __int64 a8)
{
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int8 *v10; // r12
  __int64 *v11; // rdi
  __int64 *v13; // rbx
  __int64 *v14; // r13
  _QWORD *v15; // r12
  const char *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned int v20; // ecx
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // r15
  unsigned __int8 *v25; // rbx
  int v26; // eax
  unsigned __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  char *v32; // rax
  __int64 v33; // rdx
  char *v34; // rsi
  __int64 v35; // rdx
  char *v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // rbx
  int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r8
  __int64 *v42; // rcx
  int v43; // r9d
  int n; // edx
  __int64 v45; // r10
  int v46; // edx
  __int64 *v47; // rbx
  __int64 v48; // r15
  unsigned __int8 *v49; // r13
  void *v50; // rsi
  __int64 v51; // r14
  __int64 v52; // rax
  char *v53; // r15
  __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // r13
  size_t v57; // rax
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // r15
  __m128i v61; // xmm2
  unsigned int v62; // r13d
  unsigned __int64 *v63; // r13
  __int64 v64; // r8
  unsigned __int64 *v65; // r15
  unsigned __int64 v66; // rdi
  unsigned __int64 *v67; // r13
  unsigned __int64 *v68; // r15
  unsigned __int64 v69; // rdi
  _QWORD *v70; // r14
  _QWORD *v71; // r13
  __int64 v72; // rax
  __int64 *v73; // r13
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned int v76; // r8d
  unsigned __int8 v77; // bl
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  _QWORD *v82; // r14
  _QWORD *v83; // r13
  __int64 v84; // rax
  __int64 *v85; // r13
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rsi
  __m128i *v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r12
  __m128i *v92; // r13
  __int64 v93; // rbx
  int v94; // edi
  unsigned int v95; // r13d
  __int64 *v96; // rcx
  __int64 v97; // r9
  __int64 *v98; // rax
  __int64 v99; // r8
  int v100; // edx
  __int64 v101; // rax
  unsigned __int64 v102; // rdx
  const char *v103; // rax
  __int64 v104; // rdx
  _QWORD *v105; // rsi
  unsigned __int64 *v106; // rcx
  _QWORD *v107; // rdi
  __int64 v108; // r8
  unsigned int v109; // eax
  int v110; // eax
  unsigned __int64 v111; // rax
  unsigned __int64 v112; // rax
  int v113; // ebx
  __int64 v114; // r13
  _QWORD *v115; // rax
  _QWORD *i; // rdx
  _QWORD *v117; // r8
  signed __int64 v118; // rdx
  __int64 *v119; // rcx
  int v120; // edi
  int v121; // r9d
  int m; // edx
  __int64 *v123; // rcx
  __int64 v124; // r10
  int v125; // edx
  int j; // edi
  __int64 *v127; // rcx
  __int64 *v128; // rcx
  int v129; // esi
  int k; // edx
  __int64 v131; // rdi
  int v132; // edx
  int v133; // edx
  int v134; // edx
  int v135; // edx
  __int64 *v136; // [rsp+0h] [rbp-830h]
  unsigned int v137; // [rsp+Ch] [rbp-824h]
  __int64 v138; // [rsp+10h] [rbp-820h]
  __int64 v140; // [rsp+30h] [rbp-800h]
  unsigned __int8 *v141; // [rsp+30h] [rbp-800h]
  __int64 *v143; // [rsp+48h] [rbp-7E8h]
  unsigned __int8 v147; // [rsp+78h] [rbp-7B8h]
  __int64 *v148; // [rsp+98h] [rbp-798h]
  _QWORD *v149; // [rsp+A0h] [rbp-790h]
  __int64 v150; // [rsp+A0h] [rbp-790h]
  __int64 v151; // [rsp+A8h] [rbp-788h]
  void *v152; // [rsp+B8h] [rbp-778h] BYREF
  __m128i v153; // [rsp+C0h] [rbp-770h] BYREF
  __int64 v154[2]; // [rsp+D0h] [rbp-760h] BYREF
  __int64 *v155; // [rsp+E0h] [rbp-750h]
  unsigned __int64 v156[2]; // [rsp+F0h] [rbp-740h] BYREF
  __int64 v157; // [rsp+100h] [rbp-730h] BYREF
  __int64 *v158; // [rsp+110h] [rbp-720h]
  __int64 v159; // [rsp+120h] [rbp-710h] BYREF
  unsigned __int64 v160[2]; // [rsp+140h] [rbp-6F0h] BYREF
  __int64 v161; // [rsp+150h] [rbp-6E0h] BYREF
  __int64 *v162; // [rsp+160h] [rbp-6D0h]
  __int64 v163; // [rsp+170h] [rbp-6C0h] BYREF
  unsigned __int64 v164[2]; // [rsp+190h] [rbp-6A0h] BYREF
  __int64 v165; // [rsp+1A0h] [rbp-690h] BYREF
  __int64 *v166; // [rsp+1B0h] [rbp-680h]
  __int64 v167; // [rsp+1C0h] [rbp-670h] BYREF
  __int64 *v168; // [rsp+1E0h] [rbp-650h] BYREF
  __int64 v169; // [rsp+1E8h] [rbp-648h]
  _BYTE v170[128]; // [rsp+1F0h] [rbp-640h] BYREF
  __int64 v171; // [rsp+270h] [rbp-5C0h] BYREF
  _QWORD *v172; // [rsp+278h] [rbp-5B8h]
  __int64 v173; // [rsp+280h] [rbp-5B0h]
  __int64 v174; // [rsp+288h] [rbp-5A8h]
  __int64 *v175; // [rsp+290h] [rbp-5A0h] BYREF
  __int64 v176; // [rsp+298h] [rbp-598h]
  _BYTE v177[128]; // [rsp+2A0h] [rbp-590h] BYREF
  _QWORD v178[5]; // [rsp+320h] [rbp-510h] BYREF
  _BYTE *v179; // [rsp+348h] [rbp-4E8h]
  __int64 v180; // [rsp+350h] [rbp-4E0h]
  _BYTE v181[32]; // [rsp+358h] [rbp-4D8h] BYREF
  _BYTE *v182; // [rsp+378h] [rbp-4B8h]
  __int64 v183; // [rsp+380h] [rbp-4B0h]
  _BYTE v184[192]; // [rsp+388h] [rbp-4A8h] BYREF
  _BYTE *v185; // [rsp+448h] [rbp-3E8h]
  __int64 v186; // [rsp+450h] [rbp-3E0h]
  _BYTE v187[72]; // [rsp+458h] [rbp-3D8h] BYREF
  void *v188; // [rsp+4A0h] [rbp-390h] BYREF
  int v189; // [rsp+4A8h] [rbp-388h]
  char v190; // [rsp+4ACh] [rbp-384h]
  __int64 v191; // [rsp+4B0h] [rbp-380h]
  __m128i v192; // [rsp+4B8h] [rbp-378h]
  __int64 v193; // [rsp+4C8h] [rbp-368h]
  __m128i v194; // [rsp+4D0h] [rbp-360h]
  __m128i v195; // [rsp+4E0h] [rbp-350h]
  unsigned __int64 *v196; // [rsp+4F0h] [rbp-340h] BYREF
  __int64 v197; // [rsp+4F8h] [rbp-338h]
  _BYTE v198[324]; // [rsp+500h] [rbp-330h] BYREF
  int v199; // [rsp+644h] [rbp-1ECh]
  __int64 v200; // [rsp+648h] [rbp-1E8h]
  _QWORD v201[12]; // [rsp+650h] [rbp-1E0h] BYREF
  char v202; // [rsp+6B0h] [rbp-180h] BYREF

  v8 = *(_QWORD *)(a1 + 32);
  v175 = (__int64 *)v177;
  v176 = 0x1000000000LL;
  v169 = 0x1000000000LL;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v168 = (__int64 *)v170;
  v151 = a1 + 24;
  v147 = 0;
  if ( v8 == a1 + 24 )
    goto LABEL_10;
  do
  {
    v9 = v8;
    v149 = (_QWORD *)v8;
    v8 = *(_QWORD *)(v8 + 8);
    v10 = (unsigned __int8 *)(v9 - 56);
    if ( (unsigned __int8)sub_B2D610(v9 - 56, 49) || sub_B2FC80((__int64)v10) || sub_30D6380(v10) )
      continue;
    ++v171;
    if ( !(_DWORD)v173 )
    {
      if ( !HIDWORD(v173) )
        goto LABEL_27;
      v21 = (unsigned int)v174;
      if ( (unsigned int)v174 > 0x40 )
      {
        sub_C7D6A0((__int64)v172, 8LL * (unsigned int)v174, 8);
        v172 = 0;
        v173 = 0;
        LODWORD(v174) = 0;
        goto LABEL_27;
      }
LABEL_24:
      v22 = v172;
      v23 = &v172[v21];
      if ( v172 != v23 )
      {
        do
          *v22++ = -4096;
        while ( v23 != v22 );
      }
      v173 = 0;
      goto LABEL_27;
    }
    v20 = 4 * v173;
    v21 = (unsigned int)v174;
    if ( (unsigned int)(4 * v173) < 0x40 )
      v20 = 64;
    if ( v20 >= (unsigned int)v174 )
      goto LABEL_24;
    v107 = v172;
    v108 = (unsigned int)v174;
    if ( (_DWORD)v173 == 1 )
    {
      v114 = 1024;
      v113 = 128;
LABEL_193:
      sub_C7D6A0((__int64)v172, v108 * 8, 8);
      LODWORD(v174) = v113;
      v115 = (_QWORD *)sub_C7D670(v114, 8);
      v173 = 0;
      v172 = v115;
      for ( i = &v115[(unsigned int)v174]; i != v115; ++v115 )
      {
        if ( v115 )
          *v115 = -4096;
      }
      goto LABEL_27;
    }
    _BitScanReverse(&v109, v173 - 1);
    v110 = 1 << (33 - (v109 ^ 0x1F));
    if ( v110 < 64 )
      v110 = 64;
    if ( v110 != (_DWORD)v174 )
    {
      v111 = (4 * v110 / 3u + 1) | ((unsigned __int64)(4 * v110 / 3u + 1) >> 1);
      v112 = ((v111 | (v111 >> 2)) >> 4)
           | v111
           | (v111 >> 2)
           | ((((v111 | (v111 >> 2)) >> 4) | v111 | (v111 >> 2)) >> 8);
      v113 = (v112 | (v112 >> 16)) + 1;
      v114 = 8 * ((v112 | (v112 >> 16)) + 1);
      goto LABEL_193;
    }
    v173 = 0;
    v117 = &v172[v108];
    do
    {
      if ( v107 )
        *v107 = -4096;
      ++v107;
    }
    while ( v117 != v107 );
LABEL_27:
    LODWORD(v176) = 0;
    v24 = *(v149 - 5);
    if ( v24 )
    {
      while ( 1 )
      {
        v25 = *(unsigned __int8 **)(v24 + 24);
        v26 = *v25;
        if ( (unsigned __int8)v26 <= 0x1Cu )
          goto LABEL_29;
        v27 = (unsigned int)(v26 - 34);
        if ( (unsigned __int8)v27 > 0x33u )
          goto LABEL_29;
        v28 = 0x8000000000041LL;
        if ( !_bittest64(&v28, v27) )
          goto LABEL_29;
        v29 = *((_QWORD *)v25 - 4);
        if ( !v29
          || *(_BYTE *)v29
          || *(_QWORD *)(v29 + 24) != *((_QWORD *)v25 + 10)
          || v10 != (unsigned __int8 *)v29
          || !(unsigned __int8)sub_A73ED0((_QWORD *)v25 + 9, 3) && !(unsigned __int8)sub_B49560((__int64)v25, 3) )
        {
          goto LABEL_29;
        }
        v201[0] = *((_QWORD *)v25 + 9);
        if ( (unsigned __int8)sub_A73ED0(v201, 31) )
          goto LABEL_29;
        if ( !(_DWORD)v173 )
          break;
        if ( !(_DWORD)v174 )
        {
          ++v171;
          goto LABEL_235;
        }
        v94 = 1;
        v95 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
        v96 = 0;
        v97 = ((_DWORD)v174 - 1) & v95;
        v98 = &v172[v97];
        v99 = *v98;
        if ( v25 != (unsigned __int8 *)*v98 )
        {
          while ( v99 != -4096 )
          {
            if ( v99 == -8192 && !v96 )
              v96 = v98;
            v97 = ((_DWORD)v174 - 1) & (unsigned int)(v94 + v97);
            v98 = &v172[(unsigned int)v97];
            v99 = *v98;
            if ( v25 == (unsigned __int8 *)*v98 )
              goto LABEL_29;
            ++v94;
          }
          if ( v96 )
            v98 = v96;
          v100 = v173 + 1;
          ++v171;
          if ( 4 * ((int)v173 + 1) < (unsigned int)(3 * v174) )
          {
            if ( (int)v174 - HIDWORD(v173) - v100 <= (unsigned int)v174 >> 3 )
            {
              sub_24FB720((__int64)&v171, v174);
              if ( !(_DWORD)v174 )
                goto LABEL_277;
              v97 = (__int64)v172;
              v98 = 0;
              v125 = (v174 - 1) & v95;
              for ( j = 1; ; ++j )
              {
                v127 = &v172[v125];
                v99 = *v127;
                if ( v25 == (unsigned __int8 *)*v127 )
                {
                  v100 = v173 + 1;
                  v98 = v127;
                  goto LABEL_180;
                }
                if ( v99 == -4096 )
                  break;
                if ( v99 != -8192 || v98 )
                  v127 = v98;
                v133 = j + v125;
                v98 = v127;
                v125 = (v174 - 1) & v133;
              }
              v100 = v173 + 1;
              if ( !v98 )
                v98 = v127;
            }
LABEL_180:
            LODWORD(v173) = v100;
            if ( *v98 != -4096 )
              --HIDWORD(v173);
            *v98 = (__int64)v25;
            v101 = (unsigned int)v176;
            v102 = (unsigned int)v176 + 1LL;
            if ( v102 > HIDWORD(v176) )
            {
              sub_C8D5F0((__int64)&v175, v177, v102, 8u, v99, v97);
              v101 = (unsigned int)v176;
            }
            v175[v101] = (__int64)v25;
            LODWORD(v176) = v176 + 1;
            goto LABEL_29;
          }
LABEL_235:
          sub_24FB720((__int64)&v171, 2 * v174);
          if ( !(_DWORD)v174 )
          {
LABEL_277:
            LODWORD(v173) = v173 + 1;
            BUG();
          }
          v99 = (unsigned int)(v174 - 1);
          v97 = (__int64)v172;
          v128 = 0;
          v129 = 1;
          for ( k = v99 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)); ; k = v99 & v134 )
          {
            v98 = &v172[k];
            v131 = *v98;
            if ( v25 == (unsigned __int8 *)*v98 )
            {
              v100 = v173 + 1;
              goto LABEL_180;
            }
            if ( v131 == -4096 )
              break;
            if ( v131 != -8192 || v128 )
              v98 = v128;
            v134 = v129 + k;
            v128 = v98;
            ++v129;
          }
          v100 = v173 + 1;
          if ( v128 )
            v98 = v128;
          goto LABEL_180;
        }
LABEL_29:
        v24 = *(_QWORD *)(v24 + 8);
        if ( !v24 )
        {
          v143 = &v175[(unsigned int)v176];
          if ( v143 != v175 )
          {
            v138 = v8;
            v47 = v175;
            do
            {
              v48 = *v47;
              v49 = (unsigned __int8 *)sub_B491C0(*v47);
              sub_1049690(v154, (__int64)v49);
              v50 = *(void **)(v48 + 48);
              v152 = v50;
              if ( v50 )
                sub_B96E90((__int64)&v152, (__int64)v50, 1);
              v51 = *(_QWORD *)(v48 + 40);
              v178[3] = 0;
              v178[4] = 0;
              v178[0] = a5;
              v185 = v187;
              v178[1] = a6;
              v187[64] = 1;
              v178[2] = a3;
              v179 = v181;
              v180 = 0x400000000LL;
              v182 = v184;
              v183 = 0x800000000LL;
              v186 = 0x800000000LL;
              v52 = a7(a8, v10);
              v53 = (char *)sub_29F2700(v48, v178, 1, v52, a2, 0);
              if ( v53 )
              {
                v140 = v154[0];
                v54 = sub_B2BE50(v154[0]);
                if ( sub_B6EA50(v54)
                  || (v86 = sub_B2BE50(v140),
                      v87 = sub_B6F970(v86),
                      (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v87 + 48LL))(v87)) )
                {
                  sub_B157E0((__int64)&v153, &v152);
                  sub_B17640((__int64)v201, (__int64)"inline", (__int64)"NotInlined", 10, &v153, v51);
                  sub_B18290((__int64)v201, "'", 1u);
                  sub_B16080((__int64)v164, "Callee", 6, v10);
                  v55 = sub_2445430((__int64)v201, (__int64)v164);
                  sub_B18290(v55, "' is not inlined into '", 0x17u);
                  sub_B16080((__int64)v160, "Caller", 6, v49);
                  v56 = sub_2445430(v55, (__int64)v160);
                  sub_B18290(v56, "': ", 3u);
                  v57 = strlen(v53);
                  sub_B16430((__int64)v156, "Reason", 6u, v53, v57);
                  v60 = sub_2445430(v56, (__int64)v156);
                  v189 = *(_DWORD *)(v60 + 8);
                  v190 = *(_BYTE *)(v60 + 12);
                  v191 = *(_QWORD *)(v60 + 16);
                  v192 = _mm_loadu_si128((const __m128i *)(v60 + 24));
                  v188 = &unk_49D9D40;
                  v193 = *(_QWORD *)(v60 + 40);
                  v194 = _mm_loadu_si128((const __m128i *)(v60 + 48));
                  v61 = _mm_loadu_si128((const __m128i *)(v60 + 64));
                  v196 = (unsigned __int64 *)v198;
                  v197 = 0x400000000LL;
                  v195 = v61;
                  v62 = *(_DWORD *)(v60 + 88);
                  if ( v62 && &v196 != (unsigned __int64 **)(v60 + 80) )
                  {
                    v88 = v62;
                    v89 = (__m128i *)v198;
                    if ( v62 > 4 )
                    {
                      sub_11F02D0((__int64)&v196, v62, v60 + 80, (__int64)v198, v58, v59);
                      v89 = (__m128i *)v196;
                      v88 = *(unsigned int *)(v60 + 88);
                    }
                    v90 = *(_QWORD *)(v60 + 80);
                    if ( v90 != v90 + 80 * v88 )
                    {
                      v141 = v10;
                      v91 = v90 + 80 * v88;
                      v137 = v62;
                      v92 = v89;
                      v136 = v47;
                      v93 = *(_QWORD *)(v60 + 80);
                      do
                      {
                        if ( v92 )
                        {
                          v92->m128i_i64[0] = (__int64)v92[1].m128i_i64;
                          sub_24FB300(v92->m128i_i64, *(_BYTE **)v93, *(_QWORD *)v93 + *(_QWORD *)(v93 + 8));
                          v92[2].m128i_i64[0] = (__int64)v92[3].m128i_i64;
                          sub_24FB300(
                            v92[2].m128i_i64,
                            *(_BYTE **)(v93 + 32),
                            *(_QWORD *)(v93 + 32) + *(_QWORD *)(v93 + 40));
                          v92[4] = _mm_loadu_si128((const __m128i *)(v93 + 64));
                        }
                        v93 += 80;
                        v92 += 5;
                      }
                      while ( v91 != v93 );
                      v10 = v141;
                      v62 = v137;
                      v47 = v136;
                    }
                    LODWORD(v197) = v62;
                  }
                  v198[320] = *(_BYTE *)(v60 + 416);
                  v199 = *(_DWORD *)(v60 + 420);
                  v200 = *(_QWORD *)(v60 + 424);
                  v188 = &unk_49D9DB0;
                  if ( v158 != &v159 )
                    j_j___libc_free_0((unsigned __int64)v158);
                  if ( (__int64 *)v156[0] != &v157 )
                    j_j___libc_free_0(v156[0]);
                  if ( v162 != &v163 )
                    j_j___libc_free_0((unsigned __int64)v162);
                  if ( (__int64 *)v160[0] != &v161 )
                    j_j___libc_free_0(v160[0]);
                  if ( v166 != &v167 )
                    j_j___libc_free_0((unsigned __int64)v166);
                  if ( (__int64 *)v164[0] != &v165 )
                    j_j___libc_free_0(v164[0]);
                  v63 = (unsigned __int64 *)v201[10];
                  v201[0] = &unk_49D9D40;
                  v64 = 80LL * LODWORD(v201[11]);
                  v65 = (unsigned __int64 *)(v201[10] + v64);
                  if ( v201[10] != v201[10] + v64 )
                  {
                    do
                    {
                      v65 -= 10;
                      v66 = v65[4];
                      if ( (unsigned __int64 *)v66 != v65 + 6 )
                        j_j___libc_free_0(v66);
                      if ( (unsigned __int64 *)*v65 != v65 + 2 )
                        j_j___libc_free_0(*v65);
                    }
                    while ( v63 != v65 );
                    v65 = (unsigned __int64 *)v201[10];
                  }
                  if ( v65 != (unsigned __int64 *)&v202 )
                    _libc_free((unsigned __int64)v65);
                  sub_1049740(v154, (__int64)&v188);
                  v67 = v196;
                  v188 = &unk_49D9D40;
                  v68 = &v196[10 * (unsigned int)v197];
                  if ( v196 != v68 )
                  {
                    do
                    {
                      v68 -= 10;
                      v69 = v68[4];
                      if ( (unsigned __int64 *)v69 != v68 + 6 )
                        j_j___libc_free_0(v69);
                      if ( (unsigned __int64 *)*v68 != v68 + 2 )
                        j_j___libc_free_0(*v68);
                    }
                    while ( v67 != v68 );
                    v68 = v196;
                  }
                  if ( v68 != (unsigned __int64 *)v198 )
                    _libc_free((unsigned __int64)v68);
                }
                if ( v185 != v187 )
                  _libc_free((unsigned __int64)v185);
                v70 = v182;
                v71 = &v182[24 * (unsigned int)v183];
                if ( v182 != (_BYTE *)v71 )
                {
                  do
                  {
                    v72 = *(v71 - 1);
                    v71 -= 3;
                    if ( v72 != 0 && v72 != -4096 && v72 != -8192 )
                      sub_BD60C0(v71);
                  }
                  while ( v70 != v71 );
                  v71 = v182;
                }
                if ( v71 != (_QWORD *)v184 )
                  _libc_free((unsigned __int64)v71);
                if ( v179 != v181 )
                  _libc_free((unsigned __int64)v179);
                if ( v152 )
                  sub_B91220((__int64)&v152, (__int64)v152);
                v73 = v155;
                if ( v155 )
                {
                  sub_FDC110(v155);
                  j_j___libc_free_0((unsigned __int64)v73);
                }
              }
              else
              {
                LODWORD(v201[1]) = 0;
                v201[0] = 0x80000000LL;
                v201[2] = "always inline attribute";
                LOBYTE(v201[7]) = 0;
                v188 = v152;
                if ( v152 )
                  sub_B96E90((__int64)&v188, (__int64)v152, 1);
                sub_30CD680(
                  (unsigned int)v154,
                  (unsigned int)&v188,
                  v51,
                  (_DWORD)v10,
                  (_DWORD)v49,
                  (unsigned int)v201,
                  0,
                  (__int64)"inline");
                if ( v188 )
                  sub_B91220((__int64)&v188, (__int64)v188);
                if ( LOBYTE(v201[7]) )
                {
                  LOBYTE(v201[7]) = 0;
                  if ( LODWORD(v201[6]) > 0x40 && v201[5] )
                    j_j___libc_free_0_0(v201[5]);
                  if ( LODWORD(v201[4]) > 0x40 && v201[3] )
                    j_j___libc_free_0_0(v201[3]);
                }
                if ( a4 )
                {
                  memset(v201, 0, sizeof(v201));
                  v201[1] = &v201[4];
                  LODWORD(v201[2]) = 2;
                  BYTE4(v201[3]) = 1;
                  v201[7] = &v201[10];
                  LODWORD(v201[8]) = 2;
                  BYTE4(v201[9]) = 1;
                  sub_BBE020(a4, (__int64)v49, (__int64)v201, 0);
                  if ( !BYTE4(v201[9]) )
                    _libc_free(v201[7]);
                  if ( !BYTE4(v201[3]) )
                    _libc_free(v201[1]);
                }
                if ( v185 != v187 )
                  _libc_free((unsigned __int64)v185);
                v82 = v182;
                v83 = &v182[24 * (unsigned int)v183];
                if ( v182 != (_BYTE *)v83 )
                {
                  do
                  {
                    v84 = *(v83 - 1);
                    v83 -= 3;
                    if ( v84 != -4096 && v84 != 0 && v84 != -8192 )
                      sub_BD60C0(v83);
                  }
                  while ( v82 != v83 );
                  v83 = v182;
                }
                if ( v83 != (_QWORD *)v184 )
                  _libc_free((unsigned __int64)v83);
                if ( v179 != v181 )
                  _libc_free((unsigned __int64)v179);
                if ( v152 )
                  sub_B91220((__int64)&v152, (__int64)v152);
                v85 = v155;
                if ( v155 )
                {
                  sub_FDC110(v155);
                  j_j___libc_free_0((unsigned __int64)v85);
                }
                v147 = 1;
              }
              ++v47;
            }
            while ( v143 != v47 );
            v8 = v138;
          }
          goto LABEL_122;
        }
      }
      v32 = (char *)v175;
      v33 = (unsigned int)v176;
      v34 = (char *)&v175[v33];
      v35 = (v33 * 8) >> 5;
      if ( v35 )
      {
        v36 = (char *)&v175[4 * v35];
        while ( v25 != *(unsigned __int8 **)v32 )
        {
          if ( v25 == *((unsigned __int8 **)v32 + 1) )
          {
            v32 += 8;
            break;
          }
          if ( v25 == *((unsigned __int8 **)v32 + 2) )
          {
            v32 += 16;
            break;
          }
          if ( v25 == *((unsigned __int8 **)v32 + 3) )
          {
            v32 += 24;
            break;
          }
          v32 += 32;
          if ( v36 == v32 )
            goto LABEL_205;
        }
LABEL_48:
        if ( v34 != v32 )
          goto LABEL_29;
        goto LABEL_49;
      }
LABEL_205:
      v118 = v34 - v32;
      if ( v34 - v32 != 16 )
      {
        if ( v118 != 24 )
        {
          if ( v118 != 8 )
            goto LABEL_49;
          goto LABEL_208;
        }
        if ( v25 == *(unsigned __int8 **)v32 )
          goto LABEL_48;
        v32 += 8;
      }
      if ( v25 == *(unsigned __int8 **)v32 )
        goto LABEL_48;
      v32 += 8;
LABEL_208:
      if ( v25 == *(unsigned __int8 **)v32 )
        goto LABEL_48;
LABEL_49:
      if ( (unsigned __int64)(unsigned int)v176 + 1 > HIDWORD(v176) )
      {
        sub_C8D5F0((__int64)&v175, v177, (unsigned int)v176 + 1LL, 8u, v30, v31);
        v34 = (char *)&v175[(unsigned int)v176];
      }
      *(_QWORD *)v34 = v25;
      v37 = (unsigned int)(v176 + 1);
      LODWORD(v176) = v37;
      if ( (unsigned int)v37 > 0x10 )
      {
        v38 = v175;
        v148 = &v175[v37];
        while ( (_DWORD)v174 )
        {
          v39 = (v174 - 1) & (((unsigned int)*v38 >> 9) ^ ((unsigned int)*v38 >> 4));
          v40 = &v172[v39];
          v41 = *v40;
          if ( *v40 != *v38 )
          {
            v119 = 0;
            v120 = 1;
            while ( v41 != -4096 )
            {
              if ( v41 == -8192 && !v119 )
                v119 = v40;
              v39 = (v174 - 1) & (v120 + v39);
              v40 = &v172[v39];
              v41 = *v40;
              if ( *v38 == *v40 )
                goto LABEL_54;
              ++v120;
            }
            if ( v119 )
              v40 = v119;
            ++v171;
            v46 = v173 + 1;
            if ( 4 * ((int)v173 + 1) < (unsigned int)(3 * v174) )
            {
              if ( (int)v174 - HIDWORD(v173) - v46 <= (unsigned int)v174 >> 3 )
              {
                sub_24FB720((__int64)&v171, v174);
                if ( !(_DWORD)v174 )
                {
LABEL_278:
                  LODWORD(v173) = v173 + 1;
                  BUG();
                }
                v121 = 1;
                v40 = 0;
                for ( m = (v174 - 1) & (((unsigned int)*v38 >> 9) ^ ((unsigned int)*v38 >> 4)); ; m = (v174 - 1) & v132 )
                {
                  v123 = &v172[m];
                  v124 = *v123;
                  if ( *v38 == *v123 )
                  {
                    v46 = v173 + 1;
                    v40 = v123;
                    goto LABEL_61;
                  }
                  if ( v124 == -4096 )
                    break;
                  if ( v40 || v124 != -8192 )
                    v123 = v40;
                  v132 = v121 + m;
                  v40 = v123;
                  ++v121;
                }
                v46 = v173 + 1;
                if ( !v40 )
                  v40 = v123;
              }
              goto LABEL_61;
            }
LABEL_57:
            sub_24FB720((__int64)&v171, 2 * v174);
            if ( !(_DWORD)v174 )
              goto LABEL_278;
            v42 = 0;
            v43 = 1;
            for ( n = (v174 - 1) & (((unsigned int)*v38 >> 9) ^ ((unsigned int)*v38 >> 4)); ; n = (v174 - 1) & v135 )
            {
              v40 = &v172[n];
              v45 = *v40;
              if ( *v38 == *v40 )
              {
                v46 = v173 + 1;
                goto LABEL_61;
              }
              if ( v45 == -4096 )
                break;
              if ( v45 != -8192 || v42 )
                v40 = v42;
              v135 = v43 + n;
              v42 = v40;
              ++v43;
            }
            v46 = v173 + 1;
            if ( v42 )
              v40 = v42;
LABEL_61:
            LODWORD(v173) = v46;
            if ( *v40 != -4096 )
              --HIDWORD(v173);
            *v40 = *v38;
          }
LABEL_54:
          if ( v148 == ++v38 )
            goto LABEL_29;
        }
        ++v171;
        goto LABEL_57;
      }
      goto LABEL_29;
    }
LABEL_122:
    sub_AD0030((__int64)v10);
    if ( (unsigned __int8)sub_B2D610((__int64)v10, 3) )
    {
      v77 = sub_B2E360((__int64)v10, 3, v74, v75, v76);
      if ( v77 )
      {
        if ( *(v149 - 1) )
        {
          v80 = (unsigned int)v169;
          v81 = (unsigned int)v169 + 1LL;
          if ( v81 > HIDWORD(v169) )
          {
            sub_C8D5F0((__int64)&v168, v170, v81, 8u, v78, v79);
            v80 = (unsigned int)v169;
          }
          v168[v80] = (__int64)v10;
          LODWORD(v169) = v169 + 1;
        }
        else
        {
          if ( a4 )
          {
            v103 = sub_BD5D20((__int64)v10);
            sub_BBB260(a4, (__int64)v10, (__int64)v103, v104);
          }
          sub_BA8570(v151, (__int64)v10);
          v105 = v149;
          v150 = *v149;
          v106 = (unsigned __int64 *)v105[1];
          *v106 = v150 & 0xFFFFFFFFFFFFFFF8LL | *v106 & 7;
          *(_QWORD *)((v150 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v106;
          *v105 &= 7uLL;
          v105[1] = 0;
          sub_B2E780(v10);
          sub_BD2DD0((__int64)v10);
          v147 = v77;
        }
      }
    }
  }
  while ( v151 != v8 );
  if ( (_DWORD)v169 )
  {
    sub_2A3F1A0(&v168);
    v11 = v168;
    v13 = &v168[(unsigned int)v169];
    if ( v13 != v168 )
    {
      v14 = v168;
      do
      {
        v15 = (_QWORD *)*v14;
        if ( a4 )
        {
          v16 = sub_BD5D20(*v14);
          sub_BBB260(a4, (__int64)v15, (__int64)v16, v17);
        }
        if ( !v15 )
        {
          sub_BA8570(v8, -56);
          BUG();
        }
        ++v14;
        sub_BA8570(v8, (__int64)v15);
        v18 = (unsigned __int64 *)v15[8];
        v19 = v15[7] & 0xFFFFFFFFFFFFFFF8LL;
        *v18 = v19 | *v18 & 7;
        *(_QWORD *)(v19 + 8) = v18;
        v15[7] &= 7uLL;
        v15[8] = 0;
        sub_B2E780(v15);
        sub_BD2DD0((__int64)v15);
      }
      while ( v13 != v14 );
      v147 = 1;
      v11 = v168;
    }
  }
  else
  {
    v11 = v168;
  }
  if ( v11 != (__int64 *)v170 )
    _libc_free((unsigned __int64)v11);
LABEL_10:
  if ( v175 != (__int64 *)v177 )
    _libc_free((unsigned __int64)v175);
  sub_C7D6A0((__int64)v172, 8LL * (unsigned int)v174, 8);
  return v147;
}
