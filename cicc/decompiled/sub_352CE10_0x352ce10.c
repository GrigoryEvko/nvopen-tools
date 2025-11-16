// Function: sub_352CE10
// Address: 0x352ce10
//
__int64 __fastcall sub_352CE10(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r13d
  __int64 *v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r9
  __int64 v26; // r12
  unsigned int v27; // edx
  __int64 v28; // r8
  _QWORD *v29; // rbx
  __int64 v30; // rcx
  unsigned __int64 v31; // rdi
  __int64 v32; // rbx
  __int64 v33; // r12
  int v34; // ecx
  __int64 v35; // rsi
  int v36; // ecx
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // r8
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // r14
  __int64 v46; // r12
  __int64 v47; // rdx
  unsigned int v48; // eax
  __int64 v49; // r10
  int v50; // r11d
  _QWORD *v51; // rbx
  unsigned int v52; // edx
  _QWORD *v53; // r8
  __int64 v54; // rax
  __int64 **v55; // r10
  __int64 v56; // rax
  _QWORD *v57; // rbx
  __int64 *v58; // r12
  _QWORD *v59; // rsi
  __int64 **v60; // r9
  __int64 **v61; // r10
  __int64 **v62; // r9
  __int64 *v63; // r13
  __int64 **v64; // r9
  __int64 **v65; // r10
  __int64 v66; // rax
  _QWORD *v67; // rbx
  _QWORD *v68; // r12
  unsigned __int64 v69; // rdi
  int v71; // eax
  _QWORD *v72; // r12
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r13
  unsigned int v77; // edx
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rdx
  unsigned __int64 v81; // r8
  __int64 *v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  unsigned int v86; // edx
  __int64 *v87; // r10
  __int64 v88; // rdi
  _QWORD *v89; // r15
  int v90; // ebx
  __int64 v91; // r9
  __int64 v92; // r14
  __int64 v93; // r12
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 *v96; // r10
  int v97; // r13d
  __int64 *v98; // rcx
  __int64 *v99; // rax
  int v100; // eax
  int v101; // r8d
  __int64 **v102; // r11
  unsigned int v103; // edx
  __int64 v104; // rsi
  int v105; // r11d
  _QWORD *v106; // rdi
  _QWORD *v107; // rsi
  unsigned int v108; // r13d
  __int64 v109; // rcx
  int v110; // r8d
  __int64 v111; // rsi
  __m128i *v112; // rdx
  const __m128i *v113; // rax
  unsigned __int64 v114; // rsi
  __int64 v115; // rsi
  __m128i *v116; // rdx
  const __m128i *v117; // rax
  const __m128i *v118; // rsi
  unsigned int v119; // r8d
  __int64 v120; // rcx
  __int64 v121; // [rsp+10h] [rbp-970h]
  __int64 v122; // [rsp+18h] [rbp-968h]
  __int64 v123; // [rsp+20h] [rbp-960h]
  _BYTE *v126; // [rsp+50h] [rbp-930h]
  _BYTE *v127; // [rsp+68h] [rbp-918h]
  __int64 v128; // [rsp+70h] [rbp-910h]
  __int64 v129; // [rsp+70h] [rbp-910h]
  __int64 v130; // [rsp+78h] [rbp-908h]
  __int64 v131; // [rsp+78h] [rbp-908h]
  __int64 v132; // [rsp+80h] [rbp-900h]
  __int64 v133; // [rsp+80h] [rbp-900h]
  const void *v134; // [rsp+88h] [rbp-8F8h]
  __int64 v135; // [rsp+88h] [rbp-8F8h]
  __int64 v136; // [rsp+88h] [rbp-8F8h]
  unsigned int v137; // [rsp+88h] [rbp-8F8h]
  __int64 *v138; // [rsp+90h] [rbp-8F0h]
  __int64 v139; // [rsp+98h] [rbp-8E8h]
  __int64 *v140; // [rsp+98h] [rbp-8E8h]
  __int64 v141[4]; // [rsp+A0h] [rbp-8E0h] BYREF
  __int64 v142; // [rsp+C0h] [rbp-8C0h] BYREF
  _QWORD *v143; // [rsp+C8h] [rbp-8B8h]
  __int64 v144; // [rsp+D0h] [rbp-8B0h]
  unsigned int v145; // [rsp+D8h] [rbp-8A8h]
  __int64 v146; // [rsp+E0h] [rbp-8A0h] BYREF
  __int64 v147; // [rsp+E8h] [rbp-898h]
  __int64 v148; // [rsp+F0h] [rbp-890h]
  unsigned int v149; // [rsp+F8h] [rbp-888h]
  _BYTE *v150; // [rsp+100h] [rbp-880h] BYREF
  __int64 v151; // [rsp+108h] [rbp-878h]
  _BYTE v152[64]; // [rsp+110h] [rbp-870h] BYREF
  unsigned __int64 v153[38]; // [rsp+150h] [rbp-830h] BYREF
  __int64 v154; // [rsp+280h] [rbp-700h] BYREF
  __int64 *v155; // [rsp+288h] [rbp-6F8h]
  int v156; // [rsp+290h] [rbp-6F0h]
  int v157; // [rsp+294h] [rbp-6ECh]
  int v158; // [rsp+298h] [rbp-6E8h]
  char v159; // [rsp+29Ch] [rbp-6E4h]
  __int64 v160; // [rsp+2A0h] [rbp-6E0h] BYREF
  __int64 *v161; // [rsp+2E0h] [rbp-6A0h]
  unsigned int v162; // [rsp+2E8h] [rbp-698h]
  int v163; // [rsp+2ECh] [rbp-694h]
  __int64 v164[24]; // [rsp+2F0h] [rbp-690h] BYREF
  char v165[8]; // [rsp+3B0h] [rbp-5D0h] BYREF
  unsigned __int64 v166; // [rsp+3B8h] [rbp-5C8h]
  char v167; // [rsp+3CCh] [rbp-5B4h]
  char v168[64]; // [rsp+3D0h] [rbp-5B0h] BYREF
  __m128i *v169; // [rsp+410h] [rbp-570h] BYREF
  __int64 v170; // [rsp+418h] [rbp-568h]
  _BYTE v171[192]; // [rsp+420h] [rbp-560h] BYREF
  char v172[8]; // [rsp+4E0h] [rbp-4A0h] BYREF
  unsigned __int64 v173; // [rsp+4E8h] [rbp-498h]
  char v174; // [rsp+4FCh] [rbp-484h]
  char *v175; // [rsp+540h] [rbp-440h]
  char v176; // [rsp+550h] [rbp-430h] BYREF
  _QWORD *v177; // [rsp+610h] [rbp-370h] BYREF
  unsigned __int64 v178; // [rsp+618h] [rbp-368h]
  _BYTE v179[16]; // [rsp+620h] [rbp-360h] BYREF
  char v180[64]; // [rsp+630h] [rbp-350h] BYREF
  __m128i *v181; // [rsp+670h] [rbp-310h] BYREF
  __int64 v182; // [rsp+678h] [rbp-308h]
  _BYTE v183[192]; // [rsp+680h] [rbp-300h] BYREF
  __int64 *v184; // [rsp+740h] [rbp-240h] BYREF
  unsigned __int64 v185; // [rsp+748h] [rbp-238h]
  __int64 v186; // [rsp+750h] [rbp-230h] BYREF
  __int64 v187; // [rsp+758h] [rbp-228h]
  char *v188; // [rsp+7A0h] [rbp-1E0h]
  char v189; // [rsp+7B0h] [rbp-1D0h] BYREF

  v2 = *(_QWORD *)(a1 + 144);
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  sub_2E63F80((_QWORD *)(a1 + 48), v2);
  v141[2] = (__int64)&v146;
  v150 = v152;
  memset(v153, 0, sizeof(v153));
  v155 = &v160;
  v153[1] = (unsigned __int64)&v153[4];
  v153[12] = (unsigned __int64)&v153[14];
  v3 = *(_QWORD *)(v2 + 328);
  v141[0] = a2;
  v4 = *(_QWORD *)(v3 + 112);
  v5 = *(unsigned int *)(v3 + 120);
  v141[1] = a1;
  v151 = 0x800000000LL;
  v160 = v3;
  v164[0] = v4 + 8 * v5;
  v164[1] = v4;
  v164[2] = v3;
  LODWORD(v153[2]) = 8;
  BYTE4(v153[3]) = 1;
  HIDWORD(v153[13]) = 8;
  v156 = 8;
  v158 = 0;
  v159 = 1;
  v161 = v164;
  v163 = 8;
  v157 = 1;
  v154 = 1;
  v162 = 1;
  sub_2EFF3E0((__int64)&v154, v2, v4, v164[0], v6, v7);
  sub_C8CD80((__int64)&v177, (__int64)v180, (__int64)v153, v8, v9, v10);
  v182 = 0x800000000LL;
  v14 = v153[13];
  v181 = (__m128i *)v183;
  if ( LODWORD(v153[13]) )
  {
    v111 = LODWORD(v153[13]);
    v112 = (__m128i *)v183;
    if ( LODWORD(v153[13]) > 8 )
    {
      sub_2E3C030((__int64)&v181, LODWORD(v153[13]), (__int64)v183, v11, v12, v13);
      v112 = v181;
      v111 = LODWORD(v153[13]);
    }
    v113 = (const __m128i *)v153[12];
    v114 = v153[12] + 24 * v111;
    if ( v153[12] != v114 )
    {
      do
      {
        if ( v112 )
        {
          *v112 = _mm_loadu_si128(v113);
          v112[1].m128i_i64[0] = v113[1].m128i_i64[0];
        }
        v113 = (const __m128i *)((char *)v113 + 24);
        v112 = (__m128i *)((char *)v112 + 24);
      }
      while ( (const __m128i *)v114 != v113 );
    }
    LODWORD(v182) = v14;
  }
  v15 = (__int64 *)&v184;
  sub_2EFF5C0((__int64)&v184, (__int64)&v177);
  sub_C8CD80((__int64)v165, (__int64)v168, (__int64)&v154, v16, v17, v18);
  v21 = v162;
  v169 = (__m128i *)v171;
  v170 = 0x800000000LL;
  if ( v162 )
  {
    v115 = v162;
    v116 = (__m128i *)v171;
    if ( v162 > 8 )
    {
      v137 = v162;
      sub_2E3C030((__int64)&v169, v162, (__int64)v171, v162, v19, v20);
      v116 = v169;
      v115 = v162;
      v21 = v137;
    }
    v117 = (const __m128i *)v161;
    v118 = (const __m128i *)&v161[3 * v115];
    if ( v161 != (__int64 *)v118 )
    {
      do
      {
        if ( v116 )
        {
          *v116 = _mm_loadu_si128(v117);
          v116[1].m128i_i64[0] = v117[1].m128i_i64[0];
        }
        v117 = (const __m128i *)((char *)v117 + 24);
        v116 = (__m128i *)((char *)v116 + 24);
      }
      while ( v118 != v117 );
    }
    LODWORD(v170) = v21;
  }
  sub_2EFF5C0((__int64)v172, (__int64)v165);
  sub_2EFF7A0((__int64)v172, (__int64)&v184, (__int64)&v150, v22, v23, v24);
  if ( v175 != &v176 )
    _libc_free((unsigned __int64)v175);
  if ( !v174 )
    _libc_free(v173);
  if ( v169 != (__m128i *)v171 )
    _libc_free((unsigned __int64)v169);
  if ( !v167 )
    _libc_free(v166);
  if ( v188 != &v189 )
    _libc_free((unsigned __int64)v188);
  if ( !BYTE4(v187) )
    _libc_free(v185);
  if ( v181 != (__m128i *)v183 )
    _libc_free((unsigned __int64)v181);
  if ( !v179[12] )
    _libc_free(v178);
  if ( v161 != v164 )
    _libc_free((unsigned __int64)v161);
  if ( !v159 )
    _libc_free((unsigned __int64)v155);
  if ( (unsigned __int64 *)v153[12] != &v153[14] )
    _libc_free(v153[12]);
  if ( !BYTE4(v153[3]) )
    _libc_free(v153[1]);
  v177 = v179;
  v178 = 0x800000000LL;
  v126 = v150;
  v127 = &v150[8 * (unsigned int)v151];
  if ( v150 != v127 )
  {
    while ( 1 )
    {
      v26 = *((_QWORD *)v127 - 1);
      LODWORD(v178) = 0;
      if ( v145 )
      {
        v27 = (v145 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = 5LL * v27;
        v29 = &v143[11 * v27];
        v30 = *v29;
        if ( v26 == *v29 )
        {
LABEL_30:
          if ( v29 != &v143[11 * v145] )
          {
            sub_352AEB0((__int64)&v177, (char **)v29 + 1, 5LL * v145, v30, v28, v25);
            v31 = v29[1];
            if ( (_QWORD *)v31 != v29 + 3 )
              _libc_free(v31);
            *v29 = -8192;
            LODWORD(v144) = v144 - 1;
            ++HIDWORD(v144);
          }
        }
        else
        {
          v110 = 1;
          while ( v30 != -4096 )
          {
            v25 = (unsigned int)(v110 + 1);
            v27 = (v145 - 1) & (v110 + v27);
            v28 = 5LL * v27;
            v29 = &v143[11 * v27];
            v30 = *v29;
            if ( v26 == *v29 )
              goto LABEL_30;
            v110 = v25;
          }
        }
      }
      v32 = v26 + 48;
      if ( v26 + 48 != *(_QWORD *)(v26 + 56) )
      {
        v139 = v26;
        v33 = *(_QWORD *)(v26 + 56);
        do
        {
          while ( 1 )
          {
            v34 = *(_DWORD *)(a1 + 184);
            v35 = *(_QWORD *)(a1 + 168);
            if ( v34 )
            {
              v36 = v34 - 1;
              v37 = v36 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v38 = (__int64 *)(v35 + 16LL * v37);
              v39 = *v38;
              if ( *v38 == v33 )
              {
LABEL_39:
                v40 = v38[1];
                if ( v40 )
                  sub_352C300(v141, v40, v33, (__int64)&v177);
              }
              else
              {
                v100 = 1;
                while ( v39 != -4096 )
                {
                  v101 = v100 + 1;
                  v37 = v36 & (v100 + v37);
                  v38 = (__int64 *)(v35 + 16LL * v37);
                  v39 = *v38;
                  if ( *v38 == v33 )
                    goto LABEL_39;
                  v100 = v101;
                }
              }
            }
            if ( (unsigned int)sub_352B010(v33) != 3 )
            {
              v42 = (unsigned int)v178;
              v43 = (unsigned int)v178 + 1LL;
              if ( v43 > HIDWORD(v178) )
              {
                sub_C8D5F0((__int64)&v177, v179, v43, 8u, v41, v25);
                v42 = (unsigned int)v178;
              }
              v177[v42] = v33;
              LODWORD(v178) = v178 + 1;
            }
            if ( !v33 )
              BUG();
            if ( (*(_BYTE *)v33 & 4) == 0 )
              break;
            v33 = *(_QWORD *)(v33 + 8);
            if ( v32 == v33 )
              goto LABEL_48;
          }
          while ( (*(_BYTE *)(v33 + 44) & 8) != 0 )
            v33 = *(_QWORD *)(v33 + 8);
          v33 = *(_QWORD *)(v33 + 8);
        }
        while ( v32 != v33 );
LABEL_48:
        v26 = v139;
      }
      v44 = *(__int64 **)(v26 + 112);
      v138 = &v44[*(unsigned int *)(v26 + 120)];
      if ( v44 != v138 )
        break;
LABEL_69:
      v127 -= 8;
      if ( v126 == v127 )
      {
        if ( v177 != (_QWORD *)v179 )
          _libc_free((unsigned __int64)v177);
        v127 = v150;
        goto LABEL_73;
      }
    }
    v140 = *(__int64 **)(v26 + 112);
    v45 = a2;
    while ( 1 )
    {
      v46 = *v140;
      if ( *v140 )
      {
        v47 = (unsigned int)(*(_DWORD *)(v46 + 24) + 1);
        v48 = *(_DWORD *)(v46 + 24) + 1;
      }
      else
      {
        v47 = 0;
        v48 = 0;
      }
      v49 = 0;
      if ( v48 < *(_DWORD *)(v45 + 32) )
        v49 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 8 * v47);
      if ( v145 )
      {
        v50 = 1;
        v51 = 0;
        v52 = (v145 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v53 = &v143[11 * v52];
        v54 = *v53;
        if ( v46 == *v53 )
        {
LABEL_57:
          v55 = (__int64 **)v53[1];
          v56 = *((unsigned int *)v53 + 4);
          v25 = (__int64)&v55[v56];
          if ( v55 == (__int64 **)v25 )
          {
            v102 = &v55[v56];
          }
          else
          {
            do
            {
              while ( 1 )
              {
                v57 = v177;
                v58 = *v55;
                v184 = *v55;
                v59 = &v177[(unsigned int)v178];
                if ( v59 == sub_352ADF0(v177, (__int64)v59, v15) )
                  break;
                v55 = v61 + 1;
                v102 = v55;
                if ( v60 == v55 )
                  goto LABEL_66;
              }
              v62 = v60 - 1;
              if ( v62 == v61 )
                break;
              while ( 1 )
              {
                v63 = *v62;
                v184 = *v62;
                if ( v59 != sub_352ADF0(v57, (__int64)v59, v15) )
                  break;
                v62 = v64 - 1;
                if ( v62 == v65 )
                  goto LABEL_66;
              }
              *v65 = v63;
              v55 = v65 + 1;
              *v64 = v58;
              v102 = v55;
            }
            while ( v64 != v55 );
LABEL_66:
            v25 = v53[1];
          }
          *((_DWORD *)v53 + 4) = ((__int64)v102 - v25) >> 3;
          goto LABEL_68;
        }
        while ( v54 != -4096 )
        {
          if ( !v51 && v54 == -8192 )
            v51 = v53;
          v25 = (unsigned int)(v50 + 1);
          v52 = (v145 - 1) & (v50 + v52);
          v53 = &v143[11 * v52];
          v54 = *v53;
          if ( v46 == *v53 )
            goto LABEL_57;
          ++v50;
        }
        if ( !v51 )
          v51 = v53;
        ++v142;
        v71 = v144 + 1;
        if ( 4 * ((int)v144 + 1) < 3 * v145 )
        {
          if ( v145 - HIDWORD(v144) - v71 <= v145 >> 3 )
          {
            v136 = v49;
            sub_352CBC0((__int64)&v142, v145);
            if ( !v145 )
            {
LABEL_181:
              LODWORD(v144) = v144 + 1;
              BUG();
            }
            v107 = 0;
            v49 = v136;
            v108 = (v145 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
            v25 = 1;
            v51 = &v143[11 * v108];
            v109 = *v51;
            v71 = v144 + 1;
            if ( v46 != *v51 )
            {
              while ( v109 != -4096 )
              {
                if ( !v107 && v109 == -8192 )
                  v107 = v51;
                v119 = v25 + 1;
                v120 = (v145 - 1) & (v108 + (_DWORD)v25);
                v25 = 5 * v120;
                v108 = v120;
                v51 = &v143[11 * v120];
                v109 = *v51;
                if ( v46 == *v51 )
                  goto LABEL_97;
                v25 = v119;
              }
              if ( v107 )
                v51 = v107;
            }
          }
          goto LABEL_97;
        }
      }
      else
      {
        ++v142;
      }
      v135 = v49;
      sub_352CBC0((__int64)&v142, 2 * v145);
      if ( !v145 )
        goto LABEL_181;
      v49 = v135;
      v103 = (v145 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v51 = &v143[11 * v103];
      v104 = *v51;
      v71 = v144 + 1;
      if ( v46 != *v51 )
      {
        v105 = 1;
        v106 = 0;
        while ( v104 != -4096 )
        {
          if ( v104 == -8192 && !v106 )
            v106 = v51;
          v25 = (unsigned int)(v105 + 1);
          v103 = (v145 - 1) & (v103 + v105);
          v51 = &v143[11 * v103];
          v104 = *v51;
          if ( v46 == *v51 )
            goto LABEL_97;
          ++v105;
        }
        if ( v106 )
          v51 = v106;
      }
LABEL_97:
      LODWORD(v144) = v71;
      if ( *v51 != -4096 )
        --HIDWORD(v144);
      *v51 = v46;
      v51[1] = v51 + 3;
      v134 = v51 + 3;
      v51[2] = 0x800000000LL;
      v72 = v177;
      if ( &v177[(unsigned int)v178] == v177 )
        goto LABEL_68;
      v73 = *v177;
      v25 = (__int64)&v177[(unsigned int)v178];
      v74 = *(_QWORD *)(*v177 + 24LL);
      if ( v74 )
      {
        while ( 1 )
        {
          v75 = (unsigned int)(*(_DWORD *)(v74 + 24) + 1);
          if ( (unsigned int)(*(_DWORD *)(v74 + 24) + 1) < *(_DWORD *)(v45 + 32) )
            break;
LABEL_118:
          if ( v49 )
            goto LABEL_68;
LABEL_114:
          v80 = *((unsigned int *)v51 + 4);
          v81 = v80 + 1;
          if ( v80 + 1 > (unsigned __int64)*((unsigned int *)v51 + 5) )
            goto LABEL_125;
LABEL_115:
          ++v72;
          *(_QWORD *)(v51[1] + 8 * v80) = v73;
          ++*((_DWORD *)v51 + 4);
          if ( (_QWORD *)v25 == v72 )
            goto LABEL_68;
          v73 = *v72;
          v74 = *(_QWORD *)(*v72 + 24LL);
          if ( !v74 )
            goto LABEL_117;
        }
      }
      else
      {
LABEL_117:
        v75 = 0;
        if ( !*(_DWORD *)(v45 + 32) )
          goto LABEL_118;
      }
      v76 = *(_QWORD *)(*(_QWORD *)(v45 + 24) + 8 * v75);
      if ( !v49 || v49 == v76 )
        goto LABEL_114;
      if ( !v76 )
        goto LABEL_68;
      if ( v76 == *(_QWORD *)(v49 + 8) )
        goto LABEL_114;
      if ( v49 != *(_QWORD *)(v76 + 8) && *(_DWORD *)(v76 + 16) < *(_DWORD *)(v49 + 16) )
      {
        if ( !*(_BYTE *)(v45 + 112) )
        {
          v77 = *(_DWORD *)(v45 + 116) + 1;
          *(_DWORD *)(v45 + 116) = v77;
          if ( v77 <= 0x20 )
          {
            v78 = v49;
            do
            {
              v79 = v78;
              v78 = *(_QWORD *)(v78 + 8);
            }
            while ( v78 && *(_DWORD *)(v76 + 16) <= *(_DWORD *)(v78 + 16) );
            if ( v76 != v79 )
              goto LABEL_68;
            goto LABEL_114;
          }
          v83 = *(_QWORD *)(v45 + 96);
          HIDWORD(v185) = 32;
          v184 = &v186;
          if ( v83 )
          {
            v84 = *(_QWORD *)(v83 + 24);
            v186 = v83;
            v85 = (__int64)v72;
            LODWORD(v185) = 1;
            v187 = v84;
            *(_DWORD *)(v83 + 72) = 0;
            v86 = 1;
            v131 = v49;
            v87 = &v186;
            v88 = (__int64)v15;
            v89 = v51;
            v129 = v25;
            v90 = 1;
            v91 = v45;
            v92 = v76;
            v133 = v73;
            do
            {
              v97 = v90++;
              v98 = &v87[2 * v86 - 2];
              v99 = (__int64 *)v98[1];
              if ( v99 == (__int64 *)(*(_QWORD *)(*v98 + 24) + 8LL * *(unsigned int *)(*v98 + 32)) )
              {
                --v86;
                *(_DWORD *)(*v98 + 76) = v97;
                LODWORD(v185) = v86;
              }
              else
              {
                v93 = *v99;
                v98[1] = (__int64)(v99 + 1);
                v94 = (unsigned int)v185;
                v95 = *(_QWORD *)(v93 + 24);
                if ( (unsigned __int64)(unsigned int)v185 + 1 > HIDWORD(v185) )
                {
                  v121 = v91;
                  v122 = v85;
                  v123 = *(_QWORD *)(v93 + 24);
                  sub_C8D5F0(v88, &v186, (unsigned int)v185 + 1LL, 0x10u, v85, v91);
                  v87 = v184;
                  v94 = (unsigned int)v185;
                  v91 = v121;
                  v85 = v122;
                  v95 = v123;
                }
                v96 = &v87[2 * v94];
                *v96 = v93;
                v96[1] = v95;
                v86 = v185 + 1;
                LODWORD(v185) = v185 + 1;
                *(_DWORD *)(v93 + 72) = v97;
                v87 = v184;
              }
            }
            while ( v86 );
            v51 = v89;
            v15 = (__int64 *)v88;
            v76 = v92;
            v82 = v87;
            v45 = v91;
            v73 = v133;
            v72 = (_QWORD *)v85;
            *(_DWORD *)(v91 + 116) = 0;
            v49 = v131;
            *(_BYTE *)(v91 + 112) = 1;
            v25 = v129;
            if ( v82 != &v186 )
            {
              _libc_free((unsigned __int64)v82);
              v73 = v133;
              v49 = v131;
              v25 = v129;
            }
          }
        }
        if ( *(_DWORD *)(v49 + 72) >= *(_DWORD *)(v76 + 72) && *(_DWORD *)(v49 + 76) <= *(_DWORD *)(v76 + 76) )
        {
          v80 = *((unsigned int *)v51 + 4);
          v81 = v80 + 1;
          if ( v80 + 1 <= (unsigned __int64)*((unsigned int *)v51 + 5) )
            goto LABEL_115;
LABEL_125:
          v128 = v25;
          v130 = v49;
          v132 = v73;
          sub_C8D5F0((__int64)(v51 + 1), v134, v81, 8u, v81, v25);
          v80 = *((unsigned int *)v51 + 4);
          v25 = v128;
          v49 = v130;
          v73 = v132;
          goto LABEL_115;
        }
      }
LABEL_68:
      if ( v138 == ++v140 )
        goto LABEL_69;
    }
  }
LABEL_73:
  if ( v127 != v152 )
    _libc_free((unsigned __int64)v127);
  sub_C7D6A0(v147, 16LL * v149, 8);
  v66 = v145;
  if ( v145 )
  {
    v67 = v143;
    v68 = &v143[11 * v145];
    do
    {
      if ( *v67 != -8192 && *v67 != -4096 )
      {
        v69 = v67[1];
        if ( (_QWORD *)v69 != v67 + 3 )
          _libc_free(v69);
      }
      v67 += 11;
    }
    while ( v68 != v67 );
    v66 = v145;
  }
  return sub_C7D6A0((__int64)v143, 88 * v66, 8);
}
