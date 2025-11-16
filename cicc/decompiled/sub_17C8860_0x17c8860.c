// Function: sub_17C8860
// Address: 0x17c8860
//
void __fastcall sub_17C8860(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r11
  _QWORD *v3; // r12
  unsigned int v4; // eax
  _QWORD *v5; // r9
  _QWORD *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r15
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rdi
  __int64 *v20; // r14
  unsigned __int64 v21; // rdi
  _QWORD *v22; // r8
  __int64 *v23; // rbx
  int v24; // r9d
  char v25; // dl
  __int64 v26; // r13
  __int64 *v27; // rsi
  __int64 *v28; // rax
  __int64 *v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // r13
  bool v35; // zf
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // r14
  __int64 v39; // rbx
  unsigned int v40; // esi
  __int64 v41; // r11
  unsigned int v42; // r13d
  __int64 v43; // rcx
  __int64 *v44; // rax
  __int64 v45; // r8
  __int64 v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rdx
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  unsigned __int64 v53; // rdi
  __int64 v54; // rdx
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  __int64 *v57; // r13
  __int64 *v58; // r12
  __int64 v59; // r15
  __int64 *v60; // rbx
  __int64 *v61; // r14
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // rax
  void *v65; // rdi
  unsigned int v66; // eax
  __int64 v67; // rdx
  unsigned __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdi
  unsigned __int64 *v72; // rbx
  unsigned __int64 *v73; // r12
  unsigned __int64 v74; // rdi
  unsigned __int64 *v75; // r12
  _QWORD *v76; // rbx
  _QWORD *v77; // r12
  __int64 v78; // r13
  __int64 v79; // rdi
  unsigned int v80; // ecx
  _QWORD *v81; // rdi
  unsigned int v82; // eax
  __int64 v83; // rax
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rax
  int v86; // ebx
  __int64 v87; // r12
  _QWORD *v88; // rax
  _QWORD *i; // rdx
  int v90; // eax
  int v91; // r9d
  unsigned __int64 *v92; // rax
  unsigned __int64 *v93; // r12
  unsigned __int64 *v94; // rbx
  unsigned __int64 v95; // rdi
  unsigned __int64 *v96; // rbx
  unsigned __int64 v97; // rdi
  int v98; // r10d
  int v99; // ecx
  __int64 v100; // r14
  _QWORD *v101; // rdx
  __int64 v102; // rsi
  int v103; // r10d
  __int64 *v104; // rdi
  int v105; // ecx
  int v106; // edx
  int v107; // r11d
  int v108; // r11d
  __int64 v109; // r10
  unsigned int v110; // ecx
  int v111; // edi
  __int64 *v112; // rsi
  __int64 v113; // r8
  int v114; // r11d
  int v115; // r11d
  __int64 v116; // r10
  unsigned int v117; // ecx
  __int64 v118; // r8
  int v119; // edi
  _QWORD *v120; // rax
  _QWORD *v121; // r10
  int v122; // esi
  __int64 v123; // rbx
  _QWORD *v124; // rcx
  char *v126; // [rsp+20h] [rbp-3E0h]
  __int64 v127; // [rsp+48h] [rbp-3B8h]
  __int64 v128; // [rsp+58h] [rbp-3A8h]
  char *v129; // [rsp+60h] [rbp-3A0h]
  __int64 v130; // [rsp+68h] [rbp-398h]
  unsigned __int64 v131; // [rsp+70h] [rbp-390h]
  unsigned __int64 v132; // [rsp+78h] [rbp-388h]
  __int64 v133; // [rsp+80h] [rbp-380h]
  _QWORD *v134; // [rsp+88h] [rbp-378h]
  unsigned __int64 v135; // [rsp+90h] [rbp-370h]
  unsigned __int64 v136; // [rsp+98h] [rbp-368h]
  __int64 v137; // [rsp+A0h] [rbp-360h]
  __int64 *v138; // [rsp+A8h] [rbp-358h]
  _QWORD *v139; // [rsp+B0h] [rbp-350h]
  _QWORD *v140; // [rsp+B0h] [rbp-350h]
  _QWORD *v141; // [rsp+B0h] [rbp-350h]
  __int64 v142; // [rsp+B8h] [rbp-348h]
  __int64 v143; // [rsp+B8h] [rbp-348h]
  __int64 v144; // [rsp+B8h] [rbp-348h]
  __int64 v145; // [rsp+C0h] [rbp-340h] BYREF
  _QWORD *v146; // [rsp+C8h] [rbp-338h]
  __int64 v147; // [rsp+D0h] [rbp-330h]
  unsigned int v148; // [rsp+D8h] [rbp-328h]
  __int64 *v149; // [rsp+E0h] [rbp-320h] BYREF
  __int64 v150; // [rsp+E8h] [rbp-318h]
  _QWORD v151[2]; // [rsp+F0h] [rbp-310h] BYREF
  char *v152; // [rsp+100h] [rbp-300h] BYREF
  int v153; // [rsp+108h] [rbp-2F8h]
  char v154; // [rsp+110h] [rbp-2F0h] BYREF
  unsigned __int64 v155[2]; // [rsp+130h] [rbp-2D0h] BYREF
  _BYTE v156[32]; // [rsp+140h] [rbp-2C0h] BYREF
  unsigned __int64 v157[2]; // [rsp+160h] [rbp-2A0h] BYREF
  char v158; // [rsp+170h] [rbp-290h] BYREF
  __int64 v159; // [rsp+178h] [rbp-288h]
  _QWORD *v160; // [rsp+180h] [rbp-280h]
  __int64 v161; // [rsp+188h] [rbp-278h]
  unsigned int v162; // [rsp+190h] [rbp-270h]
  __int64 v163; // [rsp+1A0h] [rbp-260h]
  char v164; // [rsp+1A8h] [rbp-258h]
  int v165; // [rsp+1ACh] [rbp-254h]
  __int64 *v166; // [rsp+1B0h] [rbp-250h] BYREF
  __int64 v167; // [rsp+1B8h] [rbp-248h]
  _BYTE v168[64]; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 (__fastcall **v169)(); // [rsp+200h] [rbp-200h] BYREF
  _QWORD *v170; // [rsp+208h] [rbp-1F8h]
  _QWORD *v171; // [rsp+210h] [rbp-1F0h]
  __int64 v172; // [rsp+218h] [rbp-1E8h]
  unsigned __int64 v173; // [rsp+220h] [rbp-1E0h]
  _QWORD v174[9]; // [rsp+228h] [rbp-1D8h] BYREF
  __int64 v175; // [rsp+270h] [rbp-190h] BYREF
  _QWORD *v176; // [rsp+278h] [rbp-188h]
  __int64 v177; // [rsp+280h] [rbp-180h]
  unsigned int v178; // [rsp+288h] [rbp-178h]
  __int64 *v179; // [rsp+290h] [rbp-170h]
  __int64 *v180; // [rsp+298h] [rbp-168h]
  __int64 v181; // [rsp+2A0h] [rbp-160h]
  unsigned __int64 v182; // [rsp+2A8h] [rbp-158h]
  unsigned __int64 v183; // [rsp+2B0h] [rbp-150h]
  unsigned __int64 *v184; // [rsp+2B8h] [rbp-148h]
  unsigned int v185; // [rsp+2C0h] [rbp-140h]
  char v186; // [rsp+2C8h] [rbp-138h] BYREF
  unsigned __int64 *v187; // [rsp+2E8h] [rbp-118h]
  unsigned int v188; // [rsp+2F0h] [rbp-110h]
  __int64 v189; // [rsp+2F8h] [rbp-108h] BYREF
  __int64 *v190; // [rsp+310h] [rbp-F0h] BYREF
  _BYTE *v191; // [rsp+318h] [rbp-E8h] BYREF
  __int64 v192; // [rsp+320h] [rbp-E0h]
  _BYTE v193[64]; // [rsp+328h] [rbp-D8h] BYREF
  _BYTE *v194; // [rsp+368h] [rbp-98h] BYREF
  __int64 v195; // [rsp+370h] [rbp-90h]
  _BYTE v196[64]; // [rsp+378h] [rbp-88h] BYREF
  __int64 v197; // [rsp+3B8h] [rbp-48h]
  __int64 *v198; // [rsp+3C0h] [rbp-40h]

  v157[0] = (unsigned __int64)&v158;
  v157[1] = 0x100000000LL;
  v163 = a2;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v164 = 0;
  v165 = 0;
  sub_15D3930((__int64)v157);
  sub_14019E0((__int64)&v175, (__int64)v157);
  v2 = (_QWORD *)a1[27];
  v3 = (_QWORD *)a1[26];
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  if ( v3 != v2 )
  {
    while ( 1 )
    {
      if ( v178 )
      {
        v10 = *v3;
        v11 = *(_QWORD *)(*v3 + 40LL);
        v12 = (v178 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v13 = &v176[2 * v12];
        v14 = *v13;
        if ( v11 != *v13 )
        {
          v90 = 1;
          while ( v14 != -8 )
          {
            v91 = v90 + 1;
            v12 = (v178 - 1) & (v90 + v12);
            v13 = &v176[2 * v12];
            v14 = *v13;
            if ( v11 == *v13 )
              goto LABEL_11;
            v90 = v91;
          }
          goto LABEL_8;
        }
LABEL_11:
        v15 = v13[1];
        if ( v15 )
          break;
      }
LABEL_8:
      v3 += 2;
      if ( v2 == v3 )
        goto LABEL_20;
    }
    v16 = v3[1];
    if ( v148 )
    {
      v4 = (v148 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      LODWORD(v5) = 9 * v4;
      v6 = &v146[19 * v4];
      v7 = *v6;
      if ( v15 == *v6 )
      {
LABEL_4:
        LODWORD(v8) = *((_DWORD *)v6 + 4);
        if ( (unsigned int)v8 >= *((_DWORD *)v6 + 5) )
        {
          v140 = v2;
          v143 = v3[1];
          sub_16CD150((__int64)(v6 + 1), v6 + 3, 0, 16, v16, (int)v5);
          v2 = v140;
          v16 = v143;
          v8 = *((unsigned int *)v6 + 4);
          v9 = (_QWORD *)(v6[1] + 16 * v8);
        }
        else
        {
          v9 = (_QWORD *)(v6[1] + 16LL * (unsigned int)v8);
        }
        if ( !v9 )
          goto LABEL_7;
        goto LABEL_19;
      }
      v98 = 1;
      v5 = 0;
      while ( v7 != -8 )
      {
        if ( !v5 && v7 == -16 )
          v5 = v6;
        v4 = (v148 - 1) & (v98 + v4);
        v6 = &v146[19 * v4];
        v7 = *v6;
        if ( v15 == *v6 )
          goto LABEL_4;
        ++v98;
      }
      if ( v5 )
        v6 = v5;
      ++v145;
      v18 = v147 + 1;
      if ( 4 * ((int)v147 + 1) < 3 * v148 )
      {
        if ( v148 - HIDWORD(v147) - v18 <= v148 >> 3 )
        {
          v141 = v2;
          v144 = v16;
          sub_17C7F60((__int64)&v145, v148);
          if ( !v148 )
          {
LABEL_229:
            LODWORD(v147) = v147 + 1;
            BUG();
          }
          v16 = v144;
          v99 = 1;
          LODWORD(v100) = (v148 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v2 = v141;
          v101 = 0;
          v6 = &v146[19 * (unsigned int)v100];
          v102 = *v6;
          v18 = v147 + 1;
          if ( v15 != *v6 )
          {
            while ( v102 != -8 )
            {
              if ( !v101 && v102 == -16 )
                v101 = v6;
              v100 = (v148 - 1) & ((_DWORD)v100 + v99);
              v6 = &v146[19 * v100];
              v102 = *v6;
              if ( v15 == *v6 )
                goto LABEL_16;
              ++v99;
            }
            if ( v101 )
              v6 = v101;
          }
        }
        goto LABEL_16;
      }
    }
    else
    {
      ++v145;
    }
    v139 = v2;
    v142 = v16;
    sub_17C7F60((__int64)&v145, 2 * v148);
    if ( !v148 )
      goto LABEL_229;
    v16 = v142;
    v2 = v139;
    v6 = &v146[19 * ((v148 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))];
    v17 = *v6;
    v18 = v147 + 1;
    if ( v15 != *v6 )
    {
      v121 = &v146[19 * ((v148 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)))];
      v122 = 1;
      LODWORD(v123) = (v148 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v124 = 0;
      while ( v17 != -8 )
      {
        if ( v17 == -16 && !v124 )
          v124 = v121;
        v123 = (v148 - 1) & ((_DWORD)v123 + v122);
        v121 = &v146[19 * v123];
        v17 = *v121;
        if ( v15 == *v121 )
        {
          v6 = &v146[19 * v123];
          goto LABEL_16;
        }
        ++v122;
      }
      v6 = v121;
      if ( v124 )
        v6 = v124;
    }
LABEL_16:
    LODWORD(v147) = v18;
    if ( *v6 != -8 )
      --HIDWORD(v147);
    v9 = v6 + 3;
    *v6 = v15;
    v6[1] = v6 + 3;
    v6[2] = 0x800000000LL;
LABEL_19:
    *v9 = v10;
    v9[1] = v16;
    LODWORD(v8) = *((_DWORD *)v6 + 4);
LABEL_7:
    *((_DWORD *)v6 + 4) = v8 + 1;
    goto LABEL_8;
  }
LABEL_20:
  sub_13FB8B0((__int64)&v152, (__int64)&v175);
  v126 = v152;
  v129 = &v152[8 * v153];
  if ( v152 == v129 )
    goto LABEL_66;
  do
  {
    v19 = *((_QWORD *)v129 - 1);
    v194 = v196;
    v169 = 0;
    v190 = &v145;
    v198 = &v175;
    v191 = v193;
    v192 = 0x800000000LL;
    v195 = 0x800000000LL;
    v167 = 0x800000000LL;
    v197 = v19;
    v170 = v174;
    v171 = v174;
    v166 = (__int64 *)v168;
    v172 = 8;
    LODWORD(v173) = 0;
    sub_13F9EC0(v19, (__int64)&v166);
    v20 = v166;
    v21 = (unsigned __int64)v171;
    v22 = v170;
    v23 = &v166[(unsigned int)v167];
    if ( v166 != v23 )
    {
      do
      {
        while ( 1 )
        {
          v26 = *v20;
          if ( (_QWORD *)v21 != v22 )
            break;
          v27 = (__int64 *)(v21 + 8LL * HIDWORD(v172));
          v24 = HIDWORD(v172);
          if ( v27 != (__int64 *)v21 )
          {
            v28 = (__int64 *)v21;
            v29 = 0;
            while ( v26 != *v28 )
            {
              if ( *v28 == -2 )
                v29 = v28;
              if ( v27 == ++v28 )
              {
                if ( !v29 )
                  goto LABEL_124;
                *v29 = v26;
                LODWORD(v173) = v173 - 1;
                v169 = (__int64 (__fastcall **)())((char *)v169 + 1);
                goto LABEL_34;
              }
            }
            goto LABEL_24;
          }
LABEL_124:
          if ( HIDWORD(v172) >= (unsigned int)v172 )
            break;
          v24 = ++HIDWORD(v172);
          *v27 = v26;
          v30 = (unsigned int)v192;
          v169 = (__int64 (__fastcall **)())((char *)v169 + 1);
          if ( (unsigned int)v192 >= HIDWORD(v192) )
          {
LABEL_126:
            sub_16CD150((__int64)&v191, v193, 0, 8, (int)v22, v24);
            v30 = (unsigned int)v192;
          }
LABEL_35:
          *(_QWORD *)&v191[8 * v30] = v26;
          LODWORD(v192) = v192 + 1;
          v31 = sub_157EE30(v26);
          v34 = v31 - 24;
          v35 = v31 == 0;
          v36 = (unsigned int)v195;
          if ( v35 )
            v34 = 0;
          if ( (unsigned int)v195 >= HIDWORD(v195) )
          {
            sub_16CD150((__int64)&v194, v196, 0, 8, v32, v33);
            v36 = (unsigned int)v195;
          }
          ++v20;
          *(_QWORD *)&v194[8 * v36] = v34;
          v21 = (unsigned __int64)v171;
          LODWORD(v195) = v195 + 1;
          v22 = v170;
          if ( v23 == v20 )
            goto LABEL_40;
        }
        sub_16CCBA0((__int64)&v169, *v20);
        v21 = (unsigned __int64)v171;
        v22 = v170;
        if ( v25 )
        {
LABEL_34:
          v30 = (unsigned int)v192;
          if ( (unsigned int)v192 >= HIDWORD(v192) )
            goto LABEL_126;
          goto LABEL_35;
        }
LABEL_24:
        ++v20;
      }
      while ( v23 != v20 );
    }
LABEL_40:
    if ( (_QWORD *)v21 != v22 )
      _libc_free(v21);
    if ( v166 != (__int64 *)v168 )
      _libc_free((unsigned __int64)v166);
    if ( (_DWORD)v192 )
    {
      v37 = sub_17C8270((__int64 *)&v190, v197);
      if ( v37 )
      {
        v38 = (__int64)v190;
        v39 = v197;
        v40 = *((_DWORD *)v190 + 6);
        if ( v40 )
        {
          v41 = v190[1];
          v42 = ((unsigned int)v197 >> 9) ^ ((unsigned int)v197 >> 4);
          LODWORD(v43) = (v40 - 1) & v42;
          v44 = (__int64 *)(v41 + 152LL * (unsigned int)v43);
          v45 = *v44;
          if ( v197 == *v44 )
          {
LABEL_48:
            v46 = v44[1];
            v127 = v46 + 16LL * *((unsigned int *)v44 + 4);
            if ( v46 != v127 )
            {
              v128 = v46 + 16LL * (unsigned int)(v37 - 1);
              while ( 1 )
              {
                v155[0] = (unsigned __int64)v156;
                v155[1] = 0x400000000LL;
                sub_1B3B830(&v166, v155);
                v130 = sub_15A0680(**(_QWORD **)v46, 0, 0);
                v135 = (unsigned __int64)v191;
                v131 = _mm_cvtsi32_si128(v195).m128i_u64[0];
                v132 = _mm_cvtsi32_si128(v192).m128i_u64[0];
                v137 = (__int64)v190;
                v136 = (unsigned __int64)v194;
                v138 = v198;
                v133 = sub_13FC520(v197);
                v49 = *(_QWORD *)(v46 + 8);
                v149 = *(__int64 **)v46;
                v150 = v49;
                v134 = (_QWORD *)v49;
                sub_1B3BD80(&v169, &v149, 2, &v166, 0, 0);
                v169 = off_49F03C0;
                v171 = v134;
                v172 = v135;
                v174[0] = v136;
                v174[2] = v137;
                v174[3] = v138;
                v173 = v132;
                v174[1] = v131;
                sub_1B3BE00(&v166, v133, v130);
                v50 = *(__int64 **)v46;
                v151[1] = *(_QWORD *)(v46 + 8);
                v149 = v151;
                v151[0] = v50;
                v150 = 0x200000002LL;
                sub_1B40B80(&v169, &v149);
                if ( v149 != v151 )
                  _libc_free((unsigned __int64)v149);
                if ( v128 == v46 )
                  break;
                v47 = dword_4FA3600;
                v48 = a1[31] + 1LL;
                a1[31] = v48;
                if ( (_DWORD)v47 != -1 && v48 >= v47 )
                  break;
                sub_1B3B860(&v166);
                if ( (_BYTE *)v155[0] != v156 )
                  _libc_free(v155[0]);
                v46 += 16;
                if ( v127 == v46 )
                  goto LABEL_60;
              }
              sub_1B3B860(&v166);
              if ( (_BYTE *)v155[0] != v156 )
                _libc_free(v155[0]);
            }
            goto LABEL_60;
          }
          v103 = 1;
          v104 = 0;
          while ( v45 != -8 )
          {
            if ( v45 == -16 && !v104 )
              v104 = v44;
            v43 = (v40 - 1) & ((_DWORD)v43 + v103);
            v44 = (__int64 *)(v41 + 152 * v43);
            v45 = *v44;
            if ( v197 == *v44 )
              goto LABEL_48;
            ++v103;
          }
          v105 = *((_DWORD *)v190 + 4);
          if ( v104 )
            v44 = v104;
          ++*v190;
          v106 = v105 + 1;
          if ( 4 * (v105 + 1) < 3 * v40 )
          {
            if ( v40 - *(_DWORD *)(v38 + 20) - v106 > v40 >> 3 )
              goto LABEL_173;
            sub_17C7F60(v38, v40);
            v107 = *(_DWORD *)(v38 + 24);
            if ( !v107 )
              goto LABEL_230;
            v108 = v107 - 1;
            v109 = *(_QWORD *)(v38 + 8);
            v110 = v108 & v42;
            v111 = 1;
            v106 = *(_DWORD *)(v38 + 16) + 1;
            v112 = 0;
            v44 = (__int64 *)(v109 + 152LL * (v108 & v42));
            v113 = *v44;
            if ( v39 == *v44 )
              goto LABEL_173;
            while ( v113 != -8 )
            {
              if ( !v112 && v113 == -16 )
                v112 = v44;
              v110 = v108 & (v111 + v110);
              v44 = (__int64 *)(v109 + 152LL * v110);
              v113 = *v44;
              if ( v39 == *v44 )
                goto LABEL_173;
              ++v111;
            }
LABEL_189:
            if ( v112 )
              v44 = v112;
LABEL_173:
            *(_DWORD *)(v38 + 16) = v106;
            if ( *v44 != -8 )
              --*(_DWORD *)(v38 + 20);
            *v44 = v39;
            v44[1] = (__int64)(v44 + 3);
            v44[2] = 0x800000000LL;
            goto LABEL_60;
          }
        }
        else
        {
          ++*v190;
        }
        sub_17C7F60(v38, 2 * v40);
        v114 = *(_DWORD *)(v38 + 24);
        if ( !v114 )
        {
LABEL_230:
          ++*(_DWORD *)(v38 + 16);
          BUG();
        }
        v115 = v114 - 1;
        v116 = *(_QWORD *)(v38 + 8);
        v117 = v115 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v106 = *(_DWORD *)(v38 + 16) + 1;
        v44 = (__int64 *)(v116 + 152LL * v117);
        v118 = *v44;
        if ( *v44 == v39 )
          goto LABEL_173;
        v119 = 1;
        v112 = 0;
        while ( v118 != -8 )
        {
          if ( v118 == -16 && !v112 )
            v112 = v44;
          v117 = v115 & (v119 + v117);
          v44 = (__int64 *)(v116 + 152LL * v117);
          v118 = *v44;
          if ( v39 == *v44 )
            goto LABEL_173;
          ++v119;
        }
        goto LABEL_189;
      }
    }
LABEL_60:
    if ( v194 != v196 )
      _libc_free((unsigned __int64)v194);
    if ( v191 != v193 )
      _libc_free((unsigned __int64)v191);
    v129 -= 8;
  }
  while ( v126 != v129 );
  v129 = v152;
LABEL_66:
  if ( v129 != &v154 )
    _libc_free((unsigned __int64)v129);
  if ( v148 )
  {
    v51 = v146;
    v52 = &v146[19 * v148];
    do
    {
      if ( *v51 != -16 && *v51 != -8 )
      {
        v53 = v51[1];
        if ( (_QWORD *)v53 != v51 + 3 )
          _libc_free(v53);
      }
      v51 += 19;
    }
    while ( v52 != v51 );
  }
  j___libc_free_0(v146);
  ++v175;
  if ( (_DWORD)v177 )
  {
    v80 = 4 * v177;
    v54 = v178;
    if ( (unsigned int)(4 * v177) < 0x40 )
      v80 = 64;
    if ( v80 >= v178 )
    {
LABEL_78:
      v55 = v176;
      v56 = &v176[2 * v54];
      if ( v176 != v56 )
      {
        do
        {
          *v55 = -8;
          v55 += 2;
        }
        while ( v56 != v55 );
      }
      v177 = 0;
      goto LABEL_81;
    }
    v81 = v176;
    if ( (_DWORD)v177 == 1 )
    {
      v87 = 2048;
      v86 = 128;
    }
    else
    {
      _BitScanReverse(&v82, v177 - 1);
      v83 = (unsigned int)(1 << (33 - (v82 ^ 0x1F)));
      if ( (int)v83 < 64 )
        v83 = 64;
      if ( (_DWORD)v83 == v178 )
      {
        v177 = 0;
        v120 = &v176[2 * v83];
        do
        {
          if ( v81 )
            *v81 = -8;
          v81 += 2;
        }
        while ( v120 != v81 );
        goto LABEL_81;
      }
      v84 = (4 * (int)v83 / 3u + 1) | ((unsigned __int64)(4 * (int)v83 / 3u + 1) >> 1);
      v85 = ((v84 | (v84 >> 2)) >> 4) | v84 | (v84 >> 2) | ((((v84 | (v84 >> 2)) >> 4) | v84 | (v84 >> 2)) >> 8);
      v86 = (v85 | (v85 >> 16)) + 1;
      v87 = 16 * ((v85 | (v85 >> 16)) + 1);
    }
    j___libc_free_0(v176);
    v178 = v86;
    v88 = (_QWORD *)sub_22077B0(v87);
    v177 = 0;
    v176 = v88;
    for ( i = &v88[2 * v178]; i != v88; v88 += 2 )
    {
      if ( v88 )
        *v88 = -8;
    }
    goto LABEL_81;
  }
  if ( HIDWORD(v177) )
  {
    v54 = v178;
    if ( v178 <= 0x40 )
      goto LABEL_78;
    j___libc_free_0(v176);
    v176 = 0;
    v177 = 0;
    v178 = 0;
  }
LABEL_81:
  v57 = v180;
  v58 = v179;
  if ( v179 != v180 )
  {
    do
    {
      v59 = *v58;
      v60 = *(__int64 **)(*v58 + 8);
      v61 = *(__int64 **)(*v58 + 16);
      if ( v60 == v61 )
      {
        *(_BYTE *)(v59 + 160) = 1;
      }
      else
      {
        do
        {
          v62 = *v60++;
          sub_13FACC0(v62);
        }
        while ( v61 != v60 );
        *(_BYTE *)(v59 + 160) = 1;
        v63 = *(_QWORD *)(v59 + 8);
        if ( v63 != *(_QWORD *)(v59 + 16) )
          *(_QWORD *)(v59 + 16) = v63;
      }
      v64 = *(_QWORD *)(v59 + 32);
      if ( v64 != *(_QWORD *)(v59 + 40) )
        *(_QWORD *)(v59 + 40) = v64;
      ++*(_QWORD *)(v59 + 56);
      v65 = *(void **)(v59 + 72);
      if ( v65 == *(void **)(v59 + 64) )
      {
        *(_QWORD *)v59 = 0;
      }
      else
      {
        v66 = 4 * (*(_DWORD *)(v59 + 84) - *(_DWORD *)(v59 + 88));
        v67 = *(unsigned int *)(v59 + 80);
        if ( v66 < 0x20 )
          v66 = 32;
        if ( v66 < (unsigned int)v67 )
          sub_16CC920(v59 + 56);
        else
          memset(v65, -1, 8 * v67);
        v68 = *(_QWORD *)(v59 + 72);
        v69 = *(_QWORD *)(v59 + 64);
        *(_QWORD *)v59 = 0;
        if ( v69 != v68 )
          _libc_free(v68);
      }
      v70 = *(_QWORD *)(v59 + 32);
      if ( v70 )
        j_j___libc_free_0(v70, *(_QWORD *)(v59 + 48) - v70);
      v71 = *(_QWORD *)(v59 + 8);
      if ( v71 )
        j_j___libc_free_0(v71, *(_QWORD *)(v59 + 24) - v71);
      ++v58;
    }
    while ( v57 != v58 );
    if ( v179 != v180 )
      v180 = v179;
  }
  v72 = v187;
  v73 = &v187[2 * v188];
  if ( v187 != v73 )
  {
    do
    {
      v74 = *v72;
      v72 += 2;
      _libc_free(v74);
    }
    while ( v73 != v72 );
  }
  v188 = 0;
  if ( v185 )
  {
    v92 = v184;
    v189 = 0;
    v93 = &v184[v185];
    v94 = v184 + 1;
    v182 = *v184;
    v183 = v182 + 4096;
    if ( v93 != v184 + 1 )
    {
      do
      {
        v95 = *v94++;
        _libc_free(v95);
      }
      while ( v93 != v94 );
      v92 = v184;
    }
    v185 = 1;
    _libc_free(*v92);
    v96 = v187;
    v75 = &v187[2 * v188];
    if ( v187 != v75 )
    {
      do
      {
        v97 = *v96;
        v96 += 2;
        _libc_free(v97);
      }
      while ( v75 != v96 );
      goto LABEL_105;
    }
  }
  else
  {
LABEL_105:
    v75 = v187;
  }
  if ( v75 != (unsigned __int64 *)&v189 )
    _libc_free((unsigned __int64)v75);
  if ( v184 != (unsigned __int64 *)&v186 )
    _libc_free((unsigned __int64)v184);
  if ( v179 )
    j_j___libc_free_0(v179, v181 - (_QWORD)v179);
  j___libc_free_0(v176);
  if ( v162 )
  {
    v76 = v160;
    v77 = &v160[2 * v162];
    do
    {
      if ( *v76 != -8 && *v76 != -16 )
      {
        v78 = v76[1];
        if ( v78 )
        {
          v79 = *(_QWORD *)(v78 + 24);
          if ( v79 )
            j_j___libc_free_0(v79, *(_QWORD *)(v78 + 40) - v79);
          j_j___libc_free_0(v78, 56);
        }
      }
      v76 += 2;
    }
    while ( v77 != v76 );
  }
  j___libc_free_0(v160);
  if ( (char *)v157[0] != &v158 )
    _libc_free(v157[0]);
}
