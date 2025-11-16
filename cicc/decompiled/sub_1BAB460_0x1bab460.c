// Function: sub_1BAB460
// Address: 0x1bab460
//
__int64 *__fastcall sub_1BAB460(__int64 *a1, __int64 *a2, int *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rdi
  size_t *v15; // rbx
  __int64 v16; // rax
  __int64 i; // r10
  _BYTE *v18; // rax
  size_t *v19; // r12
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned int v23; // esi
  size_t v24; // rcx
  __int64 v25; // r8
  unsigned int v26; // edx
  size_t *v27; // rax
  size_t v28; // rdi
  __int128 v29; // rdi
  __int64 v30; // rdx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // r13
  __m128i v34; // kr00_16
  _QWORD *v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rax
  __m128i *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rbx
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  _QWORD *v46; // r13
  __int64 v47; // rax
  __int64 *v48; // rdi
  __int64 *v49; // rbx
  _QWORD *v50; // rcx
  _BYTE *p_src; // rdi
  __int64 v52; // r8
  _BYTE *v53; // rdx
  __int64 v54; // rcx
  const char *v55; // rax
  char v56; // al
  __int64 v57; // rdx
  __m128i *v58; // rdx
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // r8
  int v63; // r9d
  unsigned int v64; // edx
  __int64 *v65; // rax
  __int64 v66; // r10
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rcx
  int v70; // r9d
  unsigned int v71; // edx
  _QWORD *v72; // rbx
  __m128i *v73; // rcx
  unsigned int v74; // edx
  __m128i **v75; // rax
  _QWORD *v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 *v79; // r8
  unsigned int v80; // ecx
  __m128i *v81; // rdx
  __int64 v82; // r8
  __int64 *v83; // rsi
  __int64 v84; // rbx
  __int64 v85; // r12
  _QWORD *v86; // rdi
  __int64 v87; // rsi
  _QWORD *v88; // rax
  int v89; // r8d
  _QWORD *v90; // rdi
  __int64 v91; // rsi
  _QWORD *v92; // rax
  int v93; // r8d
  __int64 (__fastcall *v94)(_QWORD *); // rax
  unsigned __int64 *v95; // r13
  unsigned __int64 *v96; // rbx
  unsigned __int64 *v97; // rax
  unsigned __int64 v98; // rcx
  unsigned __int64 *v99; // r13
  unsigned __int64 *v100; // rax
  unsigned __int64 v101; // rcx
  unsigned __int64 v102; // rdi
  unsigned __int64 v103; // rdi
  __int64 v104; // rdx
  __int64 v105; // rcx
  int v106; // r8d
  unsigned int v107; // ebx
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rsi
  unsigned int v111; // ebx
  __int64 v112; // rdx
  __int64 v113; // rcx
  int v114; // r8d
  __int64 v115; // rsi
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rsi
  _QWORD *v119; // rbx
  _BYTE *v120; // rdi
  size_t v121; // rcx
  __int64 v122; // rdx
  __int64 v123; // r8
  __int64 v125; // rdx
  __int64 v126; // r11
  int v127; // r13d
  int v128; // eax
  int v129; // ebx
  __int64 v130; // rax
  int v131; // eax
  __m128i **v132; // r11
  int v133; // ecx
  __m128i *v134; // rdx
  int v135; // edx
  int v136; // r10d
  size_t *v137; // r11
  int v138; // edi
  int v139; // edi
  unsigned int v140; // r8d
  unsigned int v141; // edx
  __m128i *v142; // rax
  __int64 v143; // r10
  int v144; // esi
  __m128i v145; // xmm0
  size_t v146; // rdx
  int v147; // r11d
  __int64 v148; // rax
  __int64 v149; // r11
  __int64 v150; // r9
  int v151; // eax
  int v152; // eax
  int v153; // r9d
  __int64 v154; // [rsp+8h] [rbp-298h]
  __int64 v155; // [rsp+10h] [rbp-290h]
  __int32 v157; // [rsp+30h] [rbp-270h]
  int v158; // [rsp+30h] [rbp-270h]
  __int64 v160; // [rsp+40h] [rbp-260h]
  __int64 v161; // [rsp+40h] [rbp-260h]
  __int64 *v162; // [rsp+48h] [rbp-258h]
  __int64 v164; // [rsp+58h] [rbp-248h]
  __int64 v165; // [rsp+58h] [rbp-248h]
  _QWORD *v166; // [rsp+58h] [rbp-248h]
  _QWORD *v167; // [rsp+58h] [rbp-248h]
  __int64 v168; // [rsp+58h] [rbp-248h]
  __int64 v169; // [rsp+58h] [rbp-248h]
  int v170; // [rsp+58h] [rbp-248h]
  __int64 v171; // [rsp+68h] [rbp-238h]
  __int64 v172[2]; // [rsp+70h] [rbp-230h] BYREF
  __int64 *v173; // [rsp+80h] [rbp-220h] BYREF
  __int64 *v174; // [rsp+88h] [rbp-218h]
  __int64 *v175; // [rsp+90h] [rbp-210h]
  __m128i v176; // [rsp+A0h] [rbp-200h] BYREF
  __int64 v177; // [rsp+B0h] [rbp-1F0h]
  __int64 v178; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 v179; // [rsp+C8h] [rbp-1D8h]
  __int64 v180; // [rsp+D0h] [rbp-1D0h]
  int v181; // [rsp+D8h] [rbp-1C8h]
  __int64 v182; // [rsp+E0h] [rbp-1C0h] BYREF
  __m128i *v183; // [rsp+E8h] [rbp-1B8h]
  __int64 v184; // [rsp+F0h] [rbp-1B0h]
  unsigned int v185; // [rsp+F8h] [rbp-1A8h]
  __m128i v186; // [rsp+100h] [rbp-1A0h] BYREF
  _WORD v187[8]; // [rsp+110h] [rbp-190h] BYREF
  void (__fastcall *v188)(_WORD *, _WORD *, __int64); // [rsp+120h] [rbp-180h]
  __m128i *v189; // [rsp+128h] [rbp-178h]
  size_t v190[2]; // [rsp+130h] [rbp-170h] BYREF
  _QWORD v191[2]; // [rsp+140h] [rbp-160h] BYREF
  void (__fastcall *v192)(_QWORD *, _QWORD *, __int64); // [rsp+150h] [rbp-150h]
  __int64 v193; // [rsp+158h] [rbp-148h]
  _QWORD v194[5]; // [rsp+160h] [rbp-140h] BYREF
  __int64 v195; // [rsp+188h] [rbp-118h]
  __int64 v196; // [rsp+190h] [rbp-110h]
  __int64 v197; // [rsp+198h] [rbp-108h]
  __m128i v198; // [rsp+1A0h] [rbp-100h] BYREF
  _BYTE *src; // [rsp+1B0h] [rbp-F0h] BYREF
  _BYTE *v200; // [rsp+1B8h] [rbp-E8h]
  void (__fastcall *v201)(_WORD *, _BYTE **, __int64); // [rsp+1C0h] [rbp-E0h]
  __m128i *v202; // [rsp+1C8h] [rbp-D8h]
  size_t v203; // [rsp+1D0h] [rbp-D0h]
  size_t v204; // [rsp+1D8h] [rbp-C8h]
  _BYTE v205[16]; // [rsp+1E0h] [rbp-C0h] BYREF
  void (__fastcall *v206)(_QWORD *, _BYTE *, __int64); // [rsp+1F0h] [rbp-B0h]
  __int64 v207; // [rsp+1F8h] [rbp-A8h]
  _QWORD v208[2]; // [rsp+200h] [rbp-A0h] BYREF
  __int64 v209; // [rsp+210h] [rbp-90h]
  __int64 v210; // [rsp+218h] [rbp-88h]
  __int64 v211; // [rsp+220h] [rbp-80h]
  __int64 *v212; // [rsp+228h] [rbp-78h]
  __int64 v213; // [rsp+230h] [rbp-70h]
  __int64 v214; // [rsp+238h] [rbp-68h]
  __int64 v215; // [rsp+240h] [rbp-60h]
  int v216; // [rsp+248h] [rbp-58h]
  __int64 v217; // [rsp+250h] [rbp-50h]
  __int64 v218; // [rsp+258h] [rbp-48h]
  __int64 v219; // [rsp+260h] [rbp-40h]
  int v220; // [rsp+268h] [rbp-38h]

  v155 = a2[4];
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v208[0] = "Pre-Entry";
  LOWORD(v209) = 259;
  v7 = sub_22077B0(128);
  v8 = v7;
  if ( v7 )
    sub_1B90E50(v7, (__int64)v208, 0);
  v9 = sub_22077B0(472);
  if ( v9 )
  {
    *(_QWORD *)v9 = v8;
    *(_QWORD *)(v9 + 8) = v9 + 24;
    *(_QWORD *)(v9 + 56) = v9 + 40;
    *(_QWORD *)(v9 + 64) = v9 + 40;
    *(_QWORD *)(v9 + 80) = v9 + 96;
    *(_QWORD *)(v9 + 120) = v9 + 152;
    *(_QWORD *)(v9 + 128) = v9 + 152;
    *(_QWORD *)(v9 + 16) = 0x200000000LL;
    *(_QWORD *)(v9 + 384) = v9 + 400;
    *(_DWORD *)(v9 + 40) = 0;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 72) = 0;
    *(_QWORD *)(v9 + 88) = 0;
    *(_BYTE *)(v9 + 96) = 0;
    *(_QWORD *)(v9 + 112) = 0;
    *(_QWORD *)(v9 + 136) = 16;
    *(_DWORD *)(v9 + 144) = 0;
    *(_QWORD *)(v9 + 280) = 0;
    *(_QWORD *)(v9 + 288) = 0;
    *(_QWORD *)(v9 + 296) = 0;
    *(_DWORD *)(v9 + 304) = 0;
    *(_QWORD *)(v9 + 312) = 0;
    *(_QWORD *)(v9 + 320) = 0;
    *(_QWORD *)(v9 + 328) = 0;
    *(_DWORD *)(v9 + 336) = 0;
    *(_QWORD *)(v9 + 344) = 0;
    *(_QWORD *)(v9 + 352) = 0;
    *(_QWORD *)(v9 + 360) = 0;
    *(_QWORD *)(v9 + 368) = 0;
    *(_QWORD *)(v9 + 376) = 0;
    *(_QWORD *)(v9 + 392) = 0x400000000LL;
    *(_QWORD *)(v9 + 432) = v9 + 448;
    *(_QWORD *)(v9 + 440) = 0;
    *(_QWORD *)(v9 + 448) = 0;
    *(_QWORD *)(v9 + 456) = 1;
  }
  v213 = 0;
  v214 = 0;
  *a1 = v9;
  v10 = a2[5];
  v11 = a2[4];
  v12 = a2[3];
  v215 = 0;
  v13 = a2[2];
  v14 = *a2;
  v211 = v10;
  v209 = v12;
  v15 = *(size_t **)(a4 + 16);
  v208[0] = v14;
  v208[1] = v13;
  v210 = v11;
  v212 = a2 + 12;
  v216 = 0;
  v217 = 0;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  if ( v15 == *(size_t **)(a4 + 8) )
    v16 = *(unsigned int *)(a4 + 28);
  else
    v16 = *(unsigned int *)(a4 + 24);
  for ( i = (__int64)&v15[v16]; (size_t *)i != v15; ++v15 )
  {
    if ( *v15 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  v198.m128i_i64[0] = i;
  v198.m128i_i64[1] = i;
  v164 = i;
  sub_19E4730((__int64)&v198);
  v18 = *(_BYTE **)a4;
  src = (_BYTE *)a4;
  v200 = v18;
  if ( v15 != (size_t *)v198.m128i_i64[0] )
  {
    v160 = a5;
    v19 = (size_t *)v164;
    while ( 1 )
    {
      v20 = *a1;
      v190[0] = *v15;
      v21 = sub_22077B0(40);
      v22 = v21;
      if ( v21 )
      {
        *(_BYTE *)v21 = 0;
        *(_QWORD *)(v21 + 8) = v21 + 24;
        *(_QWORD *)(v21 + 16) = 0x100000000LL;
        *(_QWORD *)(v21 + 32) = 0;
      }
      v23 = *(_DWORD *)(v20 + 304);
      if ( !v23 )
        break;
      v24 = v190[0];
      v25 = *(_QWORD *)(v20 + 288);
      v26 = (v23 - 1) & ((LODWORD(v190[0]) >> 9) ^ (LODWORD(v190[0]) >> 4));
      v27 = (size_t *)(v25 + 16LL * v26);
      v28 = *v27;
      if ( v190[0] != *v27 )
      {
        v170 = 1;
        v137 = 0;
        while ( v28 != -8 )
        {
          if ( v28 == -16 && !v137 )
            v137 = v27;
          v26 = (v23 - 1) & (v170 + v26);
          v27 = (size_t *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v190[0] == *v27 )
            goto LABEL_16;
          ++v170;
        }
        v138 = *(_DWORD *)(v20 + 296);
        if ( v137 )
          v27 = v137;
        ++*(_QWORD *)(v20 + 280);
        v139 = v138 + 1;
        if ( 4 * v139 < 3 * v23 )
        {
          if ( v23 - *(_DWORD *)(v20 + 300) - v139 > v23 >> 3 )
          {
LABEL_189:
            *(_DWORD *)(v20 + 296) = v139;
            if ( *v27 != -8 )
              --*(_DWORD *)(v20 + 300);
            *v27 = v24;
            v27[1] = 0;
            goto LABEL_16;
          }
LABEL_194:
          sub_1BA21E0(v20 + 280, v23);
          sub_1BA0BD0(v20 + 280, (__int64 *)v190, v194);
          v27 = (size_t *)v194[0];
          v24 = v190[0];
          v139 = *(_DWORD *)(v20 + 296) + 1;
          goto LABEL_189;
        }
LABEL_193:
        v23 *= 2;
        goto LABEL_194;
      }
LABEL_16:
      ++v15;
      for ( v27[1] = v22; v19 != v15; ++v15 )
      {
        if ( *v15 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
      if ( v15 == (size_t *)v198.m128i_i64[0] )
      {
        a5 = v160;
        goto LABEL_21;
      }
    }
    ++*(_QWORD *)(v20 + 280);
    goto LABEL_193;
  }
LABEL_21:
  sub_1AFCDB0((__int64)v194, *a2);
  *((_QWORD *)&v29 + 1) = a2[1];
  *(_QWORD *)&v29 = v194;
  sub_13FF3D0(v29);
  v154 = v195;
  v161 = v196;
  if ( v196 != v195 )
  {
    while ( 1 )
    {
      v171 = *(_QWORD *)(v161 - 8);
      v186.m128i_i64[0] = (__int64)sub_1649960(v171);
      v186.m128i_i64[1] = v30;
      LOWORD(v191[0]) = 261;
      v190[0] = (size_t)&v186;
      v33 = sub_22077B0(128);
      if ( v33 )
      {
        sub_16E2FC0(v198.m128i_i64, (__int64)v190);
        *(_BYTE *)(v33 + 8) = 0;
        v34 = v198;
        *(_QWORD *)v33 = &unk_49F6D50;
        *(_QWORD *)(v33 + 16) = v33 + 32;
        sub_1B8E960((__int64 *)(v33 + 16), v34.m128i_i64[0], v34.m128i_i64[0] + v34.m128i_i64[1]);
        v35 = (_QWORD *)v198.m128i_i64[0];
        *(_QWORD *)(v33 + 56) = v33 + 72;
        *(_QWORD *)(v33 + 80) = v33 + 96;
        *(_QWORD *)(v33 + 48) = 0;
        *(_QWORD *)(v33 + 64) = 0x100000000LL;
        *(_QWORD *)(v33 + 88) = 0x100000000LL;
        *(_QWORD *)(v33 + 104) = 0;
        if ( v35 != &src )
          j_j___libc_free_0(v35, src + 1);
        v36 = v33 + 112;
        *(_QWORD *)(v33 + 120) = v33 + 112;
        *(_QWORD *)v33 = &unk_49F7110;
        *(_QWORD *)(v33 + 112) = (v33 + 112) | 4;
      }
      else
      {
        v36 = 112;
      }
      v37 = *(unsigned int *)(v8 + 88);
      if ( (unsigned int)v37 >= *(_DWORD *)(v8 + 92) )
      {
        sub_16CD150(v8 + 80, (const void *)(v8 + 96), 0, 8, v31, v32);
        v37 = *(unsigned int *)(v8 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(v8 + 80) + 8 * v37) = v33;
      ++*(_DWORD *)(v8 + 88);
      v38 = *(unsigned int *)(v33 + 64);
      if ( (unsigned int)v38 >= *(_DWORD *)(v33 + 68) )
      {
        sub_16CD150(v33 + 56, (const void *)(v33 + 72), 0, 8, v31, v32);
        v38 = *(unsigned int *)(v33 + 64);
      }
      v39 = &v198;
      *(_QWORD *)(*(_QWORD *)(v33 + 56) + 8 * v38) = v8;
      ++*(_DWORD *)(v33 + 64);
      v40 = *(_QWORD *)(v8 + 48);
      v173 = 0;
      *(_QWORD *)(v33 + 48) = v40;
      v174 = 0;
      a2[12] = v33;
      a2[13] = v36;
      v175 = 0;
      sub_1580910(&v198);
      v188 = 0;
      v186 = v198;
      if ( v201 )
      {
        v39 = (__m128i *)v187;
        v201(v187, &src, 2);
        v189 = v202;
        v188 = (void (__fastcall *)(_WORD *, _WORD *, __int64))v201;
      }
      v192 = 0;
      v190[0] = v203;
      v190[1] = v204;
      if ( v206 )
      {
        v39 = (__m128i *)v191;
        v206(v191, v205, 2);
        v193 = v207;
        v192 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v206;
      }
      v165 = v33;
      while ( 1 )
      {
        v41 = v186.m128i_i64[0];
        v42 = v186.m128i_i64[0];
        if ( v186.m128i_i64[0] == v190[0] )
          break;
        while ( 1 )
        {
          if ( !v42 )
          {
            v172[0] = 0;
            BUG();
          }
          v43 = v42 - 24;
          v172[0] = v42 - 24;
          if ( *(_BYTE *)(v42 - 8) == 26 )
            goto LABEL_43;
          v44 = *(_QWORD **)(a5 + 16);
          v45 = *(_QWORD **)(a5 + 8);
          if ( v44 == v45 )
          {
            v76 = &v45[*(unsigned int *)(a5 + 28)];
            if ( v45 == v76 )
            {
              v46 = *(_QWORD **)(a5 + 8);
            }
            else
            {
              do
              {
                if ( v43 == *v45 )
                  break;
                ++v45;
              }
              while ( v76 != v45 );
              v46 = v76;
            }
            goto LABEL_95;
          }
          v39 = (__m128i *)a5;
          v46 = &v44[*(unsigned int *)(a5 + 24)];
          v45 = sub_16CC9F0(a5, v42 - 24);
          if ( v43 == *v45 )
          {
            v77 = *(_QWORD *)(a5 + 16);
            if ( v77 == *(_QWORD *)(a5 + 8) )
              v78 = *(unsigned int *)(a5 + 28);
            else
              v78 = *(unsigned int *)(a5 + 24);
            v76 = (_QWORD *)(v77 + 8 * v78);
            goto LABEL_95;
          }
          v47 = *(_QWORD *)(a5 + 16);
          if ( v47 == *(_QWORD *)(a5 + 8) )
          {
            v76 = (_QWORD *)(v47 + 8LL * *(unsigned int *)(a5 + 28));
            v45 = v76;
LABEL_95:
            while ( v76 != v45 && *v45 >= 0xFFFFFFFFFFFFFFFELL )
              ++v45;
            goto LABEL_42;
          }
          v45 = (_QWORD *)(v47 + 8LL * *(unsigned int *)(a5 + 24));
LABEL_42:
          v41 = v186.m128i_i64[0];
          if ( v46 != v45 )
            goto LABEL_43;
          v59 = v172[0];
          v39 = (__m128i *)a2[5];
          v60 = v39[24].m128i_i64[0];
          v61 = *(unsigned int *)(v60 + 72);
          if ( !(_DWORD)v61 )
            goto LABEL_82;
          v62 = *(_QWORD *)(v60 + 56);
          v63 = v61 - 1;
          v64 = (v61 - 1) & ((LODWORD(v172[0]) >> 9) ^ (LODWORD(v172[0]) >> 4));
          v65 = (__int64 *)(v62 + 16LL * v64);
          v66 = *v65;
          if ( v172[0] != *v65 )
          {
            v126 = *v65;
            v127 = (v61 - 1) & ((LODWORD(v172[0]) >> 9) ^ (LODWORD(v172[0]) >> 4));
            v128 = 1;
            while ( v126 != -8 )
            {
              v129 = v128 + 1;
              v130 = v63 & (unsigned int)(v127 + v128);
              v127 = v130;
              v126 = *(_QWORD *)(v62 + 16 * v130);
              if ( v172[0] == v126 )
              {
                v131 = 1;
                while ( v66 != -8 )
                {
                  v147 = v131 + 1;
                  v148 = v63 & (v64 + v131);
                  v64 = v148;
                  v65 = (__int64 *)(v62 + 16 * v148);
                  v66 = *v65;
                  if ( v172[0] == *v65 )
                    goto LABEL_79;
                  v131 = v147;
                }
                v65 = (__int64 *)(v62 + 16 * v61);
                goto LABEL_79;
              }
              v128 = v129;
            }
            goto LABEL_82;
          }
LABEL_79:
          v67 = v65[1];
          if ( !v67 || v172[0] == *(_QWORD *)(v67 + 56) || (unsigned int)*a3 <= 1 )
            goto LABEL_82;
          if ( (unsigned int)sub_1B99570((__int64)v39, v172[0], *a3) != 3 )
          {
            v59 = v172[0];
LABEL_82:
            v68 = *(unsigned int *)(v155 + 360);
            if ( (_DWORD)v68 )
            {
              v69 = *(_QWORD *)(v155 + 344);
              v70 = 1;
              v71 = (v68 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
              v72 = (_QWORD *)(v69 + 16LL * v71);
              v39 = (__m128i *)*v72;
              if ( *v72 == v59 )
              {
LABEL_84:
                if ( v72 != (_QWORD *)(v69 + 16 * v68) )
                {
                  if ( v185 )
                  {
                    v73 = (__m128i *)v72[1];
                    v74 = (v185 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
                    v75 = (__m128i **)&v183[v74];
                    v39 = *v75;
                    if ( v73 == *v75 )
                    {
LABEL_87:
                      v75[1] = (__m128i *)v59;
                      v41 = v186.m128i_i64[0];
                      goto LABEL_43;
                    }
                    v158 = 1;
                    v132 = 0;
                    while ( v39 != (__m128i *)-8LL )
                    {
                      if ( !v132 && v39 == (__m128i *)-16LL )
                        v132 = v75;
                      v74 = (v185 - 1) & (v158 + v74);
                      v75 = (__m128i **)&v183[v74];
                      v39 = *v75;
                      if ( v73 == *v75 )
                        goto LABEL_87;
                      ++v158;
                    }
                    if ( v132 )
                      v75 = v132;
                    ++v182;
                    v133 = v184 + 1;
                    if ( 4 * ((int)v184 + 1) < 3 * v185 )
                    {
                      if ( v185 - HIDWORD(v184) - v133 > v185 >> 3 )
                      {
LABEL_176:
                        LODWORD(v184) = v133;
                        if ( *v75 != (__m128i *)-8LL )
                          --HIDWORD(v184);
                        v134 = (__m128i *)v72[1];
                        v75[1] = 0;
                        *v75 = v134;
                        v59 = v172[0];
                        goto LABEL_87;
                      }
                      v39 = (__m128i *)&v182;
                      v144 = v185;
LABEL_202:
                      sub_1A63AC0((__int64)&v182, v144);
                      sub_1BA1140((__int64)&v182, v72 + 1, &v176);
                      v75 = (__m128i **)v176.m128i_i64[0];
                      v133 = v184 + 1;
                      goto LABEL_176;
                    }
                  }
                  else
                  {
                    ++v182;
                  }
                  v144 = 2 * v185;
                  v39 = (__m128i *)&v182;
                  goto LABEL_202;
                }
              }
              else
              {
                while ( v39 != (__m128i *)-8LL )
                {
                  v71 = (v68 - 1) & (v70 + v71);
                  v72 = (_QWORD *)(v69 + 16LL * v71);
                  v39 = (__m128i *)*v72;
                  if ( *v72 == v59 )
                    goto LABEL_84;
                  ++v70;
                }
              }
            }
            v79 = v174;
            if ( v174 == v175 )
            {
              v39 = (__m128i *)&v173;
              sub_170B610((__int64)&v173, v174, v172);
              v59 = v172[0];
            }
            else
            {
              if ( v174 )
              {
                *v174 = v59;
                v79 = v174;
                v59 = v172[0];
              }
              v174 = v79 + 1;
            }
            if ( v185 )
            {
              v39 = v183;
              v80 = (v185 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
              v81 = &v183[v80];
              v82 = v81->m128i_i64[0];
              if ( v59 == v81->m128i_i64[0] )
              {
LABEL_109:
                if ( v81 != &v183[v185] )
                {
                  v83 = v174;
                  if ( v174 != v175 )
                  {
                    if ( v174 )
                    {
                      *v174 = v81->m128i_i64[1];
                      v83 = v174;
                    }
LABEL_113:
                    v41 = v186.m128i_i64[0];
                    v174 = v83 + 1;
                    goto LABEL_43;
                  }
                  v39 = (__m128i *)&v173;
                  sub_170B610((__int64)&v173, v174, &v81->m128i_i64[1]);
                }
              }
              else
              {
                v135 = 1;
                while ( v82 != -8 )
                {
                  v136 = v135 + 1;
                  v80 = (v185 - 1) & (v135 + v80);
                  v81 = &v183[v80];
                  v82 = v81->m128i_i64[0];
                  if ( v59 == v81->m128i_i64[0] )
                    goto LABEL_109;
                  v135 = v136;
                }
              }
            }
LABEL_158:
            v41 = v186.m128i_i64[0];
            goto LABEL_43;
          }
          if ( !v185 )
            goto LABEL_158;
          v39 = (__m128i *)v172[0];
          v140 = v185 - 1;
          v141 = (v185 - 1) & ((LODWORD(v172[0]) >> 9) ^ (LODWORD(v172[0]) >> 4));
          v142 = &v183[v141];
          v143 = v142->m128i_i64[0];
          if ( v172[0] != v142->m128i_i64[0] )
          {
            v149 = v142->m128i_i64[0];
            LODWORD(v150) = (v185 - 1) & ((LODWORD(v172[0]) >> 9) ^ (LODWORD(v172[0]) >> 4));
            v151 = 1;
            while ( v149 != -8 )
            {
              v150 = v140 & ((_DWORD)v150 + v151);
              v149 = v183[v150].m128i_i64[0];
              if ( v172[0] == v149 )
              {
                v152 = 1;
                while ( v143 != -8 )
                {
                  v153 = v152 + 1;
                  v141 = v140 & (v152 + v141);
                  v142 = &v183[v141];
                  v143 = v142->m128i_i64[0];
                  if ( v172[0] == v142->m128i_i64[0] )
                    goto LABEL_197;
                  v152 = v153;
                }
                v142 = &v183[v185];
                goto LABEL_197;
              }
              ++v151;
            }
            goto LABEL_158;
          }
LABEL_197:
          v83 = v174;
          if ( v174 != v175 )
          {
            if ( v174 )
            {
              *v174 = v142->m128i_i64[1];
              v83 = v174;
            }
            goto LABEL_113;
          }
          v39 = (__m128i *)&v173;
          sub_170B610((__int64)&v173, v174, &v142->m128i_i64[1]);
          v41 = v186.m128i_i64[0];
LABEL_43:
          v41 = *(_QWORD *)(v41 + 8);
          v186.m128i_i64[0] = v41;
          v42 = v41;
          if ( v41 != v186.m128i_i64[1] )
            break;
LABEL_49:
          if ( v190[0] == v42 )
            goto LABEL_50;
        }
        while ( 1 )
        {
          if ( v42 )
            v42 -= 24;
          if ( !v188 )
            sub_4263D6(v39, v42, v41);
          v39 = (__m128i *)v187;
          if ( ((unsigned __int8 (__fastcall *)(_WORD *, __int64))v189)(v187, v42) )
            break;
          v42 = *(_QWORD *)(v186.m128i_i64[0] + 8);
          v186.m128i_i64[0] = v42;
          v41 = v42;
          if ( v186.m128i_i64[1] == v42 )
            goto LABEL_49;
        }
      }
LABEL_50:
      if ( v192 )
        v192(v191, v191, 3);
      if ( v188 )
        v188(v187, v187, 3);
      if ( v206 )
        v206(v205, v205, 3);
      if ( v201 )
        v201(&src, &src, 3);
      v48 = v173;
      v162 = v174;
      if ( v174 != v173 )
        break;
      v8 = v165;
LABEL_115:
      if ( v48 )
        j_j___libc_free_0(v48, (char *)v175 - (char *)v48);
      v161 -= 8;
      if ( v154 == v161 )
        goto LABEL_118;
    }
    v8 = v165;
    v49 = v173;
    v157 = 0;
    while ( 1 )
    {
      v168 = *v49;
      if ( !(unsigned __int8)sub_1BAA5F0((unsigned __int64)v208, *v49, a3, a1, v8) )
      {
        v54 = sub_1BAAE10((__int64)v208, v168, a3, v8, (__int64)&v178, (__int64)a1);
        if ( v54 != v8 )
          break;
      }
LABEL_68:
      if ( v162 == ++v49 )
      {
        v48 = v173;
        goto LABEL_115;
      }
    }
    if ( (*(_BYTE *)(v171 + 23) & 0x20) != 0 )
    {
      v169 = v54;
      v187[0] = 265;
      v186.m128i_i32[0] = v157;
      v55 = sub_1649960(v171);
      v54 = v169;
      v172[0] = (__int64)v55;
      v176.m128i_i64[0] = (__int64)v172;
      v176.m128i_i64[1] = (__int64)".";
      v56 = v187[0];
      v172[1] = v57;
      LOWORD(v177) = 773;
      if ( LOBYTE(v187[0]) )
      {
        if ( LOBYTE(v187[0]) == 1 )
        {
          v145 = _mm_loadu_si128(&v176);
          ++v157;
          v191[0] = v177;
          *(__m128i *)v190 = v145;
        }
        else
        {
          v58 = (__m128i *)v186.m128i_i64[0];
          if ( HIBYTE(v187[0]) != 1 )
          {
            v58 = &v186;
            v56 = 2;
          }
          v190[1] = (size_t)v58;
          v190[0] = (size_t)&v176;
          LOBYTE(v191[0]) = 2;
          BYTE1(v191[0]) = v56;
          ++v157;
        }
      }
      else
      {
        ++v157;
        LOWORD(v191[0]) = 256;
      }
    }
    else
    {
      LOWORD(v191[0]) = 257;
    }
    v166 = (_QWORD *)v54;
    sub_16E2FC0(v198.m128i_i64, (__int64)v190);
    v50 = v166;
    p_src = (_BYTE *)v166[2];
    if ( (_BYTE **)v198.m128i_i64[0] == &src )
    {
      v125 = v198.m128i_i64[1];
      if ( v198.m128i_i64[1] )
      {
        if ( v198.m128i_i64[1] == 1 )
        {
          *p_src = (_BYTE)src;
        }
        else
        {
          memcpy(p_src, &src, v198.m128i_u64[1]);
          v50 = v166;
        }
        v125 = v198.m128i_i64[1];
        p_src = (_BYTE *)v166[2];
      }
      v50[3] = v125;
      p_src[v125] = 0;
      p_src = (_BYTE *)v198.m128i_i64[0];
      goto LABEL_65;
    }
    v52 = v198.m128i_i64[1];
    if ( p_src == (_BYTE *)(v166 + 4) )
    {
      v166[2] = v198.m128i_i64[0];
      v166[3] = v52;
      v166[4] = src;
    }
    else
    {
      v166[2] = v198.m128i_i64[0];
      v53 = (_BYTE *)v166[4];
      v166[3] = v52;
      v166[4] = src;
      if ( p_src )
      {
        v198.m128i_i64[0] = (__int64)p_src;
        src = v53;
LABEL_65:
        v198.m128i_i64[1] = 0;
        *p_src = 0;
        if ( (_BYTE **)v198.m128i_i64[0] != &src )
        {
          v167 = v50;
          j_j___libc_free_0(v198.m128i_i64[0], src + 1);
          v50 = v167;
        }
        v8 = (__int64)v50;
        goto LABEL_68;
      }
    }
    v198.m128i_i64[0] = (__int64)&src;
    p_src = &src;
    goto LABEL_65;
  }
LABEL_118:
  v84 = 0;
  v85 = *(_QWORD *)*a1;
  if ( *(_DWORD *)(v85 + 88) == 1 )
    v84 = **(_QWORD **)(v85 + 80);
  *(_QWORD *)*a1 = v84;
  v198.m128i_i64[0] = v84;
  v86 = *(_QWORD **)(v85 + 80);
  v87 = (__int64)&v86[*(unsigned int *)(v85 + 88)];
  v88 = sub_1B8E5C0(v86, v87, v198.m128i_i64);
  if ( (_QWORD *)v87 != v88 + 1 )
  {
    memmove(v88, v88 + 1, v87 - (_QWORD)(v88 + 1));
    v89 = *(_DWORD *)(v85 + 88);
  }
  *(_DWORD *)(v85 + 88) = v89 - 1;
  v198.m128i_i64[0] = v85;
  v90 = *(_QWORD **)(v84 + 56);
  v91 = (__int64)&v90[*(unsigned int *)(v84 + 64)];
  v92 = sub_1B8E5C0(v90, v91, v198.m128i_i64);
  if ( (_QWORD *)v91 != v92 + 1 )
  {
    memmove(v92, v92 + 1, v91 - (_QWORD)(v92 + 1));
    v93 = *(_DWORD *)(v84 + 64);
  }
  *(_DWORD *)(v84 + 64) = v93 - 1;
  v94 = *(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)v85 + 8LL);
  if ( v94 == sub_1B8F6B0 )
  {
    v95 = *(unsigned __int64 **)(v85 + 120);
    v96 = (unsigned __int64 *)(v85 + 112);
    *(_QWORD *)v85 = &unk_49F7110;
    if ( (unsigned __int64 *)(v85 + 112) != v95 )
    {
      do
      {
        v97 = v95;
        v95 = (unsigned __int64 *)v95[1];
        v98 = *v97 & 0xFFFFFFFFFFFFFFF8LL;
        *v95 = v98 | *v95 & 7;
        *(_QWORD *)(v98 + 8) = v95;
        v97[1] = 0;
        *v97 &= 7u;
        (*(void (__fastcall **)(unsigned __int64 *))(*(v97 - 1) + 8))(v97 - 1);
      }
      while ( v96 != v95 );
      v99 = *(unsigned __int64 **)(v85 + 120);
      while ( v96 != v99 )
      {
        v100 = v99;
        v99 = (unsigned __int64 *)v99[1];
        v101 = *v100 & 0xFFFFFFFFFFFFFFF8LL;
        *v99 = v101 | *v99 & 7;
        *(_QWORD *)(v101 + 8) = v99;
        v100[1] = 0;
        *v100 &= 7u;
        (*(void (__fastcall **)(unsigned __int64 *))(*(v100 - 1) + 8))(v100 - 1);
      }
    }
    v102 = *(_QWORD *)(v85 + 80);
    *(_QWORD *)v85 = &unk_49F6D50;
    if ( v102 != v85 + 96 )
      _libc_free(v102);
    v103 = *(_QWORD *)(v85 + 56);
    if ( v103 != v85 + 72 )
      _libc_free(v103);
    sub_2240A30(v85 + 16);
    j_j___libc_free_0(v85, 128);
  }
  else
  {
    v94((_QWORD *)v85);
  }
  v186.m128i_i64[1] = 0;
  v186.m128i_i64[0] = (__int64)v187;
  v107 = *a3;
  LOBYTE(v187[0]) = 0;
  LODWORD(v201) = 1;
  v198.m128i_i64[0] = (__int64)&unk_49EFBE0;
  LODWORD(v190[0]) = v107;
  v202 = &v186;
  v200 = 0;
  src = 0;
  v108 = *a1;
  v198.m128i_i64[1] = 0;
  sub_1B94BE0(v108 + 8, (unsigned int *)v190, v104, v105, v106);
  v109 = sub_1263B40((__int64)&v198, "Initial VPlan for VF={");
  v110 = v107;
  v111 = 2 * v107;
  sub_16E7A90(v109, v110);
  while ( a3[1] > v111 )
  {
    while ( 1 )
    {
      v116 = *a1;
      LODWORD(v190[0]) = v111;
      sub_1B94BE0(v116 + 8, (unsigned int *)v190, v112, v113, v114);
      if ( src == v200 )
        break;
      *v200 = 44;
      v115 = v111;
      v111 *= 2;
      ++v200;
      sub_16E7A90((__int64)&v198, v115);
      if ( a3[1] <= v111 )
        goto LABEL_139;
    }
    v117 = sub_16E7EE0((__int64)&v198, ",", 1u);
    v118 = v111;
    v111 *= 2;
    sub_16E7A90(v117, v118);
  }
LABEL_139:
  sub_1263B40((__int64)&v198, "},UF>=1");
  if ( v200 != (_BYTE *)v198.m128i_i64[1] )
    sub_16E7BA0(v198.m128i_i64);
  v119 = (_QWORD *)*a1;
  LOWORD(v177) = 260;
  v176.m128i_i64[0] = (__int64)&v186;
  sub_16E2FC0((__int64 *)v190, (__int64)&v176);
  v120 = (_BYTE *)v119[10];
  if ( (_QWORD *)v190[0] == v191 )
  {
    v146 = v190[1];
    if ( v190[1] )
    {
      if ( v190[1] == 1 )
        *v120 = v191[0];
      else
        memcpy(v120, (const void *)v190[0], v190[1]);
      v146 = v190[1];
      v120 = (_BYTE *)v119[10];
    }
    v119[11] = v146;
    v120[v146] = 0;
    v120 = (_BYTE *)v190[0];
  }
  else
  {
    v121 = v190[1];
    v122 = v191[0];
    if ( v120 == (_BYTE *)(v119 + 12) )
    {
      v119[10] = v190[0];
      v119[11] = v121;
      v119[12] = v122;
    }
    else
    {
      v123 = v119[12];
      v119[10] = v190[0];
      v119[11] = v121;
      v119[12] = v122;
      if ( v120 )
      {
        v190[0] = (size_t)v120;
        v191[0] = v123;
        goto LABEL_145;
      }
    }
    v190[0] = (size_t)v191;
    v120 = v191;
  }
LABEL_145:
  v190[1] = 0;
  *v120 = 0;
  sub_2240A30(v190);
  sub_16E7BC0(v198.m128i_i64);
  sub_2240A30(&v186);
  if ( v195 )
    j_j___libc_free_0(v195, v197 - v195);
  j___libc_free_0(v194[2]);
  j___libc_free_0(v218);
  j___libc_free_0(v214);
  j___libc_free_0(v183);
  j___libc_free_0(v179);
  return a1;
}
