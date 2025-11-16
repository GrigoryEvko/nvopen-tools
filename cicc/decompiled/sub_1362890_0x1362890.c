// Function: sub_1362890
// Address: 0x1362890
//
__int64 __fastcall sub_1362890(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned __int8 v11; // al
  unsigned __int8 v14; // al
  __int64 v15; // r15
  unsigned __int8 v16; // al
  unsigned __int8 v17; // al
  __int64 v18; // r13
  int v19; // r12d
  unsigned int v20; // r15d
  __int64 v21; // rsi
  _BYTE *v22; // r12
  _BYTE *v23; // r13
  int v24; // eax
  unsigned __int8 v25; // al
  __int64 v26; // rdi
  __int64 v27; // r14
  bool v28; // bl
  size_t v29; // rdx
  char *v30; // r14
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r8d
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned int v37; // r8d
  __int8 v38; // bl
  __int64 v39; // r15
  __int64 v40; // r14
  __int64 v41; // r15
  __int64 v42; // r14
  __m128i v43; // xmm5
  __m128i v44; // xmm6
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __m128i v49; // xmm4
  __int64 v50; // r14
  unsigned __int64 v52; // r9
  unsigned int v53; // esi
  unsigned int v54; // edi
  char v55; // al
  int v56; // ecx
  __int64 v57; // r13
  _QWORD *v58; // r12
  __int32 v59; // ecx
  __int64 v60; // rax
  unsigned __int64 v61; // r8
  _QWORD *v62; // r13
  unsigned __int64 v63; // rax
  __int128 *v64; // rdx
  unsigned __int64 v65; // r13
  __int64 *v66; // r12
  __int64 *v67; // r13
  __m128i *v68; // rdx
  __int32 v69; // ecx
  __int64 v70; // r13
  __int64 v71; // r12
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // r13
  __int64 v75; // r12
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // r12
  __int64 v79; // r13
  __int64 v80; // rdx
  _QWORD *v81; // r12
  __int128 *v82; // rax
  unsigned __int64 v83; // r8
  __int64 v84; // rax
  unsigned __int64 v85; // r12
  unsigned __int64 v86; // rdx
  __int64 v87; // r13
  __int64 *v88; // r12
  unsigned __int64 v89; // rcx
  __m128i *v90; // rax
  int v91; // r8d
  __int64 v92; // r13
  __int64 v93; // r12
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // r13
  __int64 v97; // r12
  __int64 v98; // rax
  __int64 v99; // rcx
  __int64 v100; // rcx
  __int64 v101; // rsi
  __int64 v102; // rax
  _QWORD *v103; // r12
  _QWORD *v104; // r13
  unsigned int v105; // eax
  int v106; // eax
  unsigned int v107; // eax
  int v108; // eax
  __int64 v109; // r12
  unsigned int v110; // eax
  unsigned __int64 v111; // r13
  __int64 v112; // rax
  __int64 v113; // r8
  __int64 v114; // rax
  size_t v115; // rdx
  char *v116; // r14
  __int64 v117; // rax
  __int64 *v118; // rax
  unsigned __int8 v119; // al
  unsigned __int8 v120; // al
  unsigned __int8 v121; // al
  char v122; // al
  unsigned int v123; // eax
  char v124; // al
  __int64 v125; // rdx
  __int64 *v126; // rax
  __int64 v127; // rcx
  __m128i *v128; // rsi
  __int64 *v129; // rdi
  _DWORD *v130; // rdi
  __int64 v131; // rcx
  __int64 *v132; // rsi
  __int64 v133; // rsi
  _BYTE *v134; // rt1
  __m128i v135; // xmm6
  unsigned __int64 v136; // rsi
  unsigned __int64 v137; // rax
  char v138; // al
  __int64 *v139; // rax
  __int64 v140; // rcx
  __m128i *v141; // rsi
  __int64 *v142; // rdi
  _DWORD *v143; // rdi
  __int64 *v144; // rsi
  __int64 i; // rcx
  __int64 v146; // rsi
  _BYTE *v147; // rt2
  __m128i v148; // xmm5
  unsigned __int64 v149; // rdi
  unsigned __int64 v150; // rax
  unsigned __int64 v151; // rdi
  __m128i v152; // xmm7
  _BYTE *v153; // rtt
  __int64 v154; // rax
  __int64 v155; // rcx
  unsigned __int64 v156; // rdi
  int v157; // edx
  __int64 v158; // rax
  __int64 v159; // rax
  unsigned __int64 v160; // r12
  __int64 v161; // rcx
  __int64 v162; // rsi
  _QWORD *v163; // rax
  _QWORD *v164; // rdx
  __int64 v165; // rax
  unsigned __int64 v166; // rdx
  __int64 v167; // r12
  __int64 v168; // rax
  size_t v169; // rdx
  const char *v170; // r15
  __int64 v171; // r14
  __int64 v172; // rax
  size_t v173; // rdx
  __int64 v174; // rbx
  size_t v175; // rdx
  __int64 v176; // r15
  size_t v177; // rdx
  __int64 v178; // rax
  __int64 v179; // rcx
  __int64 v180; // rax
  unsigned __int64 v181; // [rsp+0h] [rbp-220h]
  unsigned __int64 v182; // [rsp+0h] [rbp-220h]
  __int64 v183; // [rsp+8h] [rbp-218h]
  unsigned __int64 v184; // [rsp+8h] [rbp-218h]
  int v185; // [rsp+8h] [rbp-218h]
  unsigned __int64 v186; // [rsp+8h] [rbp-218h]
  __int64 v187; // [rsp+10h] [rbp-210h]
  unsigned __int64 v188; // [rsp+10h] [rbp-210h]
  __int64 v189; // [rsp+18h] [rbp-208h]
  unsigned __int64 v190; // [rsp+18h] [rbp-208h]
  __int64 v191; // [rsp+20h] [rbp-200h]
  int v192; // [rsp+20h] [rbp-200h]
  unsigned __int64 v193; // [rsp+20h] [rbp-200h]
  unsigned int v194; // [rsp+28h] [rbp-1F8h]
  __int64 v195; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v196; // [rsp+28h] [rbp-1F8h]
  int v197; // [rsp+28h] [rbp-1F8h]
  unsigned int v198; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v199; // [rsp+38h] [rbp-1E8h]
  __int64 v200; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v202; // [rsp+48h] [rbp-1D8h]
  unsigned __int64 v203; // [rsp+50h] [rbp-1D0h]
  unsigned __int64 v204; // [rsp+58h] [rbp-1C8h]
  unsigned __int64 v205; // [rsp+60h] [rbp-1C0h]
  unsigned __int64 *v207; // [rsp+70h] [rbp-1B0h] BYREF
  unsigned int v208; // [rsp+78h] [rbp-1A8h]
  __int64 *v209; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v210; // [rsp+88h] [rbp-198h]
  _QWORD v211[8]; // [rsp+90h] [rbp-190h] BYREF
  __m128i v212; // [rsp+D0h] [rbp-150h] BYREF
  __int128 v213; // [rsp+E0h] [rbp-140h] BYREF
  __m128i v214; // [rsp+F0h] [rbp-130h] BYREF
  unsigned __int64 v215; // [rsp+100h] [rbp-120h] BYREF
  __int128 v216; // [rsp+108h] [rbp-118h] BYREF
  __int64 v217; // [rsp+118h] [rbp-108h]
  __m128i v218; // [rsp+160h] [rbp-C0h] BYREF
  __m128i v219; // [rsp+170h] [rbp-B0h] BYREF
  __m128i v220; // [rsp+180h] [rbp-A0h] BYREF
  __m128i v221; // [rsp+190h] [rbp-90h] BYREF
  __m128i v222; // [rsp+1A0h] [rbp-80h]
  char v223; // [rsp+1B0h] [rbp-70h]

  v205 = a3;
  v204 = a5;
  if ( !a3 || !a5 )
    return 0;
  v11 = *(_BYTE *)(a2 + 16);
  if ( v11 > 0x17u )
  {
    if ( v11 != 56 )
      goto LABEL_5;
LABEL_74:
    v16 = *(_BYTE *)(a4 + 16);
    v15 = a2;
    v203 = 0;
    if ( v16 > 0x17u )
      goto LABEL_10;
    goto LABEL_75;
  }
  if ( v11 == 5 && *(_WORD *)(a2 + 18) == 32 )
    goto LABEL_74;
LABEL_5:
  v203 = sub_1CCAE90(a2, 0);
  v14 = *(_BYTE *)(v203 + 16);
  if ( v14 > 0x17u )
  {
    v15 = 0;
    if ( v14 == 56 )
      v15 = v203;
  }
  else
  {
    v15 = 0;
    if ( v14 == 5 && *(_WORD *)(v203 + 18) == 32 )
      v15 = v203;
  }
  v16 = *(_BYTE *)(a4 + 16);
  if ( v16 > 0x17u )
  {
LABEL_10:
    if ( v16 != 56 )
      goto LABEL_11;
LABEL_77:
    v202 = 0;
    v52 = a4;
    goto LABEL_78;
  }
LABEL_75:
  if ( v16 == 5 && *(_WORD *)(a4 + 18) == 32 )
    goto LABEL_77;
LABEL_11:
  v202 = sub_1CCAE90(a4, 0);
  v17 = *(_BYTE *)(v202 + 16);
  if ( v17 <= 0x17u )
  {
    if ( v17 != 5 )
      goto LABEL_13;
    v52 = v202;
    if ( *(_WORD *)(v202 + 18) != 32 )
      goto LABEL_13;
  }
  else
  {
    if ( v17 != 56 )
      goto LABEL_13;
    v52 = v202;
  }
LABEL_78:
  if ( !byte_4F97E20 || v205 == -1 || v15 == 0 || v204 == -1 )
    goto LABEL_13;
  v53 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  v54 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
  v198 = v53;
  v55 = *(_BYTE *)(v15 + 23) & 0x40;
  if ( v53 != v54 )
  {
    v56 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
    if ( v53 <= v54 )
      v56 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
    v194 = v56;
LABEL_85:
    if ( v55 )
      v57 = *(_QWORD *)(v15 - 8);
    else
      v57 = v15 - 24LL * v53;
    v58 = (_QWORD *)(v57 + 24);
    v212.m128i_i64[1] = 0x1000000000LL;
    v59 = 0;
    v60 = 24LL * v194;
    v212.m128i_i64[0] = (__int64)&v213;
    v61 = v60 - 24;
    v62 = (_QWORD *)(v60 + v57);
    v63 = 0xAAAAAAAAAAAAAAABLL * ((v60 - 24) >> 3);
    v64 = &v213;
    if ( v61 > 0x180 )
    {
      v181 = v61;
      v184 = v52;
      v188 = v63;
      sub_16CD150(&v212, &v213, v63, 8);
      v59 = v212.m128i_i32[2];
      v61 = v181;
      v52 = v184;
      v63 = v188;
      v64 = (__int128 *)(v212.m128i_i64[0] + 8LL * v212.m128i_u32[2]);
    }
    if ( v62 != v58 )
    {
      do
      {
        if ( v64 )
          *(_QWORD *)v64 = *v58;
        v58 += 3;
        v64 = (__int128 *)((char *)v64 + 8);
      }
      while ( v62 != v58 );
      v59 = v212.m128i_i32[2];
    }
    v212.m128i_i32[2] = v63 + v59;
    if ( (*(_BYTE *)(v52 + 23) & 0x40) != 0 )
      v65 = *(_QWORD *)(v52 - 8);
    else
      v65 = v52 - 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
    v66 = (__int64 *)(v65 + 24);
    v67 = (__int64 *)(24LL * v194 + v65);
    v218.m128i_i64[1] = 0x1000000000LL;
    v68 = &v219;
    v69 = 0;
    v218.m128i_i64[0] = (__int64)&v219;
    if ( v61 > 0x180 )
    {
      v182 = v52;
      v185 = v63;
      sub_16CD150(&v218, &v219, v63, 8);
      v69 = v218.m128i_i32[2];
      v52 = v182;
      LODWORD(v63) = v185;
      v68 = (__m128i *)(v218.m128i_i64[0] + 8LL * v218.m128i_u32[2]);
    }
    if ( v67 != v66 )
    {
      do
      {
        if ( v68 )
          v68->m128i_i64[0] = *v66;
        v66 += 3;
        v68 = (__m128i *)((char *)v68 + 8);
      }
      while ( v67 != v66 );
      v69 = v218.m128i_i32[2];
    }
    v70 = v212.m128i_i64[0];
    v71 = v212.m128i_u32[2];
    v187 = v52;
    v218.m128i_i32[2] = v69 + v63;
    v72 = sub_16348C0(v15);
    v73 = sub_15F9F50(v72, v70, v71);
    v74 = v218.m128i_u32[2];
    v75 = v73;
    v183 = v218.m128i_i64[0];
    v76 = sub_16348C0(v187);
    if ( v75 != sub_15F9F50(v76, v183, v74) || *(_BYTE *)(v75 + 8) != 13 )
      goto LABEL_104;
    v110 = v54;
    if ( v53 >= v54 )
      v110 = v53;
    if ( v110 - v194 != 1 )
      goto LABEL_104;
    if ( v53 > v54 || (v111 = v205, v15 = v187, v53 >= v54) )
      v111 = v204;
    v112 = sub_1644900(*(_QWORD *)v75, 64);
    v211[0] = sub_159C470(v112, 0, 0);
    v209 = v211;
    v210 = 0x800000001LL;
    if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      v113 = *(_QWORD *)(v15 - 8);
    else
      v113 = v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
    v114 = *(_QWORD *)(v113 + 24LL * v194);
    LODWORD(v210) = 2;
    v211[1] = v114;
    if ( v111 > sub_15A9FF0(a1[1], v75, v211, 2) )
    {
      if ( v209 != v211 )
        _libc_free((unsigned __int64)v209);
      goto LABEL_104;
    }
    if ( v209 != v211 )
      _libc_free((unsigned __int64)v209);
    v151 = v218.m128i_i64[0];
    if ( (__m128i *)v218.m128i_i64[0] != &v219 )
      goto LABEL_239;
    goto LABEL_240;
  }
  v194 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
  if ( v53 <= 2 )
    goto LABEL_85;
  v77 = 24LL * v53;
  v78 = v15 - v77;
  if ( v55 )
    v78 = *(_QWORD *)(v15 - 8);
  v79 = v78 + v77 - 24;
  v80 = v77 - 48;
  v81 = (_QWORD *)(v78 + 24);
  v212.m128i_i64[1] = 0x1000000000LL;
  v212.m128i_i64[0] = (__int64)&v213;
  v82 = &v213;
  v83 = 0xAAAAAAAAAAAAAAABLL * (v80 >> 3);
  if ( (unsigned __int64)v80 > 0x180 )
  {
    v193 = v52;
    v197 = -1431655765 * (v80 >> 3);
    sub_16CD150(&v212, &v213, 0xAAAAAAAAAAAAAAABLL * (v80 >> 3), 8);
    v52 = v193;
    LODWORD(v83) = v197;
    v82 = (__int128 *)(v212.m128i_i64[0] + 8LL * v212.m128i_u32[2]);
  }
  for ( ; (_QWORD *)v79 != v81; v82 = (__int128 *)((char *)v82 + 8) )
  {
    if ( v82 )
      *(_QWORD *)v82 = *v81;
    v81 += 3;
  }
  v212.m128i_i32[2] += v83;
  v84 = 24LL * (*(_DWORD *)(v52 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v52 + 23) & 0x40) != 0 )
    v85 = *(_QWORD *)(v52 - 8);
  else
    v85 = v52 - v84;
  v86 = v84 - 48;
  v87 = v85 + v84 - 24;
  v88 = (__int64 *)(v85 + 24);
  v218.m128i_i64[1] = 0x1000000000LL;
  v218.m128i_i64[0] = (__int64)&v219;
  v89 = 0xAAAAAAAAAAAAAAABLL * ((v84 - 48) >> 3);
  v90 = &v219;
  v91 = v89;
  if ( v86 > 0x180 )
  {
    v190 = v52;
    v192 = v89;
    sub_16CD150(&v218, &v219, v89, 8);
    v52 = v190;
    v91 = v192;
    v90 = (__m128i *)(v218.m128i_i64[0] + 8LL * v218.m128i_u32[2]);
  }
  for ( ; (__int64 *)v87 != v88; v90 = (__m128i *)((char *)v90 + 8) )
  {
    if ( v90 )
      v90->m128i_i64[0] = *v88;
    v88 += 3;
  }
  v92 = v212.m128i_i64[0];
  v93 = v212.m128i_u32[2];
  v218.m128i_i32[2] += v91;
  v195 = v52;
  v94 = sub_16348C0(v15);
  v95 = sub_15F9F50(v94, v92, v93);
  v96 = v218.m128i_i64[0];
  v97 = v218.m128i_u32[2];
  v191 = v95;
  v98 = sub_16348C0(v195);
  if ( v191 != sub_15F9F50(v98, v96, v97) || *(_BYTE *)(v191 + 8) != 13 )
    goto LABEL_104;
  v99 = (*(_BYTE *)(v15 + 23) & 0x40) != 0 ? *(_QWORD *)(v15 - 8) : v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
  v100 = *(_QWORD *)(v99 + 24LL * (v53 - 1));
  v101 = (*(_BYTE *)(v195 + 23) & 0x40) != 0
       ? *(_QWORD *)(v195 - 8)
       : v195 - 24LL * (*(_DWORD *)(v195 + 20) & 0xFFFFFFF);
  if ( *(_BYTE *)(v100 + 16) != 13 )
    goto LABEL_104;
  v102 = *(_QWORD *)(v101 + 24LL * (v198 - 1));
  if ( *(_BYTE *)(v102 + 16) != 13 )
    goto LABEL_104;
  v103 = *(_QWORD **)(v100 + 24);
  if ( *(_DWORD *)(v100 + 32) > 0x40u )
    v103 = (_QWORD *)*v103;
  v104 = *(_QWORD **)(v102 + 24);
  if ( *(_DWORD *)(v102 + 32) > 0x40u )
    v104 = (_QWORD *)*v104;
  v189 = v195;
  v199 = *(_QWORD *)(sub_15A9930(a1[1], v191) + 8LL * (unsigned int)v103 + 16);
  v196 = *(_QWORD *)(sub_15A9930(a1[1], v191) + 8LL * (unsigned int)v104 + 16);
  v105 = sub_135E120(v15);
  v106 = sub_15A9520(a1[1], v105);
  sub_135E0D0((__int64)&v207, 8 * v106, 0, 0);
  v107 = sub_135E120(v189);
  v108 = sub_15A9520(a1[1], v107);
  sub_135E0D0((__int64)&v209, 8 * v108, 0, 0);
  v109 = sub_16348C0(v15);
  if ( v109 == sub_16348C0(v189)
    && (unsigned __int8)sub_1634900(v15, a1[1], &v207)
    && (unsigned __int8)sub_1634900(v189, a1[1], &v209) )
  {
    if ( v208 > 0x40 )
      v160 = *v207;
    else
      v160 = (__int64)((_QWORD)v207 << (64 - (unsigned __int8)v208)) >> (64 - (unsigned __int8)v208);
    v199 = v160;
    if ( (unsigned int)v210 > 0x40 )
      v161 = *v209;
    else
      v161 = (__int64)((_QWORD)v209 << (64 - (unsigned __int8)v210)) >> (64 - (unsigned __int8)v210);
    v196 = v161;
    v162 = *(_QWORD *)v218.m128i_i64[0];
    v163 = *(_QWORD **)(*(_QWORD *)v212.m128i_i64[0] + 24LL);
    if ( *(_DWORD *)(*(_QWORD *)v212.m128i_i64[0] + 32LL) > 0x40u )
      v163 = (_QWORD *)*v163;
    v164 = *(_QWORD **)(v162 + 24);
    if ( *(_DWORD *)(v162 + 32) > 0x40u )
      v164 = (_QWORD *)*v164;
    v186 = v161;
    if ( v164 != v163 )
    {
      v200 = a1[1];
      v165 = sub_16348C0(v15);
      v166 = v160 % sub_12BE0A0(v200, v165);
      v167 = a1[1];
      v199 = v166;
      v168 = sub_16348C0(v189);
      v196 = v186 % sub_12BE0A0(v167, v168);
    }
  }
  if ( v199 >= v196 )
  {
    if ( v199 <= v196 || v204 + v196 > v199 )
      goto LABEL_146;
LABEL_264:
    sub_135E100((__int64 *)&v209);
    sub_135E100((__int64 *)&v207);
    v151 = v218.m128i_i64[0];
    if ( (__m128i *)v218.m128i_i64[0] != &v219 )
LABEL_239:
      _libc_free(v151);
LABEL_240:
    if ( (__int128 *)v212.m128i_i64[0] != &v213 )
      _libc_free(v212.m128i_u64[0]);
    return 0;
  }
  if ( v205 + v199 <= v196 )
    goto LABEL_264;
LABEL_146:
  sub_135E100((__int64 *)&v209);
  sub_135E100((__int64 *)&v207);
LABEL_104:
  if ( (__m128i *)v218.m128i_i64[0] != &v219 )
    _libc_free(v218.m128i_u64[0]);
  if ( (__int128 *)v212.m128i_i64[0] != &v213 )
    _libc_free(v212.m128i_u64[0]);
LABEL_13:
  v18 = a1[1];
  v19 = sub_135D680((_QWORD *)a2, v18, (unsigned __int8)byte_4F97D40 ^ 1u);
  if ( ((unsigned int)sub_135D680((_QWORD *)a4, v18, (unsigned __int8)byte_4F97D40 ^ 1u) & v19) == 0 )
    return 0;
  if ( !v203 )
    v203 = sub_1CCAE90(a2, 0);
  if ( !v202 )
    v202 = sub_1CCAE90(a4, 0);
  if ( *(_BYTE *)(v203 + 16) == 9 || *(_BYTE *)(v202 + 16) == 9 )
    return 0;
  v20 = 3;
  if ( !(unsigned __int8)sub_1360E90((__int64)a1, v203, v202) )
  {
    if ( *(_BYTE *)(*(_QWORD *)v203 + 8LL) != 15 || *(_BYTE *)(*(_QWORD *)v202 + 8LL) != 15 )
      return 0;
    if ( !a6 )
      a6 = sub_14AD280(v203, a1[1], 6);
    if ( !a11 )
      a11 = sub_14AD280(v202, a1[1], 6);
    v21 = 0;
    v22 = (_BYTE *)sub_1CCAE90(a6, 0);
    v23 = (_BYTE *)sub_1CCAE90(a11, 0);
    if ( v22[16] == 15 )
    {
      v21 = *(_DWORD *)(*(_QWORD *)v22 + 8LL) >> 8;
      if ( !(unsigned __int8)sub_15E4690(a1[2], v21) )
        return 0;
    }
    if ( v23[16] == 15 )
    {
      v21 = *(_DWORD *)(*(_QWORD *)v23 + 8LL) >> 8;
      if ( !(unsigned __int8)sub_15E4690(a1[2], v21) )
        return 0;
    }
    if ( v22 == v23 )
      goto LABEL_57;
    if ( v22[16] == 3 && v23[16] == 3 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)a4 + 8LL) == 15 )
    {
      v24 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8;
      if ( *(_DWORD *)(*(_QWORD *)a4 + 8LL) >> 8 == v24 && v24 == 3 )
      {
        v158 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
        if ( *(_BYTE *)(v158 + 8) == 14 && !*(_QWORD *)(v158 + 32) )
        {
          v159 = *(_QWORD *)(*(_QWORD *)v23 + 24LL);
          if ( *(_BYTE *)(v159 + 8) == 14 && !*(_QWORD *)(v159 + 32) )
            return 1;
        }
      }
    }
    v25 = *(_BYTE *)(v202 + 16);
    if ( *(_BYTE *)(v203 + 16) <= 0x17u )
    {
      if ( v25 <= 0x17u )
        goto LABEL_49;
      v27 = *(_QWORD *)(*(_QWORD *)(v202 + 40) + 56LL);
      v28 = v27 != 0;
    }
    else
    {
      v26 = *(_QWORD *)(*(_QWORD *)(v203 + 40) + 56LL);
      if ( v25 <= 0x17u )
      {
        if ( !v26 || !(unsigned __int8)((__int64 (*)(void))sub_1CCAC90)() )
          goto LABEL_49;
        goto LABEL_43;
      }
      v27 = *(_QWORD *)(*(_QWORD *)(v202 + 40) + 56LL);
      v28 = v27 != 0;
      if ( v26 && (unsigned __int8)((__int64 (*)(void))sub_1CCAC90)() )
      {
        if ( v27 && (unsigned __int8)sub_1CCAC90(v27) )
        {
          v169 = 0;
          v170 = off_4CD4978[0];
          if ( off_4CD4978[0] )
            v169 = strlen(off_4CD4978[0]);
          v171 = *(_QWORD *)(v203 + 48);
          if ( v171 || *(__int16 *)(v203 + 18) < 0 )
          {
            v172 = sub_1625940(v203, v170, v169);
            v170 = off_4CD4978[0];
            v171 = v172;
          }
          v173 = 0;
          if ( v170 )
            v173 = strlen(v170);
          v174 = *(_QWORD *)(v202 + 48);
          if ( v174 || *(__int16 *)(v202 + 18) < 0 )
            v174 = sub_1625940(v202, v170, v173);
          v175 = 0;
          v21 = (__int64)off_4CD4970[0];
          if ( off_4CD4970[0] )
          {
            v21 = (__int64)off_4CD4970[0];
            v175 = strlen(off_4CD4970[0]);
          }
          v176 = *(_QWORD *)(v203 + 48);
          if ( v176 || *(__int16 *)(v203 + 18) < 0 )
          {
            v176 = sub_1625940(v203, v21, v175);
            v21 = (__int64)off_4CD4970[0];
          }
          v177 = 0;
          if ( v21 )
            v177 = strlen((const char *)v21);
          if ( *(_QWORD *)(v202 + 48) || *(__int16 *)(v202 + 18) < 0 )
          {
            v178 = sub_1625940(v202, v21, v177);
            if ( v176 )
            {
              v21 = 1LL - *(unsigned int *)(v176 + 8);
              v179 = *(_QWORD *)(v176 + 8 * v21);
              if ( v179 )
              {
                if ( v178 )
                {
                  v21 = *(unsigned int *)(v178 + 8);
                  v180 = *(_QWORD *)(v178 + 8 * (1 - v21));
                  if ( v179 == v180 && v180 && v174 | v171 )
                    return 0;
                }
              }
            }
          }
LABEL_49:
          if ( (unsigned __int8)sub_134E860((__int64)v22) && (unsigned __int8)sub_134E860((__int64)v23) )
            return 0;
          if ( v22[16] <= 0x10u && (unsigned __int8)sub_134E860((__int64)v23) )
          {
            if ( v23[16] > 0x10u )
              return 0;
          }
          else if ( v23[16] > 0x10u )
          {
            goto LABEL_52;
          }
          if ( (unsigned __int8)sub_134E860((__int64)v22) )
          {
            if ( v22[16] > 0x10u )
              return 0;
            goto LABEL_53;
          }
LABEL_52:
          if ( v22[16] == 17 && (unsigned __int8)sub_134E8E0((__int64)v23) )
            return 0;
LABEL_53:
          if ( v23[16] == 17 && (unsigned __int8)sub_134E8E0((__int64)v22) )
            return 0;
          if ( (unsigned __int8)sub_135DA10((__int64)v22, v21, v32, v33, v34) )
          {
            v21 = (__int64)v23;
            if ( (unsigned __int8)sub_1361F30(a1, (__int64)v23) )
              return 0;
          }
          if ( (unsigned __int8)sub_135DA10((__int64)v23, v21, v35, v36, v37)
            && (unsigned __int8)sub_1361F30(a1, (__int64)v22) )
          {
            return 0;
          }
LABEL_57:
          v38 = sub_15E4690(a1[2], 0);
          if ( v205 != -1 )
          {
            v39 = a1[3];
            v40 = a1[1];
            if ( (unsigned __int8)sub_134E860((__int64)v23) )
            {
              v212.m128i_i16[0] = 256;
              v212.m128i_i8[2] = v38;
              if ( (unsigned __int8)sub_140E950(v23, &v218, v40, v39, v212.m128i_i64[0]) )
              {
                if ( v205 > v218.m128i_i64[0] )
                  return 0;
              }
            }
          }
          if ( v204 != -1 )
          {
            v41 = a1[3];
            v42 = a1[1];
            if ( (unsigned __int8)sub_134E860((__int64)v22) )
            {
              v212.m128i_i16[0] = 256;
              v212.m128i_i8[2] = v38;
              if ( (unsigned __int8)sub_140E950(v22, &v218, v42, v41, v212.m128i_i64[0]) )
              {
                if ( v204 > v218.m128i_i64[0] )
                  return 0;
              }
            }
          }
          v215 = v204;
          v213 = a7;
          v216 = a9;
          v212.m128i_i64[0] = v203;
          v212.m128i_i64[1] = v205;
          v214.m128i_i64[0] = a8;
          v214.m128i_i64[1] = v202;
          v217 = a10;
          if ( v203 > v202 )
          {
            v43 = _mm_loadu_si128((const __m128i *)&v214.m128i_u64[1]);
            v44 = _mm_loadu_si128((const __m128i *)&v216);
            v214.m128i_i64[0] = a10;
            v214.m128i_i64[1] = v203;
            v215 = v205;
            v216 = a7;
            v217 = a8;
            v212 = v43;
            v213 = (__int128)v44;
          }
          v45 = _mm_loadu_si128(&v212);
          v46 = _mm_loadu_si128((const __m128i *)&v213);
          v47 = _mm_loadu_si128(&v214);
          v48 = _mm_loadu_si128((const __m128i *)&v215);
          v49 = _mm_loadu_si128((const __m128i *)((char *)&v216 + 8));
          v50 = (__int64)(a1 + 8);
          v223 = 1;
          v218 = v45;
          v219 = v46;
          v220 = v47;
          v221 = v48;
          v222 = v49;
          if ( (unsigned __int8)sub_1361B70((__int64)(a1 + 8), v218.m128i_i64, &v209) )
            return *((unsigned __int8 *)v209 + 80);
          v118 = sub_1362710(v50, v218.m128i_i64, v209);
          *(__m128i *)v118 = _mm_loadu_si128(&v218);
          *((__m128i *)v118 + 1) = _mm_loadu_si128(&v219);
          v118[4] = v220.m128i_i64[0];
          *(__m128i *)(v118 + 5) = _mm_loadu_si128((const __m128i *)&v220.m128i_u64[1]);
          *(__m128i *)(v118 + 7) = _mm_loadu_si128((const __m128i *)&v221.m128i_u64[1]);
          v118[9] = v222.m128i_i64[1];
          *((_BYTE *)v118 + 80) = v223;
          v119 = *(_BYTE *)(v203 + 16);
          if ( v119 <= 0x17u )
          {
            if ( v119 == 5 && *(_WORD *)(v203 + 18) == 32 )
              goto LABEL_205;
          }
          else if ( v119 == 56 )
          {
            goto LABEL_205;
          }
          v120 = *(_BYTE *)(v202 + 16);
          if ( v120 <= 0x17u )
          {
            if ( v120 != 5 || *(_WORD *)(v202 + 18) != 32 )
            {
LABEL_183:
              v121 = *(_BYTE *)(v203 + 16);
              if ( v121 > 0x17u )
              {
                if ( v121 != 56 )
                {
LABEL_207:
                  if ( *(_BYTE *)(v202 + 16) != 77 )
                    goto LABEL_186;
                  if ( v121 == 77 )
                  {
LABEL_249:
                    v20 = sub_1364470((_DWORD)a1, v203, v205, (unsigned int)&a7, v202, v204, (__int64)&a9, (__int64)v23);
                    if ( (_BYTE)v20 != 1 )
                      goto LABEL_246;
                    v122 = *(_BYTE *)(v203 + 16);
LABEL_187:
                    if ( *(_BYTE *)(v202 + 16) == 79 )
                    {
                      if ( v122 == 79 )
                      {
LABEL_245:
                        v20 = sub_13671D0(
                                (_DWORD)a1,
                                v203,
                                v205,
                                (unsigned int)&a7,
                                v202,
                                v204,
                                (__int64)&a9,
                                (__int64)v23);
                        if ( (_BYTE)v20 == 1 )
                          goto LABEL_189;
LABEL_246:
                        *((_BYTE *)sub_1362810(v50, (__int64)&v212) + 80) = v20;
                        return v20;
                      }
                      v152 = _mm_loadu_si128((const __m128i *)&a9);
                      v153 = v23;
                      v23 = v22;
                      v22 = v153;
                      v154 = a8;
                      v155 = a7;
                      v156 = v204;
                      *((_QWORD *)&a9 + 1) = *((_QWORD *)&a7 + 1);
                      a8 = a10;
                      a10 = v154;
                      v204 = v205;
                      v157 = v203;
                      a7 = (__int128)v152;
                      v122 = *(_BYTE *)(v202 + 16);
                      *(_QWORD *)&a9 = v155;
                      v205 = v156;
                      LODWORD(v203) = v202;
                      LODWORD(v202) = v157;
                    }
                    if ( v122 != 79 )
                    {
LABEL_189:
                      if ( v22 == v23 && v205 != -1 && v204 != -1 )
                      {
                        if ( (BYTE2(v209) = v38,
                              LOWORD(v209) = 0,
                              (unsigned __int8)sub_140E950(v22, &v218, a1[1], a1[3], v209))
                          && v218.m128i_i64[0] != -1
                          && v218.m128i_i64[0] == v205
                          || (BYTE2(v209) = v38,
                              LOWORD(v209) = 0,
                              (unsigned __int8)sub_140E950(v23, &v218, a1[1], a1[3], v209))
                          && v218.m128i_i64[0] != -1
                          && v204 == v218.m128i_i64[0] )
                        {
                          v20 = 2;
                          *((_BYTE *)sub_1362810(v50, (__int64)&v212) + 80) = 2;
                          return v20;
                        }
                      }
                      if ( *a1 )
                        v123 = sub_134CB50(*a1, (__int64)&v212, (__int64)&v214.m128i_i64[1]);
                      else
                        v123 = sub_1364F20(a1, &v212, &v214.m128i_u64[1]);
                      v20 = v123;
                      v124 = sub_1361B70(v50, v212.m128i_i64, (__int64 **)&v218);
                      v125 = v218.m128i_i64[0];
                      if ( !v124 )
                      {
                        v126 = sub_1362710(v50, v212.m128i_i64, (__int64 *)v218.m128i_i64[0]);
                        v127 = 10;
                        v128 = &v212;
                        v129 = v126;
                        v125 = (__int64)v126;
                        while ( v127 )
                        {
                          *(_DWORD *)v129 = v128->m128i_i32[0];
                          v128 = (__m128i *)((char *)v128 + 4);
                          v129 = (__int64 *)((char *)v129 + 4);
                          --v127;
                        }
                        v130 = v126 + 5;
                        v131 = 10;
                        v132 = &v214.m128i_i64[1];
                        while ( v131 )
                        {
                          *v130 = *(_DWORD *)v132;
                          v132 = (__int64 *)((char *)v132 + 4);
                          ++v130;
                          --v131;
                        }
                        *((_BYTE *)v126 + 80) = 0;
                      }
                      goto LABEL_201;
                    }
                    goto LABEL_245;
                  }
LABEL_209:
                  v133 = a10;
                  v134 = v23;
                  v23 = v22;
                  v22 = v134;
                  v135 = _mm_loadu_si128((const __m128i *)&a9);
                  a10 = a8;
                  a8 = v133;
                  v136 = v204;
                  a9 = a7;
                  v204 = v205;
                  v137 = v203;
                  v205 = v136;
                  v203 = v202;
                  v202 = v137;
                  a7 = (__int128)v135;
LABEL_186:
                  v122 = *(_BYTE *)(v203 + 16);
                  if ( v122 != 77 )
                    goto LABEL_187;
                  goto LABEL_249;
                }
              }
              else if ( v121 != 5 || *(_WORD *)(v203 + 18) != 32 )
              {
                if ( *(_BYTE *)(v202 + 16) != 77 )
                  goto LABEL_186;
                goto LABEL_209;
              }
LABEL_205:
              v20 = sub_1366100(
                      (_DWORD)a1,
                      v203,
                      v205,
                      (unsigned int)&a7,
                      v202,
                      v204,
                      (__int64)&a9,
                      (__int64)v22,
                      (__int64)v23);
              if ( (_BYTE)v20 != 1 )
              {
                v138 = sub_1361B70(v50, v212.m128i_i64, (__int64 **)&v218);
                v125 = v218.m128i_i64[0];
                if ( !v138 )
                {
                  v139 = sub_1362710(v50, v212.m128i_i64, (__int64 *)v218.m128i_i64[0]);
                  v140 = 10;
                  v141 = &v212;
                  v142 = v139;
                  v125 = (__int64)v139;
                  while ( v140 )
                  {
                    *(_DWORD *)v142 = v141->m128i_i32[0];
                    v141 = (__m128i *)((char *)v141 + 4);
                    v142 = (__int64 *)((char *)v142 + 4);
                    --v140;
                  }
                  v143 = v139 + 5;
                  v144 = &v214.m128i_i64[1];
                  for ( i = 10; i; --i )
                  {
                    *v143 = *(_DWORD *)v144;
                    v144 = (__int64 *)((char *)v144 + 4);
                    ++v143;
                  }
                  *((_BYTE *)v139 + 80) = 0;
                }
LABEL_201:
                *(_BYTE *)(v125 + 80) = v20;
                return v20;
              }
              v121 = *(_BYTE *)(v203 + 16);
              goto LABEL_207;
            }
          }
          else if ( v120 != 56 )
          {
            goto LABEL_183;
          }
          v146 = a10;
          v147 = v23;
          v23 = v22;
          v22 = v147;
          v148 = _mm_loadu_si128((const __m128i *)&a9);
          v149 = v204;
          a10 = a8;
          a8 = v146;
          a9 = a7;
          v204 = v205;
          v150 = v203;
          v205 = v149;
          v203 = v202;
          v202 = v150;
          a7 = (__int128)v148;
          goto LABEL_183;
        }
LABEL_43:
        v29 = 0;
        v30 = off_4CD4978[0];
        if ( off_4CD4978[0] )
          v29 = strlen(off_4CD4978[0]);
        if ( *(_QWORD *)(v203 + 48) || *(__int16 *)(v203 + 18) < 0 )
        {
          v21 = (__int64)v30;
          v31 = sub_1625940(v203, v30, v29);
          if ( v23[16] == 17 && v31 && (unsigned __int8)sub_15E04B0(v23) )
            return 0;
        }
        goto LABEL_49;
      }
    }
    if ( v28 && (unsigned __int8)sub_1CCAC90(v27) )
    {
      v115 = 0;
      v116 = off_4CD4978[0];
      if ( off_4CD4978[0] )
        v115 = strlen(off_4CD4978[0]);
      if ( *(_QWORD *)(v202 + 48) || *(__int16 *)(v202 + 18) < 0 )
      {
        v21 = (__int64)v116;
        v117 = sub_1625940(v202, v116, v115);
        if ( v22[16] == 17 && v117 && (unsigned __int8)sub_15E04B0(v22) )
          return 0;
      }
    }
    goto LABEL_49;
  }
  return v20;
}
