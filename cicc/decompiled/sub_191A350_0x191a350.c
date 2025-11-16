// Function: sub_191A350
// Address: 0x191a350
//
void __fastcall sub_191A350(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  _BYTE *v9; // rsi
  __m128i *v10; // rdi
  __int64 v11; // rdx
  const __m128i *v12; // rcx
  const __m128i *v13; // r8
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __m128i *v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  __m128i *v19; // rax
  __m128i *v20; // rax
  __int8 *v21; // rax
  const __m128i *v22; // rcx
  const __m128i *v23; // r8
  unsigned __int64 v24; // r14
  __int64 v25; // rax
  __m128i *v26; // rdi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  __m128i *v29; // rax
  __m128i *v30; // rax
  __int8 *v31; // rax
  const __m128i *v32; // rcx
  const __m128i *v33; // r8
  unsigned __int64 v34; // r13
  __int64 v35; // rax
  __m128i *v36; // rdi
  __m128i *v37; // rdx
  const __m128i *v38; // rax
  __m128i *v39; // rax
  __m128i *v40; // rax
  __int8 *v41; // rax
  const __m128i *v42; // rcx
  const __m128i *v43; // r8
  unsigned __int64 v44; // r14
  __int64 v45; // rax
  __m128i *v46; // rdi
  __m128i *v47; // rdx
  const __m128i *v48; // rax
  __m128i *v49; // rax
  __m128i *v50; // rax
  __int8 *v51; // rax
  const __m128i *v52; // rcx
  const __m128i *v53; // r9
  unsigned __int64 v54; // r13
  __int64 v55; // rax
  __m128i *v56; // rdi
  __m128i *v57; // rdx
  const __m128i *v58; // rax
  const __m128i *v59; // rcx
  const __m128i *v60; // r8
  unsigned __int64 v61; // r15
  __int64 v62; // rax
  __m128i *v63; // rdi
  __m128i *v64; // rdx
  const __m128i *v65; // rax
  const __m128i *v66; // rcx
  const __m128i *v67; // r8
  unsigned __int64 v68; // r12
  __int64 v69; // rax
  __m128i *v70; // rdi
  __m128i *v71; // rdx
  const __m128i *v72; // rax
  const __m128i *v73; // rcx
  const __m128i *v74; // r8
  unsigned __int64 v75; // r13
  __int64 v76; // rax
  __m128i *v77; // rdi
  __m128i *v78; // rdx
  const __m128i *v79; // rax
  unsigned __int64 v80; // rax
  __m128i *v81; // rcx
  _BYTE *v82; // rsi
  __m128i *v83; // rdx
  _BYTE *v84; // r12
  _BYTE *v85; // r11
  int v86; // r14d
  __int64 v87; // r8
  unsigned int v88; // edi
  __int64 *v89; // rax
  __int64 v90; // rcx
  unsigned int v91; // esi
  int v92; // edx
  __int64 v93; // r13
  int v94; // esi
  int v95; // esi
  __int64 v96; // r9
  unsigned int v97; // edi
  int v98; // ecx
  __int64 v99; // r8
  unsigned int v100; // ecx
  _QWORD *v101; // rdi
  unsigned int v102; // eax
  __int64 v103; // rax
  unsigned __int64 v104; // r13
  unsigned int v105; // eax
  _QWORD *v106; // rax
  __int64 v107; // rdx
  _QWORD *j; // rdx
  __int64 *v109; // r10
  int v110; // ecx
  int v111; // esi
  int v112; // esi
  __int64 v113; // r8
  __int64 *v114; // r9
  unsigned int v115; // r15d
  int v116; // r10d
  __int64 v117; // rdi
  int v118; // r15d
  __int64 *v119; // r10
  _QWORD *v120; // rax
  _BYTE *v121; // [rsp+8h] [rbp-768h]
  _BYTE *v122; // [rsp+8h] [rbp-768h]
  __int64 v123; // [rsp+10h] [rbp-760h]
  int v124; // [rsp+18h] [rbp-758h]
  int v125; // [rsp+18h] [rbp-758h]
  int v126; // [rsp+18h] [rbp-758h]
  _BYTE *v127; // [rsp+20h] [rbp-750h] BYREF
  _BYTE *v128; // [rsp+28h] [rbp-748h]
  _BYTE *v129; // [rsp+30h] [rbp-740h]
  __int64 v130; // [rsp+40h] [rbp-730h] BYREF
  _QWORD *v131; // [rsp+48h] [rbp-728h]
  _QWORD *v132; // [rsp+50h] [rbp-720h]
  __int64 v133; // [rsp+58h] [rbp-718h]
  int v134; // [rsp+60h] [rbp-710h]
  _QWORD v135[8]; // [rsp+68h] [rbp-708h] BYREF
  const __m128i *v136; // [rsp+A8h] [rbp-6C8h] BYREF
  const __m128i *v137; // [rsp+B0h] [rbp-6C0h]
  __int64 v138; // [rsp+B8h] [rbp-6B8h]
  _QWORD v139[16]; // [rsp+C0h] [rbp-6B0h] BYREF
  _QWORD v140[2]; // [rsp+140h] [rbp-630h] BYREF
  unsigned __int64 v141; // [rsp+150h] [rbp-620h]
  _BYTE v142[64]; // [rsp+168h] [rbp-608h] BYREF
  __m128i *v143; // [rsp+1A8h] [rbp-5C8h]
  __m128i *v144; // [rsp+1B0h] [rbp-5C0h]
  __int8 *v145; // [rsp+1B8h] [rbp-5B8h]
  _QWORD v146[2]; // [rsp+1C0h] [rbp-5B0h] BYREF
  unsigned __int64 v147; // [rsp+1D0h] [rbp-5A0h]
  char v148[64]; // [rsp+1E8h] [rbp-588h] BYREF
  const __m128i *v149; // [rsp+228h] [rbp-548h]
  const __m128i *v150; // [rsp+230h] [rbp-540h]
  __int8 *v151; // [rsp+238h] [rbp-538h]
  _QWORD v152[2]; // [rsp+240h] [rbp-530h] BYREF
  unsigned __int64 v153; // [rsp+250h] [rbp-520h]
  _BYTE v154[64]; // [rsp+268h] [rbp-508h] BYREF
  __m128i *v155; // [rsp+2A8h] [rbp-4C8h]
  __m128i *v156; // [rsp+2B0h] [rbp-4C0h]
  __int8 *v157; // [rsp+2B8h] [rbp-4B8h]
  _QWORD v158[2]; // [rsp+2C0h] [rbp-4B0h] BYREF
  unsigned __int64 v159; // [rsp+2D0h] [rbp-4A0h]
  char v160[64]; // [rsp+2E8h] [rbp-488h] BYREF
  const __m128i *v161; // [rsp+328h] [rbp-448h]
  const __m128i *v162; // [rsp+330h] [rbp-440h]
  __int8 *v163; // [rsp+338h] [rbp-438h]
  _QWORD v164[2]; // [rsp+340h] [rbp-430h] BYREF
  unsigned __int64 v165; // [rsp+350h] [rbp-420h]
  _BYTE v166[64]; // [rsp+368h] [rbp-408h] BYREF
  __m128i *v167; // [rsp+3A8h] [rbp-3C8h]
  __m128i *v168; // [rsp+3B0h] [rbp-3C0h]
  __int8 *v169; // [rsp+3B8h] [rbp-3B8h]
  _QWORD v170[2]; // [rsp+3C0h] [rbp-3B0h] BYREF
  unsigned __int64 v171; // [rsp+3D0h] [rbp-3A0h]
  char v172[64]; // [rsp+3E8h] [rbp-388h] BYREF
  const __m128i *v173; // [rsp+428h] [rbp-348h]
  const __m128i *v174; // [rsp+430h] [rbp-340h]
  __int8 *v175; // [rsp+438h] [rbp-338h]
  _QWORD v176[2]; // [rsp+440h] [rbp-330h] BYREF
  unsigned __int64 v177; // [rsp+450h] [rbp-320h]
  _BYTE v178[64]; // [rsp+468h] [rbp-308h] BYREF
  __m128i *v179; // [rsp+4A8h] [rbp-2C8h]
  __m128i *v180; // [rsp+4B0h] [rbp-2C0h]
  __int8 *v181; // [rsp+4B8h] [rbp-2B8h]
  _QWORD v182[2]; // [rsp+4C0h] [rbp-2B0h] BYREF
  unsigned __int64 v183; // [rsp+4D0h] [rbp-2A0h]
  char v184[64]; // [rsp+4E8h] [rbp-288h] BYREF
  const __m128i *v185; // [rsp+528h] [rbp-248h]
  const __m128i *v186; // [rsp+530h] [rbp-240h]
  __int8 *v187; // [rsp+538h] [rbp-238h]
  _QWORD v188[2]; // [rsp+540h] [rbp-230h] BYREF
  unsigned __int64 v189; // [rsp+550h] [rbp-220h]
  _BYTE v190[64]; // [rsp+568h] [rbp-208h] BYREF
  __m128i *v191; // [rsp+5A8h] [rbp-1C8h]
  __m128i *v192; // [rsp+5B0h] [rbp-1C0h]
  __int8 *v193; // [rsp+5B8h] [rbp-1B8h]
  _QWORD v194[2]; // [rsp+5C0h] [rbp-1B0h] BYREF
  unsigned __int64 v195; // [rsp+5D0h] [rbp-1A0h]
  _BYTE v196[64]; // [rsp+5E8h] [rbp-188h] BYREF
  __m128i *v197; // [rsp+628h] [rbp-148h]
  __m128i *v198; // [rsp+630h] [rbp-140h]
  __int8 *v199; // [rsp+638h] [rbp-138h]
  _QWORD v200[2]; // [rsp+640h] [rbp-130h] BYREF
  unsigned __int64 v201; // [rsp+650h] [rbp-120h]
  _BYTE v202[64]; // [rsp+668h] [rbp-108h] BYREF
  __m128i *v203; // [rsp+6A8h] [rbp-C8h]
  unsigned __int64 v204; // [rsp+6B0h] [rbp-C0h]
  __int8 *v205; // [rsp+6B8h] [rbp-B8h]
  __m128i v206; // [rsp+6C0h] [rbp-B0h] BYREF
  unsigned __int64 v207; // [rsp+6D0h] [rbp-A0h]
  _BYTE v208[64]; // [rsp+6E8h] [rbp-88h] BYREF
  __m128i *v209; // [rsp+728h] [rbp-48h]
  __m128i *v210; // [rsp+730h] [rbp-40h]
  __int8 *v211; // [rsp+738h] [rbp-38h]

  v123 = a1 + 752;
  v3 = *(_DWORD *)(a1 + 768);
  ++*(_QWORD *)(a1 + 752);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 772) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 776);
    if ( (unsigned int)v4 <= 0x40 )
      goto LABEL_4;
    j___libc_free_0(*(_QWORD *)(a1 + 760));
    *(_DWORD *)(a1 + 776) = 0;
LABEL_207:
    *(_QWORD *)(a1 + 760) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 768) = 0;
    goto LABEL_7;
  }
  v100 = 4 * v3;
  v4 = *(unsigned int *)(a1 + 776);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v100 = 64;
  if ( (unsigned int)v4 <= v100 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 760);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -8;
    goto LABEL_6;
  }
  v101 = *(_QWORD **)(a1 + 760);
  v102 = v3 - 1;
  if ( v102 )
  {
    _BitScanReverse(&v102, v102);
    v103 = (unsigned int)(1 << (33 - (v102 ^ 0x1F)));
    if ( (int)v103 < 64 )
      v103 = 64;
    if ( (_DWORD)v103 == (_DWORD)v4 )
    {
      *(_QWORD *)(a1 + 768) = 0;
      v120 = &v101[2 * v103];
      do
      {
        if ( v101 )
          *v101 = -8;
        v101 += 2;
      }
      while ( v120 != v101 );
      goto LABEL_7;
    }
    v104 = 4 * (int)v103 / 3u + 1;
  }
  else
  {
    v104 = 86;
  }
  j___libc_free_0(v101);
  v105 = sub_1454B60(v104);
  *(_DWORD *)(a1 + 776) = v105;
  if ( !v105 )
    goto LABEL_207;
  v106 = (_QWORD *)sub_22077B0(16LL * v105);
  v107 = *(unsigned int *)(a1 + 776);
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 760) = v106;
  for ( j = &v106[2 * v107]; j != v106; v106 += 2 )
  {
    if ( v106 )
      *v106 = -8;
  }
LABEL_7:
  v7 = *(_QWORD *)(a2 + 80);
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v136 = 0;
  if ( v7 )
    v7 -= 24;
  memset(v139, 0, sizeof(v139));
  LODWORD(v139[3]) = 8;
  v139[1] = &v139[5];
  v139[2] = &v139[5];
  v131 = v135;
  v132 = v135;
  v135[0] = v7;
  v137 = 0;
  v138 = 0;
  v133 = 0x100000008LL;
  v134 = 0;
  v130 = 1;
  v8 = sub_157EBA0(v7);
  v206.m128i_i64[0] = v7;
  v206.m128i_i64[1] = v8;
  LODWORD(v207) = 0;
  sub_13FDF40(&v136, 0, &v206);
  sub_13FE0F0((__int64)&v130);
  v9 = v154;
  v10 = (__m128i *)v152;
  sub_16CCCB0(v152, (__int64)v154, (__int64)v139);
  v12 = (const __m128i *)v139[14];
  v13 = (const __m128i *)v139[13];
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v14 = v139[14] - v139[13];
  if ( v139[14] == v139[13] )
  {
    v14 = 0;
    v16 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v15 = sub_22077B0(v139[14] - v139[13]);
    v12 = (const __m128i *)v139[14];
    v13 = (const __m128i *)v139[13];
    v16 = (__m128i *)v15;
  }
  v155 = v16;
  v156 = v16;
  v157 = &v16->m128i_i8[v14];
  if ( v13 != v12 )
  {
    v17 = v16;
    v18 = v13;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 24);
      v17 = (__m128i *)((char *)v17 + 24);
    }
    while ( v18 != v12 );
    v16 = (__m128i *)((char *)v16 + 8 * ((unsigned __int64)((char *)&v18[-2].m128i_u64[1] - (char *)v13) >> 3) + 24);
  }
  v156 = v16;
  sub_16CCEE0(v158, (__int64)v160, 8, (__int64)v152);
  v19 = v155;
  v10 = (__m128i *)v140;
  v9 = v142;
  v155 = 0;
  v161 = v19;
  v20 = v156;
  v156 = 0;
  v162 = v20;
  v21 = v157;
  v157 = 0;
  v163 = v21;
  sub_16CCCB0(v140, (__int64)v142, (__int64)&v130);
  v22 = v137;
  v23 = v136;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v24 = (char *)v137 - (char *)v136;
  if ( v137 == v136 )
  {
    v26 = 0;
  }
  else
  {
    if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v25 = sub_22077B0((char *)v137 - (char *)v136);
    v22 = v137;
    v23 = v136;
    v26 = (__m128i *)v25;
  }
  v143 = v26;
  v144 = v26;
  v145 = &v26->m128i_i8[v24];
  if ( v23 != v22 )
  {
    v27 = v26;
    v28 = v23;
    do
    {
      if ( v27 )
      {
        *v27 = _mm_loadu_si128(v28);
        v27[1].m128i_i64[0] = v28[1].m128i_i64[0];
      }
      v28 = (const __m128i *)((char *)v28 + 24);
      v27 = (__m128i *)((char *)v27 + 24);
    }
    while ( v28 != v22 );
    v26 = (__m128i *)((char *)v26 + 8 * ((unsigned __int64)((char *)&v28[-2].m128i_u64[1] - (char *)v23) >> 3) + 24);
  }
  v144 = v26;
  sub_16CCEE0(v146, (__int64)v148, 8, (__int64)v140);
  v29 = v143;
  v10 = (__m128i *)v176;
  v9 = v178;
  v143 = 0;
  v149 = v29;
  v30 = v144;
  v144 = 0;
  v150 = v30;
  v31 = v145;
  v145 = 0;
  v151 = v31;
  sub_16CCCB0(v176, (__int64)v178, (__int64)v158);
  v32 = v162;
  v33 = v161;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v34 = (char *)v162 - (char *)v161;
  if ( v162 == v161 )
  {
    v36 = 0;
  }
  else
  {
    if ( v34 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v35 = sub_22077B0((char *)v162 - (char *)v161);
    v32 = v162;
    v33 = v161;
    v36 = (__m128i *)v35;
  }
  v179 = v36;
  v180 = v36;
  v181 = &v36->m128i_i8[v34];
  if ( v32 != v33 )
  {
    v37 = v36;
    v38 = v33;
    do
    {
      if ( v37 )
      {
        *v37 = _mm_loadu_si128(v38);
        v37[1].m128i_i64[0] = v38[1].m128i_i64[0];
      }
      v38 = (const __m128i *)((char *)v38 + 24);
      v37 = (__m128i *)((char *)v37 + 24);
    }
    while ( v32 != v38 );
    v36 = (__m128i *)((char *)v36 + 8 * ((unsigned __int64)((char *)&v32[-2].m128i_u64[1] - (char *)v33) >> 3) + 24);
  }
  v180 = v36;
  sub_16CCEE0(v182, (__int64)v184, 8, (__int64)v176);
  v39 = v179;
  v10 = (__m128i *)v164;
  v9 = v166;
  v179 = 0;
  v185 = v39;
  v40 = v180;
  v180 = 0;
  v186 = v40;
  v41 = v181;
  v181 = 0;
  v187 = v41;
  sub_16CCCB0(v164, (__int64)v166, (__int64)v146);
  v42 = v150;
  v43 = v149;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v44 = (char *)v150 - (char *)v149;
  if ( v150 == v149 )
  {
    v46 = 0;
  }
  else
  {
    if ( v44 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v45 = sub_22077B0((char *)v150 - (char *)v149);
    v42 = v150;
    v43 = v149;
    v46 = (__m128i *)v45;
  }
  v167 = v46;
  v168 = v46;
  v169 = &v46->m128i_i8[v44];
  if ( v43 != v42 )
  {
    v47 = v46;
    v48 = v43;
    do
    {
      if ( v47 )
      {
        *v47 = _mm_loadu_si128(v48);
        v47[1].m128i_i64[0] = v48[1].m128i_i64[0];
      }
      v48 = (const __m128i *)((char *)v48 + 24);
      v47 = (__m128i *)((char *)v47 + 24);
    }
    while ( v48 != v42 );
    v46 = (__m128i *)((char *)v46 + 8 * ((unsigned __int64)((char *)&v48[-2].m128i_u64[1] - (char *)v43) >> 3) + 24);
  }
  v168 = v46;
  sub_16CCEE0(v170, (__int64)v172, 8, (__int64)v164);
  v49 = v167;
  v10 = (__m128i *)v194;
  v9 = v196;
  v167 = 0;
  v173 = v49;
  v50 = v168;
  v168 = 0;
  v174 = v50;
  v51 = v169;
  v169 = 0;
  v175 = v51;
  sub_16CCCB0(v194, (__int64)v196, (__int64)v182);
  v52 = v186;
  v53 = v185;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v54 = (char *)v186 - (char *)v185;
  if ( v186 == v185 )
  {
    v56 = 0;
  }
  else
  {
    if ( v54 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v55 = sub_22077B0((char *)v186 - (char *)v185);
    v52 = v186;
    v53 = v185;
    v56 = (__m128i *)v55;
  }
  v197 = v56;
  v198 = v56;
  v199 = &v56->m128i_i8[v54];
  if ( v52 != v53 )
  {
    v57 = v56;
    v58 = v53;
    do
    {
      if ( v57 )
      {
        *v57 = _mm_loadu_si128(v58);
        v57[1].m128i_i64[0] = v58[1].m128i_i64[0];
      }
      v58 = (const __m128i *)((char *)v58 + 24);
      v57 = (__m128i *)((char *)v57 + 24);
    }
    while ( v52 != v58 );
    v56 = (__m128i *)((char *)v56 + 8 * ((unsigned __int64)((char *)&v52[-2].m128i_u64[1] - (char *)v53) >> 3) + 24);
  }
  v198 = v56;
  v9 = v190;
  v10 = (__m128i *)v188;
  sub_16CCCB0(v188, (__int64)v190, (__int64)v170);
  v59 = v174;
  v60 = v173;
  v191 = 0;
  v192 = 0;
  v193 = 0;
  v61 = (char *)v174 - (char *)v173;
  if ( v174 == v173 )
  {
    v63 = 0;
  }
  else
  {
    if ( v61 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v62 = sub_22077B0((char *)v174 - (char *)v173);
    v59 = v174;
    v60 = v173;
    v63 = (__m128i *)v62;
  }
  v191 = v63;
  v192 = v63;
  v193 = &v63->m128i_i8[v61];
  if ( v60 != v59 )
  {
    v64 = v63;
    v65 = v60;
    do
    {
      if ( v64 )
      {
        *v64 = _mm_loadu_si128(v65);
        v64[1].m128i_i64[0] = v65[1].m128i_i64[0];
      }
      v65 = (const __m128i *)((char *)v65 + 24);
      v64 = (__m128i *)((char *)v64 + 24);
    }
    while ( v65 != v59 );
    v63 = (__m128i *)((char *)v63 + 8 * ((unsigned __int64)((char *)&v65[-2].m128i_u64[1] - (char *)v60) >> 3) + 24);
  }
  v192 = v63;
  v9 = v208;
  v10 = &v206;
  sub_16CCCB0(&v206, (__int64)v208, (__int64)v194);
  v66 = v198;
  v67 = v197;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v68 = (char *)v198 - (char *)v197;
  if ( v198 == v197 )
  {
    v70 = 0;
  }
  else
  {
    if ( v68 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_223;
    v69 = sub_22077B0((char *)v198 - (char *)v197);
    v66 = v198;
    v67 = v197;
    v70 = (__m128i *)v69;
  }
  v209 = v70;
  v210 = v70;
  v211 = &v70->m128i_i8[v68];
  if ( v67 != v66 )
  {
    v71 = v70;
    v72 = v67;
    do
    {
      if ( v71 )
      {
        *v71 = _mm_loadu_si128(v72);
        v71[1].m128i_i64[0] = v72[1].m128i_i64[0];
      }
      v72 = (const __m128i *)((char *)v72 + 24);
      v71 = (__m128i *)((char *)v71 + 24);
    }
    while ( v72 != v66 );
    v70 = (__m128i *)((char *)v70 + 8 * ((unsigned __int64)((char *)&v72[-2].m128i_u64[1] - (char *)v67) >> 3) + 24);
  }
  v210 = v70;
  v9 = v202;
  v10 = (__m128i *)v200;
  sub_16CCCB0(v200, (__int64)v202, (__int64)v188);
  v73 = v192;
  v74 = v191;
  v203 = 0;
  v204 = 0;
  v205 = 0;
  v75 = (char *)v192 - (char *)v191;
  if ( v192 == v191 )
  {
    v77 = 0;
    goto LABEL_75;
  }
  if ( v75 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_223:
    sub_4261EA(v10, v9, v11);
  v76 = sub_22077B0((char *)v192 - (char *)v191);
  v73 = v192;
  v74 = v191;
  v77 = (__m128i *)v76;
LABEL_75:
  v203 = v77;
  v204 = (unsigned __int64)v77;
  v205 = &v77->m128i_i8[v75];
  if ( v73 == v74 )
  {
    v80 = (unsigned __int64)v77;
  }
  else
  {
    v78 = v77;
    v79 = v74;
    do
    {
      if ( v78 )
      {
        *v78 = _mm_loadu_si128(v79);
        v78[1].m128i_i64[0] = v79[1].m128i_i64[0];
      }
      v79 = (const __m128i *)((char *)v79 + 24);
      v78 = (__m128i *)((char *)v78 + 24);
    }
    while ( v79 != v73 );
    v80 = (unsigned __int64)&v77[1].m128i_u64[((unsigned __int64)((char *)&v79[-2].m128i_u64[1] - (char *)v74) >> 3) + 1];
  }
  v204 = v80;
  while ( 1 )
  {
    v81 = v209;
    if ( v80 - (_QWORD)v77 != (char *)v210 - (char *)v209 )
      goto LABEL_83;
    if ( (__m128i *)v80 == v77 )
      break;
    v83 = v77;
    while ( v83->m128i_i64[0] == v81->m128i_i64[0] && v83[1].m128i_i32[0] == v81[1].m128i_i32[0] )
    {
      v83 = (__m128i *)((char *)v83 + 24);
      v81 = (__m128i *)((char *)v81 + 24);
      if ( (__m128i *)v80 == v83 )
        goto LABEL_94;
    }
LABEL_83:
    v82 = v128;
    if ( v128 == v129 )
    {
      sub_1292090((__int64)&v127, v128, (_QWORD *)(v80 - 24));
      v80 = v204;
    }
    else
    {
      if ( v128 )
      {
        *(_QWORD *)v128 = *(_QWORD *)(v80 - 24);
        v82 = v128;
        v80 = v204;
      }
      v128 = v82 + 8;
    }
    v77 = v203;
    v80 -= 24LL;
    v204 = v80;
    if ( (__m128i *)v80 != v203 )
    {
      sub_13FE0F0((__int64)v200);
      v77 = v203;
      v80 = v204;
    }
  }
LABEL_94:
  if ( v77 )
    j_j___libc_free_0(v77, v205 - (__int8 *)v77);
  if ( v201 != v200[1] )
    _libc_free(v201);
  if ( v209 )
    j_j___libc_free_0(v209, v211 - (__int8 *)v209);
  if ( v207 != v206.m128i_i64[1] )
    _libc_free(v207);
  if ( v191 )
    j_j___libc_free_0(v191, v193 - (__int8 *)v191);
  if ( v189 != v188[1] )
    _libc_free(v189);
  if ( v197 )
    j_j___libc_free_0(v197, v199 - (__int8 *)v197);
  if ( v195 != v194[1] )
    _libc_free(v195);
  if ( v173 )
    j_j___libc_free_0(v173, v175 - (__int8 *)v173);
  if ( v171 != v170[1] )
    _libc_free(v171);
  if ( v167 )
    j_j___libc_free_0(v167, v169 - (__int8 *)v167);
  if ( v165 != v164[1] )
    _libc_free(v165);
  if ( v185 )
    j_j___libc_free_0(v185, v187 - (__int8 *)v185);
  if ( v183 != v182[1] )
    _libc_free(v183);
  if ( v179 )
    j_j___libc_free_0(v179, v181 - (__int8 *)v179);
  if ( v177 != v176[1] )
    _libc_free(v177);
  if ( v149 )
    j_j___libc_free_0(v149, v151 - (__int8 *)v149);
  if ( v147 != v146[1] )
    _libc_free(v147);
  if ( v143 )
    j_j___libc_free_0(v143, v145 - (__int8 *)v143);
  if ( v141 != v140[1] )
    _libc_free(v141);
  if ( v161 )
    j_j___libc_free_0(v161, v163 - (__int8 *)v161);
  if ( v159 != v158[1] )
    _libc_free(v159);
  if ( v155 )
    j_j___libc_free_0(v155, v157 - (__int8 *)v155);
  if ( v153 != v152[1] )
    _libc_free(v153);
  if ( v136 )
    j_j___libc_free_0(v136, v138 - (_QWORD)v136);
  if ( v132 != v131 )
    _libc_free((unsigned __int64)v132);
  if ( v139[13] )
    j_j___libc_free_0(v139[13], v139[15] - v139[13]);
  if ( v139[2] != v139[1] )
    _libc_free(v139[2]);
  v84 = v128;
  v85 = v127;
  if ( v127 != v128 )
  {
    v86 = 1;
    while ( 1 )
    {
      v91 = *(_DWORD *)(a1 + 776);
      v92 = v86;
      v93 = *((_QWORD *)v84 - 1);
      ++v86;
      if ( !v91 )
        break;
      v87 = *(_QWORD *)(a1 + 760);
      v88 = (v91 - 1) & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
      v89 = (__int64 *)(v87 + 16LL * v88);
      v90 = *v89;
      if ( v93 == *v89 )
      {
LABEL_153:
        v84 -= 8;
        *((_DWORD *)v89 + 2) = v92;
        if ( v85 == v84 )
          goto LABEL_161;
      }
      else
      {
        v125 = 1;
        v109 = 0;
        while ( v90 != -8 )
        {
          if ( !v109 && v90 == -16 )
            v109 = v89;
          v88 = (v91 - 1) & (v125 + v88);
          v89 = (__int64 *)(v87 + 16LL * v88);
          v90 = *v89;
          if ( v93 == *v89 )
            goto LABEL_153;
          ++v125;
        }
        v110 = *(_DWORD *)(a1 + 768);
        if ( v109 )
          v89 = v109;
        ++*(_QWORD *)(a1 + 752);
        v98 = v110 + 1;
        if ( 4 * v98 < 3 * v91 )
        {
          if ( v91 - *(_DWORD *)(a1 + 772) - v98 <= v91 >> 3 )
          {
            v122 = v85;
            v126 = v92;
            sub_1917AE0(v123, v91);
            v111 = *(_DWORD *)(a1 + 776);
            if ( !v111 )
            {
LABEL_232:
              ++*(_DWORD *)(a1 + 768);
              BUG();
            }
            v112 = v111 - 1;
            v113 = *(_QWORD *)(a1 + 760);
            v114 = 0;
            v115 = v112 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
            v92 = v126;
            v85 = v122;
            v116 = 1;
            v98 = *(_DWORD *)(a1 + 768) + 1;
            v89 = (__int64 *)(v113 + 16LL * v115);
            v117 = *v89;
            if ( v93 != *v89 )
            {
              while ( v117 != -8 )
              {
                if ( v117 != -16 || v114 )
                  v89 = v114;
                v115 = v112 & (v116 + v115);
                v117 = *(_QWORD *)(v113 + 16LL * v115);
                if ( v93 == v117 )
                {
                  v89 = (__int64 *)(v113 + 16LL * v115);
                  goto LABEL_158;
                }
                ++v116;
                v114 = v89;
                v89 = (__int64 *)(v113 + 16LL * v115);
              }
              if ( v114 )
                v89 = v114;
            }
          }
          goto LABEL_158;
        }
LABEL_156:
        v121 = v85;
        v124 = v92;
        sub_1917AE0(v123, 2 * v91);
        v94 = *(_DWORD *)(a1 + 776);
        if ( !v94 )
          goto LABEL_232;
        v95 = v94 - 1;
        v96 = *(_QWORD *)(a1 + 760);
        v92 = v124;
        v85 = v121;
        v97 = v95 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
        v98 = *(_DWORD *)(a1 + 768) + 1;
        v89 = (__int64 *)(v96 + 16LL * v97);
        v99 = *v89;
        if ( v93 != *v89 )
        {
          v118 = 1;
          v119 = 0;
          while ( v99 != -8 )
          {
            if ( v119 || v99 != -16 )
              v89 = v119;
            v97 = v95 & (v118 + v97);
            v99 = *(_QWORD *)(v96 + 16LL * v97);
            if ( v93 == v99 )
            {
              v89 = (__int64 *)(v96 + 16LL * v97);
              goto LABEL_158;
            }
            ++v118;
            v119 = v89;
            v89 = (__int64 *)(v96 + 16LL * v97);
          }
          if ( v119 )
            v89 = v119;
        }
LABEL_158:
        *(_DWORD *)(a1 + 768) = v98;
        if ( *v89 != -8 )
          --*(_DWORD *)(a1 + 772);
        v84 -= 8;
        *((_DWORD *)v89 + 2) = 0;
        *v89 = v93;
        *((_DWORD *)v89 + 2) = v92;
        if ( v85 == v84 )
        {
LABEL_161:
          v84 = v127;
          goto LABEL_162;
        }
      }
    }
    ++*(_QWORD *)(a1 + 752);
    goto LABEL_156;
  }
LABEL_162:
  *(_BYTE *)(a1 + 784) = 0;
  if ( v84 )
    j_j___libc_free_0(v84, v129 - v84);
}
