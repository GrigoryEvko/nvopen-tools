// Function: sub_1BE1020
// Address: 0x1be1020
//
__int64 __fastcall sub_1BE1020(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __m128i a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int64 a19)
{
  __int64 v20; // r13
  __int64 v21; // r15
  __int64 v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // r13
  __int64 v25; // r12
  _QWORD *v26; // r14
  _QWORD *v27; // r15
  __int64 v28; // rax
  unsigned int v29; // r12d
  __int64 v31; // rsi
  char *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // r13
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r10
  __int64 v38; // r11
  __int64 *v39; // r15
  __int64 v40; // rcx
  unsigned __int64 v41; // rsi
  _QWORD *v42; // rax
  _DWORD *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  unsigned __int64 v46; // rsi
  _QWORD *v47; // rax
  _DWORD *v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // r13
  __m128 *v52; // rax
  const __m128i *v53; // rax
  __int64 v54; // rax
  char *v55; // rax
  const __m128i *v56; // rax
  __int64 v57; // rax
  char *v58; // rax
  __int64 v59; // rax
  __m128 *v60; // rax
  char *v61; // rax
  _BYTE *v62; // rsi
  __int64 *v63; // rdi
  __int64 v64; // rdx
  const __m128i *v65; // rcx
  const __m128i *v66; // r8
  unsigned __int64 v67; // r15
  __int64 v68; // rax
  __m128 *v69; // rdi
  __m128 *v70; // rdx
  const __m128i *v71; // rax
  __int64 *v72; // r9
  double v73; // xmm4_8
  double v74; // xmm5_8
  const __m128i *v75; // rcx
  __int64 v76; // r8
  unsigned __int64 v77; // r14
  __int64 v78; // rax
  __m128 *v79; // rdi
  __m128 *v80; // rdx
  const __m128i *v81; // rax
  const __m128i *v82; // rax
  __int64 v83; // rcx
  __int64 v84; // r15
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 *v87; // r8
  __int64 *v88; // r9
  double v89; // xmm4_8
  double v90; // xmm5_8
  __int64 v91; // rcx
  __int64 v92; // rdx
  __int64 v93; // rsi
  __int64 v94; // rax
  _DWORD *v95; // r8
  _DWORD *v96; // rdi
  __int64 v97; // rcx
  __int64 v98; // rdx
  __int64 v99; // rax
  _DWORD *v100; // r8
  _DWORD *v101; // rdi
  __int64 v102; // rcx
  __int64 v103; // rdx
  __int64 v104; // [rsp+10h] [rbp-940h]
  _QWORD v106[16]; // [rsp+20h] [rbp-930h] BYREF
  __int64 v107; // [rsp+A0h] [rbp-8B0h] BYREF
  _QWORD *v108; // [rsp+A8h] [rbp-8A8h]
  _QWORD *v109; // [rsp+B0h] [rbp-8A0h]
  __int64 v110; // [rsp+B8h] [rbp-898h]
  int v111; // [rsp+C0h] [rbp-890h]
  _QWORD v112[8]; // [rsp+C8h] [rbp-888h] BYREF
  const __m128i *v113; // [rsp+108h] [rbp-848h] BYREF
  __int64 v114; // [rsp+110h] [rbp-840h]
  char *v115; // [rsp+118h] [rbp-838h]
  __int64 v116; // [rsp+120h] [rbp-830h] BYREF
  __int64 v117; // [rsp+128h] [rbp-828h]
  unsigned __int64 v118; // [rsp+130h] [rbp-820h]
  _BYTE v119[64]; // [rsp+148h] [rbp-808h] BYREF
  const __m128i *v120; // [rsp+188h] [rbp-7C8h]
  __int64 v121; // [rsp+190h] [rbp-7C0h]
  char *v122; // [rsp+198h] [rbp-7B8h]
  __int64 v123; // [rsp+1A0h] [rbp-7B0h] BYREF
  __int64 v124; // [rsp+1A8h] [rbp-7A8h]
  unsigned __int64 v125; // [rsp+1B0h] [rbp-7A0h]
  _BYTE v126[64]; // [rsp+1C8h] [rbp-788h] BYREF
  __m128 *v127; // [rsp+208h] [rbp-748h]
  __m128 *v128; // [rsp+210h] [rbp-740h]
  char *v129; // [rsp+218h] [rbp-738h]
  __m128i v130; // [rsp+220h] [rbp-730h] BYREF
  unsigned __int64 v131; // [rsp+230h] [rbp-720h]
  char v132[64]; // [rsp+248h] [rbp-708h] BYREF
  const __m128i *v133; // [rsp+288h] [rbp-6C8h]
  const __m128i *v134; // [rsp+290h] [rbp-6C0h]
  char *v135; // [rsp+298h] [rbp-6B8h]
  _QWORD v136[2]; // [rsp+2A0h] [rbp-6B0h] BYREF
  unsigned __int64 v137; // [rsp+2B0h] [rbp-6A0h]
  char v138[64]; // [rsp+2C8h] [rbp-688h] BYREF
  __int64 v139; // [rsp+308h] [rbp-648h]
  const __m128i *v140; // [rsp+310h] [rbp-640h]
  char *v141; // [rsp+318h] [rbp-638h]
  __int64 v142[5]; // [rsp+320h] [rbp-630h] BYREF
  char v143; // [rsp+348h] [rbp-608h] BYREF
  _QWORD v144[4]; // [rsp+388h] [rbp-5C8h] BYREF
  int v145; // [rsp+3A8h] [rbp-5A8h]
  _BYTE v146[128]; // [rsp+3B0h] [rbp-5A0h] BYREF
  __int64 v147; // [rsp+430h] [rbp-520h]
  __int64 v148; // [rsp+438h] [rbp-518h]
  __int64 v149; // [rsp+440h] [rbp-510h]
  int v150; // [rsp+448h] [rbp-508h]
  char *v151; // [rsp+450h] [rbp-500h]
  __int64 v152; // [rsp+458h] [rbp-4F8h]
  char v153; // [rsp+460h] [rbp-4F0h] BYREF
  char *v154; // [rsp+4A0h] [rbp-4B0h]
  __int64 v155; // [rsp+4A8h] [rbp-4A8h]
  char v156; // [rsp+4B0h] [rbp-4A0h] BYREF
  _QWORD v157[4]; // [rsp+630h] [rbp-320h] BYREF
  int v158; // [rsp+650h] [rbp-300h]
  _BYTE v159[256]; // [rsp+658h] [rbp-2F8h] BYREF
  __int64 v160; // [rsp+758h] [rbp-1F8h]
  __int64 v161; // [rsp+760h] [rbp-1F0h]
  __int64 v162; // [rsp+768h] [rbp-1E8h]
  __int64 v163; // [rsp+770h] [rbp-1E0h]
  __int64 v164; // [rsp+778h] [rbp-1D8h]
  __int64 v165; // [rsp+780h] [rbp-1D0h]
  __int64 v166; // [rsp+788h] [rbp-1C8h]
  __int64 v167; // [rsp+790h] [rbp-1C0h]
  __int64 v168; // [rsp+798h] [rbp-1B8h]
  __int64 v169; // [rsp+7A0h] [rbp-1B0h]
  __int64 v170; // [rsp+7A8h] [rbp-1A8h]
  __int64 v171; // [rsp+7B0h] [rbp-1A0h]
  __int64 v172; // [rsp+7B8h] [rbp-198h]
  __int64 v173; // [rsp+7C0h] [rbp-190h]
  __int64 v174; // [rsp+7C8h] [rbp-188h]
  __int64 v175; // [rsp+7D0h] [rbp-180h]
  __int64 v176; // [rsp+7D8h] [rbp-178h]
  int v177; // [rsp+7E0h] [rbp-170h]
  __int64 v178; // [rsp+7E8h] [rbp-168h]
  __int64 v179; // [rsp+7F0h] [rbp-160h]
  __int64 v180; // [rsp+7F8h] [rbp-158h]
  __int64 v181; // [rsp+800h] [rbp-150h]
  __int64 v182; // [rsp+808h] [rbp-148h]
  __int64 v183; // [rsp+810h] [rbp-140h]
  __int64 v184; // [rsp+818h] [rbp-138h]
  __int64 v185; // [rsp+820h] [rbp-130h]
  int v186; // [rsp+828h] [rbp-128h]
  int v187; // [rsp+830h] [rbp-120h]
  __int64 v188; // [rsp+838h] [rbp-118h]
  __int64 v189; // [rsp+840h] [rbp-110h]
  __int64 *v190; // [rsp+848h] [rbp-108h]
  __int64 v191; // [rsp+850h] [rbp-100h]
  __int64 v192; // [rsp+858h] [rbp-F8h]
  __int64 v193; // [rsp+860h] [rbp-F0h]
  __int64 v194; // [rsp+868h] [rbp-E8h]
  __int64 v195; // [rsp+870h] [rbp-E0h]
  __int64 v196; // [rsp+878h] [rbp-D8h]
  __int64 v197; // [rsp+880h] [rbp-D0h]
  __int64 v198; // [rsp+888h] [rbp-C8h]
  int v199; // [rsp+890h] [rbp-C0h]
  int v200; // [rsp+894h] [rbp-BCh]
  __int64 v201; // [rsp+898h] [rbp-B8h]
  __int64 v202; // [rsp+8A0h] [rbp-B0h]
  __int64 v203; // [rsp+8A8h] [rbp-A8h]
  __int64 v204; // [rsp+8B0h] [rbp-A0h]
  __int64 v205; // [rsp+8B8h] [rbp-98h]
  int v206; // [rsp+8C0h] [rbp-90h]
  __int64 v207; // [rsp+8C8h] [rbp-88h]
  __int64 v208; // [rsp+8D0h] [rbp-80h]
  __int64 v209; // [rsp+8E0h] [rbp-70h]
  __int64 v210; // [rsp+8E8h] [rbp-68h]
  __int64 v211; // [rsp+8F0h] [rbp-60h]
  int v212; // [rsp+8F8h] [rbp-58h]
  __int64 v213; // [rsp+900h] [rbp-50h]
  __int64 v214; // [rsp+908h] [rbp-48h]
  __int64 v215; // [rsp+910h] [rbp-40h]

  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 32) = a15;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 40) = a16;
  *(_QWORD *)(a1 + 24) = a6;
  *(_QWORD *)(a1 + 48) = a17;
  *(_QWORD *)(a1 + 56) = a18;
  *(_QWORD *)(a1 + 64) = sub_1632FA0(*(_QWORD *)(a2 + 40));
  sub_196A810(a1 + 72);
  v20 = *(_QWORD *)(a1 + 104);
  v21 = *(_QWORD *)(a1 + 112);
  if ( v20 != v21 )
  {
    v22 = *(_QWORD *)(a1 + 104);
    do
    {
      v23 = *(_QWORD *)(v22 + 8);
      if ( v23 != v22 + 24 )
        _libc_free(v23);
      v22 += 88;
    }
    while ( v21 != v22 );
    *(_QWORD *)(a1 + 112) = v20;
  }
  sub_196A810(a1 + 128);
  v24 = *(_QWORD *)(a1 + 168);
  v104 = *(_QWORD *)(a1 + 160);
  if ( v104 != v24 )
  {
    v25 = *(_QWORD *)(a1 + 160);
    do
    {
      v26 = *(_QWORD **)(v25 + 8);
      v27 = &v26[3 * *(unsigned int *)(v25 + 16)];
      if ( v26 != v27 )
      {
        do
        {
          v28 = *(v27 - 1);
          v27 -= 3;
          if ( v28 != -8 && v28 != 0 && v28 != -16 )
            sub_1649B30(v27);
        }
        while ( v26 != v27 );
        v27 = *(_QWORD **)(v25 + 8);
      }
      if ( v27 != (_QWORD *)(v25 + 24) )
        _libc_free((unsigned __int64)v27);
      v25 += 216;
    }
    while ( v24 != v25 );
    *(_QWORD *)(a1 + 168) = v104;
  }
  if ( !(unsigned int)sub_14A3140(*(__int64 **)(a1 + 8), 1u) )
    return 0;
  v29 = sub_1560180(a2 + 112, 25);
  if ( (_BYTE)v29 )
    return 0;
  memset(v142, 0, 32);
  v31 = *(_QWORD *)(a1 + 64);
  v32 = &v143;
  v33 = *(_QWORD *)(a1 + 56);
  v34 = *(_QWORD *)(a1 + 48);
  v35 = *(_QWORD *)(a1 + 40);
  v142[4] = 1;
  v36 = *(_QWORD *)(a1 + 32);
  v37 = *(_QWORD *)(a1 + 24);
  v38 = *(_QWORD *)(a1 + 16);
  v39 = *(__int64 **)(a1 + 8);
  v40 = *(_QWORD *)a1;
  do
  {
    *(_QWORD *)v32 = -8;
    v32 += 16;
  }
  while ( v32 != (char *)v144 );
  v144[0] = 0;
  v144[1] = v146;
  v144[2] = v146;
  v151 = &v153;
  v152 = 0x800000000LL;
  v154 = &v156;
  v155 = 0x1000000000LL;
  v157[1] = v159;
  v157[2] = v159;
  v144[3] = 16;
  v145 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v157[0] = 0;
  v157[3] = 32;
  v158 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v189 = v40;
  v188 = a2;
  v191 = v38;
  v192 = v37;
  v193 = v36;
  v194 = v35;
  v195 = v34;
  v197 = v31;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v190 = v39;
  v196 = v33;
  v198 = a19;
  v204 = sub_15E0530(*(_QWORD *)(v40 + 24));
  v201 = 0;
  v203 = 0;
  v205 = 0;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v202 = 0;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  sub_14D07D0(v188, v34, (__int64)v157);
  v41 = sub_16D5D50();
  v42 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_32;
  v43 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v44 = v42[2];
      v45 = v42[3];
      if ( v41 <= v42[4] )
        break;
      v42 = (_QWORD *)v42[3];
      if ( !v45 )
        goto LABEL_30;
    }
    v43 = v42;
    v42 = (_QWORD *)v42[2];
  }
  while ( v44 );
LABEL_30:
  if ( v43 == dword_4FA0208 )
    goto LABEL_32;
  if ( v41 < *((_QWORD *)v43 + 4) )
    goto LABEL_32;
  v99 = *((_QWORD *)v43 + 7);
  v100 = v43 + 12;
  if ( !v99 )
    goto LABEL_32;
  v101 = v43 + 12;
  do
  {
    while ( 1 )
    {
      v102 = *(_QWORD *)(v99 + 16);
      v103 = *(_QWORD *)(v99 + 24);
      if ( *(_DWORD *)(v99 + 32) >= dword_4FB92E8 )
        break;
      v99 = *(_QWORD *)(v99 + 24);
      if ( !v103 )
        goto LABEL_124;
    }
    v101 = (_DWORD *)v99;
    v99 = *(_QWORD *)(v99 + 16);
  }
  while ( v102 );
LABEL_124:
  if ( v100 != v101 && dword_4FB92E8 >= v101[8] && v101[9] )
    v199 = dword_4FB9380;
  else
LABEL_32:
    v199 = sub_14A3170(v190, 1u);
  v46 = sub_16D5D50();
  v47 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_40;
  v48 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v49 = v47[2];
      v50 = v47[3];
      if ( v46 <= v47[4] )
        break;
      v47 = (_QWORD *)v47[3];
      if ( !v50 )
        goto LABEL_38;
    }
    v48 = v47;
    v47 = (_QWORD *)v47[2];
  }
  while ( v49 );
LABEL_38:
  if ( v48 == dword_4FA0208 )
    goto LABEL_40;
  if ( v46 < *((_QWORD *)v48 + 4) )
    goto LABEL_40;
  v94 = *((_QWORD *)v48 + 7);
  v95 = v48 + 12;
  if ( !v94 )
    goto LABEL_40;
  v96 = v48 + 12;
  do
  {
    while ( 1 )
    {
      v97 = *(_QWORD *)(v94 + 16);
      v98 = *(_QWORD *)(v94 + 24);
      if ( *(_DWORD *)(v94 + 32) >= dword_4FB9128 )
        break;
      v94 = *(_QWORD *)(v94 + 24);
      if ( !v98 )
        goto LABEL_115;
    }
    v96 = (_DWORD *)v94;
    v94 = *(_QWORD *)(v94 + 16);
  }
  while ( v97 );
LABEL_115:
  if ( v95 != v96 && dword_4FB9128 >= v96[8] && v96[9] )
    v200 = dword_4FB91C0;
  else
LABEL_40:
    v200 = sub_14A31A0((__int64)v190);
  v113 = 0;
  v51 = *(_QWORD *)(a2 + 80);
  v114 = 0;
  v115 = 0;
  v111 = 0;
  if ( v51 )
    v51 -= 24;
  memset(v106, 0, sizeof(v106));
  LODWORD(v106[3]) = 8;
  v106[1] = &v106[5];
  v106[2] = &v106[5];
  v108 = v112;
  v109 = v112;
  v112[0] = v51;
  v110 = 0x100000008LL;
  v107 = 1;
  v130.m128i_i64[1] = sub_157EBA0(v51);
  v130.m128i_i64[0] = v51;
  LODWORD(v131) = 0;
  sub_13FDF40(&v113, 0, &v130);
  sub_13FE0F0((__int64)&v107);
  sub_16CCEE0(&v123, (__int64)v126, 8, (__int64)v106);
  v52 = (__m128 *)v106[13];
  memset(&v106[13], 0, 24);
  v127 = v52;
  v128 = (__m128 *)v106[14];
  v129 = (char *)v106[15];
  sub_16CCEE0(&v116, (__int64)v119, 8, (__int64)&v107);
  v53 = v113;
  v113 = 0;
  v120 = v53;
  v54 = v114;
  v114 = 0;
  v121 = v54;
  v55 = v115;
  v115 = 0;
  v122 = v55;
  sub_16CCEE0(&v130, (__int64)v132, 8, (__int64)&v116);
  v56 = v120;
  v120 = 0;
  v133 = v56;
  v57 = v121;
  v121 = 0;
  v134 = (const __m128i *)v57;
  v58 = v122;
  v122 = 0;
  v135 = v58;
  sub_16CCEE0(v136, (__int64)v138, 8, (__int64)&v123);
  v59 = (__int64)v127;
  v127 = 0;
  v139 = v59;
  v60 = v128;
  v128 = 0;
  v140 = (const __m128i *)v60;
  v61 = v129;
  v129 = 0;
  v141 = v61;
  if ( v120 )
    j_j___libc_free_0(v120, v122 - (char *)v120);
  if ( v118 != v117 )
    _libc_free(v118);
  if ( v127 )
    j_j___libc_free_0(v127, v129 - (char *)v127);
  if ( v125 != v124 )
    _libc_free(v125);
  if ( v113 )
    j_j___libc_free_0(v113, v115 - (char *)v113);
  if ( v109 != v108 )
    _libc_free((unsigned __int64)v109);
  if ( v106[13] )
    j_j___libc_free_0(v106[13], v106[15] - v106[13]);
  if ( v106[2] != v106[1] )
    _libc_free(v106[2]);
  v62 = v119;
  v63 = &v116;
  sub_16CCCB0(&v116, (__int64)v119, (__int64)&v130);
  v65 = v134;
  v66 = v133;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v67 = (char *)v134 - (char *)v133;
  if ( v134 == v133 )
  {
    v69 = 0;
  }
  else
  {
    if ( v67 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_130;
    v68 = sub_22077B0((char *)v134 - (char *)v133);
    v65 = v134;
    v66 = v133;
    v69 = (__m128 *)v68;
  }
  v120 = (const __m128i *)v69;
  v121 = (__int64)v69;
  v122 = (char *)v69 + v67;
  if ( v66 != v65 )
  {
    v70 = v69;
    v71 = v66;
    do
    {
      if ( v70 )
      {
        a7 = _mm_loadu_si128(v71);
        *v70 = (__m128)a7;
        v70[1].m128_u64[0] = v71[1].m128i_u64[0];
      }
      v71 = (const __m128i *)((char *)v71 + 24);
      v70 = (__m128 *)((char *)v70 + 24);
    }
    while ( v65 != v71 );
    v69 = (__m128 *)((char *)v69 + 8 * ((unsigned __int64)((char *)&v65[-2].m128i_u64[1] - (char *)v66) >> 3) + 24);
  }
  v62 = v126;
  v121 = (__int64)v69;
  v63 = &v123;
  sub_16CCCB0(&v123, (__int64)v126, (__int64)v136);
  v75 = v140;
  v76 = v139;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v77 = (unsigned __int64)v140 - v139;
  if ( v140 == (const __m128i *)v139 )
  {
    v79 = 0;
    goto LABEL_71;
  }
  if ( v77 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_130:
    sub_4261EA(v63, v62, v64);
  v78 = sub_22077B0((char *)v140 - v139);
  v75 = v140;
  v76 = v139;
  v79 = (__m128 *)v78;
LABEL_71:
  v127 = v79;
  v80 = v79;
  v128 = v79;
  v129 = (char *)v79 + v77;
  if ( v75 != (const __m128i *)v76 )
  {
    v81 = (const __m128i *)v76;
    do
    {
      if ( v80 )
      {
        a8 = _mm_loadu_si128(v81);
        *v80 = (__m128)a8;
        v80[1].m128_u64[0] = v81[1].m128i_u64[0];
      }
      v81 = (const __m128i *)((char *)v81 + 24);
      v80 = (__m128 *)((char *)v80 + 24);
    }
    while ( v75 != v81 );
    v80 = (__m128 *)((char *)v79 + 8 * (((unsigned __int64)&v75[-2].m128i_u64[1] - v76) >> 3) + 24);
  }
  v128 = v80;
  v82 = v120;
  v83 = v121;
  while ( 1 )
  {
    v92 = (char *)v80 - (char *)v79;
    if ( v83 - (_QWORD)v82 != v92 )
      goto LABEL_78;
    if ( (const __m128i *)v83 == v82 )
      break;
    v92 = (__int64)v79;
    while ( v82->m128i_i64[0] == *(_QWORD *)v92 && v82[1].m128i_i32[0] == *(_DWORD *)(v92 + 16) )
    {
      v82 = (const __m128i *)((char *)v82 + 24);
      v92 += 24;
      if ( (const __m128i *)v83 == v82 )
        goto LABEL_91;
    }
LABEL_78:
    v84 = *(_QWORD *)(v83 - 24);
    sub_1BC3F50((_QWORD *)a1, v84);
    if ( *(_QWORD *)(a1 + 112) != *(_QWORD *)(a1 + 104) )
      v29 |= sub_1BDB370(
               a1,
               (__int64)v142,
               a7,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               *(double *)a10.m128i_i64,
               v89,
               v90,
               a13,
               a14,
               v85,
               v86,
               v87,
               v88);
    v29 |= sub_1BE0300(a1, v84, v142, a7, a8, a9, a10, v89, v90, a13, a14, v86, (int)v87, (int)v88);
    if ( *(_QWORD *)(a1 + 168) != *(_QWORD *)(a1 + 160) )
      v29 |= sub_1BDCC60(
               (__int64 *)a1,
               a7,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               *(double *)a10.m128i_i64,
               v73,
               v74,
               a13,
               a14,
               v84,
               (__int64)v142,
               v91,
               v76,
               v72);
    v82 = v120;
    v121 -= 24;
    v83 = (__int64)v120;
    if ( (const __m128i *)v121 != v120 )
    {
      sub_13FE0F0((__int64)&v116);
      v82 = v120;
      v83 = v121;
    }
    v79 = v127;
    v80 = v128;
  }
LABEL_91:
  v93 = v129 - (char *)v79;
  if ( v79 )
    j_j___libc_free_0(v79, v93);
  if ( v125 != v124 )
    _libc_free(v125);
  if ( v120 )
  {
    v93 = v122 - (char *)v120;
    j_j___libc_free_0(v120, v122 - (char *)v120);
  }
  if ( v118 != v117 )
    _libc_free(v118);
  if ( v139 )
  {
    v93 = (__int64)&v141[-v139];
    j_j___libc_free_0(v139, &v141[-v139]);
  }
  if ( v137 != v136[1] )
    _libc_free(v137);
  if ( v133 )
  {
    v93 = v135 - (char *)v133;
    j_j___libc_free_0(v133, v135 - (char *)v133);
  }
  if ( v131 != v130.m128i_i64[1] )
    _libc_free(v131);
  if ( (_BYTE)v29 )
    sub_1BC0610(
      (__int64)v142,
      (__m128)a7,
      *(double *)a8.m128i_i64,
      *(double *)a9.m128i_i64,
      *(double *)a10.m128i_i64,
      v73,
      v74,
      a13,
      a14,
      v93,
      v92,
      v83,
      v76,
      (__int64)v72);
  sub_1BBE790(
    (__int64)v142,
    (__m128)a7,
    *(double *)a8.m128i_i64,
    *(double *)a9.m128i_i64,
    *(double *)a10.m128i_i64,
    v73,
    v74,
    a13,
    a14);
  return v29;
}
