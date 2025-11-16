// Function: sub_30CEA20
// Address: 0x30cea20
//
__int64 __fastcall sub_30CEA20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(__int64 **, __int64),
        __int64 a5,
        __int64 *a6,
        int *a7,
        _BOOL4 a8)
{
  __int64 v10; // r12
  _BOOL4 v11; // ebx
  __int64 v12; // r15
  int v13; // r8d
  unsigned __int64 v14; // rsi
  char *v15; // rax
  char *v16; // r9
  char *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  int *v24; // rax
  int v25; // eax
  int *v26; // rax
  int v27; // r9d
  __int64 v28; // rcx
  int i; // esi
  __int64 v30; // rax
  int v31; // edx
  char v32; // cl
  __int64 v33; // rsi
  __int64 v34; // rdi
  int j; // r9d
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rbx
  char v39; // al
  __int64 v41; // rax
  char v42; // al
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // r9d
  int v47; // r9d
  __int64 v48; // r14
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  __m128i v52; // xmm3
  signed __int64 v53; // rdx
  __m128i v54; // xmm5
  __int64 v55; // rcx
  unsigned __int64 *v56; // r8
  unsigned __int64 *v57; // r12
  unsigned __int64 *v58; // rbx
  unsigned __int64 v59; // rdi
  unsigned __int64 *v60; // r15
  unsigned __int64 *v61; // rbx
  unsigned __int64 v62; // rdi
  int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rcx
  _BOOL4 v66; // r15d
  __int64 v67; // r12
  unsigned __int8 v68; // cl
  _BOOL4 v69; // esi
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // r14
  __int64 v73; // rax
  __int64 v74; // r14
  __int64 v75; // r14
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r9
  __int64 v79; // r15
  __m128i v80; // xmm0
  __m128i v81; // xmm2
  __int64 v82; // r8
  unsigned __int64 *v83; // r15
  unsigned __int64 *v84; // rbx
  unsigned __int64 v85; // rdi
  unsigned __int64 *v86; // rbx
  unsigned __int64 v87; // rdi
  bool v88; // r8
  __int8 v89; // bl
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // r14
  __int64 v93; // r14
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // r15
  __m128i v99; // xmm7
  __m128i v100; // xmm1
  unsigned __int64 *v101; // r8
  unsigned __int64 *v102; // r12
  unsigned __int64 *v103; // rbx
  unsigned __int64 v104; // rdi
  unsigned __int64 *v105; // rbx
  unsigned __int64 v106; // rdi
  __int64 v107; // r13
  __int64 v108; // rax
  __int64 v109; // r13
  __int64 v110; // r13
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __m128i v114; // xmm6
  __m128i v115; // xmm0
  __int64 v116; // rdx
  unsigned __int64 *v117; // r13
  unsigned __int64 *v118; // r15
  unsigned __int64 v119; // rdi
  unsigned __int64 *v120; // rbx
  unsigned __int64 *v121; // r13
  unsigned __int64 v122; // rdi
  __int64 v123; // r14
  __int64 v124; // rax
  __int64 v125; // rcx
  __int64 v126; // r8
  __int64 v127; // r9
  __int64 v128; // rax
  __m128i v129; // xmm4
  signed __int64 v130; // rdx
  __m128i v131; // xmm6
  unsigned __int64 *v132; // r8
  unsigned __int64 *v133; // r12
  unsigned __int64 *v134; // rbx
  unsigned __int64 v135; // rdi
  unsigned __int64 *v136; // rbx
  unsigned __int64 v137; // rdi
  __int64 v138; // rax
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rax
  __int64 v145; // rax
  char v146; // al
  __int64 v147; // rax
  __int64 v148; // rax
  __int64 v149; // [rsp+8h] [rbp-508h]
  __int64 v150; // [rsp+10h] [rbp-500h]
  _QWORD *v151; // [rsp+18h] [rbp-4F8h]
  _QWORD *v152; // [rsp+18h] [rbp-4F8h]
  int v153; // [rsp+18h] [rbp-4F8h]
  int v154; // [rsp+18h] [rbp-4F8h]
  char v155; // [rsp+18h] [rbp-4F8h]
  char v156; // [rsp+18h] [rbp-4F8h]
  int v157; // [rsp+20h] [rbp-4F0h]
  bool v158; // [rsp+26h] [rbp-4EAh]
  int v160; // [rsp+28h] [rbp-4E8h]
  int v161; // [rsp+28h] [rbp-4E8h]
  int v162; // [rsp+28h] [rbp-4E8h]
  int v163; // [rsp+30h] [rbp-4E0h]
  __int64 v165; // [rsp+40h] [rbp-4D0h]
  __int64 v166; // [rsp+40h] [rbp-4D0h]
  __int64 v167; // [rsp+40h] [rbp-4D0h]
  __int64 v168; // [rsp+40h] [rbp-4D0h]
  __int64 v169; // [rsp+40h] [rbp-4D0h]
  __int64 v170; // [rsp+40h] [rbp-4D0h]
  _BYTE v172[12]; // [rsp+50h] [rbp-4C0h] BYREF
  __int64 v173; // [rsp+60h] [rbp-4B0h]
  unsigned __int64 v174; // [rsp+68h] [rbp-4A8h] BYREF
  unsigned int v175; // [rsp+70h] [rbp-4A0h]
  unsigned __int64 v176; // [rsp+78h] [rbp-498h] BYREF
  unsigned int v177; // [rsp+80h] [rbp-490h]
  char v178; // [rsp+88h] [rbp-488h]
  __int64 v179[2]; // [rsp+90h] [rbp-480h] BYREF
  __int64 v180; // [rsp+A0h] [rbp-470h] BYREF
  __int64 *v181; // [rsp+B0h] [rbp-460h]
  __int64 v182; // [rsp+C0h] [rbp-450h] BYREF
  unsigned __int64 v183[2]; // [rsp+E0h] [rbp-430h] BYREF
  _QWORD v184[2]; // [rsp+F0h] [rbp-420h] BYREF
  _QWORD *v185; // [rsp+100h] [rbp-410h]
  _QWORD v186[4]; // [rsp+110h] [rbp-400h] BYREF
  _BYTE *v187; // [rsp+130h] [rbp-3E0h] BYREF
  __int64 v188; // [rsp+138h] [rbp-3D8h]
  _QWORD v189[2]; // [rsp+140h] [rbp-3D0h] BYREF
  _BYTE *v190; // [rsp+150h] [rbp-3C0h]
  __int64 v191; // [rsp+158h] [rbp-3B8h]
  _QWORD v192[2]; // [rsp+160h] [rbp-3B0h] BYREF
  __m128i v193; // [rsp+170h] [rbp-3A0h] BYREF
  __int64 *v194; // [rsp+180h] [rbp-390h] BYREF
  int v195; // [rsp+188h] [rbp-388h]
  char v196; // [rsp+18Ch] [rbp-384h]
  __int64 v197; // [rsp+190h] [rbp-380h] BYREF
  __m128i v198; // [rsp+198h] [rbp-378h] BYREF
  unsigned __int64 v199; // [rsp+1A8h] [rbp-368h] BYREF
  __m128i v200; // [rsp+1B0h] [rbp-360h] BYREF
  __m128i v201; // [rsp+1C0h] [rbp-350h]
  unsigned __int64 *v202; // [rsp+1D0h] [rbp-340h] BYREF
  __int64 v203; // [rsp+1D8h] [rbp-338h]
  _BYTE v204[320]; // [rsp+1E0h] [rbp-330h] BYREF
  char v205; // [rsp+320h] [rbp-1F0h]
  int v206; // [rsp+324h] [rbp-1ECh]
  __int64 v207; // [rsp+328h] [rbp-1E8h]
  __int64 *v208; // [rsp+330h] [rbp-1E0h] BYREF
  size_t v209; // [rsp+338h] [rbp-1D8h]
  __int64 v210; // [rsp+340h] [rbp-1D0h] BYREF
  unsigned __int64 v211; // [rsp+348h] [rbp-1C8h] BYREF
  unsigned int v212; // [rsp+350h] [rbp-1C0h]
  unsigned __int64 v213; // [rsp+358h] [rbp-1B8h] BYREF
  unsigned int v214; // [rsp+360h] [rbp-1B0h]
  char v215; // [rsp+368h] [rbp-1A8h]
  unsigned __int64 *v216; // [rsp+380h] [rbp-190h]
  unsigned int v217; // [rsp+388h] [rbp-188h]
  _BYTE v218[384]; // [rsp+390h] [rbp-180h] BYREF

  v10 = a2;
  v11 = a8;
  ((void (__fastcall *)(_BYTE *, __int64, __int64))a4)(v172, a5, a2);
  v12 = *(_QWORD *)(a2 - 32);
  if ( v12 )
  {
    if ( *(_BYTE *)v12 )
    {
      v12 = 0;
    }
    else if ( *(_QWORD *)(a2 + 80) != *(_QWORD *)(v12 + 24) )
    {
      v12 = 0;
    }
  }
  v165 = sub_B491C0(a2);
  if ( *(_DWORD *)v172 == 0x80000000 )
  {
    v38 = a1;
    v41 = *(_QWORD *)&v172[4];
    *(_DWORD *)a1 = 0x80000000;
    *(_QWORD *)(a1 + 4) = v41;
    goto LABEL_60;
  }
  if ( *(int *)v172 < *(int *)&v172[4] )
    goto LABEL_50;
  if ( *(_DWORD *)v172 == 0x7FFFFFFF )
  {
    if ( !(unsigned __int8)sub_30CA750(v12) )
    {
      v72 = *a6;
      v73 = sub_B2BE50(*a6);
      if ( sub_B6EA50(v73)
        || (v140 = sub_B2BE50(v72),
            v141 = sub_B6F970(v140),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v141 + 48LL))(v141)) )
      {
        sub_B176B0((__int64)&v208, (__int64)"inline", (__int64)"NeverInline", 11, a2);
        sub_B18290((__int64)&v208, "'", 1u);
        sub_B16080((__int64)&v187, "Callee", 6, (unsigned __int8 *)v12);
        v74 = sub_2445430((__int64)&v208, (__int64)&v187);
        sub_B18290(v74, "' not inlined into '", 0x14u);
        sub_B16080((__int64)v183, "Caller", 6, (unsigned __int8 *)v165);
        v75 = sub_2445430(v74, (__int64)v183);
        sub_B18290(v75, "' because it should never be inlined ", 0x25u);
        v79 = sub_30CCC80(v75, (__int64)v172);
        v195 = *(_DWORD *)(v79 + 8);
        v196 = *(_BYTE *)(v79 + 12);
        v197 = *(_QWORD *)(v79 + 16);
        v80 = _mm_loadu_si128((const __m128i *)(v79 + 24));
        v194 = (__int64 *)&unk_49D9D40;
        v198 = v80;
        v199 = *(_QWORD *)(v79 + 40);
        v200 = _mm_loadu_si128((const __m128i *)(v79 + 48));
        v81 = _mm_loadu_si128((const __m128i *)(v79 + 64));
        v202 = (unsigned __int64 *)v204;
        v203 = 0x400000000LL;
        v201 = v81;
        v82 = *(unsigned int *)(v79 + 88);
        if ( (_DWORD)v82 )
          sub_30CDBD0((__int64)&v202, v79 + 80, v76, v77, v82, v78);
        v205 = *(_BYTE *)(v79 + 416);
        v206 = *(_DWORD *)(v79 + 420);
        v207 = *(_QWORD *)(v79 + 424);
        v194 = (__int64 *)&unk_49D9DB0;
        if ( v185 != v186 )
          j_j___libc_free_0((unsigned __int64)v185);
        if ( (_QWORD *)v183[0] != v184 )
          j_j___libc_free_0(v183[0]);
        if ( v190 != (_BYTE *)v192 )
          j_j___libc_free_0((unsigned __int64)v190);
        if ( v187 != (_BYTE *)v189 )
          j_j___libc_free_0((unsigned __int64)v187);
        v208 = (__int64 *)&unk_49D9D40;
        v83 = &v216[10 * v217];
        if ( v216 != v83 )
        {
          v84 = v216;
          do
          {
            v83 -= 10;
            v85 = v83[4];
            if ( (unsigned __int64 *)v85 != v83 + 6 )
              j_j___libc_free_0(v85);
            if ( (unsigned __int64 *)*v83 != v83 + 2 )
              j_j___libc_free_0(*v83);
          }
          while ( v84 != v83 );
          v83 = v216;
        }
        if ( v83 != (unsigned __int64 *)v218 )
          _libc_free((unsigned __int64)v83);
        sub_1049740(a6, (__int64)&v194);
        v86 = v202;
        v194 = (__int64 *)&unk_49D9D40;
        v60 = &v202[10 * (unsigned int)v203];
        if ( v202 == v60 )
          goto LABEL_106;
        do
        {
          v60 -= 10;
          v87 = v60[4];
          if ( (unsigned __int64 *)v87 != v60 + 6 )
            j_j___libc_free_0(v87);
          if ( (unsigned __int64 *)*v60 != v60 + 2 )
            j_j___libc_free_0(*v60);
        }
        while ( v86 != v60 );
        goto LABEL_105;
      }
    }
LABEL_27:
    sub_30CAD10((__int64 *)&v208, (__int64)v172);
    sub_30CB170(v10, v208, v209);
    if ( v208 != &v210 )
      j_j___libc_free_0((unsigned __int64)v208);
    goto LABEL_29;
  }
  v151 = sub_C52410();
  v14 = sub_C959E0();
  v15 = (char *)v151[2];
  v16 = (char *)(v151 + 1);
  if ( v15 )
  {
    v17 = (char *)(v151 + 1);
    do
    {
      while ( 1 )
      {
        v18 = *((_QWORD *)v15 + 2);
        v19 = *((_QWORD *)v15 + 3);
        if ( v14 <= *((_QWORD *)v15 + 4) )
          break;
        v15 = (char *)*((_QWORD *)v15 + 3);
        if ( !v19 )
          goto LABEL_13;
      }
      v17 = v15;
      v15 = (char *)*((_QWORD *)v15 + 2);
    }
    while ( v18 );
LABEL_13:
    if ( v16 != v17 && v14 >= *((_QWORD *)v17 + 4) )
      v16 = v17;
  }
  v152 = v16;
  if ( v16 == (char *)sub_C52410() + 8 )
    goto LABEL_36;
  v20 = v152[7];
  if ( !v20 )
    goto LABEL_36;
  v21 = v152 + 6;
  do
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(v20 + 16);
      v23 = *(_QWORD *)(v20 + 24);
      if ( *(_DWORD *)(v20 + 32) >= dword_502F668 )
        break;
      v20 = *(_QWORD *)(v20 + 24);
      if ( !v23 )
        goto LABEL_22;
    }
    v21 = (_DWORD *)v20;
    v20 = *(_QWORD *)(v20 + 16);
  }
  while ( v22 );
LABEL_22:
  if ( v21 == (_DWORD *)(v152 + 6) || dword_502F668 < v21[8] || (int)v21[9] <= 0 )
  {
LABEL_36:
    v24 = (int *)sub_C94E20((__int64)qword_4F86370);
    if ( v24 )
      v25 = *v24;
    else
      v25 = qword_4F86370[2];
    if ( v25 <= 2 )
      goto LABEL_26;
  }
  else if ( !(_BYTE)qword_502F6E8 )
  {
LABEL_26:
    if ( (unsigned __int8)sub_30CA750(v12) )
      goto LABEL_27;
    v123 = *a6;
    v124 = sub_B2BE50(*a6);
    if ( !sub_B6EA50(v124) )
    {
      v142 = sub_B2BE50(v123);
      v143 = sub_B6F970(v142);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v143 + 48LL))(v143) )
        goto LABEL_27;
    }
    sub_B176B0((__int64)&v208, (__int64)"inline", (__int64)"OptLevel", 8, v10);
    sub_B16080((__int64)&v187, "Callee", 6, (unsigned __int8 *)v12);
    v194 = &v197;
    sub_30CA4D0((__int64 *)&v194, v187, (__int64)&v187[v188]);
    v198.m128i_i64[1] = (__int64)&v200;
    sub_30CA4D0(&v198.m128i_i64[1], v190, (__int64)&v190[v191]);
    v201 = _mm_loadu_si128(&v193);
    sub_B180C0((__int64)&v208, (unsigned __int64)&v194);
    if ( (__m128i *)v198.m128i_i64[1] != &v200 )
      j_j___libc_free_0(v198.m128i_u64[1]);
    if ( v194 != &v197 )
      j_j___libc_free_0((unsigned __int64)v194);
    sub_B18290((__int64)&v208, " not inlined into ", 0x12u);
    sub_B16080((__int64)v183, "Caller", 6, (unsigned __int8 *)v165);
    v169 = sub_2445430((__int64)&v208, (__int64)v183);
    sub_B18290(v169, " because opt level doesn't allow aggressive inlining", 0x34u);
    v128 = v169;
    v195 = *(_DWORD *)(v169 + 8);
    v196 = *(_BYTE *)(v169 + 12);
    v197 = *(_QWORD *)(v169 + 16);
    v129 = _mm_loadu_si128((const __m128i *)(v169 + 24));
    v194 = (__int64 *)&unk_49D9D40;
    v198 = v129;
    v130 = *(_QWORD *)(v169 + 40);
    v199 = v130;
    v200 = _mm_loadu_si128((const __m128i *)(v169 + 48));
    v131 = _mm_loadu_si128((const __m128i *)(v169 + 64));
    v202 = (unsigned __int64 *)v204;
    v203 = 0x400000000LL;
    v201 = v131;
    if ( *(_DWORD *)(v169 + 88) )
    {
      sub_30CDBD0((__int64)&v202, v169 + 80, v130, v125, v126, v127);
      v128 = v169;
    }
    v205 = *(_BYTE *)(v128 + 416);
    v206 = *(_DWORD *)(v128 + 420);
    v207 = *(_QWORD *)(v128 + 424);
    v194 = (__int64 *)&unk_49D9DB0;
    if ( v185 != v186 )
      j_j___libc_free_0((unsigned __int64)v185);
    if ( (_QWORD *)v183[0] != v184 )
      j_j___libc_free_0(v183[0]);
    if ( v190 != (_BYTE *)v192 )
      j_j___libc_free_0((unsigned __int64)v190);
    if ( v187 != (_BYTE *)v189 )
      j_j___libc_free_0((unsigned __int64)v187);
    v208 = (__int64 *)&unk_49D9D40;
    v132 = &v216[10 * v217];
    if ( v216 != v132 )
    {
      v170 = v10;
      v133 = &v216[10 * v217];
      v134 = v216;
      do
      {
        v133 -= 10;
        v135 = v133[4];
        if ( (unsigned __int64 *)v135 != v133 + 6 )
          j_j___libc_free_0(v135);
        if ( (unsigned __int64 *)*v133 != v133 + 2 )
          j_j___libc_free_0(*v133);
      }
      while ( v134 != v133 );
      v10 = v170;
      v132 = v216;
    }
    if ( v132 != (unsigned __int64 *)v218 )
      _libc_free((unsigned __int64)v132);
    sub_1049740(a6, (__int64)&v194);
    v136 = v202;
    v194 = (__int64 *)&unk_49D9D40;
    v60 = &v202[10 * (unsigned int)v203];
    if ( v202 == v60 )
    {
LABEL_106:
      if ( v60 != (unsigned __int64 *)v204 )
        _libc_free((unsigned __int64)v60);
      goto LABEL_27;
    }
    do
    {
      v60 -= 10;
      v137 = v60[4];
      if ( (unsigned __int64 *)v137 != v60 + 6 )
        j_j___libc_free_0(v137);
      if ( (unsigned __int64 *)*v60 != v60 + 2 )
        j_j___libc_free_0(*v60);
    }
    while ( v136 != v60 );
LABEL_105:
    v60 = v202;
    goto LABEL_106;
  }
  v215 = 0;
  v208 = *(__int64 **)v172;
  LODWORD(v209) = *(_DWORD *)&v172[8];
  v210 = v173;
  if ( v178 )
  {
    v212 = v175;
    if ( v175 > 0x40 )
      sub_C43780((__int64)&v211, (const void **)&v174);
    else
      v211 = v174;
    v214 = v177;
    if ( v177 > 0x40 )
      sub_C43780((__int64)&v213, (const void **)&v176);
    else
      v213 = v176;
    v26 = a7;
    v215 = 1;
    v13 = a7[3];
    if ( !v13 )
      goto LABEL_229;
  }
  else
  {
    v26 = a7;
    v13 = a7[3];
    if ( !v13 )
      goto LABEL_186;
  }
  v27 = *v26;
  if ( v12 )
  {
    v28 = *(_QWORD *)(v12 + 80);
    for ( i = 0; v12 + 72 != v28; i += v31 )
    {
      if ( !v28 )
        BUG();
      v30 = *(_QWORD *)(v28 + 32);
      if ( v28 + 24 == v30 )
      {
        v31 = 0;
      }
      else
      {
        v31 = 0;
        do
        {
          v30 = *(_QWORD *)(v30 + 8);
          ++v31;
        }
        while ( v28 + 24 != v30 );
      }
      v28 = *(_QWORD *)(v28 + 8);
    }
  }
  else
  {
    i = -1;
  }
  v32 = v215;
  if ( v13 / 100 >= i || v27 + i <= v13 )
  {
    if ( !v215 )
      goto LABEL_50;
    goto LABEL_230;
  }
  if ( !v215 )
    goto LABEL_186;
LABEL_229:
  v32 = 0;
LABEL_230:
  v215 = 0;
  if ( v214 > 0x40 && v213 )
  {
    v155 = v32;
    j_j___libc_free_0_0(v213);
    v32 = v155;
  }
  if ( v212 > 0x40 && v211 )
  {
    v156 = v32;
    j_j___libc_free_0_0(v211);
    v32 = v156;
  }
  if ( !v32 )
  {
LABEL_186:
    if ( !(unsigned __int8)sub_30CA750(v12) )
    {
      v90 = *a6;
      v91 = sub_B2BE50(*a6);
      if ( sub_B6EA50(v91)
        || (v147 = sub_B2BE50(v90),
            v148 = sub_B6F970(v147),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v148 + 48LL))(v148)) )
      {
        sub_B176B0((__int64)&v208, (__int64)"inline", (__int64)"TooCostly", 9, v10);
        sub_B18290((__int64)&v208, "'", 1u);
        sub_B16080((__int64)&v187, "Callee", 6, (unsigned __int8 *)v12);
        v92 = sub_2445430((__int64)&v208, (__int64)&v187);
        sub_B18290(v92, "' not inlined into '", 0x14u);
        sub_B16080((__int64)v183, "Caller", 6, (unsigned __int8 *)v165);
        v93 = sub_2445430(v92, (__int64)v183);
        sub_B18290(v93, "' because too costly to inline ", 0x1Fu);
        sub_B18290(v93, " ", 1u);
        v98 = sub_30CCC80(v93, (__int64)v172);
        v195 = *(_DWORD *)(v98 + 8);
        v196 = *(_BYTE *)(v98 + 12);
        v197 = *(_QWORD *)(v98 + 16);
        v99 = _mm_loadu_si128((const __m128i *)(v98 + 24));
        v194 = (__int64 *)&unk_49D9D40;
        v198 = v99;
        v199 = *(_QWORD *)(v98 + 40);
        v200 = _mm_loadu_si128((const __m128i *)(v98 + 48));
        v100 = _mm_loadu_si128((const __m128i *)(v98 + 64));
        v202 = (unsigned __int64 *)v204;
        v203 = 0x400000000LL;
        v201 = v100;
        if ( *(_DWORD *)(v98 + 88) )
          sub_30CDBD0((__int64)&v202, v98 + 80, v94, v95, v96, v97);
        v205 = *(_BYTE *)(v98 + 416);
        v206 = *(_DWORD *)(v98 + 420);
        v207 = *(_QWORD *)(v98 + 424);
        v194 = (__int64 *)&unk_49D9DB0;
        if ( v185 != v186 )
          j_j___libc_free_0((unsigned __int64)v185);
        if ( (_QWORD *)v183[0] != v184 )
          j_j___libc_free_0(v183[0]);
        if ( v190 != (_BYTE *)v192 )
          j_j___libc_free_0((unsigned __int64)v190);
        if ( v187 != (_BYTE *)v189 )
          j_j___libc_free_0((unsigned __int64)v187);
        v101 = v216;
        v208 = (__int64 *)&unk_49D9D40;
        if ( v216 != &v216[10 * v217] )
        {
          v168 = v10;
          v102 = &v216[10 * v217];
          v103 = v216;
          do
          {
            v102 -= 10;
            v104 = v102[4];
            if ( (unsigned __int64 *)v104 != v102 + 6 )
              j_j___libc_free_0(v104);
            if ( (unsigned __int64 *)*v102 != v102 + 2 )
              j_j___libc_free_0(*v102);
          }
          while ( v103 != v102 );
          v10 = v168;
          v101 = v216;
        }
        if ( v101 != (unsigned __int64 *)v218 )
          _libc_free((unsigned __int64)v101);
        sub_1049740(a6, (__int64)&v194);
        v60 = v202;
        v194 = (__int64 *)&unk_49D9D40;
        v105 = &v202[10 * (unsigned int)v203];
        if ( v202 == v105 )
          goto LABEL_106;
        do
        {
          v105 -= 10;
          v106 = v105[4];
          if ( (unsigned __int64 *)v106 != v105 + 6 )
            j_j___libc_free_0(v106);
          if ( (unsigned __int64 *)*v105 != v105 + 2 )
            j_j___libc_free_0(*v105);
        }
        while ( v60 != v105 );
        goto LABEL_105;
      }
    }
    goto LABEL_27;
  }
LABEL_50:
  if ( v12 )
  {
    v33 = *(_QWORD *)(v12 + 80);
    v34 = v12 + 72;
    for ( j = 0; v34 != v33; j += v37 )
    {
      while ( 1 )
      {
        if ( !v33 )
          BUG();
        v36 = *(_QWORD *)(v33 + 32);
        if ( v33 + 24 != v36 )
          break;
        v33 = *(_QWORD *)(v33 + 8);
        if ( v34 == v33 )
          goto LABEL_57;
      }
      v37 = 0;
      do
      {
        v36 = *(_QWORD *)(v36 + 8);
        ++v37;
      }
      while ( v33 + 24 != v36 );
      v33 = *(_QWORD *)(v33 + 8);
    }
  }
  else
  {
    j = -1;
  }
LABEL_57:
  if ( j + a7[1] > a7[2] )
  {
    v153 = j;
    v42 = sub_CE9220(v165);
    j = v153;
    if ( !v42 )
    {
      if ( !(unsigned __int8)sub_30CA750(v12) )
      {
        v43 = *a6;
        v44 = sub_B2BE50(*a6);
        v45 = sub_B6EA50(v44);
        v46 = v153;
        if ( v45
          || (v144 = sub_B2BE50(v43),
              v145 = sub_B6F970(v144),
              v146 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v145 + 48LL))(v145),
              v46 = v153,
              v146) )
        {
          v160 = v46;
          sub_B176B0((__int64)&v208, (__int64)"inline", (__int64)"TooCostly", 9, v10);
          sub_B16080((__int64)&v187, "Callee", 6, (unsigned __int8 *)v12);
          v194 = &v197;
          sub_30CA4D0((__int64 *)&v194, v187, (__int64)&v187[v188]);
          v198.m128i_i64[1] = (__int64)&v200;
          sub_30CA4D0(&v198.m128i_i64[1], v190, (__int64)&v190[v191]);
          v201 = _mm_loadu_si128(&v193);
          sub_B180C0((__int64)&v208, (unsigned __int64)&v194);
          v47 = v160;
          if ( (__m128i *)v198.m128i_i64[1] != &v200 )
          {
            j_j___libc_free_0(v198.m128i_u64[1]);
            v47 = v160;
          }
          if ( v194 != &v197 )
          {
            v161 = v47;
            j_j___libc_free_0((unsigned __int64)v194);
            v47 = v161;
          }
          v162 = v47;
          sub_B18290((__int64)&v208, " not inlined into ", 0x12u);
          sub_B16080((__int64)v183, "Caller", 6, (unsigned __int8 *)v165);
          v48 = sub_2445430((__int64)&v208, (__int64)v183);
          sub_B18290(v48, " because callee doesn't have forceinline", 0x28u);
          sub_B18290(v48, " attribute and inlining it would exceed total Inline Budget.", 0x3Cu);
          sub_B18290(v48, " (CalleeSize = ", 0xFu);
          sub_B16530(v179, "CalleeSize", 10, v162);
          v166 = sub_2445430(v48, (__int64)v179);
          sub_B18290(v166, ")", 1u);
          v51 = v166;
          v195 = *(_DWORD *)(v166 + 8);
          v196 = *(_BYTE *)(v166 + 12);
          v197 = *(_QWORD *)(v166 + 16);
          v52 = _mm_loadu_si128((const __m128i *)(v166 + 24));
          v194 = (__int64 *)&unk_49D9D40;
          v198 = v52;
          v53 = *(_QWORD *)(v166 + 40);
          v199 = v53;
          v200 = _mm_loadu_si128((const __m128i *)(v166 + 48));
          v54 = _mm_loadu_si128((const __m128i *)(v166 + 64));
          v202 = (unsigned __int64 *)v204;
          v203 = 0x400000000LL;
          v201 = v54;
          v55 = *(unsigned int *)(v166 + 88);
          if ( (_DWORD)v55 )
          {
            sub_30CDBD0((__int64)&v202, v166 + 80, v53, v55, v49, v50);
            v51 = v166;
          }
          v205 = *(_BYTE *)(v51 + 416);
          v206 = *(_DWORD *)(v51 + 420);
          v207 = *(_QWORD *)(v51 + 424);
          v194 = (__int64 *)&unk_49D9DB0;
          if ( v181 != &v182 )
            j_j___libc_free_0((unsigned __int64)v181);
          if ( (__int64 *)v179[0] != &v180 )
            j_j___libc_free_0(v179[0]);
          if ( v185 != v186 )
            j_j___libc_free_0((unsigned __int64)v185);
          if ( (_QWORD *)v183[0] != v184 )
            j_j___libc_free_0(v183[0]);
          if ( v190 != (_BYTE *)v192 )
            j_j___libc_free_0((unsigned __int64)v190);
          if ( v187 != (_BYTE *)v189 )
            j_j___libc_free_0((unsigned __int64)v187);
          v56 = v216;
          v208 = (__int64 *)&unk_49D9D40;
          if ( v216 != &v216[10 * v217] )
          {
            v167 = v10;
            v57 = v216;
            v58 = &v216[10 * v217];
            do
            {
              v58 -= 10;
              v59 = v58[4];
              if ( (unsigned __int64 *)v59 != v58 + 6 )
                j_j___libc_free_0(v59);
              if ( (unsigned __int64 *)*v58 != v58 + 2 )
                j_j___libc_free_0(*v58);
            }
            while ( v57 != v58 );
            v10 = v167;
            v56 = v216;
          }
          if ( v56 != (unsigned __int64 *)v218 )
            _libc_free((unsigned __int64)v56);
          sub_1049740(a6, (__int64)&v194);
          v60 = v202;
          v194 = (__int64 *)&unk_49D9D40;
          v61 = &v202[10 * (unsigned int)v203];
          if ( v202 == v61 )
            goto LABEL_106;
          do
          {
            v61 -= 10;
            v62 = v61[4];
            if ( (unsigned __int64 *)v62 != v61 + 6 )
              j_j___libc_free_0(v62);
            if ( (unsigned __int64 *)*v61 != v61 + 2 )
              j_j___libc_free_0(*v61);
          }
          while ( v60 != v61 );
          goto LABEL_105;
        }
      }
      goto LABEL_27;
    }
  }
  *a7 += j;
  if ( !a8 )
    goto LABEL_59;
  v200.m128i_i8[8] = 0;
  v194 = *(__int64 **)v172;
  v195 = *(_DWORD *)&v172[8];
  v197 = v173;
  if ( v178 )
  {
    v198.m128i_i32[2] = v175;
    if ( v175 > 0x40 )
      sub_C43780((__int64)&v198, (const void **)&v174);
    else
      v198.m128i_i64[0] = v174;
    v200.m128i_i32[0] = v177;
    if ( v177 > 0x40 )
      sub_C43780((__int64)&v199, (const void **)&v176);
    else
      v199 = v176;
    v200.m128i_i8[8] = 1;
  }
  v63 = *(_BYTE *)(v165 + 32) & 0xF;
  if ( v63 == 7 )
  {
    v154 = (int)v194;
    if ( (int)v194 <= 0 )
      goto LABEL_172;
    v65 = *(_QWORD *)(v165 + 16);
LABEL_241:
    v64 = v65;
    if ( !v65 )
      goto LABEL_172;
    LOBYTE(v13) = *(_QWORD *)(v65 + 8) != 0;
    goto LABEL_114;
  }
  LOBYTE(v13) = (*(_BYTE *)(v165 + 32) & 0xF) != 3 && v63 != 8;
  if ( (_BYTE)v13 )
    goto LABEL_172;
  v154 = (int)v194;
  if ( (int)v194 <= 0 )
    goto LABEL_172;
  v64 = *(_QWORD *)(v165 + 16);
  v65 = v64;
  if ( v63 == 8 )
    goto LABEL_241;
  if ( !v64 )
    goto LABEL_172;
LABEL_114:
  v150 = v12;
  v66 = v13;
  v149 = v10;
  v67 = v64;
  v163 = 0;
  v158 = 0;
  v157 = 0;
  do
  {
    v71 = *(_QWORD *)(v67 + 24);
    if ( *(_BYTE *)v71 <= 0x1Cu
      || (v68 = *(_BYTE *)v71 - 34, v68 > 0x33u)
      || (v69 = ((0x8000000000041uLL >> v68) & 1) == 0, ((0x8000000000041uLL >> v68) & 1) == 0) )
    {
      v69 = 0;
      goto LABEL_120;
    }
    v70 = *(_QWORD *)(v71 - 32);
    if ( !v70 )
      goto LABEL_120;
    if ( *(_BYTE *)v70 )
      goto LABEL_120;
    LOBYTE(v11) = v165 != v70 || *(_QWORD *)(v70 + 24) != *(_QWORD *)(v71 + 80);
    if ( v11 )
      goto LABEL_120;
    a4(&v208, a5);
    if ( (int)v208 >= SHIDWORD(v208) )
      goto LABEL_165;
    if ( (_DWORD)v208 == 0x80000000 )
    {
      v11 = v66;
LABEL_165:
      if ( !v215 )
        goto LABEL_166;
      v215 = 0;
      if ( v214 > 0x40 && v213 )
        j_j___libc_free_0_0(v213);
      if ( v212 > 0x40 && v211 )
      {
        j_j___libc_free_0_0(v211);
        v69 = v11;
      }
      else
      {
LABEL_166:
        v69 = v11;
      }
LABEL_120:
      v66 = v69;
      goto LABEL_121;
    }
    if ( v154 > HIDWORD(v208) - (int)v208 )
    {
      v163 += (int)v208;
      ++v157;
      v158 = a8;
    }
    if ( v215 )
    {
      v215 = 0;
      if ( v214 > 0x40 && v213 )
        j_j___libc_free_0_0(v213);
      if ( v212 > 0x40 && v211 )
        j_j___libc_free_0_0(v211);
    }
LABEL_121:
    v67 = *(_QWORD *)(v67 + 8);
  }
  while ( v67 );
  v88 = v66;
  v10 = v149;
  v12 = v150;
  if ( !v158 )
    goto LABEL_172;
  if ( v88 )
    v163 -= sub_DF9440(a3);
  if ( (int)qword_502F8A8 >= 0 )
  {
    if ( (int)v194 * v157 + v163 < (int)qword_502F8A8 * (int)v194 )
      goto LABEL_222;
LABEL_172:
    if ( v200.m128i_i8[8] )
    {
      v89 = 0;
      goto LABEL_174;
    }
    goto LABEL_59;
  }
  if ( (int)v194 <= v163 )
    goto LABEL_172;
LABEL_222:
  v89 = v200.m128i_i8[8];
  if ( !v200.m128i_i8[8] )
  {
LABEL_181:
    if ( !(unsigned __int8)sub_30CA750(v12) )
    {
      v107 = *a6;
      v108 = sub_B2BE50(*a6);
      if ( sub_B6EA50(v108)
        || (v138 = sub_B2BE50(v107),
            v139 = sub_B6F970(v138),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v139 + 48LL))(v139)) )
      {
        sub_B176B0((__int64)&v208, (__int64)"inline", (__int64)"IncreaseCostInOtherContexts", 27, v10);
        sub_B18290((__int64)&v208, "Not inlining. Cost of inlining '", 0x20u);
        sub_B16080((__int64)&v187, "Callee", 6, (unsigned __int8 *)v12);
        v109 = sub_2445430((__int64)&v208, (__int64)&v187);
        sub_B18290(v109, "' increases the cost of inlining '", 0x22u);
        sub_B16080((__int64)v183, "Caller", 6, (unsigned __int8 *)v165);
        v110 = sub_2445430(v109, (__int64)v183);
        sub_B18290(v110, "' in other contexts", 0x13u);
        v195 = *(_DWORD *)(v110 + 8);
        v196 = *(_BYTE *)(v110 + 12);
        v197 = *(_QWORD *)(v110 + 16);
        v114 = _mm_loadu_si128((const __m128i *)(v110 + 24));
        v194 = (__int64 *)&unk_49D9D40;
        v198 = v114;
        v199 = *(_QWORD *)(v110 + 40);
        v200 = _mm_loadu_si128((const __m128i *)(v110 + 48));
        v115 = _mm_loadu_si128((const __m128i *)(v110 + 64));
        v202 = (unsigned __int64 *)v204;
        v203 = 0x400000000LL;
        v201 = v115;
        v116 = *(unsigned int *)(v110 + 88);
        if ( (_DWORD)v116 )
          sub_30CDBD0((__int64)&v202, v110 + 80, v116, v111, v112, v113);
        v205 = *(_BYTE *)(v110 + 416);
        v206 = *(_DWORD *)(v110 + 420);
        v207 = *(_QWORD *)(v110 + 424);
        v194 = (__int64 *)&unk_49D9DB0;
        if ( v185 != v186 )
          j_j___libc_free_0((unsigned __int64)v185);
        if ( (_QWORD *)v183[0] != v184 )
          j_j___libc_free_0(v183[0]);
        if ( v190 != (_BYTE *)v192 )
          j_j___libc_free_0((unsigned __int64)v190);
        if ( v187 != (_BYTE *)v189 )
          j_j___libc_free_0((unsigned __int64)v187);
        v117 = v216;
        v208 = (__int64 *)&unk_49D9D40;
        v118 = &v216[10 * v217];
        if ( v216 != v118 )
        {
          do
          {
            v118 -= 10;
            v119 = v118[4];
            if ( (unsigned __int64 *)v119 != v118 + 6 )
              j_j___libc_free_0(v119);
            if ( (unsigned __int64 *)*v118 != v118 + 2 )
              j_j___libc_free_0(*v118);
          }
          while ( v117 != v118 );
          v118 = v216;
        }
        if ( v118 != (unsigned __int64 *)v218 )
          _libc_free((unsigned __int64)v118);
        sub_1049740(a6, (__int64)&v194);
        v120 = v202;
        v194 = (__int64 *)&unk_49D9D40;
        v121 = &v202[10 * (unsigned int)v203];
        if ( v202 != v121 )
        {
          do
          {
            v121 -= 10;
            v122 = v121[4];
            if ( (unsigned __int64 *)v122 != v121 + 6 )
              j_j___libc_free_0(v122);
            if ( (unsigned __int64 *)*v121 != v121 + 2 )
              j_j___libc_free_0(*v121);
          }
          while ( v120 != v121 );
          v121 = v202;
        }
        if ( v121 != (unsigned __int64 *)v204 )
          _libc_free((unsigned __int64)v121);
      }
    }
    sub_30CB170(v10, "deferred", 8u);
LABEL_29:
    *(_BYTE *)(a1 + 64) = 0;
    if ( !v178 )
      return a1;
    goto LABEL_30;
  }
LABEL_174:
  v200.m128i_i8[8] = 0;
  if ( v200.m128i_i32[0] > 0x40u && v199 )
    j_j___libc_free_0_0(v199);
  if ( v198.m128i_i32[2] > 0x40u && v198.m128i_i64[0] )
    j_j___libc_free_0_0(v198.m128i_u64[0]);
  if ( v89 )
    goto LABEL_181;
LABEL_59:
  v38 = a1;
  *(_QWORD *)a1 = *(_QWORD *)v172;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)&v172[8];
LABEL_60:
  *(_QWORD *)(v38 + 16) = v173;
  v39 = v178;
  *(_BYTE *)(v38 + 56) = 0;
  if ( v39 )
  {
    *(_DWORD *)(v38 + 32) = v175;
    *(_QWORD *)(v38 + 24) = v174;
    v175 = 0;
    *(_DWORD *)(v38 + 48) = v177;
    *(_QWORD *)(v38 + 40) = v176;
    v177 = 0;
    *(_BYTE *)(v38 + 56) = 1;
  }
  *(_BYTE *)(a1 + 64) = 1;
  if ( v39 )
  {
LABEL_30:
    v178 = 0;
    if ( v177 > 0x40 && v176 )
      j_j___libc_free_0_0(v176);
    if ( v175 > 0x40 && v174 )
      j_j___libc_free_0_0(v174);
  }
  return a1;
}
