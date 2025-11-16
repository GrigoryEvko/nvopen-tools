// Function: sub_19DF5F0
// Address: 0x19df5f0
//
_BOOL8 __fastcall sub_19DF5F0(
        __int64 a1,
        __m128i a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  int v10; // r12d
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned int v14; // eax
  _QWORD *v15; // r14
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 *v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rsi
  __int64 *v32; // rdi
  unsigned __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 *v37; // rax
  char v38; // r8
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // r8
  unsigned __int64 v41; // rbx
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rax
  char v46; // si
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // rdx
  __int64 v50; // r14
  __int64 *v51; // r15
  _QWORD *v52; // r12
  __int64 v53; // r15
  bool v54; // bl
  unsigned int v55; // ecx
  __int64 *v56; // rax
  char v57; // dl
  __int64 v58; // rbx
  __int64 *v59; // rax
  __int64 *v60; // rcx
  __int64 *v61; // rsi
  unsigned __int64 v62; // rcx
  char v63; // si
  char v64; // al
  bool v65; // al
  __int64 v67; // r12
  __int64 *v68; // rax
  __int64 v69; // rdi
  double v70; // xmm4_8
  double v71; // xmm5_8
  __int64 v72; // rcx
  __int64 v73; // rcx
  __int64 v74; // rax
  unsigned int v75; // esi
  __int64 v76; // rcx
  __int64 v77; // r10
  unsigned int v78; // ebx
  unsigned int v79; // edi
  __int64 *v80; // rax
  __int64 v81; // r9
  bool v82; // bl
  unsigned int v83; // esi
  unsigned __int64 *v84; // rdi
  unsigned __int64 v85; // rsi
  bool v86; // zf
  unsigned int v87; // esi
  __int64 v88; // r10
  unsigned int v89; // ecx
  __int64 *v90; // rax
  __int64 v91; // rdi
  unsigned int v92; // ecx
  unsigned __int64 *v93; // rdi
  unsigned __int64 v94; // rcx
  int v95; // r8d
  __int64 *v96; // rdx
  int v97; // edi
  int v98; // edi
  _QWORD *v99; // rax
  _QWORD *v100; // r14
  __int64 v101; // rdx
  int v102; // eax
  int v103; // r8d
  __int64 v104; // rdx
  __int64 v105; // rsi
  int v106; // r10d
  __int64 *v107; // r9
  __int64 v108; // r11
  int v109; // eax
  int v110; // r8d
  __int64 v111; // rdx
  __int64 v112; // rsi
  __int64 v113; // r11
  int v114; // r10d
  int v115; // edx
  __int64 *v116; // r9
  int v117; // edi
  int v118; // ecx
  int v119; // eax
  int v120; // r8d
  __int64 v121; // rdx
  int v122; // r9d
  __int64 v123; // rsi
  __int64 *v124; // rdi
  __int64 v125; // r10
  int v126; // eax
  int v127; // r8d
  __int64 v128; // rdx
  __int64 v129; // rsi
  __int64 v130; // r10
  int v131; // r9d
  int v132; // edx
  __int64 v133; // rbx
  unsigned int v134; // eax
  _QWORD *v135; // rdi
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rax
  _QWORD *v138; // rax
  __int64 v139; // rdx
  _QWORD *i; // rdx
  _QWORD *v141; // rax
  __int64 *v142; // [rsp+0h] [rbp-370h]
  __int64 *v143; // [rsp+0h] [rbp-370h]
  __int64 v144; // [rsp+0h] [rbp-370h]
  __int64 v145; // [rsp+8h] [rbp-368h]
  bool v146; // [rsp+17h] [rbp-359h]
  __int64 v147; // [rsp+18h] [rbp-358h]
  __int64 *v148; // [rsp+20h] [rbp-350h]
  __int64 v149; // [rsp+28h] [rbp-348h]
  __int64 v150; // [rsp+28h] [rbp-348h]
  __int64 v151; // [rsp+28h] [rbp-348h]
  __int64 v152; // [rsp+28h] [rbp-348h]
  __int64 v153; // [rsp+28h] [rbp-348h]
  __int64 *v154; // [rsp+28h] [rbp-348h]
  __int64 v155; // [rsp+28h] [rbp-348h]
  __int64 v156; // [rsp+28h] [rbp-348h]
  __int64 v157; // [rsp+38h] [rbp-338h]
  __int64 *v158; // [rsp+38h] [rbp-338h]
  __int64 *v159; // [rsp+38h] [rbp-338h]
  __int64 *v160; // [rsp+38h] [rbp-338h]
  _QWORD *v161; // [rsp+38h] [rbp-338h]
  _QWORD v162[16]; // [rsp+40h] [rbp-330h] BYREF
  __int64 v163; // [rsp+C0h] [rbp-2B0h] BYREF
  _QWORD *v164; // [rsp+C8h] [rbp-2A8h]
  _QWORD *v165; // [rsp+D0h] [rbp-2A0h]
  __int64 v166; // [rsp+D8h] [rbp-298h]
  int v167; // [rsp+E0h] [rbp-290h]
  _QWORD v168[8]; // [rsp+E8h] [rbp-288h] BYREF
  unsigned __int64 v169; // [rsp+128h] [rbp-248h] BYREF
  unsigned __int64 v170; // [rsp+130h] [rbp-240h]
  unsigned __int64 v171; // [rsp+138h] [rbp-238h]
  __int64 v172; // [rsp+140h] [rbp-230h] BYREF
  __int64 *v173; // [rsp+148h] [rbp-228h]
  __int64 *v174; // [rsp+150h] [rbp-220h]
  unsigned int v175; // [rsp+158h] [rbp-218h]
  unsigned int v176; // [rsp+15Ch] [rbp-214h]
  int v177; // [rsp+160h] [rbp-210h]
  _BYTE v178[64]; // [rsp+168h] [rbp-208h] BYREF
  unsigned __int64 v179; // [rsp+1A8h] [rbp-1C8h] BYREF
  unsigned __int64 v180; // [rsp+1B0h] [rbp-1C0h]
  unsigned __int64 v181; // [rsp+1B8h] [rbp-1B8h]
  __int64 v182; // [rsp+1C0h] [rbp-1B0h] BYREF
  __int64 v183; // [rsp+1C8h] [rbp-1A8h]
  unsigned __int64 v184; // [rsp+1D0h] [rbp-1A0h]
  _BYTE v185[64]; // [rsp+1E8h] [rbp-188h] BYREF
  unsigned __int64 v186; // [rsp+228h] [rbp-148h]
  unsigned __int64 v187; // [rsp+230h] [rbp-140h]
  unsigned __int64 v188; // [rsp+238h] [rbp-138h]
  _QWORD v189[2]; // [rsp+240h] [rbp-130h] BYREF
  unsigned __int64 v190; // [rsp+250h] [rbp-120h]
  char v191[64]; // [rsp+268h] [rbp-108h] BYREF
  __int64 *v192; // [rsp+2A8h] [rbp-C8h]
  __int64 *v193; // [rsp+2B0h] [rbp-C0h]
  unsigned __int64 v194; // [rsp+2B8h] [rbp-B8h]
  _QWORD v195[2]; // [rsp+2C0h] [rbp-B0h] BYREF
  unsigned __int64 v196; // [rsp+2D0h] [rbp-A0h]
  char v197[64]; // [rsp+2E8h] [rbp-88h] BYREF
  unsigned __int64 v198; // [rsp+328h] [rbp-48h]
  unsigned __int64 v199; // [rsp+330h] [rbp-40h]
  unsigned __int64 v200; // [rsp+338h] [rbp-38h]

  v10 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  v145 = a1 + 48;
  if ( !v10 && !*(_DWORD *)(a1 + 68) )
    goto LABEL_20;
  v11 = *(_QWORD *)(a1 + 56);
  v12 = *(unsigned int *)(a1 + 72);
  v13 = v11 + 72 * v12;
  v14 = 4 * v10;
  if ( (unsigned int)(4 * v10) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v12 <= v14 )
  {
    for ( ; v11 != v13; v11 += 72 )
    {
      if ( *(_QWORD *)v11 != -8 )
      {
        if ( *(_QWORD *)v11 != -16 )
        {
          v15 = *(_QWORD **)(v11 + 8);
          v16 = &v15[3 * *(unsigned int *)(v11 + 16)];
          if ( v15 != v16 )
          {
            do
            {
              v17 = *(v16 - 1);
              v16 -= 3;
              if ( v17 != -8 && v17 != 0 && v17 != -16 )
                sub_1649B30(v16);
            }
            while ( v15 != v16 );
            v16 = *(_QWORD **)(v11 + 8);
          }
          if ( v16 != (_QWORD *)(v11 + 24) )
            _libc_free((unsigned __int64)v16);
        }
        *(_QWORD *)v11 = -8;
      }
    }
    goto LABEL_19;
  }
  do
  {
    while ( *(_QWORD *)v11 == -16 )
    {
LABEL_168:
      v11 += 72;
      if ( v11 == v13 )
        goto LABEL_218;
    }
    if ( *(_QWORD *)v11 != -8 )
    {
      v99 = *(_QWORD **)(v11 + 8);
      v100 = &v99[3 * *(unsigned int *)(v11 + 16)];
      if ( v99 != v100 )
      {
        do
        {
          v101 = *(v100 - 1);
          v100 -= 3;
          if ( v101 != 0 && v101 != -8 && v101 != -16 )
          {
            v161 = v99;
            sub_1649B30(v100);
            v99 = v161;
          }
        }
        while ( v99 != v100 );
        v100 = *(_QWORD **)(v11 + 8);
      }
      if ( v100 != (_QWORD *)(v11 + 24) )
        _libc_free((unsigned __int64)v100);
      goto LABEL_168;
    }
    v11 += 72;
  }
  while ( v11 != v13 );
LABEL_218:
  v132 = *(_DWORD *)(a1 + 72);
  if ( !v10 )
  {
    if ( v132 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 56));
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_DWORD *)(a1 + 72) = 0;
      goto LABEL_20;
    }
LABEL_19:
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_20;
  }
  v133 = 64;
  if ( v10 != 1 )
  {
    _BitScanReverse(&v134, v10 - 1);
    v133 = (unsigned int)(1 << (33 - (v134 ^ 0x1F)));
    if ( (int)v133 < 64 )
      v133 = 64;
  }
  v135 = *(_QWORD **)(a1 + 56);
  if ( (_DWORD)v133 == v132 )
  {
    *(_QWORD *)(a1 + 64) = 0;
    v141 = &v135[9 * v133];
    do
    {
      if ( v135 )
        *v135 = -8;
      v135 += 9;
    }
    while ( v141 != v135 );
  }
  else
  {
    j___libc_free_0(v135);
    v136 = ((((((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v133 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
            | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v133 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 8)
          | (((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v133 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v133 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 16;
    v137 = (v136
          | (((((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v133 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
            | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v133 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 8)
          | (((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v133 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v133 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1))
         + 1;
    *(_DWORD *)(a1 + 72) = v137;
    v138 = (_QWORD *)sub_22077B0(72 * v137);
    v139 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v138;
    for ( i = &v138[9 * v139]; i != v138; v138 += 9 )
    {
      if ( v138 )
        *v138 = -8;
    }
  }
LABEL_20:
  v18 = *(_QWORD *)(a1 + 16);
  memset(v162, 0, sizeof(v162));
  v162[1] = &v162[5];
  v162[2] = &v162[5];
  v19 = *(_QWORD *)(v18 + 56);
  v166 = 0x100000008LL;
  v164 = v168;
  v165 = v168;
  v168[0] = v19;
  v189[0] = v19;
  LODWORD(v162[3]) = 8;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v167 = 0;
  v163 = 1;
  LOBYTE(v190) = 0;
  sub_13B8390(&v169, (__int64)v189);
  sub_16CCEE0(&v182, (__int64)v185, 8, (__int64)v162);
  v20 = v162[13];
  memset(&v162[13], 0, 24);
  v186 = v20;
  v187 = v162[14];
  v188 = v162[15];
  sub_16CCEE0(&v172, (__int64)v178, 8, (__int64)&v163);
  v21 = v169;
  v169 = 0;
  v179 = v21;
  v22 = v170;
  v170 = 0;
  v180 = v22;
  v23 = v171;
  v171 = 0;
  v181 = v23;
  sub_16CCEE0(v189, (__int64)v191, 8, (__int64)&v172);
  v24 = v179;
  v179 = 0;
  v192 = (__int64 *)v24;
  v25 = (__int64 *)v180;
  v180 = 0;
  v193 = v25;
  v26 = v181;
  v181 = 0;
  v194 = v26;
  sub_16CCEE0(v195, (__int64)v197, 8, (__int64)&v182);
  v27 = v186;
  v186 = 0;
  v198 = v27;
  v28 = v187;
  v187 = 0;
  v199 = v28;
  v29 = v188;
  v188 = 0;
  v200 = v29;
  if ( v179 )
    j_j___libc_free_0(v179, v181 - v179);
  if ( v174 != v173 )
    _libc_free((unsigned __int64)v174);
  if ( v186 )
    j_j___libc_free_0(v186, v188 - v186);
  if ( v184 != v183 )
    _libc_free(v184);
  if ( v169 )
    j_j___libc_free_0(v169, v171 - v169);
  if ( v165 != v164 )
    _libc_free((unsigned __int64)v165);
  if ( v162[13] )
    j_j___libc_free_0(v162[13], v162[15] - v162[13]);
  if ( v162[2] != v162[1] )
    _libc_free(v162[2]);
  sub_16CCCB0(&v172, (__int64)v178, (__int64)v189);
  v31 = v193;
  v32 = v192;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v33 = (char *)v193 - (char *)v192;
  if ( v193 != v192 )
  {
    if ( v33 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v34 = sub_22077B0((char *)v193 - (char *)v192);
      v31 = v193;
      v32 = v192;
      v35 = v34;
      goto LABEL_39;
    }
LABEL_248:
    sub_4261EA(v32, v31, v30);
  }
  v33 = 0;
  v35 = 0;
LABEL_39:
  v179 = v35;
  v180 = v35;
  v181 = v35 + v33;
  if ( v31 != v32 )
  {
    v36 = v35;
    v37 = v32;
    do
    {
      if ( v36 )
      {
        *(_QWORD *)v36 = *v37;
        v38 = *((_BYTE *)v37 + 16);
        *(_BYTE *)(v36 + 16) = v38;
        if ( v38 )
          *(_QWORD *)(v36 + 8) = v37[1];
      }
      v37 += 3;
      v36 += 24LL;
    }
    while ( v37 != v31 );
    v35 += 8 * ((unsigned __int64)((char *)(v37 - 3) - (char *)v32) >> 3) + 24;
  }
  v31 = (__int64 *)v185;
  v32 = &v182;
  v180 = v35;
  sub_16CCCB0(&v182, (__int64)v185, (__int64)v195);
  v39 = v199;
  v40 = v198;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v41 = v199 - v198;
  if ( v199 == v198 )
  {
    v43 = 0;
  }
  else
  {
    if ( v41 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_248;
    v42 = sub_22077B0(v199 - v198);
    v39 = v199;
    v40 = v198;
    v43 = v42;
  }
  v186 = v43;
  v187 = v43;
  v188 = v43 + v41;
  if ( v39 == v40 )
  {
    v47 = v43;
  }
  else
  {
    v44 = v43;
    v45 = v40;
    do
    {
      if ( v44 )
      {
        *(_QWORD *)v44 = *(_QWORD *)v45;
        v46 = *(_BYTE *)(v45 + 16);
        *(_BYTE *)(v44 + 16) = v46;
        if ( v46 )
          *(_QWORD *)(v44 + 8) = *(_QWORD *)(v45 + 8);
      }
      v45 += 24LL;
      v44 += 24LL;
    }
    while ( v45 != v39 );
    v47 = v43 + 8 * ((v45 - 24 - v40) >> 3) + 24;
  }
  v48 = v180;
  v49 = v179;
  v50 = a1;
  v51 = &v172;
  v187 = v47;
  v146 = 0;
  if ( v180 - v179 == v47 - v43 )
    goto LABEL_81;
  while ( 2 )
  {
    while ( 2 )
    {
      v52 = *(_QWORD **)(v48 - 24);
      v147 = *v52;
      if ( *(_QWORD *)(*v52 + 48LL) == *v52 + 40LL )
        goto LABEL_66;
      v148 = v51;
      v53 = *(_QWORD *)(*v52 + 48LL);
      while ( 2 )
      {
        if ( !v53 )
          goto LABEL_257;
        v54 = sub_1456C80(*(_QWORD *)(v50 + 24), *(_QWORD *)(v53 - 24));
        if ( v54 )
        {
          v55 = *(unsigned __int8 *)(v53 - 8) - 24;
          if ( v55 <= 0x20 && ((1LL << v55) & 0x100008800LL) != 0 )
          {
            v67 = v53 - 24;
            v157 = sub_146F1B0(*(_QWORD *)(v50 + 24), v53 - 24);
            v68 = sub_19DF200((__int64 *)v50, v53 - 24, a2, a3, a4);
            v69 = *(_QWORD *)(v50 + 24);
            v149 = (__int64)v68;
            if ( v68 )
            {
              sub_1464C80(v69, v53 - 24);
              sub_164D160(v53 - 24, v149, (__m128)a2, *(double *)a3.m128i_i64, a4, a5, v70, v71, a8, a9);
              v72 = v149;
              v163 = 4;
              v164 = 0;
              v165 = (_QWORD *)v149;
              if ( v149 != -8 && v149 != -16 )
              {
                sub_164C220((__int64)&v163);
                v72 = v149;
              }
              v150 = v72;
              sub_1AEB370(v53 - 24, *(_QWORD *)(v50 + 32));
              v73 = v150;
              if ( !v165 )
              {
                v146 = v54;
                v53 = *(_QWORD *)(v147 + 48);
                goto LABEL_63;
              }
              v53 = v150 + 24;
              if ( v165 != (_QWORD *)-8LL && v165 != (_QWORD *)-16LL )
              {
                sub_1649B30(&v163);
                v73 = v150;
              }
              v146 = v54;
              v69 = *(_QWORD *)(v50 + 24);
              v67 = v73;
            }
            v74 = sub_146F1B0(v69, v67);
            v75 = *(_DWORD *)(v50 + 72);
            v76 = v74;
            if ( v75 )
            {
              v77 = *(_QWORD *)(v50 + 56);
              v78 = ((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4);
              v79 = (v75 - 1) & v78;
              v80 = (__int64 *)(v77 + 72LL * v79);
              v81 = *v80;
              if ( v76 == *v80 )
                goto LABEL_119;
              v95 = 1;
              v96 = 0;
              while ( v81 != -8 )
              {
                if ( v81 == -16 && !v96 )
                  v96 = v80;
                v79 = (v75 - 1) & (v95 + v79);
                v80 = (__int64 *)(v77 + 72LL * v79);
                v81 = *v80;
                if ( v76 == *v80 )
                  goto LABEL_119;
                ++v95;
              }
              v97 = *(_DWORD *)(v50 + 64);
              if ( v96 )
                v80 = v96;
              ++*(_QWORD *)(v50 + 48);
              v98 = v97 + 1;
              if ( 4 * v98 < 3 * v75 )
              {
                if ( v75 - *(_DWORD *)(v50 + 68) - v98 > v75 >> 3 )
                  goto LABEL_157;
                v155 = v76;
                sub_19DF220(v145, v75);
                v102 = *(_DWORD *)(v50 + 72);
                if ( !v102 )
                  goto LABEL_256;
                v103 = v102 - 1;
                v104 = *(_QWORD *)(v50 + 56);
                LODWORD(v105) = (v102 - 1) & v78;
                v76 = v155;
                v106 = 1;
                v107 = 0;
                v98 = *(_DWORD *)(v50 + 64) + 1;
                v80 = (__int64 *)(v104 + 72LL * (unsigned int)v105);
                v108 = *v80;
                if ( v155 == *v80 )
                  goto LABEL_157;
                while ( v108 != -8 )
                {
                  if ( !v107 && v108 == -16 )
                    v107 = v80;
                  v105 = v103 & (unsigned int)(v105 + v106);
                  v80 = (__int64 *)(v104 + 72 * v105);
                  v108 = *v80;
                  if ( v155 == *v80 )
                    goto LABEL_157;
                  ++v106;
                }
LABEL_177:
                if ( v107 )
                  v80 = v107;
LABEL_157:
                *(_DWORD *)(v50 + 64) = v98;
                if ( *v80 != -8 )
                  --*(_DWORD *)(v50 + 68);
                *v80 = v76;
                v80[1] = (__int64)(v80 + 3);
                v80[2] = 0x200000000LL;
LABEL_119:
                v165 = (_QWORD *)v67;
                v163 = 6;
                v164 = 0;
                v82 = v67 != -8 && v67 != -16;
                if ( v82 )
                {
                  v142 = v80;
                  v151 = v76;
                  sub_164C220((__int64)&v163);
                  v80 = v142;
                  v76 = v151;
                }
                v83 = *((_DWORD *)v80 + 4);
                if ( v83 >= *((_DWORD *)v80 + 5) )
                {
                  v144 = v76;
                  v154 = v80;
                  sub_170B450((__int64)(v80 + 1), 0);
                  v80 = v154;
                  v76 = v144;
                  v83 = *((_DWORD *)v154 + 4);
                }
                v84 = (unsigned __int64 *)(v80[1] + 24LL * v83);
                if ( v84 )
                {
                  *v84 = 6;
                  v84[1] = 0;
                  v85 = (unsigned __int64)v165;
                  v86 = v165 + 1 == 0;
                  v84[2] = (unsigned __int64)v165;
                  if ( v85 == 0 || v86 || v85 == -16 )
                  {
                    v83 = *((_DWORD *)v80 + 4);
                  }
                  else
                  {
                    v143 = v80;
                    v152 = v76;
                    sub_1649AC0(v84, v163 & 0xFFFFFFFFFFFFFFF8LL);
                    v80 = v143;
                    v76 = v152;
                    v83 = *((_DWORD *)v143 + 4);
                  }
                }
                *((_DWORD *)v80 + 4) = v83 + 1;
                if ( v165 != 0 && v165 + 1 != 0 && v165 != (_QWORD *)-16LL )
                {
                  v153 = v76;
                  sub_1649B30(&v163);
                  v76 = v153;
                }
                if ( v157 == v76 )
                  goto LABEL_63;
                v87 = *(_DWORD *)(v50 + 72);
                if ( v87 )
                {
                  v88 = *(_QWORD *)(v50 + 56);
                  v89 = (v87 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
                  v90 = (__int64 *)(v88 + 72LL * v89);
                  v91 = *v90;
                  if ( v157 == *v90 )
                  {
LABEL_133:
                    v163 = 6;
                    v164 = 0;
                    v165 = (_QWORD *)v67;
                    if ( v82 )
                    {
                      v158 = v90;
                      sub_164C220((__int64)&v163);
                      v90 = v158;
                    }
                    v92 = *((_DWORD *)v90 + 4);
                    if ( v92 >= *((_DWORD *)v90 + 5) )
                    {
                      v160 = v90;
                      sub_170B450((__int64)(v90 + 1), 0);
                      v90 = v160;
                      v92 = *((_DWORD *)v160 + 4);
                    }
                    v93 = (unsigned __int64 *)(v90[1] + 24LL * v92);
                    if ( v93 )
                    {
                      *v93 = 6;
                      v93[1] = 0;
                      v94 = (unsigned __int64)v165;
                      v86 = v165 + 1 == 0;
                      v93[2] = (unsigned __int64)v165;
                      if ( v94 != 0 && !v86 && v94 != -16 )
                      {
                        v159 = v90;
                        sub_1649AC0(v93, v163 & 0xFFFFFFFFFFFFFFF8LL);
                        v90 = v159;
                      }
                      v92 = *((_DWORD *)v90 + 4);
                    }
                    *((_DWORD *)v90 + 4) = v92 + 1;
                    if ( v165 != 0 && v165 + 1 != 0 && v165 != (_QWORD *)-16LL )
                      sub_1649B30(&v163);
                    goto LABEL_63;
                  }
                  v115 = 1;
                  v116 = 0;
                  while ( v91 != -8 )
                  {
                    if ( !v116 && v91 == -16 )
                      v116 = v90;
                    v89 = (v87 - 1) & (v115 + v89);
                    v90 = (__int64 *)(v88 + 72LL * v89);
                    v91 = *v90;
                    if ( v157 == *v90 )
                      goto LABEL_133;
                    ++v115;
                  }
                  v117 = *(_DWORD *)(v50 + 64);
                  if ( v116 )
                    v90 = v116;
                  ++*(_QWORD *)(v50 + 48);
                  v118 = v117 + 1;
                  if ( 4 * (v117 + 1) < 3 * v87 )
                  {
                    if ( v87 - *(_DWORD *)(v50 + 68) - v118 > v87 >> 3 )
                      goto LABEL_198;
                    sub_19DF220(v145, v87);
                    v119 = *(_DWORD *)(v50 + 72);
                    if ( !v119 )
                      goto LABEL_256;
                    v120 = v119 - 1;
                    v121 = *(_QWORD *)(v50 + 56);
                    v122 = 1;
                    LODWORD(v123) = (v119 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
                    v118 = *(_DWORD *)(v50 + 64) + 1;
                    v124 = 0;
                    v90 = (__int64 *)(v121 + 72LL * (unsigned int)v123);
                    v125 = *v90;
                    if ( v157 == *v90 )
                      goto LABEL_198;
                    while ( v125 != -8 )
                    {
                      if ( v125 == -16 && !v124 )
                        v124 = v90;
                      v123 = v120 & (unsigned int)(v123 + v122);
                      v90 = (__int64 *)(v121 + 72 * v123);
                      v125 = *v90;
                      if ( v157 == *v90 )
                        goto LABEL_198;
                      ++v122;
                    }
LABEL_204:
                    if ( v124 )
                      v90 = v124;
LABEL_198:
                    *(_DWORD *)(v50 + 64) = v118;
                    if ( *v90 != -8 )
                      --*(_DWORD *)(v50 + 68);
                    v90[1] = (__int64)(v90 + 3);
                    *v90 = v157;
                    v90[2] = 0x200000000LL;
                    goto LABEL_133;
                  }
                }
                else
                {
                  ++*(_QWORD *)(v50 + 48);
                }
                sub_19DF220(v145, 2 * v87);
                v126 = *(_DWORD *)(v50 + 72);
                if ( !v126 )
                {
LABEL_256:
                  ++*(_DWORD *)(v50 + 64);
LABEL_257:
                  BUG();
                }
                v127 = v126 - 1;
                v128 = *(_QWORD *)(v50 + 56);
                v118 = *(_DWORD *)(v50 + 64) + 1;
                LODWORD(v129) = (v126 - 1) & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
                v90 = (__int64 *)(v128 + 72LL * (unsigned int)v129);
                v130 = *v90;
                if ( v157 == *v90 )
                  goto LABEL_198;
                v131 = 1;
                v124 = 0;
                while ( v130 != -8 )
                {
                  if ( !v124 && v130 == -16 )
                    v124 = v90;
                  v129 = v127 & (unsigned int)(v129 + v131);
                  v90 = (__int64 *)(v128 + 72 * v129);
                  v130 = *v90;
                  if ( v157 == *v90 )
                    goto LABEL_198;
                  ++v131;
                }
                goto LABEL_204;
              }
            }
            else
            {
              ++*(_QWORD *)(v50 + 48);
            }
            v156 = v76;
            sub_19DF220(v145, 2 * v75);
            v109 = *(_DWORD *)(v50 + 72);
            if ( !v109 )
              goto LABEL_256;
            v76 = v156;
            v110 = v109 - 1;
            v111 = *(_QWORD *)(v50 + 56);
            v98 = *(_DWORD *)(v50 + 64) + 1;
            LODWORD(v112) = (v109 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
            v80 = (__int64 *)(v111 + 72LL * (unsigned int)v112);
            v113 = *v80;
            if ( v156 == *v80 )
              goto LABEL_157;
            v114 = 1;
            v107 = 0;
            while ( v113 != -8 )
            {
              if ( !v107 && v113 == -16 )
                v107 = v80;
              v112 = v110 & (unsigned int)(v112 + v114);
              v80 = (__int64 *)(v111 + 72 * v112);
              v113 = *v80;
              if ( v156 == *v80 )
                goto LABEL_157;
              ++v114;
            }
            goto LABEL_177;
          }
        }
LABEL_63:
        v53 = *(_QWORD *)(v53 + 8);
        if ( v53 != v147 + 40 )
          continue;
        break;
      }
      v51 = v148;
      v48 = v180;
      while ( 2 )
      {
        v52 = *(_QWORD **)(v48 - 24);
LABEL_66:
        if ( !*(_BYTE *)(v48 - 8) )
        {
          v56 = (__int64 *)v52[3];
          *(_BYTE *)(v48 - 8) = 1;
          *(_QWORD *)(v48 - 16) = v56;
          goto LABEL_70;
        }
        while ( 1 )
        {
          v56 = *(__int64 **)(v48 - 16);
LABEL_70:
          if ( v56 == (__int64 *)v52[4] )
            break;
          *(_QWORD *)(v48 - 16) = v56 + 1;
          v58 = *v56;
          v59 = v173;
          if ( v174 == v173 )
          {
            v60 = &v173[v176];
            if ( v173 != v60 )
            {
              v61 = 0;
              while ( v58 != *v59 )
              {
                if ( *v59 == -2 )
                {
                  v61 = v59;
                  if ( v60 == v59 + 1 )
                    goto LABEL_78;
                  ++v59;
                }
                else if ( v60 == ++v59 )
                {
                  if ( !v61 )
                    goto LABEL_145;
LABEL_78:
                  *v61 = v58;
                  --v177;
                  ++v172;
                  goto LABEL_79;
                }
              }
              continue;
            }
LABEL_145:
            if ( v176 < v175 )
            {
              ++v176;
              *v60 = v58;
              ++v172;
LABEL_79:
              v163 = v58;
              LOBYTE(v165) = 0;
              sub_13B8390(&v179, (__int64)&v163);
              v49 = v179;
              v48 = v180;
              goto LABEL_80;
            }
          }
          sub_16CCBA0((__int64)v51, v58);
          if ( v57 )
            goto LABEL_79;
        }
        v180 -= 24LL;
        v49 = v179;
        v48 = v180;
        if ( v180 != v179 )
          continue;
        break;
      }
LABEL_80:
      v43 = v186;
      if ( v48 - v49 != v187 - v186 )
        continue;
      break;
    }
LABEL_81:
    if ( v48 != v49 )
    {
      v62 = v43;
      while ( *(_QWORD *)v49 == *(_QWORD *)v62 )
      {
        v63 = *(_BYTE *)(v49 + 16);
        v64 = *(_BYTE *)(v62 + 16);
        if ( v63 && v64 )
          v65 = *(_QWORD *)(v49 + 8) == *(_QWORD *)(v62 + 8);
        else
          v65 = v63 == v64;
        if ( !v65 )
          break;
        v49 += 24LL;
        v62 += 24LL;
        if ( v49 == v48 )
          goto LABEL_89;
      }
      continue;
    }
    break;
  }
LABEL_89:
  if ( v43 )
    j_j___libc_free_0(v43, v188 - v43);
  if ( v184 != v183 )
    _libc_free(v184);
  if ( v179 )
    j_j___libc_free_0(v179, v181 - v179);
  if ( v174 != v173 )
    _libc_free((unsigned __int64)v174);
  if ( v198 )
    j_j___libc_free_0(v198, v200 - v198);
  if ( v196 != v195[1] )
    _libc_free(v196);
  if ( v192 )
    j_j___libc_free_0(v192, v194 - (_QWORD)v192);
  if ( v190 != v189[1] )
    _libc_free(v190);
  return v146;
}
