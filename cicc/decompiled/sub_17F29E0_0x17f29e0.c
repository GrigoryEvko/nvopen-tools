// Function: sub_17F29E0
// Address: 0x17f29e0
//
__int64 __fastcall sub_17F29E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 *v4; // r12
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // r13
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r10
  __int64 v13; // r11
  unsigned __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 *v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // rax
  int v29; // edx
  int v30; // r15d
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int64 *v33; // rbx
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  bool v38; // zf
  __int64 v39; // rax
  __int64 v40; // rbx
  __m128i *v41; // rax
  const __m128i *v42; // rdi
  const __m128i *v43; // r13
  const __m128i *v44; // rdx
  __m128i *v45; // rcx
  __int64 *v46; // r14
  __int64 v47; // rdi
  __int64 v48; // r13
  __int64 *v49; // rax
  __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // rax
  _QWORD *v57; // r12
  unsigned __int64 *v58; // r15
  __int64 v59; // rax
  unsigned __int64 v60; // rsi
  __int64 v61; // rcx
  _QWORD *v62; // r8
  __int64 v63; // r9
  __int64 v64; // rsi
  __int64 v65; // rdx
  unsigned __int8 *v66; // rsi
  __m128i *v67; // rsi
  __m128i *v68; // rdx
  __m128i *v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r13
  _QWORD *v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // r15
  __int64 v76; // rax
  int v77; // r9d
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned int v80; // eax
  const char *v81; // rdi
  char *v82; // r15
  size_t v83; // rax
  __int64 v84; // r12
  __int64 v85; // r12
  __int64 v86; // r12
  __int64 v87; // r12
  unsigned int v88; // eax
  __int64 *v89; // rbx
  __int64 *v90; // r12
  __int64 *v91; // rdi
  __m128i *v92; // rbx
  __m128i *v93; // r12
  __m128i *v94; // rdi
  _QWORD *v95; // rbx
  _QWORD *v96; // r14
  void (__fastcall *v97)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v98; // rax
  unsigned __int64 v99; // rax
  int v100; // eax
  unsigned __int64 v101; // rcx
  unsigned int v102; // edx
  unsigned __int64 v103; // rax
  __int64 v104; // rdx
  unsigned __int64 v105; // rdx
  bool v106; // cf
  __int64 v107; // rax
  __int64 v108; // rax
  __m128i *v109; // r15
  __int64 v110; // rsi
  __int64 v111; // rbx
  __int64 v112; // r14
  unsigned __int64 v113; // [rsp-968h] [rbp-968h]
  unsigned __int64 v114; // [rsp-958h] [rbp-958h]
  __int64 v115; // [rsp-938h] [rbp-938h]
  _QWORD *v116; // [rsp-920h] [rbp-920h]
  __int64 *v117; // [rsp-920h] [rbp-920h]
  __int64 v118; // [rsp-918h] [rbp-918h]
  unsigned __int64 v119; // [rsp-918h] [rbp-918h]
  __int64 v120; // [rsp-910h] [rbp-910h]
  __int64 v121; // [rsp-910h] [rbp-910h]
  __int64 v122; // [rsp-908h] [rbp-908h]
  unsigned __int64 v123; // [rsp-908h] [rbp-908h]
  unsigned __int64 v124; // [rsp-908h] [rbp-908h]
  _QWORD *v125; // [rsp-900h] [rbp-900h]
  unsigned __int64 v126; // [rsp-900h] [rbp-900h]
  unsigned __int64 v127; // [rsp-900h] [rbp-900h]
  __int64 v128; // [rsp-8F8h] [rbp-8F8h]
  unsigned int v129; // [rsp-8F8h] [rbp-8F8h]
  unsigned int v130; // [rsp-8F8h] [rbp-8F8h]
  unsigned __int64 v131; // [rsp-8F8h] [rbp-8F8h]
  __int64 v132; // [rsp-8F8h] [rbp-8F8h]
  __int64 *v133; // [rsp-8F0h] [rbp-8F0h]
  __int64 v134; // [rsp-8E8h] [rbp-8E8h]
  __int64 *v135; // [rsp-8E8h] [rbp-8E8h]
  __int64 *v136; // [rsp-8E8h] [rbp-8E8h]
  unsigned int v137; // [rsp-8E0h] [rbp-8E0h]
  unsigned __int8 v138; // [rsp-8D9h] [rbp-8D9h]
  unsigned int v139; // [rsp-8D4h] [rbp-8D4h] BYREF
  unsigned __int64 v140; // [rsp-8D0h] [rbp-8D0h] BYREF
  const __m128i *v141; // [rsp-8C8h] [rbp-8C8h] BYREF
  __m128i *v142; // [rsp-8C0h] [rbp-8C0h]
  const __m128i *v143; // [rsp-8B8h] [rbp-8B8h]
  __int64 v144; // [rsp-8A8h] [rbp-8A8h] BYREF
  __int64 v145; // [rsp-8A0h] [rbp-8A0h]
  unsigned __int64 *v146; // [rsp-898h] [rbp-898h]
  __int64 v147; // [rsp-890h] [rbp-890h]
  __int64 v148; // [rsp-888h] [rbp-888h]
  int v149; // [rsp-880h] [rbp-880h]
  __int64 v150; // [rsp-878h] [rbp-878h]
  __int64 v151; // [rsp-870h] [rbp-870h]
  _QWORD v152[2]; // [rsp-858h] [rbp-858h] BYREF
  __int64 v153; // [rsp-848h] [rbp-848h] BYREF
  __int64 *v154; // [rsp-838h] [rbp-838h]
  __int64 v155; // [rsp-828h] [rbp-828h] BYREF
  _QWORD v156[2]; // [rsp-7F8h] [rbp-7F8h] BYREF
  __int64 v157; // [rsp-7E8h] [rbp-7E8h] BYREF
  __int64 *v158; // [rsp-7D8h] [rbp-7D8h]
  __int64 v159; // [rsp-7C8h] [rbp-7C8h] BYREF
  __int64 v160[2]; // [rsp-798h] [rbp-798h] BYREF
  __int64 v161; // [rsp-788h] [rbp-788h] BYREF
  __int64 *v162; // [rsp-778h] [rbp-778h]
  __int64 v163; // [rsp-768h] [rbp-768h] BYREF
  _QWORD v164[2]; // [rsp-738h] [rbp-738h] BYREF
  __int64 v165; // [rsp-728h] [rbp-728h] BYREF
  __int64 *v166; // [rsp-718h] [rbp-718h]
  __int64 v167; // [rsp-708h] [rbp-708h] BYREF
  __int64 *v168; // [rsp-6D8h] [rbp-6D8h] BYREF
  __int64 v169; // [rsp-6D0h] [rbp-6D0h]
  _BYTE v170[128]; // [rsp-6C8h] [rbp-6C8h] BYREF
  unsigned __int64 *v171; // [rsp-648h] [rbp-648h] BYREF
  __int64 v172; // [rsp-640h] [rbp-640h]
  _QWORD v173[16]; // [rsp-638h] [rbp-638h] BYREF
  unsigned __int64 v174[2]; // [rsp-5B8h] [rbp-5B8h] BYREF
  _QWORD v175[36]; // [rsp-5A8h] [rbp-5A8h] BYREF
  char v176; // [rsp-488h] [rbp-488h]
  __int64 v177; // [rsp-480h] [rbp-480h]
  _BYTE *v178; // [rsp-478h] [rbp-478h]
  _BYTE *v179; // [rsp-470h] [rbp-470h]
  __int64 v180; // [rsp-468h] [rbp-468h]
  int v181; // [rsp-460h] [rbp-460h]
  _BYTE v182[64]; // [rsp-458h] [rbp-458h] BYREF
  _QWORD *v183; // [rsp-418h] [rbp-418h]
  _QWORD *v184; // [rsp-410h] [rbp-410h]
  __int64 v185; // [rsp-408h] [rbp-408h]
  __int16 v186; // [rsp-400h] [rbp-400h]
  __m128i v187; // [rsp-3F8h] [rbp-3F8h] BYREF
  __int64 v188; // [rsp-3E8h] [rbp-3E8h]
  __m128i v189; // [rsp-3E0h] [rbp-3E0h]
  __int64 v190; // [rsp-3D0h] [rbp-3D0h]
  __int64 v191; // [rsp-3C8h] [rbp-3C8h]
  __m128i v192; // [rsp-3C0h] [rbp-3C0h]
  __int64 v193; // [rsp-3B0h] [rbp-3B0h]
  __m128i *v195; // [rsp-3A0h] [rbp-3A0h] BYREF
  __int64 v196; // [rsp-398h] [rbp-398h]
  _BYTE v197[356]; // [rsp-390h] [rbp-390h] BYREF
  int v198; // [rsp-22Ch] [rbp-22Ch]
  __int64 v199; // [rsp-228h] [rbp-228h]
  const char *v200; // [rsp-218h] [rbp-218h] BYREF
  __int64 *v201; // [rsp-210h] [rbp-210h]
  unsigned __int64 *v202; // [rsp-208h] [rbp-208h]
  __int64 v203; // [rsp-200h] [rbp-200h]
  __int64 v204; // [rsp-1F8h] [rbp-1F8h]
  int v205; // [rsp-1F0h] [rbp-1F0h]
  __int64 v206; // [rsp-1E8h] [rbp-1E8h]
  __int64 v207; // [rsp-1E0h] [rbp-1E0h]
  __int64 *v208; // [rsp-1C0h] [rbp-1C0h]
  unsigned int v209; // [rsp-1B8h] [rbp-1B8h]
  __int64 v210; // [rsp-1B0h] [rbp-1B0h] BYREF

  v2 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v2 + 16) )
    BUG();
  if ( *(_DWORD *)(v2 + 36) == 135 )
    return 0;
  v3 = a2;
  v4 = (__int64 *)a1;
  v138 = sub_1695460(a2, 1, dword_4FA5BE0 + 2, *(_QWORD *)(a1 + 80), &v139, &v140);
  if ( !v138 )
    return 0;
  v5 = v140;
  v6 = v140;
  if ( byte_4FA5B00 )
  {
    sub_1368C40((__int64)&v200, *(__int64 **)(a1 + 8), *(_QWORD *)(a2 + 40));
    v6 = (unsigned __int64)v200;
    if ( !(_BYTE)v201 )
      return 0;
  }
  if ( (unsigned int)dword_4FA5E80 > v6 || !v140 )
    return 0;
  v8 = *(_QWORD **)(a1 + 80);
  v140 = v6;
  v120 = v139;
  v9 = 2LL * v139;
  v168 = (__int64 *)v170;
  v169 = 0x1000000000LL;
  v171 = v173;
  v116 = v8;
  v172 = 0x1000000001LL;
  v173[0] = 0;
  v133 = &v8[v9];
  if ( &v8[v9] != v8 )
  {
    v134 = v5;
    v10 = v8;
    v11 = v6;
    v12 = 0;
    v137 = 0;
    do
    {
      v13 = *v10;
      v14 = v10[1];
      if ( byte_4FA5B00 )
      {
        if ( v14
          && (_BitScanReverse64(&v99, v14), v100 = v99 ^ 0x3F, v6)
          && (_BitScanReverse64(&v101, v6), v102 = 126 - v100 - (v101 ^ 0x3F), v102 > 0x3E) )
        {
          v103 = -1;
          if ( v102 == 63 )
          {
            v104 = v6 * (v14 >> 1);
            if ( v104 >= 0 )
            {
              v103 = 2 * v104;
              if ( (v14 & 1) != 0 )
              {
                v105 = v6 + v103;
                if ( v6 >= v103 )
                  v103 = v6;
                v106 = v105 < v103;
                v103 = -1;
                if ( !v106 )
                  v103 = v105;
              }
            }
          }
        }
        else
        {
          v103 = v6 * v14;
        }
        v14 = v103 / v5;
      }
      v15 = LODWORD(qword_4FA3C60[20]);
      if ( (v15 != v13 || !(_DWORD)v15) && *(_QWORD *)(a1 + 72) + 1LL != v13 )
      {
        if ( (unsigned int)dword_4FA5E80 > v14 || v11 * (unsigned int)dword_4FA5CC0 / 0x64 > v14 )
          break;
        v16 = (unsigned int)v169;
        if ( (unsigned int)v169 >= HIDWORD(v169) )
        {
          v119 = v12;
          v124 = v11;
          v127 = v14;
          v132 = *v10;
          sub_16CD150((__int64)&v168, v170, 0, 8, v14, v11);
          v16 = (unsigned int)v169;
          v12 = v119;
          v11 = v124;
          v14 = v127;
          v13 = v132;
        }
        v168[v16] = v13;
        v17 = (unsigned int)v172;
        LODWORD(v169) = v169 + 1;
        if ( (unsigned int)v172 >= HIDWORD(v172) )
        {
          v123 = v12;
          v126 = v11;
          v131 = v14;
          sub_16CD150((__int64)&v171, v173, 0, 8, v14, v11);
          v17 = (unsigned int)v172;
          v12 = v123;
          v11 = v126;
          v14 = v131;
        }
        v171[v17] = v14;
        LODWORD(v172) = v172 + 1;
        if ( v12 < v14 )
          v12 = v14;
        v134 -= v10[1];
        v11 -= v14;
        ++v137;
        if ( dword_4FA5BE0 )
        {
          if ( v137 > dword_4FA5BE0 )
            break;
        }
      }
      v10 += 2;
    }
    while ( v133 != v10 );
    v18 = v171;
    if ( !v137 )
    {
      v138 = 0;
LABEL_94:
      if ( v18 != v173 )
        _libc_free((unsigned __int64)v18);
      goto LABEL_96;
    }
    *v171 = v11;
    v19 = *(_QWORD *)(a2 + 40);
    if ( v11 >= v12 )
      v12 = v11;
    v115 = *(_QWORD *)(a2 + 40);
    v114 = v12;
    v113 = v140 - v11;
    v20 = sub_1368AA0((__int64 *)v4[1], v115);
    v21 = sub_1AA8CA0(v19, a2, v4[3], 0);
    v22 = *(_QWORD *)(a2 + 32);
    v23 = v21;
    if ( v22 )
      v22 -= 24;
    v24 = sub_1AA8CA0(v21, v22, v4[3], 0);
    v200 = "MemOP.Merge";
    v128 = v24;
    LOWORD(v202) = 259;
    sub_164B780(v24, (__int64 *)&v200);
    sub_136C010((__int64 *)v4[1], v128, v20);
    v200 = "MemOP.Default";
    v122 = v23;
    LOWORD(v202) = 259;
    sub_164B780(v23, (__int64 *)&v200);
    v25 = v4[3];
    v186 = 0;
    v26 = *v4;
    v174[0] = (unsigned __int64)v175;
    v175[34] = v25;
    v174[1] = 0x1000000000LL;
    v175[32] = 0;
    v175[33] = 0;
    v175[35] = 0;
    v176 = 0;
    v177 = 0;
    v178 = v182;
    v179 = v182;
    v180 = 8;
    v181 = 0;
    v183 = 0;
    v184 = 0;
    v185 = 0;
    v118 = sub_15E0530(v26);
    v27 = sub_157E9C0(v19);
    v145 = v19;
    v147 = v27;
    v144 = 0;
    v148 = 0;
    v149 = 0;
    v150 = 0;
    v151 = 0;
    v146 = (unsigned __int64 *)(v19 + 40);
    v28 = (_QWORD *)sub_157EBA0(v19);
    sub_15F20C0(v28);
    v29 = *(_DWORD *)(v3 + 20);
    LOWORD(v202) = 257;
    v30 = v169;
    v31 = *(_QWORD *)(v3 + 24 * (2LL - (v29 & 0xFFFFFFF)));
    v32 = sub_1648B60(64);
    v125 = (_QWORD *)v32;
    if ( v32 )
      sub_15FFAB0(v32, v31, v122, v30, 0);
    if ( v145 )
    {
      v33 = v146;
      sub_157E9D0(v145 + 40, (__int64)v125);
      v34 = *v33;
      v35 = v125[3];
      v125[4] = v33;
      v34 &= 0xFFFFFFFFFFFFFFF8LL;
      v125[3] = v34 | v35 & 7;
      *(_QWORD *)(v34 + 8) = v125 + 3;
      *v33 = *v33 & 7 | (unsigned __int64)(v125 + 3);
    }
    sub_164B780((__int64)v125, (__int64 *)&v200);
    if ( v144 )
    {
      v187.m128i_i64[0] = v144;
      sub_1623A60((__int64)&v187, v144, 2);
      v36 = v125[6];
      if ( v36 )
        sub_161E7C0((__int64)(v125 + 6), v36);
      v37 = (unsigned __int8 *)v187.m128i_i64[0];
      v125[6] = v187.m128i_i64[0];
      if ( v37 )
        sub_1623210((__int64)&v187, v37, (__int64)(v125 + 6));
    }
    sub_1625C10(v3, 2, 0);
    if ( v134 || v139 != v137 )
      sub_1694FA0(*(__int64 ***)(*v4 + 40), v3, &v116[2 * v137], v120 - v137, v134, 1u, v139);
    v38 = v4[3] == 0;
    v141 = 0;
    v142 = 0;
    v39 = (unsigned int)v169;
    v143 = 0;
    if ( !v38 && (_DWORD)v169 )
    {
      v40 = 2LL * (unsigned int)v169;
      v41 = (__m128i *)sub_22077B0(v40 * 16);
      v42 = v141;
      v43 = v41;
      v44 = v141;
      if ( v142 != v141 )
      {
        v45 = (__m128i *)((char *)v41 + (char *)v142 - (char *)v141);
        do
        {
          if ( v41 )
            *v41 = _mm_loadu_si128(v44);
          ++v41;
          ++v44;
        }
        while ( v41 != v45 );
      }
      if ( v42 )
        j_j___libc_free_0(v42, (char *)v143 - (char *)v42);
      v141 = v43;
      v142 = (__m128i *)v43;
      v143 = &v43[v40];
      v39 = (unsigned int)v169;
    }
    v117 = &v168[v39];
    if ( v168 != v117 )
    {
      v121 = v3;
      v46 = v168;
      v135 = v4;
      while ( 1 )
      {
        v70 = *v46;
        LOWORD(v202) = 2819;
        v160[0] = v70;
        v71 = *v135;
        v200 = "MemOP.Case.";
        v201 = v160;
        v72 = (_QWORD *)sub_22077B0(64);
        v73 = (__int64)v72;
        if ( v72 )
          sub_157FB60(v72, v118, (__int64)&v200, v71, v122);
        v74 = sub_15F4880(v121);
        v75 = v74;
        if ( *(_BYTE *)(v74 + 16) != 78
          || (v76 = *(_QWORD *)(v74 - 24), *(_BYTE *)(v76 + 16))
          || (*(_BYTE *)(v76 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v76 + 36) - 133) > 4
          || ((1LL << (*(_BYTE *)(v76 + 36) + 123)) & 0x15) == 0 )
        {
          BUG();
        }
        v47 = **(_QWORD **)(v75 + 24 * (2LL - (*(_DWORD *)(v75 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v47 + 8) != 11 )
          v47 = 0;
        v48 = sub_159C470(v47, v160[0], 0);
        v49 = (__int64 *)(v75 + 24 * (2LL - (*(_DWORD *)(v75 + 20) & 0xFFFFFFF)));
        if ( *v49 )
        {
          v50 = v49[1];
          v51 = v49[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v51 = v50;
          if ( v50 )
            *(_QWORD *)(v50 + 16) = *(_QWORD *)(v50 + 16) & 3LL | v51;
        }
        *v49 = v48;
        if ( v48 )
        {
          v52 = *(_QWORD *)(v48 + 8);
          v49[1] = v52;
          if ( v52 )
            *(_QWORD *)(v52 + 16) = (unsigned __int64)(v49 + 1) | *(_QWORD *)(v52 + 16) & 3LL;
          v49[2] = (v48 + 8) | v49[2] & 3;
          *(_QWORD *)(v48 + 8) = v49;
        }
        sub_157E9D0(v73 + 40, v75);
        v53 = *(_QWORD *)(v73 + 40);
        v54 = *(_QWORD *)(v75 + 24);
        *(_QWORD *)(v75 + 32) = v73 + 40;
        v53 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v75 + 24) = v53 | v54 & 7;
        *(_QWORD *)(v53 + 8) = v75 + 24;
        *(_QWORD *)(v73 + 40) = *(_QWORD *)(v73 + 40) & 7LL | (v75 + 24);
        v55 = sub_157E9C0(v73);
        v202 = (unsigned __int64 *)(v73 + 40);
        v203 = v55;
        v200 = 0;
        v204 = 0;
        v205 = 0;
        v206 = 0;
        v207 = 0;
        v201 = (__int64 *)v73;
        LOWORD(v188) = 257;
        v56 = sub_1648A60(56, 1u);
        v57 = v56;
        if ( v56 )
          sub_15F8320((__int64)v56, v128, 0);
        if ( v201 )
        {
          v58 = v202;
          sub_157E9D0((__int64)(v201 + 5), (__int64)v57);
          v59 = v57[3];
          v60 = *v58;
          v57[4] = v58;
          v60 &= 0xFFFFFFFFFFFFFFF8LL;
          v57[3] = v60 | v59 & 7;
          *(_QWORD *)(v60 + 8) = v57 + 3;
          *v58 = *v58 & 7 | (unsigned __int64)(v57 + 3);
        }
        sub_164B780((__int64)v57, v187.m128i_i64);
        if ( v200 )
        {
          v164[0] = v200;
          sub_1623A60((__int64)v164, (__int64)v200, 2);
          v64 = v57[6];
          v65 = (__int64)(v57 + 6);
          v62 = v164;
          if ( v64 )
          {
            sub_161E7C0((__int64)(v57 + 6), v64);
            v62 = v164;
            v65 = (__int64)(v57 + 6);
          }
          v66 = (unsigned __int8 *)v164[0];
          v57[6] = v164[0];
          if ( v66 )
            sub_1623210((__int64)v164, v66, v65);
        }
        sub_15FFFB0((__int64)v125, v48, v73, v61, (__int64)v62, v63);
        if ( !v135[3] )
          goto LABEL_80;
        v67 = v142;
        v187.m128i_i64[0] = v73;
        v68 = (__m128i *)v143;
        v187.m128i_i64[1] = v128 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v142 != v143 )
          break;
        sub_17F2860(&v141, v142, &v187);
        v187.m128i_i64[1] = v73 & 0xFFFFFFFFFFFFFFFBLL;
        v69 = v142;
        v187.m128i_i64[0] = v115;
        if ( v142 == v143 )
          goto LABEL_168;
        if ( v142 )
          goto LABEL_78;
LABEL_79:
        v142 = v69 + 1;
LABEL_80:
        if ( v200 )
          sub_161E7C0((__int64)&v200, (__int64)v200);
        if ( v117 == ++v46 )
        {
          v4 = v135;
          v3 = v121;
          goto LABEL_99;
        }
      }
      if ( v142 )
      {
        *v142 = _mm_loadu_si128(&v187);
        v67 = v142;
        v68 = (__m128i *)v143;
      }
      v69 = v67 + 1;
      v142 = v69;
      v187.m128i_i64[0] = v115;
      v187.m128i_i64[1] = v73 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v68 == v69 )
      {
LABEL_168:
        sub_17F2860(&v141, v69, &v187);
        goto LABEL_80;
      }
LABEL_78:
      *v69 = _mm_loadu_si128(&v187);
      v69 = v142;
      goto LABEL_79;
    }
LABEL_99:
    sub_38B89E0(v174, v141, v142 - v141, 0);
    if ( v141 != v142 )
      v142 = (__m128i *)v141;
    sub_17E9890(*(__int64 **)(*v4 + 40), (__int64)v125, v171, (unsigned int)v172, v114, v77);
    v136 = (__int64 *)v4[2];
    v78 = sub_15E0530(*v136);
    if ( !sub_1602790(v78) )
    {
      v107 = sub_15E0530(*v136);
      v108 = sub_16033E0(v107);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v108 + 48LL))(v108) )
      {
LABEL_146:
        if ( v141 )
          j_j___libc_free_0(v141, (char *)v143 - (char *)v141);
        if ( v144 )
          sub_161E7C0((__int64)&v144, v144);
        sub_38B9570(v174);
        v95 = v184;
        v96 = v183;
        if ( v184 != v183 )
        {
          do
          {
            v97 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v96[7];
            *v96 = &unk_49F0628;
            if ( v97 )
              v97(v96 + 5, v96 + 5, 3);
            *v96 = &unk_49EE2B0;
            v98 = v96[3];
            if ( v98 != 0 && v98 != -8 && v98 != -16 )
              sub_1649B30(v96 + 1);
            v96 += 9;
          }
          while ( v95 != v96 );
          v96 = v183;
        }
        if ( v96 )
          j_j___libc_free_0(v96, v185 - (_QWORD)v96);
        if ( v179 != v178 )
          _libc_free((unsigned __int64)v179);
        if ( (_QWORD *)v174[0] != v175 )
          _libc_free(v174[0]);
        v18 = v171;
        goto LABEL_94;
      }
    }
    sub_15CA3B0((__int64)&v200, (__int64)"pgo-memop-opt", (__int64)"memopt-opt", 10, v3);
    sub_15CAB20((__int64)&v200, "optimized ", 0xAu);
    v79 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v79 + 16) )
      BUG();
    v80 = *(_DWORD *)(v79 + 36);
    if ( v80 == 137 )
    {
      v81 = "memset";
    }
    else if ( v80 > 0x89 )
    {
      v81 = "unknown";
    }
    else
    {
      if ( v80 == 133 )
      {
        v82 = "memcpy";
        v81 = "memcpy";
        goto LABEL_109;
      }
      v81 = "unknown";
      if ( v80 == 135 )
        v81 = "memmove";
    }
    v82 = (char *)v81;
LABEL_109:
    v83 = strlen(v81);
    sub_15C9800((__int64)v164, "Intrinsic", 9, v82, v83);
    v84 = sub_17C2270((__int64)&v200, (__int64)v164);
    sub_15CAB20(v84, " with count ", 0xCu);
    sub_15C9D40((__int64)v160, "Count", 5, v113);
    v85 = sub_17C2270(v84, (__int64)v160);
    sub_15CAB20(v85, " out of ", 8u);
    sub_15C9D40((__int64)v156, "Total", 5, v140);
    v86 = sub_17C2270(v85, (__int64)v156);
    sub_15CAB20(v86, " for ", 5u);
    sub_15C9C50((__int64)v152, "Versions", 8, v137);
    v87 = sub_17C2270(v86, (__int64)v152);
    sub_15CAB20(v87, " versions", 9u);
    v187.m128i_i32[2] = *(_DWORD *)(v87 + 8);
    v187.m128i_i8[12] = *(_BYTE *)(v87 + 12);
    v188 = *(_QWORD *)(v87 + 16);
    v189 = _mm_loadu_si128((const __m128i *)(v87 + 24));
    v190 = *(_QWORD *)(v87 + 40);
    v187.m128i_i64[0] = (__int64)&unk_49ECF68;
    v191 = *(_QWORD *)(v87 + 48);
    v192 = _mm_loadu_si128((const __m128i *)(v87 + 56));
    if ( *(_BYTE *)(v87 + 80) )
      v193 = *(_QWORD *)(v87 + 72);
    v195 = (__m128i *)v197;
    v196 = 0x400000000LL;
    v88 = *(_DWORD *)(v87 + 96);
    if ( v88 && &v195 != (__m128i **)(v87 + 88) )
    {
      v109 = (__m128i *)v197;
      v110 = v88;
      if ( v88 > 4 )
      {
        v129 = *(_DWORD *)(v87 + 96);
        sub_14B3F20((__int64)&v195, v88);
        v110 = *(unsigned int *)(v87 + 96);
        v109 = v195;
        v88 = v129;
      }
      v111 = *(_QWORD *)(v87 + 88);
      v112 = v111 + 88 * v110;
      if ( v111 != v112 )
      {
        v130 = v88;
        do
        {
          if ( v109 )
          {
            v109->m128i_i64[0] = (__int64)v109[1].m128i_i64;
            sub_17F22C0(v109->m128i_i64, *(_BYTE **)v111, *(_QWORD *)v111 + *(_QWORD *)(v111 + 8));
            v109[2].m128i_i64[0] = (__int64)v109[3].m128i_i64;
            sub_17F22C0(v109[2].m128i_i64, *(_BYTE **)(v111 + 32), *(_QWORD *)(v111 + 32) + *(_QWORD *)(v111 + 40));
            v109[4] = _mm_loadu_si128((const __m128i *)(v111 + 64));
            v109[5].m128i_i64[0] = *(_QWORD *)(v111 + 80);
          }
          v111 += 88;
          v109 = (__m128i *)((char *)v109 + 88);
        }
        while ( v112 != v111 );
        v88 = v130;
      }
      LODWORD(v196) = v88;
    }
    v197[352] = *(_BYTE *)(v87 + 456);
    v198 = *(_DWORD *)(v87 + 460);
    v199 = *(_QWORD *)(v87 + 464);
    v187.m128i_i64[0] = (__int64)&unk_49ECF98;
    if ( v154 != &v155 )
      j_j___libc_free_0(v154, v155 + 1);
    if ( (__int64 *)v152[0] != &v153 )
      j_j___libc_free_0(v152[0], v153 + 1);
    if ( v158 != &v159 )
      j_j___libc_free_0(v158, v159 + 1);
    if ( (__int64 *)v156[0] != &v157 )
      j_j___libc_free_0(v156[0], v157 + 1);
    if ( v162 != &v163 )
      j_j___libc_free_0(v162, v163 + 1);
    if ( (__int64 *)v160[0] != &v161 )
      j_j___libc_free_0(v160[0], v161 + 1);
    if ( v166 != &v167 )
      j_j___libc_free_0(v166, v167 + 1);
    if ( (__int64 *)v164[0] != &v165 )
      j_j___libc_free_0(v164[0], v165 + 1);
    v89 = v208;
    v200 = (const char *)&unk_49ECF68;
    v90 = &v208[11 * v209];
    if ( v208 != v90 )
    {
      do
      {
        v90 -= 11;
        v91 = (__int64 *)v90[4];
        if ( v91 != v90 + 6 )
          j_j___libc_free_0(v91, v90[6] + 1);
        if ( (__int64 *)*v90 != v90 + 2 )
          j_j___libc_free_0(*v90, v90[2] + 1);
      }
      while ( v89 != v90 );
      v90 = v208;
    }
    if ( v90 != &v210 )
      _libc_free((unsigned __int64)v90);
    sub_143AA50(v136, (__int64)&v187);
    v92 = v195;
    v187.m128i_i64[0] = (__int64)&unk_49ECF68;
    v93 = (__m128i *)((char *)v195 + 88 * (unsigned int)v196);
    if ( v195 != v93 )
    {
      do
      {
        v93 = (__m128i *)((char *)v93 - 88);
        v94 = (__m128i *)v93[2].m128i_i64[0];
        if ( v94 != &v93[3] )
          j_j___libc_free_0(v94, v93[3].m128i_i64[0] + 1);
        if ( (__m128i *)v93->m128i_i64[0] != &v93[1] )
          j_j___libc_free_0(v93->m128i_i64[0], v93[1].m128i_i64[0] + 1);
      }
      while ( v92 != v93 );
      v93 = v195;
    }
    if ( v93 != (__m128i *)v197 )
      _libc_free((unsigned __int64)v93);
    goto LABEL_146;
  }
  v138 = 0;
LABEL_96:
  if ( v168 != (__int64 *)v170 )
    _libc_free((unsigned __int64)v168);
  return v138;
}
