// Function: sub_325BEE0
// Address: 0x325bee0
//
void __fastcall sub_325BEE0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rax
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // r10d
  __int64 v10; // r11
  int v11; // ecx
  __int64 v12; // rdx
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // r12
  int v17; // edx
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r14
  int v22; // esi
  __int64 v23; // rdi
  __int64 v24; // rbx
  __int64 *v25; // rdx
  unsigned int v26; // r12d
  __int64 v27; // rcx
  unsigned int v28; // r8d
  __int64 *v29; // rax
  __int64 v30; // rdi
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  _DWORD *v35; // rax
  _QWORD *v36; // r12
  unsigned int v37; // esi
  int v38; // ecx
  __int64 *v39; // rdx
  unsigned int v40; // r13d
  unsigned int v41; // r8d
  __int64 *v42; // rax
  _QWORD *v43; // rdi
  unsigned int v44; // r8d
  __int64 v45; // rax
  __int64 v46; // rbx
  unsigned __int8 *v47; // rax
  unsigned __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rbx
  const __m128i *v51; // r9
  unsigned __int64 v52; // r8
  __int64 v53; // r14
  signed int v54; // r13d
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rdx
  int v58; // ecx
  __int64 v59; // rdx
  int v60; // esi
  __int64 v61; // rdx
  __int64 v62; // rax
  int v63; // edx
  unsigned int v64; // ecx
  __int64 v65; // r8
  int v66; // edi
  __int64 *v67; // rsi
  unsigned __int64 v68; // rcx
  __int64 v69; // rdx
  const __m128i *v70; // r12
  __m128i *v71; // rdx
  __int64 v72; // rcx
  unsigned __int64 v73; // rsi
  __int64 v74; // rdx
  signed __int64 v75; // r12
  signed int *v76; // rdx
  __int64 v77; // rax
  int v78; // edx
  _BYTE *v79; // rax
  unsigned __int64 v80; // rbx
  _QWORD *v81; // rax
  __int64 v82; // rax
  unsigned __int64 v83; // r15
  __int64 v84; // rax
  unsigned __int64 v85; // r10
  __int64 v86; // rdx
  unsigned int v87; // eax
  __int64 *v88; // rcx
  unsigned int v89; // r10d
  int v90; // esi
  __int64 v91; // rdi
  __int64 v92; // rax
  int v93; // esi
  int v94; // r9d
  int v95; // edx
  __int64 v96; // r8
  __int64 v97; // r9
  unsigned __int64 v98; // rax
  unsigned __int64 i; // rdx
  int v100; // edi
  __int64 v101; // rsi
  __int64 v102; // rax
  __int64 *v103; // rsi
  unsigned int v104; // ecx
  int v105; // edi
  __int64 v106; // r8
  unsigned int v107; // ecx
  __int64 v108; // r8
  int v109; // edi
  __int64 v110; // [rsp+0h] [rbp-2C0h]
  _QWORD *v112; // [rsp+10h] [rbp-2B0h]
  __int64 v113; // [rsp+18h] [rbp-2A8h]
  unsigned __int64 v114; // [rsp+28h] [rbp-298h]
  _QWORD *v115; // [rsp+30h] [rbp-290h]
  unsigned __int64 v116; // [rsp+38h] [rbp-288h]
  __int64 v117; // [rsp+40h] [rbp-280h]
  __int64 v118; // [rsp+50h] [rbp-270h]
  __int64 v119; // [rsp+50h] [rbp-270h]
  unsigned __int64 v120; // [rsp+58h] [rbp-268h]
  __int64 v121; // [rsp+58h] [rbp-268h]
  unsigned __int8 *v122; // [rsp+58h] [rbp-268h]
  __int64 v123; // [rsp+60h] [rbp-260h]
  int v124; // [rsp+68h] [rbp-258h]
  _QWORD *v125; // [rsp+68h] [rbp-258h]
  const __m128i *v126; // [rsp+68h] [rbp-258h]
  __int64 v127; // [rsp+70h] [rbp-250h]
  unsigned __int8 *v128; // [rsp+70h] [rbp-250h]
  const __m128i *v129; // [rsp+80h] [rbp-240h]
  int v130; // [rsp+80h] [rbp-240h]
  unsigned __int8 *v131; // [rsp+80h] [rbp-240h]
  int v132; // [rsp+88h] [rbp-238h]
  unsigned int v133; // [rsp+88h] [rbp-238h]
  const __m128i *v134; // [rsp+88h] [rbp-238h]
  int v135; // [rsp+88h] [rbp-238h]
  const __m128i *v136; // [rsp+88h] [rbp-238h]
  unsigned int v137; // [rsp+88h] [rbp-238h]
  unsigned __int8 *v138; // [rsp+88h] [rbp-238h]
  const __m128i *v139; // [rsp+88h] [rbp-238h]
  __int64 v140; // [rsp+90h] [rbp-230h] BYREF
  __int64 v141; // [rsp+98h] [rbp-228h]
  __int64 v142; // [rsp+A0h] [rbp-220h]
  unsigned int v143; // [rsp+A8h] [rbp-218h]
  _DWORD *v144; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v145; // [rsp+B8h] [rbp-208h]
  _DWORD v146[4]; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v147; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 v148; // [rsp+D8h] [rbp-1E8h]
  unsigned __int64 v149; // [rsp+E0h] [rbp-1E0h]
  _QWORD *v150; // [rsp+E8h] [rbp-1D8h]
  unsigned __int64 v151; // [rsp+F0h] [rbp-1D0h]
  __int64 v152; // [rsp+F8h] [rbp-1C8h]
  __int64 v153; // [rsp+100h] [rbp-1C0h]
  int v154; // [rsp+108h] [rbp-1B8h]
  char v155; // [rsp+110h] [rbp-1B0h]
  int v156; // [rsp+114h] [rbp-1ACh]
  __int64 v157; // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v158; // [rsp+128h] [rbp-198h]
  _QWORD *v159; // [rsp+130h] [rbp-190h]
  _QWORD *v160; // [rsp+138h] [rbp-188h]
  unsigned __int64 v161; // [rsp+140h] [rbp-180h]
  __int64 v162; // [rsp+148h] [rbp-178h]
  __int64 v163; // [rsp+150h] [rbp-170h]
  int v164; // [rsp+158h] [rbp-168h]
  char v165; // [rsp+160h] [rbp-160h]
  int v166; // [rsp+164h] [rbp-15Ch]
  _BYTE *v167; // [rsp+170h] [rbp-150h] BYREF
  __int64 v168; // [rsp+178h] [rbp-148h]
  _BYTE v169[64]; // [rsp+180h] [rbp-140h] BYREF
  const __m128i *v170; // [rsp+1C0h] [rbp-100h] BYREF
  __int64 v171; // [rsp+1C8h] [rbp-F8h]
  _BYTE v172[240]; // [rsp+1D0h] [rbp-F0h] BYREF

  v3 = *(_QWORD **)(a1 + 8);
  v4 = a2[11];
  v140 = 0;
  v5 = v3[28];
  v141 = 0;
  v142 = 0;
  v117 = v5;
  v6 = v3[67];
  v7 = v3[50];
  v143 = 0;
  v113 = v6;
  v110 = v7;
  v170 = (const __m128i *)v172;
  v171 = 0x800000000LL;
  v132 = *(_DWORD *)(v4 + 632);
  if ( v132 <= 0 )
  {
    v24 = a2[41];
    v92 = 1;
    goto LABEL_134;
  }
  v127 = *(unsigned int *)(v4 + 632);
  v8 = 0;
  v9 = 0;
  v10 = 0;
  while ( 1 )
  {
    v124 = v8;
    v16 = *(_QWORD *)(*(_QWORD *)(v4 + 624) + 24 * v8) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 )
    {
      v11 = 1;
      v12 = 0;
      v13 = (v9 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = v10 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( v16 == *(_QWORD *)v14 )
        goto LABEL_4;
      while ( v15 != -4096 )
      {
        if ( v15 == -8192 && !v12 )
          v12 = v14;
        v13 = (v9 - 1) & (v11 + v13);
        v14 = v10 + 16LL * v13;
        v15 = *(_QWORD *)v14;
        if ( v16 == *(_QWORD *)v14 )
          goto LABEL_4;
        ++v11;
      }
      if ( v12 )
        v14 = v12;
      ++v140;
      v17 = v142 + 1;
      if ( 4 * ((int)v142 + 1) < 3 * v9 )
      {
        if ( v9 - (v17 + HIDWORD(v142)) <= v9 >> 3 )
        {
          sub_2E3ADF0((__int64)&v140, v9);
          if ( !v143 )
          {
LABEL_205:
            LODWORD(v142) = v142 + 1;
            BUG();
          }
          v20 = 0;
          LODWORD(v21) = (v143 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v17 = v142 + 1;
          v22 = 1;
          v14 = v141 + 16LL * (unsigned int)v21;
          v23 = *(_QWORD *)v14;
          if ( v16 != *(_QWORD *)v14 )
          {
            while ( v23 != -4096 )
            {
              if ( !v20 && v23 == -8192 )
                v20 = v14;
              v21 = (v143 - 1) & ((_DWORD)v21 + v22);
              v14 = v141 + 16 * v21;
              v23 = *(_QWORD *)v14;
              if ( v16 == *(_QWORD *)v14 )
                goto LABEL_10;
              ++v22;
            }
            if ( v20 )
              v14 = v20;
          }
        }
        goto LABEL_10;
      }
    }
    else
    {
      ++v140;
    }
    sub_2E3ADF0((__int64)&v140, 2 * v9);
    if ( !v143 )
      goto LABEL_205;
    v17 = v142 + 1;
    LODWORD(v18) = (v143 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v14 = v141 + 16LL * (unsigned int)v18;
    v19 = *(_QWORD *)v14;
    if ( v16 != *(_QWORD *)v14 )
    {
      v100 = 1;
      v101 = 0;
      while ( v19 != -4096 )
      {
        if ( !v101 && v19 == -8192 )
          v101 = v14;
        v18 = (v143 - 1) & ((_DWORD)v18 + v100);
        v14 = v141 + 16 * v18;
        v19 = *(_QWORD *)v14;
        if ( v16 == *(_QWORD *)v14 )
          goto LABEL_10;
        ++v100;
      }
      if ( v101 )
        v14 = v101;
    }
LABEL_10:
    LODWORD(v142) = v17;
    if ( *(_QWORD *)v14 != -4096 )
      --HIDWORD(v142);
    *(_QWORD *)v14 = v16;
    *(_DWORD *)(v14 + 8) = 0;
LABEL_4:
    ++v8;
    *(_DWORD *)(v14 + 8) = v124;
    if ( v127 == v8 )
      break;
    v10 = v141;
    v9 = v143;
  }
  v24 = a2[41];
  if ( !v143 )
  {
    v92 = v140 + 1;
LABEL_134:
    v140 = v92;
    v93 = 0;
    goto LABEL_189;
  }
  v25 = 0;
  v26 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
  v27 = 1;
  v28 = (v143 - 1) & v26;
  v29 = (__int64 *)(v141 + 16LL * v28);
  v30 = *v29;
  if ( v24 == *v29 )
  {
LABEL_31:
    *((_DWORD *)v29 + 2) = -1;
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v117 + 536LL))(v117, 0xFFFFFFFFLL, 4, v27);
    v31 = v132;
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v117 + 536LL))(v117, v132, 4);
    v168 = 0x400000000LL;
    v167 = v169;
    v32 = sub_2207820(8LL * v132);
    v145 = 0x400000000LL;
    v116 = v32;
    v144 = v146;
    if ( (unsigned __int64)v132 <= 4 )
    {
LABEL_32:
      v35 = v146;
      do
        *v35++ = v132;
      while ( &v146[v31] != v35 );
      goto LABEL_34;
    }
LABEL_178:
    sub_C8D5F0((__int64)&v144, v146, v31, 4u, v33, v34);
    v98 = (unsigned __int64)v144;
LABEL_149:
    for ( i = v98 + 4 * v31; i != v98; *(_DWORD *)(v98 - 4) = v132 )
      v98 += 4LL;
    goto LABEL_34;
  }
  while ( v30 != -4096 )
  {
    if ( !v25 && v30 == -8192 )
      v25 = v29;
    v94 = v27 + 1;
    v27 = (v143 - 1) & (v28 + (_DWORD)v27);
    v28 = v27;
    v29 = (__int64 *)(v141 + 16LL * (unsigned int)v27);
    v30 = *v29;
    if ( v24 == *v29 )
      goto LABEL_31;
    LODWORD(v27) = v94;
  }
  if ( v25 )
    v29 = v25;
  ++v140;
  v95 = v142 + 1;
  if ( 4 * ((int)v142 + 1) >= 3 * v143 )
  {
    v93 = 2 * v143;
LABEL_189:
    sub_2E3ADF0((__int64)&v140, v93);
    if ( v143 )
    {
      v107 = (v143 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v95 = v142 + 1;
      v29 = (__int64 *)(v141 + 16LL * v107);
      v108 = *v29;
      if ( *v29 == v24 )
        goto LABEL_145;
      v109 = 1;
      v103 = 0;
      while ( v108 != -4096 )
      {
        if ( !v103 && v108 == -8192 )
          v103 = v29;
        v107 = (v143 - 1) & (v109 + v107);
        v29 = (__int64 *)(v141 + 16LL * v107);
        v108 = *v29;
        if ( *v29 == v24 )
          goto LABEL_145;
        ++v109;
      }
      goto LABEL_185;
    }
LABEL_203:
    LODWORD(v142) = v142 + 1;
    BUG();
  }
  if ( v143 - HIDWORD(v142) - v95 <= v143 >> 3 )
  {
    sub_2E3ADF0((__int64)&v140, v143);
    if ( v143 )
    {
      v103 = 0;
      v104 = (v143 - 1) & v26;
      v95 = v142 + 1;
      v105 = 1;
      v29 = (__int64 *)(v141 + 16LL * v104);
      v106 = *v29;
      if ( v24 == *v29 )
        goto LABEL_145;
      while ( v106 != -4096 )
      {
        if ( v106 == -8192 && !v103 )
          v103 = v29;
        v104 = (v143 - 1) & (v105 + v104);
        v29 = (__int64 *)(v141 + 16LL * v104);
        v106 = *v29;
        if ( v24 == *v29 )
          goto LABEL_145;
        ++v105;
      }
LABEL_185:
      if ( v103 )
        v29 = v103;
      goto LABEL_145;
    }
    goto LABEL_203;
  }
LABEL_145:
  LODWORD(v142) = v95;
  if ( *v29 != -4096 )
    --HIDWORD(v142);
  *v29 = v24;
  *((_DWORD *)v29 + 2) = -1;
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v117 + 536LL))(v117, 0xFFFFFFFFLL, 4);
  v31 = v132;
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v117 + 536LL))(v117, v132, 4);
  v168 = 0x400000000LL;
  v167 = v169;
  if ( (unsigned __int64)v132 > 0xFFFFFFFFFFFFFFFLL )
  {
    v116 = sub_2207820(0xFFFFFFFFFFFFFFFFLL);
    v144 = v146;
    v145 = 0x400000000LL;
    sub_C8D5F0((__int64)&v144, v146, v132, 4u, v96, v97);
    v98 = (unsigned __int64)v144;
    goto LABEL_149;
  }
  v102 = sub_2207820(8LL * v132);
  v145 = 0x400000000LL;
  v116 = v102;
  v144 = v146;
  if ( (unsigned __int64)v132 > 4 )
    goto LABEL_178;
  if ( v132 )
    goto LABEL_32;
LABEL_34:
  v36 = (_QWORD *)a2[41];
  LODWORD(v145) = v31;
  v112 = a2 + 40;
  if ( v36 != a2 + 40 )
  {
    v37 = v143;
    if ( !v143 )
      goto LABEL_66;
LABEL_36:
    v38 = 1;
    v39 = 0;
    v40 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
    v41 = (v37 - 1) & v40;
    v42 = (__int64 *)(v141 + 16LL * v41);
    v43 = (_QWORD *)*v42;
    if ( (_QWORD *)*v42 == v36 )
      goto LABEL_37;
    while ( v43 != (_QWORD *)-4096LL )
    {
      if ( !v39 && v43 == (_QWORD *)-8192LL )
        v39 = v42;
      v41 = (v37 - 1) & (v38 + v41);
      v42 = (__int64 *)(v141 + 16LL * v41);
      v43 = (_QWORD *)*v42;
      if ( (_QWORD *)*v42 == v36 )
      {
LABEL_37:
        v44 = *((_DWORD *)v42 + 2);
        goto LABEL_38;
      }
      ++v38;
    }
    if ( v39 )
      v42 = v39;
    ++v140;
    v63 = v142 + 1;
    if ( 4 * ((int)v142 + 1) < 3 * v37 )
    {
      if ( v37 - HIDWORD(v142) - v63 > v37 >> 3 )
        goto LABEL_124;
      sub_2E3ADF0((__int64)&v140, v37);
      if ( !v143 )
      {
LABEL_204:
        LODWORD(v142) = v142 + 1;
        BUG();
      }
      v88 = 0;
      v89 = (v143 - 1) & v40;
      v63 = v142 + 1;
      v90 = 1;
      v42 = (__int64 *)(v141 + 16LL * v89);
      v91 = *v42;
      if ( (_QWORD *)*v42 == v36 )
        goto LABEL_124;
      while ( v91 != -4096 )
      {
        if ( !v88 && v91 == -8192 )
          v88 = v42;
        v89 = (v143 - 1) & (v90 + v89);
        v42 = (__int64 *)(v141 + 16LL * v89);
        v91 = *v42;
        if ( (_QWORD *)*v42 == v36 )
          goto LABEL_124;
        ++v90;
      }
      if ( v88 )
        v42 = v88;
      goto LABEL_124;
    }
LABEL_67:
    sub_2E3ADF0((__int64)&v140, 2 * v37);
    if ( !v143 )
      goto LABEL_204;
    v63 = v142 + 1;
    v64 = (v143 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
    v42 = (__int64 *)(v141 + 16LL * v64);
    v65 = *v42;
    if ( (_QWORD *)*v42 != v36 )
    {
      v66 = 1;
      v67 = 0;
      while ( v65 != -4096 )
      {
        if ( v65 == -8192 && !v67 )
          v67 = v42;
        v64 = (v143 - 1) & (v66 + v64);
        v42 = (__int64 *)(v141 + 16LL * v64);
        v65 = *v42;
        if ( (_QWORD *)*v42 == v36 )
          goto LABEL_124;
        ++v66;
      }
      if ( v67 )
        v42 = v67;
    }
LABEL_124:
    LODWORD(v142) = v63;
    if ( *v42 != -4096 )
      --HIDWORD(v142);
    *v42 = (__int64)v36;
    v44 = 0;
    *((_DWORD *)v42 + 2) = 0;
LABEL_38:
    v45 = (__int64)v36;
    do
    {
      v45 = *(_QWORD *)(v45 + 8);
      if ( v112 == (_QWORD *)v45 )
      {
        v115 = (_QWORD *)v45;
        v46 = v110;
        goto LABEL_42;
      }
    }
    while ( !*(_BYTE *)(v45 + 235) );
    v137 = v44;
    v115 = (_QWORD *)v45;
    v82 = sub_3258B50(v45);
    v44 = v137;
    v46 = v82;
LABEL_42:
    v133 = v44;
    v47 = (unsigned __int8 *)sub_3259000(a1, v46, v113);
    sub_E9A5B0(v117, v47);
    if ( v133 != -1 )
      *(_QWORD *)(v116 + 8LL * (int)v133) = v46;
    v48 = v36[7];
    v49 = *v115;
    v159 = v115;
    v160 = v115;
    v157 = v4;
    v161 = (v49 & 0xFFFFFFFFFFFFFFF8LL) + 48;
    v158 = 0;
    v165 = 0;
    v166 = -1;
    v162 = 0;
    v163 = 0;
    v164 = -1;
    sub_32588C0((__int64)&v157);
    v150 = v115;
    v151 = v48;
    v50 = -1;
    v147 = v4;
    v148 = 0;
    v149 = (unsigned __int64)v36;
    v155 = 0;
    v156 = -1;
    v152 = 0;
    v153 = 0;
    v154 = -1;
    sub_32588C0((__int64)&v147);
    v51 = (const __m128i *)&v147;
    v52 = v133;
    v53 = 0;
    v157 = v147;
    v118 = v158;
    v54 = v133;
    v158 = v148;
    v125 = v159;
    v160 = v150;
    v120 = v161;
    v161 = v151;
    v159 = (_QWORD *)v149;
    v162 = v152;
    v163 = v153;
    v164 = v154;
    v165 = v155;
    v166 = v156;
    if ( (_QWORD *)v149 == v125 )
      goto LABEL_62;
    while ( 1 )
    {
      v55 = v164;
      if ( (_DWORD)v50 == -1 )
      {
        if ( v164 == -1 )
          goto LABEL_77;
        v56 = *(_QWORD *)(v4 + 624);
        v58 = 0;
      }
      else
      {
        v56 = *(_QWORD *)(v4 + 624);
        v57 = (int)v50;
        v58 = 0;
        do
        {
          ++v58;
          LODWORD(v57) = *(_DWORD *)(v56 + 24 * v57 + 16);
        }
        while ( (_DWORD)v57 != -1 );
        if ( v164 == -1 )
        {
          v60 = 0;
LABEL_54:
          v61 = (int)v50;
          do
          {
            --v58;
            v61 = *(int *)(v56 + 24 * v61 + 16);
          }
          while ( v58 != v60 );
          if ( (_DWORD)v61 == (_DWORD)v55 )
            goto LABEL_77;
          goto LABEL_57;
        }
      }
      v59 = v164;
      v60 = 0;
      do
      {
        ++v60;
        LODWORD(v59) = *(_DWORD *)(v56 + 24 * v59 + 16);
      }
      while ( (_DWORD)v59 != -1 );
      if ( v58 < v60 )
      {
        do
        {
          --v60;
          v55 = *(int *)(v56 + 24 * v55 + 16);
        }
        while ( v58 != v60 );
      }
      if ( v60 < v58 )
        goto LABEL_54;
      v61 = (int)v50;
      while ( (_DWORD)v61 != (_DWORD)v55 )
      {
LABEL_57:
        LODWORD(v61) = *(_DWORD *)(v56 + 24 * v61 + 16);
        LODWORD(v55) = *(_DWORD *)(v56 + 24 * v55 + 16);
      }
      while ( (_DWORD)v55 != (_DWORD)v50 )
      {
        v70 = v51;
        v147 = v53;
        v149 = __PAIR64__(v54, v50);
        v68 = (unsigned __int64)v170;
        v148 = v162;
        v69 = (unsigned int)v171;
        v52 = (unsigned int)v171 + 1LL;
        if ( v52 > HIDWORD(v171) )
        {
          if ( v170 > v51 || v51 >= (const __m128i *)((char *)v170 + 24 * (unsigned int)v171) )
          {
            v129 = v51;
            v135 = v55;
            sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 0x18u, v52, (__int64)v51);
            v51 = v129;
            v68 = (unsigned __int64)v170;
            v69 = (unsigned int)v171;
            LODWORD(v55) = v135;
            v70 = v129;
          }
          else
          {
            v75 = (char *)v51 - (char *)v170;
            v130 = v55;
            v136 = v51;
            sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 0x18u, v52, (__int64)v51);
            v68 = (unsigned __int64)v170;
            v69 = (unsigned int)v171;
            v51 = v136;
            LODWORD(v55) = v130;
            v70 = (const __m128i *)((char *)v170 + v75);
          }
        }
        v71 = (__m128i *)(v68 + 24 * v69);
        *v71 = _mm_loadu_si128(v70);
        v72 = v70[1].m128i_i64[0];
        v73 = (unsigned __int64)v167;
        LODWORD(v171) = v171 + 1;
        v71[1].m128i_i64[0] = v72;
        LODWORD(v50) = *(_DWORD *)(*(_QWORD *)(v4 + 624) + 24 * v50 + 16);
        v74 = v73 + 16LL * (unsigned int)v168 - 16;
        if ( (_DWORD)v50 == *(_DWORD *)(v74 + 8) )
        {
          v53 = *(_QWORD *)v74;
          LODWORD(v168) = v168 - 1;
        }
LABEL_77:
        ;
      }
      v62 = v164;
      if ( v164 != (_DWORD)v50 )
      {
        do
        {
          v76 = &v144[v62];
          if ( *v76 > v54 )
            *v76 = v54;
          v62 = *(int *)(*(_QWORD *)(v4 + 624) + 24 * v62 + 16);
        }
        while ( (_DWORD)v62 != (_DWORD)v50 );
        v77 = (unsigned int)v168;
        v78 = v168;
        if ( (unsigned int)v168 >= (unsigned __int64)HIDWORD(v168) )
        {
          v80 = v114 & 0xFFFFFFFF00000000LL | (unsigned int)v50;
          v114 = v80;
          if ( HIDWORD(v168) < (unsigned __int64)(unsigned int)v168 + 1 )
          {
            v139 = v51;
            sub_C8D5F0((__int64)&v167, v169, (unsigned int)v168 + 1LL, 0x10u, v52, (__int64)v51);
            v77 = (unsigned int)v168;
            v51 = v139;
          }
          v81 = &v167[16 * v77];
          *v81 = v53;
          v81[1] = v80;
          LODWORD(v168) = v168 + 1;
        }
        else
        {
          v79 = &v167[16 * (unsigned int)v168];
          if ( v79 )
          {
            *(_QWORD *)v79 = v53;
            *((_DWORD *)v79 + 2) = v50;
            v78 = v168;
          }
          LODWORD(v168) = v78 + 1;
        }
        v53 = v163;
        v50 = v164;
      }
      v134 = v51;
      sub_32588C0((__int64)&v157);
      v51 = v134;
      if ( v159 == v125 )
      {
LABEL_62:
        if ( v161 == v120 && v158 == v118 )
        {
          if ( v112 == v115 )
            break;
          v37 = v143;
          v36 = v115;
          if ( v143 )
            goto LABEL_36;
LABEL_66:
          ++v140;
          goto LABEL_67;
        }
      }
    }
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v117 + 536LL))(v117, (unsigned int)v171, 4);
  v126 = (const __m128i *)((char *)v170 + 24 * (unsigned int)v171);
  if ( v126 != v170 )
  {
    v123 = v4;
    v83 = (unsigned __int64)v170;
    do
    {
      v138 = (unsigned __int8 *)sub_3259070(a1, *(_QWORD *)v83, v113);
      v131 = (unsigned __int8 *)sub_3259070(a1, *(_QWORD *)(v83 + 8), v113);
      v121 = *(_QWORD *)(v123 + 624) + 24LL * *(int *)(v83 + 16);
      v84 = sub_3258B50(*(_QWORD *)v121 & 0xFFFFFFFFFFFFFFF8LL);
      v128 = (unsigned __int8 *)sub_3259000(a1, v84, v113);
      v85 = sub_3259000(a1, *(_QWORD *)(v116 + 8LL * *(int *)(v83 + 16)), v113);
      v86 = (unsigned int)(*(_DWORD *)(v121 + 20) - 1);
      v87 = 0;
      if ( (unsigned int)v86 <= 2 )
        v87 = dword_44D6D10[v86];
      if ( *(_DWORD *)(v83 + 20) != v144[*(int *)(v83 + 16)] )
        v87 |= 8u;
      v119 = v121;
      v122 = (unsigned __int8 *)v85;
      v83 += 24LL;
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v117 + 536LL))(v117, v87, 4);
      sub_E9A5B0(v117, v138);
      sub_E9A5B0(v117, v131);
      sub_E9A5B0(v117, v128);
      sub_E9A5B0(v117, v122);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v117 + 536LL))(v117, *(unsigned int *)(v119 + 8), 4);
    }
    while ( v126 != (const __m128i *)v83 );
  }
  if ( v144 != v146 )
    _libc_free((unsigned __int64)v144);
  if ( v116 )
    j_j___libc_free_0_0(v116);
  if ( v167 != v169 )
    _libc_free((unsigned __int64)v167);
  sub_C7D6A0(v141, 16LL * v143, 8);
  if ( v170 != (const __m128i *)v172 )
    _libc_free((unsigned __int64)v170);
}
