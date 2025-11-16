// Function: sub_39AF860
// Address: 0x39af860
//
void __fastcall sub_39AF860(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rax
  unsigned __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // r10d
  unsigned __int64 v10; // r11
  unsigned int v11; // edi
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r12
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // r9
  int v18; // ecx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // r14
  int v22; // esi
  __int64 v23; // r8
  __int64 v24; // rbx
  unsigned int v25; // r12d
  __int64 v26; // r8
  __int64 *v27; // rax
  __int64 v28; // rdi
  unsigned __int64 v29; // rbx
  int v30; // r8d
  int v31; // r9d
  _BYTE *v32; // rax
  _BYTE *i; // rdx
  _QWORD *v34; // rbx
  unsigned int v35; // esi
  unsigned int v36; // r12d
  unsigned int v37; // r8d
  __int64 *v38; // rax
  _QWORD *v39; // rdi
  signed int v40; // r13d
  __int64 v41; // rax
  __int64 v42; // r14
  unsigned int *v43; // rax
  unsigned __int64 v44; // r12
  __int64 v45; // rax
  __int64 v46; // rbx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdx
  int v53; // ecx
  __int64 v54; // rdx
  int v55; // esi
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned int v58; // ecx
  int v59; // edx
  __int64 v60; // r8
  int v61; // edi
  __int64 *v62; // rsi
  __m128i *v63; // rdx
  unsigned __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rdx
  signed int *v67; // rdx
  int v68; // edx
  _BYTE *v69; // rax
  unsigned __int64 v70; // r15
  unsigned __int64 v71; // r14
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // r14
  unsigned __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // rax
  unsigned __int64 v77; // r10
  __int64 v78; // rdx
  unsigned int v79; // eax
  int v80; // ecx
  __int64 *v81; // rdx
  __int64 *v82; // rcx
  unsigned int v83; // r10d
  int v84; // esi
  __int64 v85; // rdi
  int v86; // ecx
  __int64 *v87; // rdx
  int v88; // edx
  __int64 v89; // rax
  __int64 v90; // rax
  int v91; // esi
  unsigned int v92; // ecx
  __int64 v93; // r8
  int v94; // edi
  __int64 *v95; // rsi
  int v96; // edi
  unsigned __int64 v97; // rsi
  unsigned int v98; // ecx
  int v99; // edi
  __int64 v100; // r8
  __int64 v101; // [rsp+8h] [rbp-2B8h]
  _QWORD *v103; // [rsp+20h] [rbp-2A0h]
  __int64 v104; // [rsp+28h] [rbp-298h]
  unsigned __int64 v105; // [rsp+40h] [rbp-280h]
  _QWORD *v106; // [rsp+48h] [rbp-278h]
  __int64 v107; // [rsp+50h] [rbp-270h]
  __int128 v108; // [rsp+58h] [rbp-268h]
  unsigned int *v109; // [rsp+58h] [rbp-268h]
  unsigned __int64 v110; // [rsp+60h] [rbp-260h]
  int v111; // [rsp+68h] [rbp-258h]
  _BYTE *v112; // [rsp+68h] [rbp-258h]
  __int64 v113; // [rsp+70h] [rbp-250h]
  _QWORD *v114; // [rsp+70h] [rbp-250h]
  unsigned int *v115; // [rsp+70h] [rbp-250h]
  __int64 v116; // [rsp+78h] [rbp-248h]
  __int64 v117; // [rsp+78h] [rbp-248h]
  unsigned int *v118; // [rsp+78h] [rbp-248h]
  int v119; // [rsp+88h] [rbp-238h]
  int v120; // [rsp+88h] [rbp-238h]
  __int64 v121; // [rsp+88h] [rbp-238h]
  __int64 v122; // [rsp+88h] [rbp-238h]
  unsigned int *v123; // [rsp+88h] [rbp-238h]
  __int64 v124; // [rsp+90h] [rbp-230h] BYREF
  unsigned __int64 v125; // [rsp+98h] [rbp-228h]
  __int64 v126; // [rsp+A0h] [rbp-220h]
  unsigned int v127; // [rsp+A8h] [rbp-218h]
  _BYTE *v128; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v129; // [rsp+B8h] [rbp-208h]
  _BYTE v130[16]; // [rsp+C0h] [rbp-200h] BYREF
  __m128i v131; // [rsp+D0h] [rbp-1F0h] BYREF
  unsigned __int64 v132; // [rsp+E0h] [rbp-1E0h]
  _QWORD *v133; // [rsp+E8h] [rbp-1D8h]
  unsigned __int64 v134; // [rsp+F0h] [rbp-1D0h]
  __int64 v135; // [rsp+F8h] [rbp-1C8h]
  __int64 v136; // [rsp+100h] [rbp-1C0h]
  int v137; // [rsp+108h] [rbp-1B8h]
  char v138; // [rsp+110h] [rbp-1B0h]
  int v139; // [rsp+114h] [rbp-1ACh]
  __m128i v140; // [rsp+120h] [rbp-1A0h] BYREF
  _QWORD *v141; // [rsp+130h] [rbp-190h]
  _QWORD *v142; // [rsp+138h] [rbp-188h]
  unsigned __int64 v143; // [rsp+140h] [rbp-180h]
  __int64 v144; // [rsp+148h] [rbp-178h]
  __int64 v145; // [rsp+150h] [rbp-170h]
  int v146; // [rsp+158h] [rbp-168h]
  char v147; // [rsp+160h] [rbp-160h]
  int v148; // [rsp+164h] [rbp-15Ch]
  _BYTE *v149; // [rsp+170h] [rbp-150h] BYREF
  __int64 v150; // [rsp+178h] [rbp-148h]
  _BYTE v151[64]; // [rsp+180h] [rbp-140h] BYREF
  _BYTE *v152; // [rsp+1C0h] [rbp-100h] BYREF
  __int64 v153; // [rsp+1C8h] [rbp-F8h]
  _BYTE v154[240]; // [rsp+1D0h] [rbp-F0h] BYREF

  v3 = *(_QWORD **)(a1 + 8);
  v4 = a2[11];
  v124 = 0;
  v5 = v3[32];
  v125 = 0;
  v126 = 0;
  v107 = v5;
  v6 = v3[48];
  v7 = v3[49];
  v127 = 0;
  v104 = v6;
  v101 = v7;
  v152 = v154;
  v153 = 0x800000000LL;
  v119 = *(_DWORD *)(v4 + 600);
  if ( v119 <= 0 )
  {
    v24 = a2[41];
    v90 = 1;
    goto LABEL_133;
  }
  v113 = *(unsigned int *)(v4 + 600);
  v8 = 0;
  v9 = 0;
  v10 = 0;
  while ( 1 )
  {
    v111 = v8;
    v14 = *(_QWORD *)(*(_QWORD *)(v4 + 592) + 24 * v8) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 )
    {
      v11 = (v9 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v12 = v10 + 16LL * v11;
      v13 = *(_QWORD *)v12;
      if ( v14 == *(_QWORD *)v12 )
        goto LABEL_4;
      v18 = 1;
      v19 = 0;
      while ( v13 != -8 )
      {
        if ( v13 == -16 && !v19 )
          v19 = v12;
        v11 = (v9 - 1) & (v18 + v11);
        v12 = v10 + 16LL * v11;
        v13 = *(_QWORD *)v12;
        if ( v14 == *(_QWORD *)v12 )
          goto LABEL_4;
        ++v18;
      }
      if ( v19 )
        v12 = v19;
      ++v124;
      v15 = v126 + 1;
      if ( 4 * ((int)v126 + 1) < 3 * v9 )
      {
        if ( v9 - (v15 + HIDWORD(v126)) <= v9 >> 3 )
        {
          sub_1DDD540((__int64)&v124, v9);
          if ( !v127 )
          {
LABEL_198:
            LODWORD(v126) = v126 + 1;
            BUG();
          }
          v20 = 0;
          LODWORD(v21) = (v127 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v15 = v126 + 1;
          v22 = 1;
          v12 = v125 + 16LL * (unsigned int)v21;
          v23 = *(_QWORD *)v12;
          if ( v14 != *(_QWORD *)v12 )
          {
            while ( v23 != -8 )
            {
              if ( v23 == -16 && !v20 )
                v20 = v12;
              v21 = (v127 - 1) & ((_DWORD)v21 + v22);
              v12 = v125 + 16 * v21;
              v23 = *(_QWORD *)v12;
              if ( v14 == *(_QWORD *)v12 )
                goto LABEL_10;
              ++v22;
            }
            if ( v20 )
              v12 = v20;
          }
        }
        goto LABEL_10;
      }
    }
    else
    {
      ++v124;
    }
    sub_1DDD540((__int64)&v124, 2 * v9);
    if ( !v127 )
      goto LABEL_198;
    v15 = v126 + 1;
    LODWORD(v16) = (v127 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v12 = v125 + 16LL * (unsigned int)v16;
    v17 = *(_QWORD *)v12;
    if ( v14 != *(_QWORD *)v12 )
    {
      v96 = 1;
      v97 = 0;
      while ( v17 != -8 )
      {
        if ( !v97 && v17 == -16 )
          v97 = v12;
        v16 = (v127 - 1) & ((_DWORD)v16 + v96);
        v12 = v125 + 16 * v16;
        v17 = *(_QWORD *)v12;
        if ( v14 == *(_QWORD *)v12 )
          goto LABEL_10;
        ++v96;
      }
      if ( v97 )
        v12 = v97;
    }
LABEL_10:
    LODWORD(v126) = v15;
    if ( *(_QWORD *)v12 != -8 )
      --HIDWORD(v126);
    *(_QWORD *)v12 = v14;
    *(_DWORD *)(v12 + 8) = 0;
LABEL_4:
    ++v8;
    *(_DWORD *)(v12 + 8) = v111;
    if ( v113 == v8 )
      break;
    v10 = v125;
    v9 = v127;
  }
  v24 = a2[41];
  if ( !v127 )
  {
    v90 = v124 + 1;
LABEL_133:
    v124 = v90;
    v91 = 0;
    goto LABEL_134;
  }
  v25 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
  LODWORD(v26) = (v127 - 1) & v25;
  v27 = (__int64 *)(v125 + 16LL * (unsigned int)v26);
  v28 = *v27;
  if ( v24 == *v27 )
  {
LABEL_27:
    *((_DWORD *)v27 + 2) = -1;
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v107 + 424LL))(v107, 0xFFFFFFFFLL, 4);
    v29 = v119;
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v107 + 424LL))(v107, v119, 4);
    v149 = v151;
    v150 = 0x400000000LL;
    goto LABEL_28;
  }
  v86 = 1;
  v87 = 0;
  while ( v28 != -8 )
  {
    if ( !v87 && v28 == -16 )
      v87 = v27;
    v26 = (v127 - 1) & ((_DWORD)v26 + v86);
    v27 = (__int64 *)(v125 + 16 * v26);
    v28 = *v27;
    if ( v24 == *v27 )
      goto LABEL_27;
    ++v86;
  }
  if ( v87 )
    v27 = v87;
  ++v124;
  v88 = v126 + 1;
  if ( 4 * ((int)v126 + 1) < 3 * v127 )
  {
    if ( v127 - HIDWORD(v126) - v88 > v127 >> 3 )
      goto LABEL_127;
    sub_1DDD540((__int64)&v124, v127);
    if ( v127 )
    {
      v95 = 0;
      v98 = (v127 - 1) & v25;
      v88 = v126 + 1;
      v99 = 1;
      v27 = (__int64 *)(v125 + 16LL * v98);
      v100 = *v27;
      if ( v24 != *v27 )
      {
        while ( v100 != -8 )
        {
          if ( v100 == -16 && !v95 )
            v95 = v27;
          v98 = (v127 - 1) & (v99 + v98);
          v27 = (__int64 *)(v125 + 16LL * v98);
          v100 = *v27;
          if ( v24 == *v27 )
            goto LABEL_127;
          ++v99;
        }
        goto LABEL_138;
      }
      goto LABEL_127;
    }
LABEL_196:
    LODWORD(v126) = v126 + 1;
    BUG();
  }
  v91 = 2 * v127;
LABEL_134:
  sub_1DDD540((__int64)&v124, v91);
  if ( !v127 )
    goto LABEL_196;
  v92 = (v127 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
  v88 = v126 + 1;
  v27 = (__int64 *)(v125 + 16LL * v92);
  v93 = *v27;
  if ( *v27 != v24 )
  {
    v94 = 1;
    v95 = 0;
    while ( v93 != -8 )
    {
      if ( !v95 && v93 == -16 )
        v95 = v27;
      v92 = (v127 - 1) & (v94 + v92);
      v27 = (__int64 *)(v125 + 16LL * v92);
      v93 = *v27;
      if ( *v27 == v24 )
        goto LABEL_127;
      ++v94;
    }
LABEL_138:
    if ( v95 )
      v27 = v95;
  }
LABEL_127:
  LODWORD(v126) = v88;
  if ( *v27 != -8 )
    --HIDWORD(v126);
  *v27 = v24;
  *((_DWORD *)v27 + 2) = -1;
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v107 + 424LL))(v107, 0xFFFFFFFFLL, 4);
  v29 = v119;
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v107 + 424LL))(v107, v119, 4);
  v150 = 0x400000000LL;
  v149 = v151;
  if ( (unsigned __int64)v119 <= 0xFFFFFFFFFFFFFFFLL )
  {
LABEL_28:
    v105 = sub_2207820(8 * v29);
    v128 = v130;
    v129 = 0x400000000LL;
    if ( v29 > 4 )
      goto LABEL_131;
    v32 = v130;
  }
  else
  {
    v89 = sub_2207820(0xFFFFFFFFFFFFFFFFLL);
    v129 = 0x400000000LL;
    v105 = v89;
    v128 = v130;
LABEL_131:
    sub_16CD150((__int64)&v128, v130, v29, 4, v30, v31);
    v32 = v128;
  }
  LODWORD(v129) = v29;
  for ( i = &v32[4 * (unsigned int)v29]; i != v32; *((_DWORD *)v32 - 1) = v119 )
    v32 += 4;
  v103 = a2 + 40;
  v34 = (_QWORD *)a2[41];
  if ( a2 + 40 != v34 )
  {
    v35 = v127;
    if ( !v127 )
      goto LABEL_62;
LABEL_34:
    v36 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
    v37 = (v35 - 1) & v36;
    v38 = (__int64 *)(v125 + 16LL * v37);
    v39 = (_QWORD *)*v38;
    if ( (_QWORD *)*v38 == v34 )
      goto LABEL_35;
    v80 = 1;
    v81 = 0;
    while ( v39 != (_QWORD *)-8LL )
    {
      if ( !v81 && v39 == (_QWORD *)-16LL )
        v81 = v38;
      v37 = (v35 - 1) & (v80 + v37);
      v38 = (__int64 *)(v125 + 16LL * v37);
      v39 = (_QWORD *)*v38;
      if ( (_QWORD *)*v38 == v34 )
      {
LABEL_35:
        v40 = *((_DWORD *)v38 + 2);
        goto LABEL_36;
      }
      ++v80;
    }
    if ( v81 )
      v38 = v81;
    ++v124;
    v59 = v126 + 1;
    if ( 4 * ((int)v126 + 1) < 3 * v35 )
    {
      if ( v35 - HIDWORD(v126) - v59 > v35 >> 3 )
        goto LABEL_112;
      sub_1DDD540((__int64)&v124, v35);
      if ( !v127 )
      {
LABEL_197:
        LODWORD(v126) = v126 + 1;
        BUG();
      }
      v82 = 0;
      v83 = (v127 - 1) & v36;
      v59 = v126 + 1;
      v84 = 1;
      v38 = (__int64 *)(v125 + 16LL * v83);
      v85 = *v38;
      if ( (_QWORD *)*v38 == v34 )
        goto LABEL_112;
      while ( v85 != -8 )
      {
        if ( !v82 && v85 == -16 )
          v82 = v38;
        v83 = (v127 - 1) & (v84 + v83);
        v38 = (__int64 *)(v125 + 16LL * v83);
        v85 = *v38;
        if ( (_QWORD *)*v38 == v34 )
          goto LABEL_112;
        ++v84;
      }
      if ( v82 )
        v38 = v82;
      goto LABEL_112;
    }
LABEL_63:
    sub_1DDD540((__int64)&v124, 2 * v35);
    if ( !v127 )
      goto LABEL_197;
    v58 = (v127 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v59 = v126 + 1;
    v38 = (__int64 *)(v125 + 16LL * v58);
    v60 = *v38;
    if ( (_QWORD *)*v38 != v34 )
    {
      v61 = 1;
      v62 = 0;
      while ( v60 != -8 )
      {
        if ( v60 == -16 && !v62 )
          v62 = v38;
        v58 = (v127 - 1) & (v61 + v58);
        v38 = (__int64 *)(v125 + 16LL * v58);
        v60 = *v38;
        if ( (_QWORD *)*v38 == v34 )
          goto LABEL_112;
        ++v61;
      }
      if ( v62 )
        v38 = v62;
    }
LABEL_112:
    LODWORD(v126) = v59;
    if ( *v38 != -8 )
      --HIDWORD(v126);
    *v38 = (__int64)v34;
    v40 = 0;
    *((_DWORD *)v38 + 2) = 0;
LABEL_36:
    v41 = (__int64)v34;
    do
    {
      v41 = *(_QWORD *)(v41 + 8);
      if ( v103 == (_QWORD *)v41 )
      {
        v106 = (_QWORD *)v41;
        v42 = v101;
        goto LABEL_40;
      }
    }
    while ( !*(_BYTE *)(v41 + 183) );
    v106 = (_QWORD *)v41;
    v42 = sub_39AC850(v41);
LABEL_40:
    v43 = (unsigned int *)sub_39ACC30(a1, v42, v104);
    sub_38DDD30(v107, v43);
    if ( v40 != -1 )
      *(_QWORD *)(v105 + 8LL * v40) = v42;
    v44 = v34[4];
    v45 = *v106;
    v140 = (__m128i)v4;
    v141 = v106;
    v142 = v106;
    v143 = (v45 & 0xFFFFFFFFFFFFFFF8LL) + 24;
    v147 = 0;
    v148 = -1;
    v144 = 0;
    v145 = 0;
    v146 = -1;
    sub_39AC5C0((__int64)&v140);
    v134 = v44;
    v131 = (__m128i)v4;
    v132 = (unsigned __int64)v34;
    v46 = -1;
    v133 = v106;
    v138 = 0;
    v139 = -1;
    v135 = 0;
    v136 = 0;
    v137 = -1;
    sub_39AC5C0((__int64)&v131);
    *(_QWORD *)&v108 = v140.m128i_i64[1];
    v140 = v131;
    v114 = v141;
    v142 = v133;
    *((_QWORD *)&v108 + 1) = v143;
    v143 = v134;
    v49 = 0;
    v144 = v135;
    v141 = (_QWORD *)v132;
    v145 = v136;
    v146 = v137;
    v147 = v138;
    v148 = v139;
    if ( v114 == (_QWORD *)v132 )
      goto LABEL_59;
    while ( 1 )
    {
      v50 = v146;
      if ( (_DWORD)v46 == -1 )
      {
        if ( v146 == -1 )
        {
          LODWORD(v56) = -1;
          goto LABEL_84;
        }
        v51 = *(_QWORD *)(v4 + 592);
        v53 = 0;
      }
      else
      {
        v51 = *(_QWORD *)(v4 + 592);
        v52 = (int)v46;
        v53 = 0;
        do
        {
          ++v53;
          LODWORD(v52) = *(_DWORD *)(v51 + 24 * v52 + 16);
        }
        while ( (_DWORD)v52 != -1 );
        if ( v146 == -1 )
        {
          v55 = 0;
          goto LABEL_52;
        }
      }
      v54 = v146;
      v55 = 0;
      do
      {
        ++v55;
        LODWORD(v54) = *(_DWORD *)(v51 + 24 * v54 + 16);
      }
      while ( (_DWORD)v54 != -1 );
      if ( v53 < v55 )
      {
        do
        {
          --v55;
          v50 = *(int *)(v51 + 24 * v50 + 16);
        }
        while ( v53 != v55 );
      }
      if ( v53 <= v55 )
      {
        v56 = (int)v46;
        goto LABEL_54;
      }
LABEL_52:
      v56 = (int)v46;
      do
      {
        --v53;
        v56 = *(int *)(v51 + 24 * v56 + 16);
      }
      while ( v55 != v53 );
LABEL_54:
      if ( (_DWORD)v50 != (_DWORD)v56 )
      {
        do
        {
          LODWORD(v56) = *(_DWORD *)(v51 + 24 * v56 + 16);
          LODWORD(v50) = *(_DWORD *)(v51 + 24 * v50 + 16);
        }
        while ( (_DWORD)v56 != (_DWORD)v50 );
        if ( (_DWORD)v50 == (_DWORD)v46 )
          goto LABEL_57;
        goto LABEL_73;
      }
LABEL_84:
      LODWORD(v50) = v56;
      while ( (_DWORD)v50 != (_DWORD)v46 )
      {
LABEL_73:
        v131.m128i_i64[0] = v49;
        v132 = __PAIR64__(v40, v46);
        v131.m128i_i64[1] = v144;
        v66 = (unsigned int)v153;
        if ( (unsigned int)v153 >= HIDWORD(v153) )
        {
          v120 = v50;
          sub_16CD150((__int64)&v152, v154, 0, 24, v47, v48);
          v66 = (unsigned int)v153;
          LODWORD(v50) = v120;
        }
        v63 = (__m128i *)&v152[24 * v66];
        *v63 = _mm_loadu_si128(&v131);
        v64 = (unsigned __int64)v149;
        LODWORD(v153) = v153 + 1;
        v63[1].m128i_i64[0] = v132;
        LODWORD(v46) = *(_DWORD *)(*(_QWORD *)(v4 + 592) + 24 * v46 + 16);
        v65 = v64 + 16LL * (unsigned int)v150 - 16;
        if ( (_DWORD)v46 == *(_DWORD *)(v65 + 8) )
        {
          v49 = *(_QWORD *)v65;
          LODWORD(v150) = v150 - 1;
        }
      }
LABEL_57:
      v57 = v146;
      if ( v146 != (_DWORD)v46 )
      {
        do
        {
          v67 = (signed int *)&v128[4 * v57];
          if ( *v67 > v40 )
            *v67 = v40;
          v57 = *(int *)(*(_QWORD *)(v4 + 592) + 24 * v57 + 16);
        }
        while ( (_DWORD)v57 != (_DWORD)v46 );
        v68 = v150;
        if ( (unsigned int)v150 >= HIDWORD(v150) )
        {
          sub_16CD150((__int64)&v149, v151, 0, 16, v47, v48);
          v68 = v150;
        }
        v69 = &v149[16 * v68];
        if ( v69 )
        {
          *(_QWORD *)v69 = v49;
          *((_DWORD *)v69 + 2) = v46;
          v68 = v150;
        }
        v49 = v145;
        v46 = v146;
        LODWORD(v150) = v68 + 1;
      }
      sub_39AC5C0((__int64)&v140);
      if ( v114 == v141 )
      {
LABEL_59:
        if ( __PAIR128__(v143, v140.m128i_u64[1]) == v108 )
        {
          if ( v103 == v106 )
            break;
          v35 = v127;
          v34 = v106;
          if ( v127 )
            goto LABEL_34;
LABEL_62:
          ++v124;
          goto LABEL_63;
        }
      }
    }
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v107 + 424LL))(v107, (unsigned int)v153, 4);
  if ( v152 != &v152[24 * (unsigned int)v153] )
  {
    v112 = &v152[24 * (unsigned int)v153];
    v110 = v4;
    v70 = (unsigned __int64)v152;
    do
    {
      v116 = *(_QWORD *)v70;
      v121 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
      v71 = sub_38CB470(1, v121);
      v72 = sub_39ACC30(a1, v116, v104);
      v115 = (unsigned int *)sub_38CB1F0(0, v72, v71, v121, 0);
      v117 = *(_QWORD *)(v70 + 8);
      v122 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
      v73 = sub_38CB470(1, v122);
      v74 = sub_39ACC30(a1, v117, v104);
      v118 = (unsigned int *)sub_38CB1F0(0, v74, v73, v122, 0);
      v75 = *(_QWORD *)(v110 + 592) + 24LL * *(int *)(v70 + 16);
      v76 = sub_39AC850(*(_QWORD *)v75 & 0xFFFFFFFFFFFFFFF8LL);
      v123 = (unsigned int *)sub_39ACC30(a1, v76, v104);
      v77 = sub_39ACC30(a1, *(_QWORD *)(v105 + 8LL * *(int *)(v70 + 16)), v104);
      v78 = (unsigned int)(*(_DWORD *)(v75 + 20) - 1);
      v79 = 0;
      if ( (unsigned int)v78 <= 2 )
        v79 = dword_4533148[v78];
      if ( *(_DWORD *)(v70 + 20) != *(_DWORD *)&v128[4 * *(int *)(v70 + 16)] )
        v79 |= 8u;
      v109 = (unsigned int *)v77;
      v70 += 24LL;
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v107 + 424LL))(v107, v79, 4);
      sub_38DDD30(v107, v115);
      sub_38DDD30(v107, v118);
      sub_38DDD30(v107, v123);
      sub_38DDD30(v107, v109);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v107 + 424LL))(v107, *(unsigned int *)(v75 + 8), 4);
    }
    while ( v112 != (_BYTE *)v70 );
  }
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( v105 )
    j_j___libc_free_0_0(v105);
  if ( v149 != v151 )
    _libc_free((unsigned __int64)v149);
  j___libc_free_0(v125);
  if ( v152 != v154 )
    _libc_free((unsigned __int64)v152);
}
