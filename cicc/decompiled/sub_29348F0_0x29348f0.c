// Function: sub_29348F0
// Address: 0x29348f0
//
__int64 __fastcall sub_29348F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rdx
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned int v13; // ebx
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rbx
  unsigned int v22; // edx
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // rsi
  int v26; // edx
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r12
  unsigned __int64 v31; // rsi
  int v32; // eax
  __int64 v33; // rsi
  const char *v34; // rax
  __int64 v35; // rdx
  __m128i v36; // rax
  char v37; // al
  _QWORD *v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 (__fastcall *v41)(__int64, unsigned __int64, __int64); // rax
  __int64 v42; // r12
  __int64 v43; // r15
  __int64 v44; // rdx
  unsigned int v45; // esi
  const char *v46; // rax
  __int64 v47; // rdx
  __m128i v48; // rax
  char v49; // al
  _QWORD *v50; // rcx
  _QWORD *v51; // rax
  __int64 v52; // r15
  __int64 v53; // r13
  __int64 v54; // r12
  __int64 v55; // rdx
  unsigned int v56; // esi
  int v57; // edx
  int v58; // edx
  unsigned int v59; // ecx
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rcx
  int v63; // r13d
  unsigned int v64; // r8d
  __int64 *v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // rdi
  __int64 v68; // rbx
  int v69; // edx
  __int64 v70; // r8
  int v71; // esi
  unsigned int v72; // r10d
  __int64 *v73; // rcx
  __int64 v74; // r9
  __int64 *v75; // rcx
  int v76; // edx
  _QWORD *v77; // rdx
  _QWORD *v78; // rcx
  unsigned int v79; // esi
  int v80; // edx
  int v81; // eax
  int v82; // eax
  unsigned int v83; // edx
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // r15
  __int64 v88; // r12
  __int64 v89; // rdx
  unsigned int v90; // esi
  unsigned int v91; // ecx
  __int64 *v92; // rdi
  int v93; // r8d
  unsigned int v94; // r9d
  int v95; // ecx
  int v96; // r9d
  int v97; // eax
  int v98; // r8d
  int v99; // r12d
  __int64 v100; // rdi
  __int64 v101; // rsi
  __int64 v102; // rax
  __int64 v103; // rbx
  __int64 v104; // r12
  int v106; // edx
  int v107; // r9d
  __m128i v108; // xmm3
  __m128i v109; // xmm1
  int v110; // edx
  __int64 v111; // r8
  int v112; // edx
  unsigned int v113; // ecx
  __int64 v114; // rsi
  __int64 v115; // rsi
  __int64 v116; // r8
  int v117; // r12d
  __int64 *v118; // r10
  __int64 *v119; // r9
  __int64 v120; // r12
  int v121; // r11d
  __int64 v122; // rdi
  int v123; // edx
  __int64 v124; // r8
  int v125; // edx
  unsigned int v126; // ecx
  __int64 v127; // rsi
  int v128; // r11d
  __int64 *v129; // r9
  _QWORD *v130; // rax
  int v131; // r11d
  __int64 *v132; // rdx
  int v133; // eax
  unsigned int v134; // ecx
  __int64 v135; // r11
  int v136; // edi
  __int64 *v137; // rsi
  int v138; // esi
  __int64 *v139; // rcx
  unsigned int v140; // ebx
  __int64 v141; // r8
  int v142; // r11d
  __int64 *v143; // [rsp+8h] [rbp-1B8h]
  __int64 v144; // [rsp+10h] [rbp-1B0h]
  __int64 v145; // [rsp+18h] [rbp-1A8h]
  __int64 v146; // [rsp+28h] [rbp-198h]
  unsigned int v147; // [rsp+38h] [rbp-188h]
  char v148; // [rsp+3Dh] [rbp-183h]
  __int64 v149; // [rsp+40h] [rbp-180h]
  __int64 *v150; // [rsp+48h] [rbp-178h]
  __int64 v151; // [rsp+50h] [rbp-170h]
  __int64 v153; // [rsp+60h] [rbp-160h]
  __int64 v154; // [rsp+68h] [rbp-158h]
  __int64 v155; // [rsp+70h] [rbp-150h] BYREF
  __int64 v156; // [rsp+78h] [rbp-148h]
  __int64 v157; // [rsp+80h] [rbp-140h]
  unsigned int v158; // [rsp+88h] [rbp-138h]
  __int64 v159; // [rsp+90h] [rbp-130h] BYREF
  _QWORD *v160; // [rsp+98h] [rbp-128h]
  __int64 v161; // [rsp+A0h] [rbp-120h]
  unsigned int v162; // [rsp+A8h] [rbp-118h]
  __int64 v163[4]; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v164; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v165; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v166; // [rsp+F0h] [rbp-D0h]
  const char *v167; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v168; // [rsp+108h] [rbp-B8h]
  __int16 v169; // [rsp+120h] [rbp-A0h]
  __m128i v170; // [rsp+130h] [rbp-90h] BYREF
  __m128i v171; // [rsp+140h] [rbp-80h]
  __int64 v172; // [rsp+150h] [rbp-70h]
  _QWORD v173[4]; // [rsp+160h] [rbp-60h] BYREF
  __int16 v174; // [rsp+180h] [rbp-40h]

  v3 = *(__int64 **)a3;
  v4 = *(unsigned int *)(a3 + 8);
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v143 = &v3[v4];
  if ( v3 == v143 )
  {
    v101 = 0;
    v100 = 0;
    goto LABEL_129;
  }
  v150 = v3;
  while ( 2 )
  {
    v9 = *(_QWORD *)(*v150 + 8);
    v149 = *v150;
    v153 = v9;
    sub_D5F1F0((__int64)a1, a2);
    if ( v162 )
    {
      v6 = (v162 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v7 = &v160[2 * v6];
      v8 = *v7;
      if ( v9 == *v7 )
      {
LABEL_4:
        if ( v7 != &v160[2 * v162] )
        {
          sub_BD84D0(v149, v7[1]);
          sub_B43D60((_QWORD *)v149);
          goto LABEL_6;
        }
      }
      else
      {
        v106 = 1;
        while ( v8 != -4096 )
        {
          v107 = v106 + 1;
          v6 = (v162 - 1) & (v106 + v6);
          v7 = &v160[2 * v6];
          v8 = *v7;
          if ( v9 == *v7 )
            goto LABEL_4;
          v106 = v107;
        }
      }
    }
    v173[0] = sub_BD5D20(a2);
    v173[2] = ".sroa.speculated";
    v10 = *(_DWORD *)(a2 + 4);
    v173[1] = v11;
    v174 = 773;
    v12 = sub_D5C860(a1, v9, v10 & 0x7FFFFFF, (__int64)v173);
    if ( !v162 )
    {
      ++v159;
      goto LABEL_197;
    }
    v13 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
    v14 = (v162 - 1) & v13;
    v15 = &v160[2 * v14];
    v16 = *v15;
    if ( v153 != *v15 )
    {
      v131 = 1;
      v132 = 0;
      while ( v16 != -4096 )
      {
        if ( v16 == -8192 && !v132 )
          v132 = v15;
        v14 = (v162 - 1) & (v131 + v14);
        v15 = &v160[2 * v14];
        v16 = *v15;
        if ( v153 == *v15 )
          goto LABEL_10;
        ++v131;
      }
      if ( !v132 )
        v132 = v15;
      ++v159;
      v133 = v161 + 1;
      if ( 4 * ((int)v161 + 1) < 3 * v162 )
      {
        if ( v162 - HIDWORD(v161) - v133 <= v162 >> 3 )
        {
          sub_2927F00((__int64)&v159, v162);
          if ( !v162 )
          {
LABEL_250:
            LODWORD(v161) = v161 + 1;
            BUG();
          }
          v138 = 1;
          v139 = 0;
          v140 = (v162 - 1) & v13;
          v133 = v161 + 1;
          v132 = &v160[2 * v140];
          v141 = *v132;
          if ( v153 != *v132 )
          {
            while ( v141 != -4096 )
            {
              if ( !v139 && v141 == -8192 )
                v139 = v132;
              v140 = (v162 - 1) & (v138 + v140);
              v132 = &v160[2 * v140];
              v141 = *v132;
              if ( v153 == *v132 )
                goto LABEL_187;
              ++v138;
            }
            if ( v139 )
              v132 = v139;
          }
        }
        goto LABEL_187;
      }
LABEL_197:
      sub_2927F00((__int64)&v159, 2 * v162);
      if ( !v162 )
        goto LABEL_250;
      v134 = (v162 - 1) & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
      v133 = v161 + 1;
      v132 = &v160[2 * v134];
      v135 = *v132;
      if ( v153 != *v132 )
      {
        v136 = 1;
        v137 = 0;
        while ( v135 != -4096 )
        {
          if ( !v137 && v135 == -8192 )
            v137 = v132;
          v134 = (v162 - 1) & (v136 + v134);
          v132 = &v160[2 * v134];
          v135 = *v132;
          if ( v153 == *v132 )
            goto LABEL_187;
          ++v136;
        }
        if ( v137 )
          v132 = v137;
      }
LABEL_187:
      LODWORD(v161) = v133;
      if ( *v132 != -4096 )
        --HIDWORD(v161);
      v132[1] = v12;
      *v132 = v153;
    }
LABEL_10:
    sub_BD84D0(v149, v12);
    sub_B91FC0(v163, v149);
    _BitScanReverse64(&v17, 1LL << (*(_WORD *)(v149 + 2) >> 1));
    v148 = 63 - (v17 ^ 0x3F);
    v18 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( !v18 )
      goto LABEL_65;
    v151 = (__int64)a1;
    v154 = 0;
    v146 = 8LL * v18;
    v147 = ((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4);
    do
    {
      v19 = *(_QWORD *)(a2 - 8);
      v20 = *(_QWORD *)(v19 + 4 * v154);
      v21 = *(_QWORD *)(v19 + 32LL * *(unsigned int *)(a2 + 72) + v154);
      if ( !v158 )
        goto LABEL_19;
      v22 = (v158 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v23 = (__int64 *)(v156 + 88LL * v22);
      v24 = *v23;
      if ( v21 != *v23 )
      {
        v95 = 1;
        while ( v24 != -4096 )
        {
          v96 = v95 + 1;
          v22 = (v158 - 1) & (v95 + v22);
          v23 = (__int64 *)(v156 + 88LL * v22);
          v24 = *v23;
          if ( v21 == *v23 )
            goto LABEL_14;
          v95 = v96;
        }
LABEL_19:
        v31 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v31 == v21 + 48 )
        {
          v33 = 0;
        }
        else
        {
          if ( !v31 )
            BUG();
          v32 = *(unsigned __int8 *)(v31 - 24);
          v33 = v31 - 24;
          if ( (unsigned int)(v32 - 30) >= 0xB )
            v33 = 0;
        }
        sub_D5F1F0(v151, v33);
        v34 = sub_BD5D20(v21);
        v169 = 261;
        v167 = v34;
        v168 = v35;
        v36.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        LOWORD(v166) = 773;
        v164 = v36;
        v165.m128i_i64[0] = (__int64)".sroa.speculate.cast.";
        v37 = v169;
        if ( (_BYTE)v169 )
        {
          if ( (_BYTE)v169 == 1 )
          {
            v109 = _mm_loadu_si128(&v165);
            v170 = _mm_loadu_si128(&v164);
            v172 = v166;
            v171 = v109;
          }
          else
          {
            if ( HIBYTE(v169) == 1 )
            {
              v38 = v167;
              v145 = v168;
            }
            else
            {
              v38 = &v167;
              v37 = 2;
            }
            v171.m128i_i64[0] = (__int64)v38;
            v170.m128i_i64[0] = (__int64)&v164;
            v171.m128i_i64[1] = v145;
            LOBYTE(v172) = 2;
            BYTE1(v172) = v37;
          }
        }
        else
        {
          LOWORD(v172) = 256;
        }
        v39 = *(_QWORD *)(*(_QWORD *)(v149 - 32) + 8LL);
        if ( v39 != *(_QWORD *)(v20 + 8) )
        {
          if ( *(_BYTE *)v20 > 0x15u )
          {
            v174 = 257;
            v20 = sub_B52190(v20, v39, (__int64)v173, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v151 + 88) + 16LL))(
              *(_QWORD *)(v151 + 88),
              v20,
              &v170,
              *(_QWORD *)(v151 + 56),
              *(_QWORD *)(v151 + 64));
            v87 = *(_QWORD *)v151;
            v88 = *(_QWORD *)v151 + 16LL * *(unsigned int *)(v151 + 8);
            if ( *(_QWORD *)v151 != v88 )
            {
              do
              {
                v89 = *(_QWORD *)(v87 + 8);
                v90 = *(_DWORD *)v87;
                v87 += 16;
                sub_B99FD0(v20, v90, v89);
              }
              while ( v88 != v87 );
            }
          }
          else
          {
            v40 = *(_QWORD *)(v151 + 80);
            v41 = *(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v40 + 144LL);
            if ( v41 == sub_B32D70 )
              v20 = sub_ADB060(v20, v39);
            else
              v20 = v41(v40, v20, v39);
            if ( *(_BYTE *)v20 > 0x1Cu )
            {
              (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v151 + 88) + 16LL))(
                *(_QWORD *)(v151 + 88),
                v20,
                &v170,
                *(_QWORD *)(v151 + 56),
                *(_QWORD *)(v151 + 64));
              v42 = *(_QWORD *)v151 + 16LL * *(unsigned int *)(v151 + 8);
              if ( *(_QWORD *)v151 != v42 )
              {
                v43 = *(_QWORD *)v151;
                do
                {
                  v44 = *(_QWORD *)(v43 + 8);
                  v45 = *(_DWORD *)v43;
                  v43 += 16;
                  sub_B99FD0(v20, v45, v44);
                }
                while ( v42 != v43 );
              }
            }
          }
        }
        v46 = sub_BD5D20(v21);
        v169 = 261;
        v167 = v46;
        v168 = v47;
        v48.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v164 = v48;
        v165.m128i_i64[0] = (__int64)".sroa.speculate.load.";
        v49 = v169;
        LOWORD(v166) = 773;
        if ( (_BYTE)v169 )
        {
          if ( (_BYTE)v169 == 1 )
          {
            v108 = _mm_loadu_si128(&v165);
            v170 = _mm_loadu_si128(&v164);
            v172 = v166;
            v171 = v108;
          }
          else
          {
            if ( HIBYTE(v169) == 1 )
            {
              v50 = v167;
              v144 = v168;
            }
            else
            {
              v50 = &v167;
              v49 = 2;
            }
            v171.m128i_i64[0] = (__int64)v50;
            v170.m128i_i64[0] = (__int64)&v164;
            v171.m128i_i64[1] = v144;
            LOBYTE(v172) = 2;
            BYTE1(v172) = v49;
          }
        }
        else
        {
          LOWORD(v172) = 256;
        }
        v174 = 257;
        v51 = sub_BD2C40(80, 1u);
        v52 = (__int64)v51;
        if ( v51 )
          sub_B4D190((__int64)v51, v153, v20, (__int64)v173, 0, v148, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v151 + 88) + 16LL))(
          *(_QWORD *)(v151 + 88),
          v52,
          &v170,
          *(_QWORD *)(v151 + 56),
          *(_QWORD *)(v151 + 64));
        v53 = *(_QWORD *)v151;
        v54 = *(_QWORD *)v151 + 16LL * *(unsigned int *)(v151 + 8);
        if ( *(_QWORD *)v151 != v54 )
        {
          do
          {
            v55 = *(_QWORD *)(v53 + 8);
            v56 = *(_DWORD *)v53;
            v53 += 16;
            sub_B99FD0(v52, v56, v55);
          }
          while ( v54 != v53 );
        }
        if ( v163[0] || v163[1] || v163[2] || v163[3] )
          sub_B9A100(v52, v163);
        v57 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        if ( v57 == *(_DWORD *)(v12 + 72) )
        {
          sub_B48D90(v12);
          v57 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        }
        v58 = (v57 + 1) & 0x7FFFFFF;
        v59 = v58 | *(_DWORD *)(v12 + 4) & 0xF8000000;
        v60 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v58 - 1);
        *(_DWORD *)(v12 + 4) = v59;
        if ( *(_QWORD *)v60 )
        {
          v61 = *(_QWORD *)(v60 + 8);
          **(_QWORD **)(v60 + 16) = v61;
          if ( v61 )
            *(_QWORD *)(v61 + 16) = *(_QWORD *)(v60 + 16);
        }
        *(_QWORD *)v60 = v52;
        if ( v52 )
        {
          v62 = *(_QWORD *)(v52 + 16);
          *(_QWORD *)(v60 + 8) = v62;
          if ( v62 )
            *(_QWORD *)(v62 + 16) = v60 + 8;
          *(_QWORD *)(v60 + 16) = v52 + 16;
          *(_QWORD *)(v52 + 16) = v60;
        }
        *(_QWORD *)(*(_QWORD *)(v12 - 8)
                  + 32LL * *(unsigned int *)(v12 + 72)
                  + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = v21;
        if ( v158 )
        {
          v63 = 1;
          v64 = (v158 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v65 = 0;
          v66 = v156 + 88LL * v64;
          v67 = *(_QWORD *)v66;
          if ( v21 == *(_QWORD *)v66 )
          {
LABEL_58:
            v68 = v66 + 8;
            v69 = *(_BYTE *)(v66 + 16) & 1;
            if ( v69 )
            {
LABEL_59:
              v70 = v68 + 16;
              v71 = 3;
              goto LABEL_60;
            }
LABEL_83:
            v79 = *(_DWORD *)(v68 + 24);
            v70 = *(_QWORD *)(v68 + 16);
            if ( !v79 )
            {
              v91 = *(_DWORD *)(v68 + 8);
              ++*(_QWORD *)v68;
              v92 = 0;
              v93 = (v91 >> 1) + 1;
              goto LABEL_105;
            }
            v71 = v79 - 1;
LABEL_60:
            v72 = v71 & v147;
            v73 = (__int64 *)(v70 + 16LL * (v71 & v147));
            v74 = *v73;
            if ( v153 == *v73 )
            {
LABEL_61:
              v75 = v73 + 1;
LABEL_62:
              *v75 = v52;
              goto LABEL_63;
            }
            v99 = 1;
            v92 = 0;
            while ( v74 != -4096 )
            {
              if ( v74 == -8192 && !v92 )
                v92 = v73;
              v72 = v71 & (v99 + v72);
              v73 = (__int64 *)(v70 + 16LL * v72);
              v74 = *v73;
              if ( v153 == *v73 )
                goto LABEL_61;
              ++v99;
            }
            v94 = 12;
            v79 = 4;
            if ( !v92 )
              v92 = v73;
            v91 = *(_DWORD *)(v68 + 8);
            ++*(_QWORD *)v68;
            v93 = (v91 >> 1) + 1;
            if ( (_BYTE)v69 )
            {
LABEL_106:
              if ( v94 <= 4 * v93 )
              {
                sub_2926720(v68, 2 * v79);
                if ( (*(_BYTE *)(v68 + 8) & 1) != 0 )
                {
                  v111 = v68 + 16;
                  v112 = 3;
                }
                else
                {
                  v110 = *(_DWORD *)(v68 + 24);
                  v111 = *(_QWORD *)(v68 + 16);
                  if ( !v110 )
                    goto LABEL_248;
                  v112 = v110 - 1;
                }
                v113 = v112 & v147;
                v92 = (__int64 *)(v111 + 16LL * (v112 & v147));
                v114 = *v92;
                if ( v153 == *v92 )
                  goto LABEL_148;
                v142 = 1;
                v129 = 0;
                while ( v114 != -4096 )
                {
                  if ( v114 == -8192 && !v129 )
                    v129 = v92;
                  v113 = v112 & (v142 + v113);
                  v92 = (__int64 *)(v111 + 16LL * v113);
                  v114 = *v92;
                  if ( v153 == *v92 )
                    goto LABEL_148;
                  ++v142;
                }
              }
              else
              {
                if ( v79 - *(_DWORD *)(v68 + 12) - v93 > v79 >> 3 )
                {
LABEL_108:
                  *(_DWORD *)(v68 + 8) = (2 * (v91 >> 1) + 2) | v91 & 1;
                  if ( *v92 != -4096 )
                    --*(_DWORD *)(v68 + 12);
                  v92[1] = 0;
                  v75 = v92 + 1;
                  *v92 = v153;
                  goto LABEL_62;
                }
                sub_2926720(v68, v79);
                if ( (*(_BYTE *)(v68 + 8) & 1) != 0 )
                {
                  v124 = v68 + 16;
                  v125 = 3;
                }
                else
                {
                  v123 = *(_DWORD *)(v68 + 24);
                  v124 = *(_QWORD *)(v68 + 16);
                  if ( !v123 )
                  {
LABEL_248:
                    *(_DWORD *)(v68 + 8) = (2 * (*(_DWORD *)(v68 + 8) >> 1) + 2) | *(_DWORD *)(v68 + 8) & 1;
                    BUG();
                  }
                  v125 = v123 - 1;
                }
                v126 = v125 & v147;
                v92 = (__int64 *)(v124 + 16LL * (v125 & v147));
                v127 = *v92;
                if ( v153 == *v92 )
                {
LABEL_148:
                  v91 = *(_DWORD *)(v68 + 8);
                  goto LABEL_108;
                }
                v128 = 1;
                v129 = 0;
                while ( v127 != -4096 )
                {
                  if ( v127 == -8192 && !v129 )
                    v129 = v92;
                  v126 = v125 & (v128 + v126);
                  v92 = (__int64 *)(v124 + 16LL * v126);
                  v127 = *v92;
                  if ( v153 == *v92 )
                    goto LABEL_148;
                  ++v128;
                }
              }
              if ( v129 )
                v92 = v129;
              goto LABEL_148;
            }
            v79 = *(_DWORD *)(v68 + 24);
LABEL_105:
            v94 = 3 * v79;
            goto LABEL_106;
          }
          while ( v67 != -4096 )
          {
            if ( v67 == -8192 && !v65 )
              v65 = (__int64 *)v66;
            v64 = (v158 - 1) & (v63 + v64);
            v66 = v156 + 88LL * v64;
            v67 = *(_QWORD *)v66;
            if ( v21 == *(_QWORD *)v66 )
              goto LABEL_58;
            ++v63;
          }
          if ( !v65 )
            v65 = (__int64 *)v66;
          ++v155;
          v76 = v157 + 1;
          if ( 4 * ((int)v157 + 1) < 3 * v158 )
          {
            if ( v158 - HIDWORD(v157) - v76 <= v158 >> 3 )
            {
              sub_2934680((__int64)&v155, v158);
              if ( !v158 )
              {
LABEL_249:
                LODWORD(v157) = v157 + 1;
                BUG();
              }
              v119 = 0;
              LODWORD(v120) = (v158 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
              v121 = 1;
              v65 = (__int64 *)(v156 + 88LL * (unsigned int)v120);
              v76 = v157 + 1;
              v122 = *v65;
              if ( v21 != *v65 )
              {
                while ( v122 != -4096 )
                {
                  if ( v122 == -8192 && !v119 )
                    v119 = v65;
                  v120 = (v158 - 1) & ((_DWORD)v120 + v121);
                  v65 = (__int64 *)(v156 + 88 * v120);
                  v122 = *v65;
                  if ( v21 == *v65 )
                    goto LABEL_76;
                  ++v121;
                }
                if ( v119 )
                  v65 = v119;
              }
            }
LABEL_76:
            LODWORD(v157) = v76;
            if ( *v65 != -4096 )
              --HIDWORD(v157);
            *v65 = v21;
            v77 = v65 + 3;
            v68 = (__int64)(v65 + 1);
            v78 = v65 + 11;
            *(v78 - 10) = 0;
            *(v78 - 9) = 1;
            do
            {
              if ( v77 )
                *v77 = -4096;
              v77 += 2;
            }
            while ( v77 != v78 );
            LOBYTE(v69) = *(_BYTE *)(v68 + 8) & 1;
            if ( (_BYTE)v69 )
              goto LABEL_59;
            goto LABEL_83;
          }
        }
        else
        {
          ++v155;
        }
        sub_2934680((__int64)&v155, 2 * v158);
        if ( !v158 )
          goto LABEL_249;
        LODWORD(v115) = (v158 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v65 = (__int64 *)(v156 + 88LL * (unsigned int)v115);
        v76 = v157 + 1;
        v116 = *v65;
        if ( v21 != *v65 )
        {
          v117 = 1;
          v118 = 0;
          while ( v116 != -4096 )
          {
            if ( !v118 && v116 == -8192 )
              v118 = v65;
            v115 = (v158 - 1) & ((_DWORD)v115 + v117);
            v65 = (__int64 *)(v156 + 88 * v115);
            v116 = *v65;
            if ( v21 == *v65 )
              goto LABEL_76;
            ++v117;
          }
          if ( v118 )
            v65 = v118;
        }
        goto LABEL_76;
      }
LABEL_14:
      if ( v23 == (__int64 *)(v156 + 88LL * v158) )
        goto LABEL_19;
      if ( (v23[2] & 1) != 0 )
      {
        v25 = v23 + 3;
        v26 = 3;
      }
      else
      {
        v80 = *((_DWORD *)v23 + 8);
        v25 = (__int64 *)v23[3];
        if ( !v80 )
          goto LABEL_19;
        v26 = v80 - 1;
      }
      v27 = v26 & v147;
      v28 = &v25[2 * (v26 & v147)];
      v29 = *v28;
      if ( v153 != *v28 )
      {
        v97 = 1;
        while ( v29 != -4096 )
        {
          v98 = v97 + 1;
          v27 = v26 & (v97 + v27);
          v28 = &v25[2 * v27];
          v29 = *v28;
          if ( v153 == *v28 )
            goto LABEL_18;
          v97 = v98;
        }
        goto LABEL_19;
      }
LABEL_18:
      v30 = v28[1];
      if ( !v30 )
        goto LABEL_19;
      v81 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
      if ( v81 == *(_DWORD *)(v12 + 72) )
      {
        sub_B48D90(v12);
        v81 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
      }
      v82 = (v81 + 1) & 0x7FFFFFF;
      v83 = v82 | *(_DWORD *)(v12 + 4) & 0xF8000000;
      v84 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v82 - 1);
      *(_DWORD *)(v12 + 4) = v83;
      if ( *(_QWORD *)v84 )
      {
        v85 = *(_QWORD *)(v84 + 8);
        **(_QWORD **)(v84 + 16) = v85;
        if ( v85 )
          *(_QWORD *)(v85 + 16) = *(_QWORD *)(v84 + 16);
      }
      *(_QWORD *)v84 = v30;
      v86 = *(_QWORD *)(v30 + 16);
      *(_QWORD *)(v84 + 8) = v86;
      if ( v86 )
        *(_QWORD *)(v86 + 16) = v84 + 8;
      *(_QWORD *)(v84 + 16) = v30 + 16;
      *(_QWORD *)(v30 + 16) = v84;
      *(_QWORD *)(*(_QWORD *)(v12 - 8)
                + 32LL * *(unsigned int *)(v12 + 72)
                + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = v21;
LABEL_63:
      v154 += 8;
    }
    while ( v146 != v154 );
    a1 = (__int64 *)v151;
LABEL_65:
    sub_B43D60((_QWORD *)v149);
LABEL_6:
    if ( v143 != ++v150 )
      continue;
    break;
  }
  v100 = (__int64)v160;
  v101 = 2LL * v162;
  if ( (_DWORD)v161 && &v160[v101] != v160 )
  {
    v130 = v160;
    do
    {
      if ( *v130 != -8192 && *v130 != -4096 )
        break;
      v130 += 2;
    }
    while ( &v160[v101] != v130 );
  }
LABEL_129:
  sub_C7D6A0(v100, v101 * 8, 8);
  v102 = v158;
  if ( v158 )
  {
    v103 = v156;
    v104 = v156 + 88LL * v158;
    do
    {
      if ( *(_QWORD *)v103 != -4096 && *(_QWORD *)v103 != -8192 && (*(_BYTE *)(v103 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v103 + 24), 16LL * *(unsigned int *)(v103 + 32), 8);
      v103 += 88;
    }
    while ( v104 != v103 );
    v102 = v158;
  }
  return sub_C7D6A0(v156, 88 * v102, 8);
}
