// Function: sub_33846E0
// Address: 0x33846e0
//
void __fastcall sub_33846E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD *v3; // rdx
  unsigned __int64 v4; // rax
  int v5; // edx
  unsigned __int64 v6; // rax
  bool v7; // cf
  int *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  const __m128i *v11; // rdi
  __int64 v12; // rbx
  __int64 *v13; // r8
  unsigned int v14; // esi
  __int64 v15; // r15
  unsigned int v16; // r11d
  __int64 v17; // rax
  int *v18; // rcx
  void (__fastcall *v19)(unsigned __int64 *, __int64, __int64, __int64, int *); // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  __int16 v22; // ax
  __m128i *v23; // r12
  __m128i *v24; // rax
  __int64 v25; // rsi
  int *v26; // r15
  int *v27; // rcx
  const __m128i *v28; // rdx
  unsigned int v29; // r13d
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 (__fastcall *v35)(__int64, __m128i *, __int64 *, __m128i *, __int64 *, __int64); // r15
  __int64 v36; // r9
  __m128i *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  _QWORD *v49; // rax
  _BYTE *v50; // r15
  unsigned __int64 v51; // r12
  __int64 v52; // r13
  unsigned __int64 v53; // r14
  unsigned __int64 *v54; // r13
  unsigned __int64 *v55; // r15
  unsigned __int64 *v56; // r12
  unsigned __int64 v57; // r14
  __int64 v58; // r12
  __int64 v59; // r15
  int v60; // eax
  int v61; // r9d
  int v62; // ecx
  int v63; // edx
  int v64; // ebx
  __m128i *v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rbx
  int v68; // edx
  int v69; // r12d
  _QWORD *v70; // rax
  unsigned __int64 v71; // r13
  unsigned __int64 v72; // rdi
  __int64 v73; // r14
  unsigned __int64 v74; // r12
  __int64 v75; // rbx
  unsigned __int64 v76; // r15
  unsigned __int64 *v77; // rbx
  unsigned __int64 *v78; // rbx
  unsigned __int64 *v79; // r12
  unsigned __int64 *v80; // rdi
  __m128i *v81; // rsi
  __m128i *v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rax
  unsigned __int64 v89; // rdx
  __int64 *v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  _QWORD *v94; // rax
  _BYTE *v95; // r15
  unsigned __int64 v96; // r12
  __int64 v97; // r13
  unsigned __int64 v98; // r14
  unsigned __int64 *v99; // r13
  unsigned __int64 *v100; // r15
  int v101; // r12d
  __int64 v102; // rdx
  int v103; // ecx
  int v104; // edx
  int v105; // r11d
  int v106; // r11d
  __int64 v107; // r9
  unsigned int v108; // ecx
  int *v109; // r8
  int v110; // edi
  __int64 v111; // rsi
  int v112; // r11d
  int v113; // r11d
  int v114; // edi
  __int64 v115; // r9
  unsigned int v116; // ecx
  int *v117; // r8
  __int128 v118; // [rsp-10h] [rbp-460h]
  __int64 v119; // [rsp+0h] [rbp-450h]
  __int64 v120; // [rsp+0h] [rbp-450h]
  __int64 v121; // [rsp+0h] [rbp-450h]
  int v122; // [rsp+0h] [rbp-450h]
  __int64 v123; // [rsp+0h] [rbp-450h]
  __int64 v124; // [rsp+0h] [rbp-450h]
  __int64 v125; // [rsp+8h] [rbp-448h]
  __int64 v126; // [rsp+8h] [rbp-448h]
  __int64 v127; // [rsp+8h] [rbp-448h]
  __int64 v128; // [rsp+8h] [rbp-448h]
  int *v129; // [rsp+10h] [rbp-440h]
  __int64 v131; // [rsp+20h] [rbp-430h]
  __int64 v132; // [rsp+40h] [rbp-410h]
  __int64 v133; // [rsp+48h] [rbp-408h]
  unsigned int v134; // [rsp+50h] [rbp-400h]
  const __m128i *v135; // [rsp+50h] [rbp-400h]
  int *v136; // [rsp+58h] [rbp-3F8h]
  __int64 v137; // [rsp+68h] [rbp-3E8h]
  __int64 v138; // [rsp+70h] [rbp-3E0h]
  unsigned int v139; // [rsp+A8h] [rbp-3A8h]
  __int64 v140; // [rsp+B0h] [rbp-3A0h]
  int v141; // [rsp+B0h] [rbp-3A0h]
  __int64 v143; // [rsp+B8h] [rbp-398h]
  __m128i v144; // [rsp+E0h] [rbp-370h] BYREF
  __int64 v145; // [rsp+F0h] [rbp-360h] BYREF
  int v146; // [rsp+F8h] [rbp-358h]
  unsigned __int64 v147; // [rsp+100h] [rbp-350h] BYREF
  __int64 v148; // [rsp+108h] [rbp-348h]
  __m128i v149; // [rsp+120h] [rbp-330h] BYREF
  __int64 v150; // [rsp+130h] [rbp-320h]
  _BYTE *v151; // [rsp+140h] [rbp-310h] BYREF
  __int64 v152; // [rsp+148h] [rbp-308h]
  _BYTE v153[128]; // [rsp+150h] [rbp-300h] BYREF
  _BYTE *v154; // [rsp+1D0h] [rbp-280h] BYREF
  __int64 v155; // [rsp+1D8h] [rbp-278h]
  _BYTE v156[128]; // [rsp+1E0h] [rbp-270h] BYREF
  __int64 v157; // [rsp+260h] [rbp-1F0h] BYREF
  int v158; // [rsp+268h] [rbp-1E8h]
  int v159; // [rsp+26Ch] [rbp-1E4h]
  unsigned __int64 *v160; // [rsp+270h] [rbp-1E0h] BYREF
  __int64 v161; // [rsp+278h] [rbp-1D8h]
  _BYTE v162[32]; // [rsp+280h] [rbp-1D0h] BYREF
  _BYTE *v163; // [rsp+2A0h] [rbp-1B0h] BYREF
  __int64 v164; // [rsp+2A8h] [rbp-1A8h]
  _BYTE v165[112]; // [rsp+2B0h] [rbp-1A0h] BYREF
  _QWORD *v166; // [rsp+320h] [rbp-130h] BYREF
  _QWORD v167[2]; // [rsp+330h] [rbp-120h] BYREF
  unsigned int v168; // [rsp+340h] [rbp-110h]
  __int64 v169; // [rsp+348h] [rbp-108h]
  __int16 v170; // [rsp+350h] [rbp-100h]
  __int64 v171; // [rsp+358h] [rbp-F8h]
  __int64 v172; // [rsp+360h] [rbp-F0h]
  unsigned __int64 v173[2]; // [rsp+368h] [rbp-E8h] BYREF
  _BYTE v174[64]; // [rsp+378h] [rbp-D8h] BYREF
  char *v175; // [rsp+3B8h] [rbp-98h]
  __int64 v176; // [rsp+3C0h] [rbp-90h]
  __int64 v177; // [rsp+3C8h] [rbp-88h]
  char v178; // [rsp+3D0h] [rbp-80h] BYREF
  int *v179; // [rsp+3D8h] [rbp-78h]
  __int64 v180; // [rsp+3E0h] [rbp-70h]
  _BYTE v181[16]; // [rsp+3E8h] [rbp-68h] BYREF
  _BYTE *v182; // [rsp+3F8h] [rbp-58h]
  __int64 v183; // [rsp+400h] [rbp-50h]
  _BYTE v184[72]; // [rsp+408h] [rbp-48h] BYREF

  v2 = *(_QWORD *)(a2 + 40);
  v151 = v153;
  v152 = 0x800000000LL;
  v154 = v156;
  v155 = 0x800000000LL;
  v3 = (_QWORD *)(sub_AA5510(v2) + 48);
  v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v4 == v3 )
  {
    v136 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = *(unsigned __int8 *)(v4 - 24);
    v6 = v4 - 24;
    v7 = (unsigned int)(v5 - 30) < 0xB;
    v8 = 0;
    if ( v7 )
      v8 = (int *)v6;
    v136 = v8;
  }
  v9 = *(_QWORD *)(a1 + 864);
  v138 = *(_QWORD *)(v9 + 16);
  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(v9 + 40) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(v9 + 40) + 16LL));
  v11 = *(const __m128i **)(a1 + 864);
  v12 = *(_QWORD *)(a1 + 960);
  v140 = v10;
  v13 = (__int64 *)v11[2].m128i_i64[1];
  v14 = *(_DWORD *)(v12 + 144);
  v131 = v13[4];
  if ( !v14 )
  {
    ++*(_QWORD *)(v12 + 120);
    goto LABEL_171;
  }
  v15 = *(_QWORD *)(v12 + 128);
  v139 = ((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4);
  v16 = (v14 - 1) & v139;
  v17 = v15 + 16LL * v16;
  v18 = *(int **)v17;
  if ( v136 == *(int **)v17 )
  {
LABEL_8:
    v134 = *(_DWORD *)(v17 + 8);
    goto LABEL_9;
  }
  v101 = 1;
  v102 = 0;
  while ( v18 != (int *)-4096LL )
  {
    if ( !v102 && v18 == (int *)-8192LL )
      v102 = v17;
    v16 = (v14 - 1) & (v16 + v101);
    v17 = v15 + 16LL * v16;
    v18 = *(int **)v17;
    if ( v136 == *(int **)v17 )
      goto LABEL_8;
    ++v101;
  }
  v103 = *(_DWORD *)(v12 + 136);
  if ( v102 )
    v17 = v102;
  ++*(_QWORD *)(v12 + 120);
  v104 = v103 + 1;
  if ( 4 * (v103 + 1) >= 3 * v14 )
  {
LABEL_171:
    sub_3384500(v12 + 120, 2 * v14);
    v105 = *(_DWORD *)(v12 + 144);
    if ( v105 )
    {
      v106 = v105 - 1;
      v107 = *(_QWORD *)(v12 + 128);
      v104 = *(_DWORD *)(v12 + 136) + 1;
      v108 = v106 & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
      v17 = v107 + 16LL * v108;
      v109 = *(int **)v17;
      if ( v136 == *(int **)v17 )
        goto LABEL_166;
      v110 = 1;
      v111 = 0;
      while ( v109 != (int *)-4096LL )
      {
        if ( !v111 && v109 == (int *)-8192LL )
          v111 = v17;
        v108 = v106 & (v110 + v108);
        v17 = v107 + 16LL * v108;
        v109 = *(int **)v17;
        if ( v136 == *(int **)v17 )
          goto LABEL_166;
        ++v110;
      }
LABEL_175:
      if ( v111 )
        v17 = v111;
      goto LABEL_166;
    }
LABEL_197:
    ++*(_DWORD *)(v12 + 136);
    BUG();
  }
  if ( v14 - *(_DWORD *)(v12 + 140) - v104 <= v14 >> 3 )
  {
    sub_3384500(v12 + 120, v14);
    v112 = *(_DWORD *)(v12 + 144);
    if ( v112 )
    {
      v113 = v112 - 1;
      v114 = 1;
      v111 = 0;
      v115 = *(_QWORD *)(v12 + 128);
      v116 = v113 & v139;
      v104 = *(_DWORD *)(v12 + 136) + 1;
      v17 = v115 + 16LL * (v113 & v139);
      v117 = *(int **)v17;
      if ( v136 == *(int **)v17 )
        goto LABEL_166;
      while ( v117 != (int *)-4096LL )
      {
        if ( !v111 && v117 == (int *)-8192LL )
          v111 = v17;
        v116 = v113 & (v114 + v116);
        v17 = v115 + 16LL * v116;
        v117 = *(int **)v17;
        if ( v136 == *(int **)v17 )
          goto LABEL_166;
        ++v114;
      }
      goto LABEL_175;
    }
    goto LABEL_197;
  }
LABEL_166:
  *(_DWORD *)(v12 + 136) = v104;
  if ( *(_QWORD *)v17 != -4096 )
    --*(_DWORD *)(v12 + 140);
  *(_DWORD *)(v17 + 8) = 0;
  v134 = 0;
  *(_QWORD *)v17 = v136;
  v11 = *(const __m128i **)(a1 + 864);
  v13 = (__int64 *)v11[2].m128i_i64[1];
LABEL_9:
  v144 = _mm_loadu_si128(v11 + 24);
  v19 = *(void (__fastcall **)(unsigned __int64 *, __int64, __int64, __int64, int *))(*(_QWORD *)v138 + 2456LL);
  v20 = sub_2E79000(v13);
  v19(&v147, v138, v20, v140, v136);
  v21 = v147;
  v137 = v148;
  if ( v147 != v148 )
  {
    while ( 1 )
    {
      v157 = *(_QWORD *)v21;
      v158 = *(_DWORD *)(v21 + 8);
      v159 = *(_DWORD *)(v21 + 12);
      v160 = (unsigned __int64 *)v162;
      v161 = 0x100000000LL;
      if ( *(_DWORD *)(v21 + 24) )
        sub_337B130((__int64)&v160, v21 + 16);
      v163 = v165;
      v164 = 0x200000000LL;
      if ( *(_DWORD *)(v21 + 72) )
        sub_337B7E0((__int64)&v163, v21 + 64);
      v166 = v167;
      sub_33654B0((__int64 *)&v166, *(_BYTE **)(v21 + 192), *(_QWORD *)(v21 + 192) + *(_QWORD *)(v21 + 200));
      v168 = *(_DWORD *)(v21 + 224);
      v169 = *(_QWORD *)(v21 + 232);
      v22 = *(_WORD *)(v21 + 240);
      v175 = &v178;
      v170 = v22;
      v179 = (int *)v181;
      v173[0] = (unsigned __int64)v174;
      v171 = 0;
      LODWORD(v172) = 0;
      v173[1] = 0x400000000LL;
      v176 = 0;
      v177 = 4;
      v180 = 0x400000000LL;
      v182 = v184;
      v183 = 0x400000000LL;
      v184[20] = 0;
      if ( (_DWORD)v157 != 1 )
        break;
      (*(void (__fastcall **)(__int64, __int64 *, __int64, __int64, _QWORD))(*(_QWORD *)v138 + 2480LL))(
        v138,
        &v157,
        v171,
        v172,
        *(_QWORD *)(a1 + 864));
      if ( v168 > 1 )
      {
        if ( v168 == 5 )
        {
          v145 = 0;
          v146 = 0;
          v35 = *(__int64 (__fastcall **)(__int64, __m128i *, __int64 *, __m128i *, __int64 *, __int64))(*(_QWORD *)v138 + 2528LL);
          v149.m128i_i64[0] = 0;
          v36 = *(_QWORD *)(a1 + 864);
          v37 = *(__m128i **)a1;
          v149.m128i_i32[2] = *(_DWORD *)(a1 + 848);
          if ( v37 )
          {
            if ( &v149 != &v37[3] )
            {
              v38 = v37[3].m128i_i64[0];
              v149.m128i_i64[0] = v38;
              if ( v38 )
              {
                v119 = v36;
                sub_B96E90((__int64)&v149, v38, 1);
                v36 = v119;
              }
            }
          }
          v39 = v35(v138, &v144, &v145, &v149, &v157, v36);
          v41 = v39;
          v42 = v40;
          if ( v149.m128i_i64[0] )
          {
            v120 = v39;
            v125 = v40;
            sub_B91220((__int64)&v149, v149.m128i_i64[0]);
            v41 = v120;
            v42 = v125;
          }
          v43 = (unsigned int)v155;
          ++v134;
          v44 = (unsigned int)v155 + 1LL;
          if ( v44 > HIDWORD(v155) )
          {
            v124 = v41;
            v128 = v42;
            sub_C8D5F0((__int64)&v154, v156, v44, 0x10u, v41, v42);
            v43 = (unsigned int)v155;
            v41 = v124;
            v42 = v128;
          }
          v45 = (__int64 *)&v154[16 * v43];
          *v45 = v41;
          v45[1] = v42;
          v46 = v133;
          LOWORD(v46) = v170;
          LODWORD(v155) = v155 + 1;
          v133 = v46;
          v47 = (unsigned int)v152;
          v48 = (unsigned int)v152 + 1LL;
          if ( v48 > HIDWORD(v152) )
          {
            sub_C8D5F0((__int64)&v151, v153, v48, 0x10u, v41, v42);
            v47 = (unsigned int)v152;
          }
          v49 = &v151[16 * v47];
          v49[1] = 0;
          *v49 = v133;
          LODWORD(v152) = v152 + 1;
        }
      }
      else
      {
        v23 = &v149;
        v149.m128i_i64[0] = 0;
        v24 = *(__m128i **)a1;
        v149.m128i_i32[2] = *(_DWORD *)(a1 + 848);
        if ( v24 )
        {
          if ( &v149 != &v24[3] )
          {
            v25 = v24[3].m128i_i64[0];
            v149.m128i_i64[0] = v25;
            if ( v25 )
              sub_B96E90((__int64)&v149, v25, 1);
          }
        }
        sub_336F780(*(_QWORD **)(a1 + 864), (int)&v149, (__int64)&v157, (__int64)&v157);
        if ( v149.m128i_i64[0] )
          sub_B91220((__int64)&v149, v149.m128i_i64[0]);
        v26 = v179;
        v27 = &v179[(unsigned int)v180];
        if ( v27 != v179 )
        {
          v28 = &v149;
          v29 = v134;
          while ( 1 )
          {
            v34 = v29++;
            if ( (int)v34 >= 0 )
              v30 = *(_QWORD *)(*(_QWORD *)(v131 + 304) + 8 * v34);
            else
              v30 = *(_QWORD *)(*(_QWORD *)(v131 + 56) + 16 * (v34 & 0x7FFFFFFF) + 8);
            if ( v30 )
            {
              if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
              {
                v30 = *(_QWORD *)(v30 + 32);
                if ( v30 )
                {
                  if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
                    BUG();
                }
              }
            }
            v31 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 16) + 32LL) + 48LL);
            v32 = v31 < 0
                ? *(_QWORD *)(*(_QWORD *)(v131 + 56) + 16LL * (v31 & 0x7FFFFFFF) + 8)
                : *(_QWORD *)(*(_QWORD *)(v131 + 304) + 8LL * (unsigned int)v31);
            if ( v32 )
            {
              if ( (*(_BYTE *)(v32 + 3) & 0x10) == 0 )
              {
                v32 = *(_QWORD *)(v32 + 32);
                if ( v32 )
                  break;
              }
            }
            v33 = *(_QWORD *)(v32 + 16);
            if ( *(_WORD *)(v33 + 68) == 20 )
              goto LABEL_135;
LABEL_32:
            if ( (unsigned int)(v31 - 1) <= 0x3FFFFFFE )
            {
              v80 = *(unsigned __int64 **)(*(_QWORD *)(a1 + 960) + 744LL);
              v149.m128i_i64[1] = -1;
              v149.m128i_i32[0] = (unsigned __int16)v31;
              v150 = -1;
              v81 = (__m128i *)v80[24];
              if ( v81 == (__m128i *)v80[25] )
              {
                v129 = v27;
                v122 = v31;
                v135 = v28;
                sub_2E341F0(v80 + 23, v81, v28);
                v27 = v129;
                v31 = v122;
                v28 = v135;
                goto LABEL_33;
              }
              if ( v81 )
              {
                *v81 = _mm_load_si128(&v149);
                v81[1].m128i_i64[0] = v150;
                v81 = (__m128i *)v80[24];
              }
              ++v26;
              v80[24] = (unsigned __int64)&v81[1].m128i_u64[1];
              *(v26 - 1) = v31;
              if ( v27 == v26 )
              {
LABEL_120:
                v134 = v29;
                v23 = (__m128i *)v28;
                goto LABEL_121;
              }
            }
            else
            {
LABEL_33:
              *v26++ = v31;
              if ( v27 == v26 )
                goto LABEL_120;
            }
          }
          if ( (*(_BYTE *)(v32 + 3) & 0x10) == 0 )
            BUG();
          v33 = *(_QWORD *)(v32 + 16);
          if ( *(_WORD *)(v33 + 68) != 20 )
            goto LABEL_32;
LABEL_135:
          v31 = *(_DWORD *)(*(_QWORD *)(v33 + 32) + 48LL);
          goto LABEL_32;
        }
LABEL_121:
        v149.m128i_i64[0] = 0;
        v82 = *(__m128i **)a1;
        v149.m128i_i32[2] = *(_DWORD *)(a1 + 848);
        if ( v82 )
        {
          if ( v23 != &v82[3] )
          {
            v83 = v82[3].m128i_i64[0];
            v149.m128i_i64[0] = v83;
            if ( v83 )
              sub_B96E90((__int64)v23, v83, 1);
          }
        }
        v84 = sub_3370E50(
                (__int64)v173,
                *(_QWORD *)(a1 + 864),
                *(_QWORD *)(a1 + 960),
                (__int64)v23,
                (__int64)&v144,
                0,
                v136);
        v86 = v84;
        v87 = v85;
        if ( v149.m128i_i64[0] )
        {
          v121 = v84;
          v126 = v85;
          sub_B91220((__int64)v23, v149.m128i_i64[0]);
          v86 = v121;
          v87 = v126;
        }
        v88 = (unsigned int)v155;
        v89 = (unsigned int)v155 + 1LL;
        if ( v89 > HIDWORD(v155) )
        {
          v123 = v86;
          v127 = v87;
          sub_C8D5F0((__int64)&v154, v156, v89, 0x10u, v86, v87);
          v88 = (unsigned int)v155;
          v86 = v123;
          v87 = v127;
        }
        v90 = (__int64 *)&v154[16 * v88];
        *v90 = v86;
        v90[1] = v87;
        v91 = v132;
        LOWORD(v91) = v170;
        LODWORD(v155) = v155 + 1;
        v132 = v91;
        v92 = (unsigned int)v152;
        v93 = (unsigned int)v152 + 1LL;
        if ( v93 > HIDWORD(v152) )
        {
          sub_C8D5F0((__int64)&v151, v153, v93, 0x10u, v86, v87);
          v92 = (unsigned int)v152;
        }
        v94 = &v151[16 * v92];
        v94[1] = 0;
        *v94 = v132;
        LODWORD(v152) = v152 + 1;
      }
      if ( v182 != v184 )
        _libc_free((unsigned __int64)v182);
      if ( v179 != (int *)v181 )
        _libc_free((unsigned __int64)v179);
      if ( v175 != &v178 )
        _libc_free((unsigned __int64)v175);
      if ( (_BYTE *)v173[0] != v174 )
        _libc_free(v173[0]);
      if ( v166 != v167 )
        j_j___libc_free_0((unsigned __int64)v166);
      v50 = v163;
      v51 = (unsigned __int64)&v163[56 * (unsigned int)v164];
      if ( v163 != (_BYTE *)v51 )
      {
        do
        {
          v52 = *(unsigned int *)(v51 - 40);
          v53 = *(_QWORD *)(v51 - 48);
          v51 -= 56LL;
          v54 = (unsigned __int64 *)(v53 + 32 * v52);
          if ( (unsigned __int64 *)v53 != v54 )
          {
            do
            {
              v54 -= 4;
              if ( (unsigned __int64 *)*v54 != v54 + 2 )
                j_j___libc_free_0(*v54);
            }
            while ( (unsigned __int64 *)v53 != v54 );
            v53 = *(_QWORD *)(v51 + 8);
          }
          if ( v53 != v51 + 24 )
            _libc_free(v53);
        }
        while ( v50 != (_BYTE *)v51 );
        v51 = (unsigned __int64)v163;
      }
      if ( (_BYTE *)v51 != v165 )
        _libc_free(v51);
      v55 = v160;
      v56 = &v160[4 * (unsigned int)v161];
      if ( v160 != v56 )
      {
        do
        {
          v56 -= 4;
          if ( (unsigned __int64 *)*v56 != v56 + 2 )
            j_j___libc_free_0(*v56);
        }
        while ( v55 != v56 );
LABEL_75:
        v56 = v160;
      }
LABEL_76:
      if ( v56 != (unsigned __int64 *)v162 )
        _libc_free((unsigned __int64)v56);
      v21 += 248LL;
      if ( v137 == v21 )
        goto LABEL_79;
    }
    if ( v166 != v167 )
      j_j___libc_free_0((unsigned __int64)v166);
    v95 = v163;
    v96 = (unsigned __int64)&v163[56 * (unsigned int)v164];
    if ( v163 != (_BYTE *)v96 )
    {
      do
      {
        v97 = *(unsigned int *)(v96 - 40);
        v98 = *(_QWORD *)(v96 - 48);
        v96 -= 56LL;
        v99 = (unsigned __int64 *)(v98 + 32 * v97);
        if ( (unsigned __int64 *)v98 != v99 )
        {
          do
          {
            v99 -= 4;
            if ( (unsigned __int64 *)*v99 != v99 + 2 )
              j_j___libc_free_0(*v99);
          }
          while ( (unsigned __int64 *)v98 != v99 );
          v98 = *(_QWORD *)(v96 + 8);
        }
        if ( v98 != v96 + 24 )
          _libc_free(v98);
      }
      while ( v95 != (_BYTE *)v96 );
      v96 = (unsigned __int64)v163;
    }
    if ( (_BYTE *)v96 != v165 )
      _libc_free(v96);
    v100 = v160;
    v56 = &v160[4 * (unsigned int)v161];
    if ( v160 == v56 )
      goto LABEL_76;
    do
    {
      v56 -= 4;
      if ( (unsigned __int64 *)*v56 != v56 + 2 )
        j_j___libc_free_0(*v56);
    }
    while ( v100 != v56 );
    goto LABEL_75;
  }
LABEL_79:
  v57 = (unsigned __int64)v154;
  v58 = *(_QWORD *)(a1 + 864);
  v59 = (unsigned int)v155;
  v60 = sub_33E5830(v58, v151);
  v157 = 0;
  v62 = v60;
  v64 = v63;
  v65 = *(__m128i **)a1;
  v158 = *(_DWORD *)(a1 + 848);
  if ( v65 )
  {
    if ( &v157 != (__int64 *)&v65[3] )
    {
      v66 = v65[3].m128i_i64[0];
      v157 = v66;
      if ( v66 )
      {
        v141 = v62;
        sub_B96E90((__int64)&v157, v66, 1);
        v62 = v141;
      }
    }
  }
  *((_QWORD *)&v118 + 1) = v59;
  *(_QWORD *)&v118 = v57;
  v67 = sub_3411630(v58, 55, (unsigned int)&v157, v62, v64, v61, v118);
  v69 = v68;
  if ( v157 )
    sub_B91220((__int64)&v157, v157);
  v157 = a2;
  v70 = sub_337DC20(a1 + 8, &v157);
  *v70 = v67;
  v71 = v147;
  *((_DWORD *)v70 + 2) = v69;
  v143 = v148;
  if ( v148 != v71 )
  {
    do
    {
      v72 = *(_QWORD *)(v71 + 192);
      if ( v72 != v71 + 208 )
        j_j___libc_free_0(v72);
      v73 = *(_QWORD *)(v71 + 64);
      v74 = v73 + 56LL * *(unsigned int *)(v71 + 72);
      if ( v73 != v74 )
      {
        do
        {
          v75 = *(unsigned int *)(v74 - 40);
          v76 = *(_QWORD *)(v74 - 48);
          v74 -= 56LL;
          v77 = (unsigned __int64 *)(v76 + 32 * v75);
          if ( (unsigned __int64 *)v76 != v77 )
          {
            do
            {
              v77 -= 4;
              if ( (unsigned __int64 *)*v77 != v77 + 2 )
                j_j___libc_free_0(*v77);
            }
            while ( (unsigned __int64 *)v76 != v77 );
            v76 = *(_QWORD *)(v74 + 8);
          }
          if ( v76 != v74 + 24 )
            _libc_free(v76);
        }
        while ( v73 != v74 );
        v74 = *(_QWORD *)(v71 + 64);
      }
      if ( v74 != v71 + 80 )
        _libc_free(v74);
      v78 = *(unsigned __int64 **)(v71 + 16);
      v79 = &v78[4 * *(unsigned int *)(v71 + 24)];
      if ( v78 != v79 )
      {
        do
        {
          v79 -= 4;
          if ( (unsigned __int64 *)*v79 != v79 + 2 )
            j_j___libc_free_0(*v79);
        }
        while ( v78 != v79 );
        v79 = *(unsigned __int64 **)(v71 + 16);
      }
      if ( v79 != (unsigned __int64 *)(v71 + 32) )
        _libc_free((unsigned __int64)v79);
      v71 += 248LL;
    }
    while ( v143 != v71 );
    v71 = v147;
  }
  if ( v71 )
    j_j___libc_free_0(v71);
  if ( v154 != v156 )
    _libc_free((unsigned __int64)v154);
  if ( v151 != v153 )
    _libc_free((unsigned __int64)v151);
}
