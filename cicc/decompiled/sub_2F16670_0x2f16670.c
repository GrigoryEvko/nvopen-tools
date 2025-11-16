// Function: sub_2F16670
// Address: 0x2f16670
//
void __fastcall sub_2F16670(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r13d
  int v9; // ebx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // esi
  unsigned int v15; // edi
  _DWORD *v16; // rax
  int v17; // r8d
  __int64 v18; // rax
  __int64 v19; // rdi
  const char *v20; // rdx
  char *v21; // rsi
  _BYTE *v22; // rdi
  __int64 v23; // rsi
  int v24; // edx
  __int8 v25; // al
  __int64 v26; // r15
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __int64 v30; // rdx
  size_t v31; // rdx
  __int64 v32; // r15
  size_t v33; // rdx
  __int64 v34; // r8
  __int64 *v35; // rdi
  __int64 v36; // r10
  __int64 v37; // rsi
  int v38; // r13d
  __m128i *v39; // r13
  int v40; // ecx
  unsigned __int64 v41; // rsi
  int v42; // edx
  __int64 v43; // rax
  int *v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r8
  _QWORD *v49; // r14
  _QWORD *v50; // r13
  _QWORD *v51; // r15
  __int64 v52; // rax
  unsigned __int64 v53; // rsi
  size_t v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdi
  const char *v60; // rdi
  _QWORD *v61; // rax
  __int64 v62; // r13
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rdi
  const char *v66; // rdi
  unsigned __int64 v67; // r8
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 i; // rbx
  unsigned int v71; // edi
  unsigned int *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __int8 v75; // al
  __int64 v76; // rsi
  __m128i v77; // xmm1
  __m128i v78; // xmm2
  __m128i v79; // xmm3
  __m128i v80; // xmm4
  size_t v81; // rax
  unsigned int v82; // esi
  int v83; // eax
  __int64 v84; // rsi
  int v85; // edi
  unsigned int v86; // r10d
  unsigned int *v87; // rdx
  unsigned int v88; // eax
  int v89; // r11d
  int v90; // r11d
  int v91; // ebx
  _DWORD *v92; // rdx
  int v93; // eax
  int v94; // ebx
  int v95; // ebx
  __int64 v96; // r11
  unsigned int v97; // esi
  int v98; // r8d
  int v99; // r10d
  _DWORD *v100; // rdi
  int v101; // ebx
  int v102; // ebx
  __int64 v103; // r11
  unsigned int v104; // esi
  int v105; // r8d
  int v106; // r10d
  int v107; // edx
  __int64 v108; // rsi
  int v109; // r11d
  unsigned int v110; // r10d
  size_t v111; // rdx
  __int64 v112; // rdi
  __int64 v114; // [rsp+28h] [rbp-368h]
  __int64 v115; // [rsp+38h] [rbp-358h]
  __int64 v116; // [rsp+40h] [rbp-350h]
  int v118; // [rsp+50h] [rbp-340h]
  __int64 v119; // [rsp+50h] [rbp-340h]
  int v120; // [rsp+50h] [rbp-340h]
  int v121; // [rsp+50h] [rbp-340h]
  unsigned int v122; // [rsp+58h] [rbp-338h]
  int v123; // [rsp+58h] [rbp-338h]
  _QWORD *v125; // [rsp+78h] [rbp-318h]
  __int64 v126; // [rsp+78h] [rbp-318h]
  __int64 v127; // [rsp+80h] [rbp-310h]
  _QWORD *v128; // [rsp+80h] [rbp-310h]
  __int64 v129; // [rsp+88h] [rbp-308h]
  __int64 v130; // [rsp+88h] [rbp-308h]
  __m128i *v132; // [rsp+90h] [rbp-300h] BYREF
  __int64 v133; // [rsp+98h] [rbp-2F8h]
  __m128i v134; // [rsp+A0h] [rbp-2F0h] BYREF
  int v135; // [rsp+B0h] [rbp-2E0h]
  char v136; // [rsp+B4h] [rbp-2DCh]
  unsigned __int64 v137; // [rsp+C0h] [rbp-2D0h] BYREF
  size_t n; // [rsp+C8h] [rbp-2C8h]
  _BYTE src[24]; // [rsp+D0h] [rbp-2C0h] BYREF
  __int64 v140; // [rsp+E8h] [rbp-2A8h]
  _QWORD *v141; // [rsp+F0h] [rbp-2A0h]
  _BYTE *v142; // [rsp+100h] [rbp-290h] BYREF
  __int64 v143; // [rsp+108h] [rbp-288h]
  _BYTE v144[128]; // [rsp+110h] [rbp-280h] BYREF
  size_t *v145; // [rsp+190h] [rbp-200h] BYREF
  unsigned __int64 v146; // [rsp+198h] [rbp-1F8h]
  size_t v147; // [rsp+1A0h] [rbp-1F0h] BYREF
  __m128i si128; // [rsp+1A8h] [rbp-1E8h] BYREF
  unsigned int v149; // [rsp+1B8h] [rbp-1D8h]
  char v150; // [rsp+1BCh] [rbp-1D4h]
  size_t v151[2]; // [rsp+220h] [rbp-170h] BYREF
  __int64 v152; // [rsp+230h] [rbp-160h] BYREF
  void *dest; // [rsp+238h] [rbp-158h]
  __m128i v154; // [rsp+240h] [rbp-150h] BYREF
  __int64 v155; // [rsp+250h] [rbp-140h] BYREF
  __m128i v156; // [rsp+258h] [rbp-138h] BYREF
  __int64 v157; // [rsp+268h] [rbp-128h]
  __m128i v158; // [rsp+270h] [rbp-120h] BYREF
  __m128i v159; // [rsp+280h] [rbp-110h] BYREF
  __int64 v160; // [rsp+290h] [rbp-100h]
  __m128i *v161; // [rsp+298h] [rbp-F8h] BYREF
  __int64 v162; // [rsp+2A0h] [rbp-F0h]
  __m128i v163; // [rsp+2A8h] [rbp-E8h] BYREF
  __m128i v164; // [rsp+2B8h] [rbp-D8h] BYREF
  __m128i *v165; // [rsp+2C8h] [rbp-C8h]
  __int8 *v166; // [rsp+2D0h] [rbp-C0h]
  __m128i v167; // [rsp+2D8h] [rbp-B8h] BYREF
  __m128i v168; // [rsp+2E8h] [rbp-A8h] BYREF
  __m128i *v169; // [rsp+2F8h] [rbp-98h]
  __int8 *v170; // [rsp+300h] [rbp-90h]
  __m128i v171; // [rsp+308h] [rbp-88h] BYREF
  __m128i v172; // [rsp+318h] [rbp-78h] BYREF
  __int64 v173; // [rsp+328h] [rbp-68h]
  _BYTE *v174; // [rsp+330h] [rbp-60h]
  __int64 v175; // [rsp+338h] [rbp-58h]
  _BYTE v176[16]; // [rsp+340h] [rbp-50h] BYREF
  __m128i v177[4]; // [rsp+350h] [rbp-40h] BYREF

  v4 = a1;
  v5 = *(_QWORD *)(a3 + 48);
  v115 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 16) + 200LL))(*(_QWORD *)(a3 + 16));
  v142 = v144;
  v143 = 0x2000000000LL;
  v8 = -*(_DWORD *)(v5 + 32);
  if ( *(int *)(v5 + 32) > 0 )
  {
    v67 = *(int *)(v5 + 32);
    v68 = 32;
    v69 = 0;
    if ( v67 > 0x20 )
    {
      sub_C8D5F0((__int64)&v142, v144, v67, 4u, v67, v7);
      v69 = (unsigned int)v143;
      v68 = HIDWORD(v143);
    }
    for ( i = 0; ; ++i )
    {
      v7 = v69 + 1;
      v6 = (unsigned int)i;
      if ( v69 + 1 > v68 )
      {
        sub_C8D5F0((__int64)&v142, v144, v69 + 1, 4u, (unsigned int)i, v7);
        v69 = (unsigned int)v143;
        v6 = (unsigned int)i;
      }
      *(_DWORD *)&v142[4 * v69] = -1;
      v73 = *(_DWORD *)(v5 + 32) + v8;
      LODWORD(v143) = v143 + 1;
      if ( *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40 * v73 + 8) != -1 )
        break;
LABEL_129:
      if ( !++v8 )
        goto LABEL_2;
      v69 = (unsigned int)v143;
      v68 = HIDWORD(v143);
    }
    v151[1] = 0;
    v156.m128i_i64[1] = (__int64)&v158;
    v161 = &v163;
    v165 = &v167;
    v169 = &v171;
    v152 = 0;
    LODWORD(dest) = 0;
    v154 = 0u;
    BYTE1(v155) = 0;
    v156.m128i_i16[0] = 0;
    v157 = 0;
    v158.m128i_i8[0] = 0;
    v159 = 0u;
    LOBYTE(v160) = 1;
    v162 = 0;
    v163.m128i_i8[0] = 0;
    v164 = 0u;
    v166 = 0;
    v167.m128i_i8[0] = 0;
    v168 = 0u;
    v170 = 0;
    v171.m128i_i8[0] = 0;
    v172 = 0u;
    LODWORD(v151[0]) = v6;
    v74 = 5LL * (*(_DWORD *)(v5 + 32) + v8);
    LODWORD(dest) = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 18);
    v154.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 8 * v74);
    v154.m128i_i64[1] = *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 8);
    LOBYTE(v74) = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 16);
    BYTE1(v155) = 1;
    LOBYTE(v155) = v74;
    HIDWORD(v155) = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 20);
    v75 = 0;
    if ( !*(_BYTE *)(v5 + 670) )
      v75 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 17);
    v156.m128i_i8[0] = v75;
    v156.m128i_i8[1] = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (*(_DWORD *)(v5 + 32) + v8) + 33);
    *(_DWORD *)&v142[4 * i] = 1041204193 * ((__int64)(a2[48] - a2[47]) >> 3);
    v76 = a2[48];
    if ( v76 == a2[49] )
    {
      v123 = v6;
      sub_2F15D50(a2 + 47, v76, (__int64)v151);
      LODWORD(v6) = v123;
    }
    else
    {
      if ( v76 )
      {
        *(__m128i *)v76 = _mm_load_si128((const __m128i *)v151);
        *(_QWORD *)(v76 + 16) = v152;
        *(_DWORD *)(v76 + 24) = (_DWORD)dest;
        *(__m128i *)(v76 + 32) = v154;
        *(_WORD *)(v76 + 48) = v155;
        *(_DWORD *)(v76 + 52) = HIDWORD(v155);
        *(_WORD *)(v76 + 56) = v156.m128i_i16[0];
        *(_QWORD *)(v76 + 64) = v76 + 80;
        if ( (__m128i *)v156.m128i_i64[1] == &v158 )
        {
          *(__m128i *)(v76 + 80) = _mm_load_si128(&v158);
        }
        else
        {
          *(_QWORD *)(v76 + 64) = v156.m128i_i64[1];
          *(_QWORD *)(v76 + 80) = v158.m128i_i64[0];
        }
        *(_QWORD *)(v76 + 72) = v157;
        v77 = _mm_load_si128(&v159);
        v158.m128i_i8[0] = 0;
        v156.m128i_i64[1] = (__int64)&v158;
        v157 = 0;
        *(__m128i *)(v76 + 96) = v77;
        *(_BYTE *)(v76 + 112) = v160;
        *(_QWORD *)(v76 + 120) = v76 + 136;
        if ( v161 == &v163 )
        {
          *(__m128i *)(v76 + 136) = _mm_loadu_si128(&v163);
        }
        else
        {
          *(_QWORD *)(v76 + 120) = v161;
          *(_QWORD *)(v76 + 136) = v163.m128i_i64[0];
        }
        *(_QWORD *)(v76 + 128) = v162;
        v78 = _mm_loadu_si128(&v164);
        v163.m128i_i8[0] = 0;
        v161 = &v163;
        v162 = 0;
        *(_QWORD *)(v76 + 168) = v76 + 184;
        *(__m128i *)(v76 + 152) = v78;
        if ( v165 == &v167 )
        {
          *(__m128i *)(v76 + 184) = _mm_loadu_si128(&v167);
        }
        else
        {
          *(_QWORD *)(v76 + 168) = v165;
          *(_QWORD *)(v76 + 184) = v167.m128i_i64[0];
        }
        *(_QWORD *)(v76 + 176) = v166;
        v79 = _mm_loadu_si128(&v168);
        v167.m128i_i8[0] = 0;
        v165 = &v167;
        v166 = 0;
        *(_QWORD *)(v76 + 216) = v76 + 232;
        *(__m128i *)(v76 + 200) = v79;
        if ( v169 == &v171 )
        {
          *(__m128i *)(v76 + 232) = _mm_loadu_si128(&v171);
        }
        else
        {
          *(_QWORD *)(v76 + 216) = v169;
          *(_QWORD *)(v76 + 232) = v171.m128i_i64[0];
        }
        *(_QWORD *)(v76 + 224) = v170;
        v80 = _mm_loadu_si128(&v172);
        v171.m128i_i8[0] = 0;
        v169 = &v171;
        v170 = 0;
        *(__m128i *)(v76 + 248) = v80;
        v76 = a2[48];
      }
      a2[48] = v76 + 264;
    }
    v122 = v6;
    v137 = (unsigned __int64)src;
    v119 = v4 + 48;
    sub_2F07580((__int64 *)&v137, byte_3F871B3, (__int64)byte_3F871B3);
    src[20] = 1;
    *(_DWORD *)&src[16] = v122;
    LODWORD(v145) = v8;
    v146 = (unsigned __int64)&si128;
    if ( (_BYTE *)v137 == src )
    {
      si128 = _mm_load_si128((const __m128i *)src);
    }
    else
    {
      v146 = v137;
      si128.m128i_i64[0] = *(_QWORD *)src;
    }
    v81 = n;
    v137 = (unsigned __int64)src;
    n = 0;
    v147 = v81;
    src[0] = 0;
    v82 = *(_DWORD *)(v4 + 72);
    v149 = v122;
    v150 = 1;
    if ( v82 )
    {
      v7 = *(_QWORD *)(v4 + 56);
      v71 = (v82 - 1) & (37 * v8);
      v72 = (unsigned int *)(v7 + 48LL * v71);
      v6 = *v72;
      if ( v8 == (_DWORD)v6 )
      {
LABEL_117:
        if ( (__m128i *)v146 != &si128 )
          j_j___libc_free_0(v146);
        goto LABEL_119;
      }
      v90 = 1;
      v87 = 0;
      while ( (_DWORD)v6 != 0x7FFFFFFF )
      {
        if ( v87 || (_DWORD)v6 != 0x80000000 )
          v72 = v87;
        v71 = (v82 - 1) & (v90 + v71);
        v6 = *(unsigned int *)(v7 + 48LL * v71);
        if ( v8 == (_DWORD)v6 )
          goto LABEL_117;
        ++v90;
        v87 = v72;
        v72 = (unsigned int *)(v7 + 48LL * v71);
      }
      if ( !v87 )
        v87 = v72;
      ++*(_QWORD *)(v4 + 48);
      v85 = *(_DWORD *)(v4 + 64) + 1;
      if ( 4 * v85 < 3 * v82 )
      {
        v6 = v8;
        v7 = v82 >> 3;
        if ( v82 - *(_DWORD *)(v4 + 68) - v85 <= (unsigned int)v7 )
        {
          sub_2F08370(v119, v82);
          v107 = *(_DWORD *)(v4 + 72);
          if ( !v107 )
          {
LABEL_241:
            ++*(_DWORD *)(v4 + 64);
            BUG();
          }
          v108 = *(_QWORD *)(v4 + 56);
          v7 = 0;
          v121 = v107 - 1;
          v109 = 1;
          v110 = (v107 - 1) & (37 * (_DWORD)v145);
          v85 = *(_DWORD *)(v4 + 64) + 1;
          v87 = (unsigned int *)(v108 + 48LL * v110);
          v6 = *v87;
          if ( (_DWORD)v145 != (_DWORD)v6 )
          {
            while ( (_DWORD)v6 != 0x7FFFFFFF )
            {
              if ( !v7 && (_DWORD)v6 == 0x80000000 )
                v7 = (__int64)v87;
              v110 = v121 & (v109 + v110);
              v87 = (unsigned int *)(v108 + 48LL * v110);
              v6 = *v87;
              if ( (_DWORD)v145 == (_DWORD)v6 )
                goto LABEL_172;
              ++v109;
            }
            v6 = (unsigned int)v145;
            if ( v7 )
              v87 = (unsigned int *)v7;
          }
        }
        goto LABEL_172;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 48);
    }
    sub_2F08370(v119, 2 * v82);
    v83 = *(_DWORD *)(v4 + 72);
    if ( !v83 )
      goto LABEL_241;
    v6 = (unsigned int)v145;
    v84 = *(_QWORD *)(v4 + 56);
    v120 = v83 - 1;
    v85 = *(_DWORD *)(v4 + 64) + 1;
    v86 = (v83 - 1) & (37 * (_DWORD)v145);
    v87 = (unsigned int *)(v84 + 48LL * v86);
    v88 = *v87;
    if ( *v87 != (_DWORD)v145 )
    {
      v89 = 1;
      v7 = 0;
      while ( v88 != 0x7FFFFFFF )
      {
        if ( v88 == 0x80000000 && !v7 )
          v7 = (__int64)v87;
        v86 = v120 & (v89 + v86);
        v87 = (unsigned int *)(v84 + 48LL * v86);
        v88 = *v87;
        if ( (_DWORD)v145 == *v87 )
          goto LABEL_172;
        ++v89;
      }
      if ( v7 )
        v87 = (unsigned int *)v7;
    }
LABEL_172:
    *(_DWORD *)(v4 + 64) = v85;
    if ( *v87 != 0x7FFFFFFF )
      --*(_DWORD *)(a1 + 68);
    *v87 = v6;
    *((_QWORD *)v87 + 1) = v87 + 6;
    if ( (__m128i *)v146 == &si128 )
    {
      *(__m128i *)(v87 + 6) = _mm_loadu_si128(&si128);
    }
    else
    {
      *((_QWORD *)v87 + 1) = v146;
      *((_QWORD *)v87 + 3) = si128.m128i_i64[0];
    }
    *((_QWORD *)v87 + 2) = v147;
    v87[10] = v149;
    *((_BYTE *)v87 + 44) = v150;
LABEL_119:
    if ( (_BYTE *)v137 != src )
      j_j___libc_free_0(v137);
    if ( v169 != &v171 )
      j_j___libc_free_0((unsigned __int64)v169);
    if ( v165 != &v167 )
      j_j___libc_free_0((unsigned __int64)v165);
    if ( v161 != &v163 )
      j_j___libc_free_0((unsigned __int64)v161);
    if ( (__m128i *)v156.m128i_i64[1] != &v158 )
      j_j___libc_free_0(v156.m128i_u64[1]);
    goto LABEL_129;
  }
LABEL_2:
  v145 = &v147;
  v146 = 0x2000000000LL;
  v9 = -858993459 * ((__int64)(*(_QWORD *)(v5 + 16) - *(_QWORD *)(v5 + 8)) >> 3) - *(_DWORD *)(v5 + 32);
  if ( v9 <= 0 )
    goto LABEL_54;
  v10 = 32;
  v11 = 0;
  if ( v9 > 32 )
  {
    sub_C8D5F0((__int64)&v145, &v147, v9, 4u, v6, v7);
    v11 = (unsigned int)v146;
    v10 = HIDWORD(v146);
  }
  v129 = v4;
  v12 = 0;
  v116 = (unsigned int)v9;
  while ( 1 )
  {
    v118 = v12;
    if ( v11 + 1 > v10 )
    {
      sub_C8D5F0((__int64)&v145, &v147, v11 + 1, 4u, v11 + 1, v7);
      v11 = (unsigned int)v146;
    }
    *((_DWORD *)v145 + v11) = -1;
    LODWORD(v146) = v146 + 1;
    v7 = *(_QWORD *)(v5 + 8);
    if ( *(_QWORD *)(v7 + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12) + 8) != -1 )
    {
      v171.m128i_i8[8] = 0;
      v151[1] = 0;
      dest = &v154.m128i_u64[1];
      v159.m128i_i64[1] = (__int64)&v161;
      v166 = &v167.m128i_i8[8];
      v170 = &v171.m128i_i8[8];
      v152 = 0;
      v154.m128i_i64[0] = 0;
      v154.m128i_i8[8] = 0;
      v156 = 0u;
      LODWORD(v157) = 0;
      v158 = 0u;
      v159.m128i_i8[1] = 0;
      v160 = 0;
      LOBYTE(v161) = 0;
      v163 = 0u;
      v164.m128i_i8[0] = 1;
      LOBYTE(v165) = 0;
      v167.m128i_i64[0] = 0;
      v167.m128i_i8[8] = 0;
      v168.m128i_i64[1] = 0;
      v169 = 0;
      v171.m128i_i64[0] = 0;
      v172.m128i_i64[1] = 0;
      v173 = 0;
      v174 = v176;
      v175 = 0;
      v176[0] = 0;
      v177[0] = 0u;
      LODWORD(v151[0]) = v12;
      v18 = *(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12);
      v19 = *(_QWORD *)(v18 + 24);
      if ( !v19 )
        goto LABEL_39;
      v20 = byte_3F871B3;
      v21 = (char *)byte_3F871B3;
      if ( (*(_BYTE *)(v19 + 7) & 0x10) != 0 )
      {
        v21 = (char *)sub_BD5D20(v19);
        v20 = &v21[v30];
      }
      v137 = (unsigned __int64)src;
      sub_2F07580((__int64 *)&v137, v21, (__int64)v20);
      v22 = dest;
      if ( (_BYTE *)v137 == src )
      {
        v31 = n;
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)dest = src[0];
          else
            memcpy(dest, src, n);
          v31 = n;
          v22 = dest;
        }
        v154.m128i_i64[0] = v31;
        v22[v31] = 0;
        v22 = (_BYTE *)v137;
        goto LABEL_36;
      }
      if ( dest == &v154.m128i_u64[1] )
      {
        dest = (void *)v137;
        v154 = (__m128i)__PAIR128__(*(unsigned __int64 *)src, n);
      }
      else
      {
        v23 = v154.m128i_i64[1];
        dest = (void *)v137;
        v154 = (__m128i)__PAIR128__(*(unsigned __int64 *)src, n);
        if ( v22 )
        {
          v137 = (unsigned __int64)v22;
          *(_QWORD *)src = v23;
LABEL_36:
          n = 0;
          *v22 = 0;
          if ( (_BYTE *)v137 != src )
            j_j___libc_free_0(v137);
          v18 = *(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12);
LABEL_39:
          v24 = 1;
          if ( !*(_BYTE *)(v18 + 18) )
            v24 = 2 * (*(_QWORD *)(v18 + 8) == 0);
          LODWORD(v157) = v24;
          v158 = *(__m128i *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12));
          v25 = *(_BYTE *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12) + 16);
          v159.m128i_i8[1] = 1;
          v159.m128i_i8[0] = v25;
          v159.m128i_i32[1] = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8)
                                                 + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v12)
                                                 + 20);
          *((_DWORD *)v145 + v12) = -858993459 * ((__int64)(a2[54] - a2[53]) >> 6);
          v26 = a2[54];
          if ( v26 == a2[55] )
          {
            sub_2F13700(a2 + 53, a2[54], (__int64)v151);
          }
          else
          {
            if ( v26 )
            {
              *(__m128i *)v26 = _mm_load_si128((const __m128i *)v151);
              *(_QWORD *)(v26 + 16) = v152;
              *(_QWORD *)(v26 + 24) = v26 + 40;
              sub_2F07250((__int64 *)(v26 + 24), dest, (__int64)dest + v154.m128i_i64[0]);
              *(__m128i *)(v26 + 56) = _mm_loadu_si128(&v156);
              *(_DWORD *)(v26 + 72) = v157;
              *(__m128i *)(v26 + 80) = v158;
              *(_WORD *)(v26 + 96) = v159.m128i_i16[0];
              *(_DWORD *)(v26 + 100) = v159.m128i_i32[1];
              *(_QWORD *)(v26 + 104) = v26 + 120;
              sub_2F07250((__int64 *)(v26 + 104), (_BYTE *)v159.m128i_i64[1], v159.m128i_i64[1] + v160);
              *(__m128i *)(v26 + 136) = _mm_loadu_si128(&v163);
              *(_BYTE *)(v26 + 152) = v164.m128i_i8[0];
              v27 = _mm_load_si128((const __m128i *)&v164.m128i_u64[1]);
              *(_QWORD *)(v26 + 176) = v26 + 192;
              *(__m128i *)(v26 + 160) = v27;
              sub_2F07250((__int64 *)(v26 + 176), v166, (__int64)&v166[v167.m128i_i64[0]]);
              v28 = _mm_load_si128((const __m128i *)&v168.m128i_u64[1]);
              *(_QWORD *)(v26 + 224) = v26 + 240;
              *(__m128i *)(v26 + 208) = v28;
              sub_2F07250((__int64 *)(v26 + 224), v170, (__int64)&v170[v171.m128i_i64[0]]);
              v29 = _mm_load_si128((const __m128i *)&v172.m128i_u64[1]);
              *(_QWORD *)(v26 + 272) = v26 + 288;
              *(__m128i *)(v26 + 256) = v29;
              sub_2F07250((__int64 *)(v26 + 272), v174, (__int64)&v174[v175]);
              *(__m128i *)(v26 + 304) = _mm_load_si128(v177);
              v26 = a2[54];
            }
            a2[54] = v26 + 320;
          }
          v114 = v129 + 48;
          if ( dest )
          {
            v132 = &v134;
            sub_2F07580((__int64 *)&v132, dest, (__int64)dest + v154.m128i_i64[0]);
            v135 = v12;
            v136 = 0;
            v13 = v133;
            LODWORD(v137) = v12;
            n = (size_t)&src[8];
            if ( v132 != &v134 )
            {
              n = (size_t)v132;
              *(_QWORD *)&src[8] = v134.m128i_i64[0];
              goto LABEL_8;
            }
          }
          else
          {
            v134.m128i_i8[0] = 0;
            v13 = 0;
            v135 = v12;
            v136 = 0;
            LODWORD(v137) = v12;
            n = (size_t)&src[8];
          }
          *(__m128i *)&src[8] = _mm_load_si128(&v134);
LABEL_8:
          *(_QWORD *)src = v13;
          v132 = &v134;
          v14 = *(_DWORD *)(v129 + 72);
          v134.m128i_i8[0] = 0;
          v133 = 0;
          LODWORD(v140) = v12;
          BYTE4(v140) = 0;
          if ( v14 )
          {
            v7 = *(_QWORD *)(v129 + 56);
            v15 = (v14 - 1) & (37 * v12);
            v16 = (_DWORD *)(v7 + 48LL * v15);
            v17 = *v16;
            if ( (_DWORD)v12 == *v16 )
            {
LABEL_10:
              if ( (_BYTE *)n != &src[8] )
                j_j___libc_free_0(n);
LABEL_12:
              if ( v132 != &v134 )
                j_j___libc_free_0((unsigned __int64)v132);
              if ( v174 != v176 )
                j_j___libc_free_0((unsigned __int64)v174);
              if ( v170 != (__int8 *)&v171.m128i_u64[1] )
                j_j___libc_free_0((unsigned __int64)v170);
              if ( v166 != (__int8 *)&v167.m128i_u64[1] )
                j_j___libc_free_0((unsigned __int64)v166);
              if ( (__m128i **)v159.m128i_i64[1] != &v161 )
                j_j___libc_free_0(v159.m128i_u64[1]);
              if ( dest != &v154.m128i_u64[1] )
                j_j___libc_free_0((unsigned __int64)dest);
              goto LABEL_24;
            }
            v91 = 1;
            v92 = 0;
            while ( v17 != 0x7FFFFFFF )
            {
              if ( v92 || v17 != 0x80000000 )
                v16 = v92;
              v15 = (v14 - 1) & (v91 + v15);
              v17 = *(_DWORD *)(v7 + 48LL * v15);
              if ( v17 == (_DWORD)v12 )
                goto LABEL_10;
              ++v91;
              v92 = v16;
              v16 = (_DWORD *)(v7 + 48LL * v15);
            }
            if ( !v92 )
              v92 = v16;
            ++*(_QWORD *)(v129 + 48);
            v93 = *(_DWORD *)(v129 + 64) + 1;
            if ( 4 * v93 < 3 * v14 )
            {
              if ( v14 - *(_DWORD *)(v129 + 68) - v93 > v14 >> 3 )
                goto LABEL_184;
              sub_2F08370(v114, v14);
              v101 = *(_DWORD *)(v129 + 72);
              if ( !v101 )
              {
LABEL_240:
                ++*(_DWORD *)(v129 + 64);
                BUG();
              }
              v102 = v101 - 1;
              v103 = *(_QWORD *)(v129 + 56);
              v118 = v137;
              v104 = v102 & (37 * v137);
              v92 = (_DWORD *)(v103 + 48LL * v104);
              v93 = *(_DWORD *)(v129 + 64) + 1;
              v105 = *v92;
              if ( *v92 == (_DWORD)v137 )
                goto LABEL_184;
              v106 = 1;
              v100 = 0;
              while ( v105 != 0x7FFFFFFF )
              {
                if ( !v100 && v105 == 0x80000000 )
                  v100 = v92;
                v7 = (unsigned int)(v106 + 1);
                v104 = v102 & (v106 + v104);
                v92 = (_DWORD *)(v103 + 48LL * v104);
                v105 = *v92;
                if ( (_DWORD)v137 == *v92 )
                  goto LABEL_184;
                ++v106;
              }
              goto LABEL_196;
            }
          }
          else
          {
            ++*(_QWORD *)(v129 + 48);
          }
          sub_2F08370(v114, 2 * v14);
          v94 = *(_DWORD *)(v129 + 72);
          if ( !v94 )
            goto LABEL_240;
          v95 = v94 - 1;
          v96 = *(_QWORD *)(v129 + 56);
          v118 = v137;
          v97 = v95 & (37 * v137);
          v92 = (_DWORD *)(v96 + 48LL * v97);
          v93 = *(_DWORD *)(v129 + 64) + 1;
          v98 = *v92;
          if ( *v92 == (_DWORD)v137 )
            goto LABEL_184;
          v99 = 1;
          v100 = 0;
          while ( v98 != 0x7FFFFFFF )
          {
            if ( v98 == 0x80000000 && !v100 )
              v100 = v92;
            v7 = (unsigned int)(v99 + 1);
            v97 = v95 & (v99 + v97);
            v92 = (_DWORD *)(v96 + 48LL * v97);
            v98 = *v92;
            if ( (_DWORD)v137 == *v92 )
              goto LABEL_184;
            ++v99;
          }
LABEL_196:
          if ( v100 )
            v92 = v100;
LABEL_184:
          *(_DWORD *)(v129 + 64) = v93;
          if ( *v92 != 0x7FFFFFFF )
            --*(_DWORD *)(v129 + 68);
          *v92 = v118;
          *((_QWORD *)v92 + 1) = v92 + 6;
          if ( (_BYTE *)n == &src[8] )
          {
            *(__m128i *)(v92 + 6) = _mm_loadu_si128((const __m128i *)&src[8]);
          }
          else
          {
            *((_QWORD *)v92 + 1) = n;
            *((_QWORD *)v92 + 3) = *(_QWORD *)&src[8];
          }
          *((_QWORD *)v92 + 2) = *(_QWORD *)src;
          v92[10] = v140;
          *((_BYTE *)v92 + 44) = BYTE4(v140);
          goto LABEL_12;
        }
      }
      v137 = (unsigned __int64)src;
      v22 = src;
      goto LABEL_36;
    }
LABEL_24:
    if ( v116 == ++v12 )
      break;
    v11 = (unsigned int)v146;
    v10 = HIDWORD(v146);
  }
  v4 = v129;
LABEL_54:
  v130 = *(_QWORD *)(v5 + 104);
  if ( v130 == *(_QWORD *)(v5 + 96) )
    goto LABEL_70;
  v127 = v4;
  v32 = *(_QWORD *)(v5 + 96);
  while ( 2 )
  {
    v38 = *(_DWORD *)(v32 + 4);
    if ( *(_BYTE *)(v32 + 9)
      || *(_QWORD *)(*(_QWORD *)(v5 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v5 + 32) + v38) + 8) != -1 )
    {
      v151[1] = 0;
      v151[0] = (size_t)&v152;
      LOBYTE(v152) = 0;
      v154 = 0u;
      sub_2F07630(*(_DWORD *)v32, (__int64)v151, v115);
      if ( !*(_BYTE *)(v32 + 9) )
      {
        if ( v38 < 0 )
        {
          v39 = (__m128i *)(a2[47] + 264LL * *(int *)&v142[4 * (*(_DWORD *)(v5 + 32) + v38)]);
          sub_2F074A0((__int64)v39[4].m128i_i64, (__int64)v151);
          v39[6] = _mm_load_si128(&v154);
          v39[7].m128i_i8[0] = *(_BYTE *)(v32 + 8);
          goto LABEL_61;
        }
        v33 = v151[1];
        v34 = a2[53] + 320LL * *((unsigned int *)v145 + v38);
        v35 = *(__int64 **)(v34 + 104);
        if ( (__int64 *)v151[0] == &v152 )
        {
          if ( v151[1] )
          {
            if ( v151[1] == 1 )
            {
              *(_BYTE *)v35 = v152;
              v111 = v151[1];
              v112 = *(_QWORD *)(v34 + 104);
              *(_QWORD *)(v34 + 112) = v151[1];
              *(_BYTE *)(v112 + v111) = 0;
              v35 = (__int64 *)v151[0];
              goto LABEL_60;
            }
            v126 = a2[53] + 320LL * *((unsigned int *)v145 + v38);
            memcpy(v35, &v152, v151[1]);
            v34 = v126;
            v33 = v151[1];
            v35 = *(__int64 **)(v126 + 104);
          }
          *(_QWORD *)(v34 + 112) = v33;
          *((_BYTE *)v35 + v33) = 0;
          v35 = (__int64 *)v151[0];
          goto LABEL_60;
        }
        v36 = v152;
        if ( v35 == (__int64 *)(v34 + 120) )
        {
          *(_QWORD *)(v34 + 104) = v151[0];
          *(_QWORD *)(v34 + 112) = v33;
          *(_QWORD *)(v34 + 120) = v36;
        }
        else
        {
          v37 = *(_QWORD *)(v34 + 120);
          *(_QWORD *)(v34 + 104) = v151[0];
          *(_QWORD *)(v34 + 112) = v33;
          *(_QWORD *)(v34 + 120) = v36;
          if ( v35 )
          {
            v151[0] = (size_t)v35;
            v152 = v37;
            goto LABEL_60;
          }
        }
        v151[0] = (size_t)&v152;
        v35 = &v152;
LABEL_60:
        v151[1] = 0;
        *(_BYTE *)v35 = 0;
        *(__m128i *)(v34 + 136) = _mm_load_si128(&v154);
        *(_BYTE *)(v34 + 152) = *(_BYTE *)(v32 + 8);
      }
LABEL_61:
      if ( (__int64 *)v151[0] != &v152 )
        j_j___libc_free_0(v151[0]);
    }
    v32 += 12;
    if ( v130 != v32 )
      continue;
    break;
  }
  v4 = v127;
LABEL_70:
  v40 = *(_DWORD *)(v5 + 136);
  if ( v40 )
  {
    v41 = (unsigned __int64)v145;
    v42 = 0;
    do
    {
      v43 = v42++;
      v44 = (int *)(*(_QWORD *)(v5 + 128) + 16 * v43);
      v45 = *((_QWORD *)v44 + 1);
      v46 = a2[53] + 320LL * *(unsigned int *)(v41 + 4LL * *v44);
      *(_QWORD *)(v46 + 160) = v45;
      *(_BYTE *)(v46 + 168) = 1;
    }
    while ( v42 != v40 );
  }
  if ( *(_DWORD *)(v5 + 68) != -1 )
  {
    v140 = 0x100000000LL;
    n = 0;
    v137 = (unsigned __int64)&unk_49DD210;
    memset(src, 0, sizeof(src));
    v141 = a2 + 20;
    sub_CB5980((__int64)&v137, 0, 0, 0);
    v151[0] = (size_t)&v137;
    v154.m128i_i64[0] = (__int64)&v155;
    v151[1] = a4;
    v152 = v4 + 16;
    dest = (void *)(v4 + 48);
    v154.m128i_i64[1] = 0x800000000LL;
    sub_2F11000((__int64 *)v151, *(_DWORD *)(v5 + 68));
    if ( (__int64 *)v154.m128i_i64[0] != &v155 )
      _libc_free(v154.m128i_u64[0]);
    v137 = (unsigned __int64)&unk_49DD210;
    sub_CB5840((__int64)&v137);
  }
  if ( *(_DWORD *)(v5 + 72) != -1 )
  {
    v140 = 0x100000000LL;
    n = 0;
    v137 = (unsigned __int64)&unk_49DD210;
    memset(src, 0, sizeof(src));
    v141 = a2 + 26;
    sub_CB5980((__int64)&v137, 0, 0, 0);
    v151[0] = (size_t)&v137;
    v154.m128i_i64[0] = (__int64)&v155;
    v151[1] = a4;
    v152 = v4 + 16;
    dest = (void *)(v4 + 48);
    v154.m128i_i64[1] = 0x800000000LL;
    sub_2F11000((__int64 *)v151, *(_DWORD *)(v5 + 72));
    if ( (__int64 *)v154.m128i_i64[0] != &v155 )
      _libc_free(v154.m128i_u64[0]);
    v137 = (unsigned __int64)&unk_49DD210;
    sub_CB5840((__int64)&v137);
  }
  v47 = *(_QWORD *)(a3 + 752);
  v48 = 32LL * *(unsigned int *)(a3 + 760);
  v49 = (_QWORD *)(v47 + v48);
  if ( v47 != v47 + v48 )
  {
    while ( *(_BYTE *)(v47 + 4) )
    {
      v47 += 32;
      if ( v49 == (_QWORD *)v47 )
        goto LABEL_90;
    }
    if ( v49 != (_QWORD *)v47 )
    {
      v50 = (_QWORD *)v47;
      v51 = v49;
      do
      {
        v52 = *(int *)v50;
        v53 = v50[1];
        v54 = v50[2];
        v55 = v50[3];
        if ( (int)v52 < 0 )
        {
          v128 = v51;
          v125 = v50;
          v62 = 0;
          v63 = *(int *)&v142[4 * (unsigned int)(*(_DWORD *)(v5 + 32) + v52)];
          v137 = v53;
          n = v54;
          *(_QWORD *)src = v55;
          v64 = a2[47] + 264 * v63;
          v65 = v64 + 120;
          v132 = (__m128i *)(v64 + 120);
          v133 = v64 + 168;
          v134.m128i_i64[0] = v64 + 216;
          while ( 1 )
          {
            v154.m128i_i64[1] = 0x100000000LL;
            v155 = v65;
            v151[1] = 0;
            v152 = 0;
            dest = 0;
            v154.m128i_i64[0] = 0;
            v151[0] = (size_t)&unk_49DD210;
            sub_CB5980((__int64)v151, 0, 0, 0);
            v66 = *(const char **)((char *)&v137 + v62);
            v62 += 8;
            sub_A61DC0(v66, (__int64)v151, a4, 0);
            v151[0] = (size_t)&unk_49DD210;
            sub_CB5840((__int64)v151);
            if ( v62 == 24 )
              break;
            v65 = *(__int64 *)((char *)&v132 + v62);
          }
        }
        else
        {
          v128 = v51;
          v125 = v50;
          v56 = 0;
          v57 = *((unsigned int *)v145 + v52);
          v137 = v53;
          n = v54;
          *(_QWORD *)src = v55;
          v58 = a2[53] + 320 * v57;
          v59 = v58 + 176;
          v132 = (__m128i *)(v58 + 176);
          v133 = v58 + 224;
          v134.m128i_i64[0] = v58 + 272;
          while ( 1 )
          {
            v154.m128i_i64[1] = 0x100000000LL;
            v155 = v59;
            v151[1] = 0;
            v152 = 0;
            dest = 0;
            v154.m128i_i64[0] = 0;
            v151[0] = (size_t)&unk_49DD210;
            sub_CB5980((__int64)v151, 0, 0, 0);
            v60 = *(const char **)((char *)&v137 + v56);
            v56 += 8;
            sub_A61DC0(v60, (__int64)v151, a4, 0);
            v151[0] = (size_t)&unk_49DD210;
            sub_CB5840((__int64)v151);
            if ( v56 == 24 )
              break;
            v59 = *(__int64 *)((char *)&v132 + v56);
          }
        }
        v51 = v128;
        v61 = v125 + 4;
        if ( v128 == v125 + 4 )
          break;
        while ( 1 )
        {
          v50 = v61;
          if ( !*((_BYTE *)v61 + 4) )
            break;
          v61 += 4;
          if ( v128 == v61 )
            goto LABEL_90;
        }
      }
      while ( v128 != v61 );
    }
  }
LABEL_90:
  if ( v145 != &v147 )
    _libc_free((unsigned __int64)v145);
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
}
