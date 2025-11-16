// Function: sub_28B18B0
// Address: 0x28b18b0
//
__int64 __fastcall sub_28B18B0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  void ***v4; // r15
  __int64 v5; // r12
  _QWORD *v6; // rdi
  __int64 *v7; // rax
  __m128i v8; // xmm5
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  unsigned __int8 *v11; // rsi
  __m128i *v12; // r14
  __int64 *v13; // rax
  char v14; // bl
  __m128i v16; // xmm2
  __m128i v17; // xmm1
  __m128i v18; // xmm0
  unsigned __int8 *v19; // rax
  unsigned __int8 *v20; // rbx
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  __int64 *v23; // rax
  char v24; // r13
  unsigned __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // eax
  unsigned __int8 **v28; // rcx
  unsigned __int8 *v29; // rdi
  __m128i v30; // xmm0
  _QWORD *v31; // rdi
  __m128i v32; // xmm5
  __int64 *v33; // rax
  __m128i v34; // xmm0
  __int64 *v35; // rax
  char v36; // r13
  int v37; // edx
  __int64 v38; // rcx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // r13
  __int64 *v44; // r12
  __int64 v45; // rdx
  int v46; // r10d
  __int64 v47; // r9
  unsigned int v48; // ecx
  unsigned __int8 **v49; // rdx
  unsigned __int8 *v50; // r8
  unsigned __int8 *v51; // rdx
  unsigned __int64 v52; // rdx
  __int64 *v53; // r14
  _QWORD *v54; // rbx
  __int64 v55; // r15
  _QWORD **i; // rbx
  __int64 v57; // rdx
  __int64 v58; // r13
  __int64 v59; // rcx
  __int64 v60; // r9
  __int64 v61; // rax
  int v62; // esi
  __int64 v63; // rdi
  int v64; // esi
  unsigned int v65; // edx
  __int64 *v66; // rax
  __int64 v67; // r8
  int v68; // eax
  int v69; // ecx
  const __m128i *v70; // r13
  __int64 v71; // rax
  __int64 v72; // rcx
  signed __int64 v73; // rax
  __int64 v74; // r15
  __int64 v75; // r12
  _QWORD *v76; // rdi
  __m128i v77; // xmm0
  _QWORD *v78; // rdi
  __m128i v79; // xmm6
  _QWORD *v80; // rdi
  __m128i v81; // xmm4
  _QWORD *v82; // rdi
  __m128i v83; // xmm5
  unsigned __int8 **v84; // r13
  __int64 v85; // rax
  unsigned __int8 **v86; // rcx
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // r14
  unsigned __int8 **v90; // r12
  _QWORD *v91; // rdi
  __m128i v92; // xmm2
  __m128i v93; // xmm1
  __m128i v94; // xmm0
  unsigned __int8 *v95; // rsi
  __int64 *v96; // rax
  __int64 *v97; // rax
  char v98; // r13
  __int64 v99; // rax
  __int64 v100; // rdx
  const __m128i *v101; // r13
  __m128i *v102; // rax
  _QWORD *v103; // rdi
  __m128i v104; // xmm0
  _QWORD *v105; // rdi
  __m128i v106; // xmm0
  _QWORD *v107; // rdi
  __m128i v108; // xmm6
  char *v109; // r13
  unsigned __int64 v110; // r13
  _QWORD *v111; // rcx
  int v112; // ebx
  _QWORD *v113; // rdi
  unsigned int v114; // esi
  _QWORD *v115; // rdx
  _QWORD *v116; // r14
  int v117; // edx
  int v118; // r8d
  int v119; // edx
  int v120; // edi
  __int64 v121; // [rsp+0h] [rbp-6B0h]
  void ***v122; // [rsp+8h] [rbp-6A8h]
  __m128i *v123; // [rsp+8h] [rbp-6A8h]
  const __m128i *v124; // [rsp+10h] [rbp-6A0h]
  unsigned __int8 **v125; // [rsp+10h] [rbp-6A0h]
  unsigned __int8 *v126; // [rsp+38h] [rbp-678h]
  __int64 v127; // [rsp+48h] [rbp-668h]
  unsigned __int8 *v129; // [rsp+60h] [rbp-650h]
  unsigned __int8 v130; // [rsp+7Fh] [rbp-631h]
  unsigned __int64 v131; // [rsp+88h] [rbp-628h]
  unsigned __int8 *v132; // [rsp+90h] [rbp-620h] BYREF
  __int64 v133; // [rsp+98h] [rbp-618h] BYREF
  _QWORD v134[4]; // [rsp+A0h] [rbp-610h] BYREF
  __int64 v135; // [rsp+C0h] [rbp-5F0h] BYREF
  __int64 v136; // [rsp+C8h] [rbp-5E8h]
  __int64 v137; // [rsp+D0h] [rbp-5E0h]
  __int64 v138; // [rsp+D8h] [rbp-5D8h]
  __m128i v139; // [rsp+E0h] [rbp-5D0h] BYREF
  __m128i v140; // [rsp+F0h] [rbp-5C0h] BYREF
  __m128i v141; // [rsp+100h] [rbp-5B0h] BYREF
  __m128i v142; // [rsp+110h] [rbp-5A0h] BYREF
  __m128i v143; // [rsp+120h] [rbp-590h] BYREF
  __m128i v144; // [rsp+130h] [rbp-580h] BYREF
  _OWORD v145[3]; // [rsp+140h] [rbp-570h] BYREF
  __m128i v146; // [rsp+170h] [rbp-540h] BYREF
  __m128i v147; // [rsp+180h] [rbp-530h]
  __m128i v148; // [rsp+190h] [rbp-520h]
  char v149; // [rsp+1A0h] [rbp-510h]
  _QWORD *v150; // [rsp+1B0h] [rbp-500h] BYREF
  __int64 v151; // [rsp+1B8h] [rbp-4F8h]
  _QWORD v152[8]; // [rsp+1C0h] [rbp-4F0h] BYREF
  unsigned __int8 **v153; // [rsp+200h] [rbp-4B0h] BYREF
  __int64 v154; // [rsp+208h] [rbp-4A8h]
  _BYTE v155[64]; // [rsp+210h] [rbp-4A0h] BYREF
  __m128i v156; // [rsp+250h] [rbp-460h] BYREF
  __m128i v157; // [rsp+260h] [rbp-450h] BYREF
  __m128i v158; // [rsp+270h] [rbp-440h]
  __m128i v159; // [rsp+280h] [rbp-430h]
  __m128i v160; // [rsp+3E0h] [rbp-2D0h] BYREF
  __m128i v161; // [rsp+3F0h] [rbp-2C0h] BYREF
  __m128i v162; // [rsp+400h] [rbp-2B0h] BYREF
  char v163; // [rsp+410h] [rbp-2A0h]
  void **v164; // [rsp+538h] [rbp-178h] BYREF
  __int64 v165; // [rsp+540h] [rbp-170h]
  char v166; // [rsp+548h] [rbp-168h]
  _BYTE *v167; // [rsp+550h] [rbp-160h]
  __int64 v168; // [rsp+558h] [rbp-158h]
  _BYTE v169[128]; // [rsp+560h] [rbp-150h] BYREF
  __int16 v170; // [rsp+5E0h] [rbp-D0h]
  void *v171; // [rsp+5E8h] [rbp-C8h] BYREF
  __int64 v172; // [rsp+5F0h] [rbp-C0h]
  __int64 v173; // [rsp+5F8h] [rbp-B8h]
  __int64 v174; // [rsp+600h] [rbp-B0h] BYREF
  unsigned int v175; // [rsp+608h] [rbp-A8h]
  _BYTE v176[48]; // [rsp+680h] [rbp-30h] BYREF

  v4 = &v164;
  v5 = a1;
  v133 = a2;
  v132 = a3;
  sub_D66630(&v139, a2);
  v6 = *(_QWORD **)(a1 + 8);
  v7 = &v161.m128i_i64[1];
  v8 = _mm_loadu_si128(&v139);
  v9 = _mm_loadu_si128(&v140);
  v10 = _mm_loadu_si128(&v141);
  v159.m128i_i8[0] = 1;
  v160 = (__m128i)(unsigned __int64)v6;
  v11 = v132;
  v161.m128i_i64[0] = 1;
  v156 = v8;
  v157 = v9;
  v158 = v10;
  do
  {
    *v7 = -4;
    v7 += 5;
    *(v7 - 4) = -3;
    *(v7 - 3) = -4;
    *(v7 - 2) = -3;
  }
  while ( v7 != (__int64 *)&v164 );
  v165 = 0;
  v12 = &v160;
  v164 = &v171;
  v167 = v169;
  v168 = 0x400000000LL;
  v166 = 0;
  v170 = 256;
  v171 = &unk_49DDBE8;
  v13 = &v174;
  v172 = 0;
  v173 = 1;
  do
  {
    *v13 = -4096;
    v13 += 2;
  }
  while ( v13 != (__int64 *)v176 );
  v14 = sub_CF63E0(v6, v11, &v156, (__int64)&v160);
  v171 = &unk_49DDBE8;
  if ( (v173 & 1) == 0 )
    sub_C7D6A0(v174, 16LL * v175, 8);
  nullsub_184();
  if ( v167 != v169 )
    _libc_free((unsigned __int64)v167);
  if ( (v161.m128i_i8[0] & 1) == 0 )
    sub_C7D6A0(v161.m128i_i64[1], 40LL * v162.m128i_u32[0], 8);
  v130 = 0;
  if ( v14 )
    return v130;
  v135 = 0;
  v134[0] = &v133;
  v134[1] = &v132;
  v134[2] = &v135;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v130 = sub_28B1630((__int64)v134, *(_QWORD *)(v133 - 32));
  if ( !v130 )
    goto LABEL_63;
  v16 = _mm_loadu_si128(&v139);
  v17 = _mm_loadu_si128(&v140);
  v18 = _mm_loadu_si128(&v141);
  v150 = v152;
  v152[0] = v133;
  v151 = 0x800000001LL;
  v156.m128i_i64[1] = 0x800000001LL;
  v153 = (unsigned __int8 **)v155;
  v154 = 0x800000000LL;
  v156.m128i_i64[0] = (__int64)&v157;
  v160 = v16;
  v161 = v17;
  v162 = v18;
  v157 = v16;
  v158 = v17;
  v159 = v18;
  sub_D665A0(&v142, a4);
  v129 = (unsigned __int8 *)(*(_QWORD *)(v133 + 24) & 0xFFFFFFFFFFFFFFF8LL);
  v19 = v132;
  v126 = v132 + 24;
  if ( v132 + 24 == v129 )
    goto LABEL_67;
  do
  {
    v20 = v129 - 24;
    if ( !v129 )
      v20 = 0;
    if ( !(unsigned __int8)sub_98CD80((char *)v20) )
      goto LABEL_56;
    v21 = *(_QWORD **)(v5 + 8);
    v149 = 0;
    v22 = &v161.m128i_i64[1];
    v160 = (__m128i)(unsigned __int64)v21;
    v161.m128i_i64[0] = 1;
    do
    {
      *v22 = -4;
      v22 += 5;
      *(v22 - 4) = -3;
      *(v22 - 3) = -4;
      *(v22 - 2) = -3;
    }
    while ( v22 != (__int64 *)v4 );
    v165 = 0;
    v166 = 0;
    v164 = &v171;
    v170 = 256;
    v167 = v169;
    v168 = 0x400000000LL;
    v172 = 0;
    v173 = 1;
    v171 = &unk_49DDBE8;
    v23 = &v174;
    do
    {
      *v23 = -4096;
      v23 += 2;
    }
    while ( v23 != (__int64 *)v176 );
    v24 = sub_CF63E0(v21, v20, &v146, (__int64)v12);
    v171 = &unk_49DDBE8;
    if ( (v173 & 1) == 0 )
      sub_C7D6A0(v174, 16LL * v175, 8);
    nullsub_184();
    if ( v167 != v169 )
      _libc_free((unsigned __int64)v167);
    if ( (v161.m128i_i8[0] & 1) == 0 )
      sub_C7D6A0(v161.m128i_i64[1], 40LL * v162.m128i_u32[0], 8);
    if ( (_DWORD)v138 )
    {
      v27 = (v138 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v28 = (unsigned __int8 **)(v136 + 8LL * v27);
      v29 = *v28;
      if ( v20 == *v28 )
      {
LABEL_30:
        *v28 = (unsigned __int8 *)-8192LL;
        LODWORD(v137) = v137 - 1;
        ++HIDWORD(v137);
        if ( !v24 )
          goto LABEL_48;
        goto LABEL_31;
      }
      v69 = 1;
      while ( v29 != (unsigned __int8 *)-4096LL )
      {
        v25 = (unsigned int)(v69 + 1);
        v27 = (v138 - 1) & (v69 + v27);
        v28 = (unsigned __int8 **)(v136 + 8LL * v27);
        v29 = *v28;
        if ( v20 == *v28 )
          goto LABEL_30;
        v69 = v25;
      }
    }
    if ( !v24 )
      goto LABEL_65;
    v70 = (const __m128i *)v156.m128i_i64[0];
    v71 = 48LL * v156.m128i_u32[2];
    v72 = v156.m128i_i64[0] + v71;
    v73 = 0xAAAAAAAAAAAAAAABLL * (v71 >> 4);
    v124 = (const __m128i *)v72;
    if ( !(v73 >> 2) )
    {
LABEL_125:
      if ( v73 != 2 )
      {
        if ( v73 != 3 )
        {
          if ( v73 != 1 )
            goto LABEL_96;
          goto LABEL_128;
        }
        v105 = *(_QWORD **)(v5 + 8);
        v160 = _mm_loadu_si128(v70);
        v161 = _mm_loadu_si128(v70 + 1);
        v106 = _mm_loadu_si128(v70 + 2);
        v163 = 1;
        v162 = v106;
        if ( (unsigned __int8)sub_CF6520(v105, v20, v12) )
          goto LABEL_95;
        v70 += 3;
      }
      v107 = *(_QWORD **)(v5 + 8);
      v160 = _mm_loadu_si128(v70);
      v161 = _mm_loadu_si128(v70 + 1);
      v108 = _mm_loadu_si128(v70 + 2);
      v163 = 1;
      v162 = v108;
      if ( (unsigned __int8)sub_CF6520(v107, v20, v12) )
        goto LABEL_95;
      v70 += 3;
LABEL_128:
      v103 = *(_QWORD **)(v5 + 8);
      v160 = _mm_loadu_si128(v70);
      v161 = _mm_loadu_si128(v70 + 1);
      v104 = _mm_loadu_si128(v70 + 2);
      v163 = 1;
      v162 = v104;
      if ( (unsigned __int8)sub_CF6520(v103, v20, v12) )
        goto LABEL_95;
LABEL_96:
      v84 = v153;
      v85 = 8LL * (unsigned int)v154;
      v86 = &v153[(unsigned __int64)v85 / 8];
      v87 = v85 >> 3;
      v88 = v85 >> 5;
      v125 = v86;
      if ( v88 )
      {
        v123 = v12;
        v89 = v5;
        v90 = &v153[4 * v88];
        while ( 1 )
        {
          if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v89 + 8), v20, *v84) )
          {
            v5 = v89;
            v12 = v123;
            goto LABEL_104;
          }
          if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v89 + 8), v20, v84[1]) )
          {
            v5 = v89;
            ++v84;
            v12 = v123;
            goto LABEL_104;
          }
          if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v89 + 8), v20, v84[2]) )
          {
            v5 = v89;
            v84 += 2;
            v12 = v123;
            goto LABEL_104;
          }
          if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v89 + 8), v20, v84[3]) )
            break;
          v84 += 4;
          if ( v90 == v84 )
          {
            v5 = v89;
            v12 = v123;
            v87 = v125 - v84;
            goto LABEL_138;
          }
        }
        v5 = v89;
        v84 += 3;
        v12 = v123;
        goto LABEL_104;
      }
LABEL_138:
      if ( v87 != 2 )
      {
        if ( v87 != 3 )
        {
          if ( v87 != 1 )
            goto LABEL_65;
LABEL_141:
          if ( !(unsigned __int8)sub_CF5B00(*(_QWORD **)(v5 + 8), v20, *v84) )
            goto LABEL_65;
LABEL_104:
          if ( v125 == v84 )
            goto LABEL_65;
          goto LABEL_31;
        }
        if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v5 + 8), v20, *v84) )
          goto LABEL_104;
        ++v84;
      }
      if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v5 + 8), v20, *v84) )
        goto LABEL_104;
      ++v84;
      goto LABEL_141;
    }
    v122 = v4;
    v74 = v5;
    v75 = v156.m128i_i64[0] + 192 * (v73 >> 2);
    while ( 1 )
    {
      v82 = *(_QWORD **)(v74 + 8);
      v160 = _mm_loadu_si128(v70);
      v161 = _mm_loadu_si128(v70 + 1);
      v83 = _mm_loadu_si128(v70 + 2);
      v163 = 1;
      v162 = v83;
      if ( (unsigned __int8)sub_CF6520(v82, v20, v12) )
      {
        v5 = v74;
        v4 = v122;
        goto LABEL_95;
      }
      v76 = *(_QWORD **)(v74 + 8);
      v160 = _mm_loadu_si128(v70 + 3);
      v161 = _mm_loadu_si128(v70 + 4);
      v77 = _mm_loadu_si128(v70 + 5);
      v163 = 1;
      v162 = v77;
      if ( (unsigned __int8)sub_CF6520(v76, v20, v12) )
      {
        v5 = v74;
        v70 += 3;
        v4 = v122;
        goto LABEL_95;
      }
      v78 = *(_QWORD **)(v74 + 8);
      v160 = _mm_loadu_si128(v70 + 6);
      v161 = _mm_loadu_si128(v70 + 7);
      v79 = _mm_loadu_si128(v70 + 8);
      v163 = 1;
      v162 = v79;
      if ( (unsigned __int8)sub_CF6520(v78, v20, v12) )
      {
        v5 = v74;
        v70 += 6;
        v4 = v122;
        goto LABEL_95;
      }
      v80 = *(_QWORD **)(v74 + 8);
      v160 = _mm_loadu_si128(v70 + 9);
      v161 = _mm_loadu_si128(v70 + 10);
      v81 = _mm_loadu_si128(v70 + 11);
      v163 = 1;
      v162 = v81;
      if ( (unsigned __int8)sub_CF6520(v80, v20, v12) )
        break;
      v70 += 12;
      if ( (const __m128i *)v75 == v70 )
      {
        v5 = v74;
        v4 = v122;
        v73 = 0xAAAAAAAAAAAAAAABLL * (v124 - v70);
        goto LABEL_125;
      }
    }
    v5 = v74;
    v70 += 9;
    v4 = v122;
LABEL_95:
    if ( v124 == v70 )
      goto LABEL_96;
LABEL_31:
    v30 = _mm_loadu_si128(&v142);
    v31 = *(_QWORD **)(v5 + 8);
    v149 = 1;
    v32 = _mm_loadu_si128(&v144);
    v33 = &v161.m128i_i64[1];
    v146 = v30;
    v34 = _mm_loadu_si128(&v143);
    v160 = (__m128i)(unsigned __int64)v31;
    v161.m128i_i64[0] = 1;
    v147 = v34;
    v148 = v32;
    do
    {
      *v33 = -4;
      v33 += 5;
      *(v33 - 4) = -3;
      *(v33 - 3) = -4;
      *(v33 - 2) = -3;
    }
    while ( v33 != (__int64 *)v4 );
    v165 = 0;
    v166 = 0;
    v164 = &v171;
    v170 = 256;
    v167 = v169;
    v168 = 0x400000000LL;
    v172 = 0;
    v173 = 1;
    v171 = &unk_49DDBE8;
    v35 = &v174;
    do
    {
      *v35 = -4096;
      v35 += 2;
    }
    while ( v35 != (__int64 *)v176 );
    v36 = sub_CF63E0(v31, v20, &v146, (__int64)v12);
    v171 = &unk_49DDBE8;
    if ( (v173 & 1) == 0 )
      sub_C7D6A0(v174, 16LL * v175, 8);
    nullsub_184();
    if ( v167 != v169 )
      _libc_free((unsigned __int64)v167);
    if ( (v161.m128i_i8[0] & 1) == 0 )
      sub_C7D6A0(v161.m128i_i64[1], 40LL * v162.m128i_u32[0], 8);
    if ( (v36 & 2) != 0 )
      goto LABEL_56;
    v37 = *v20;
    if ( (unsigned __int8)(v37 - 34) <= 0x33u && (v38 = 0x8000000000041LL, _bittest64(&v38, (unsigned int)(v37 - 34))) )
    {
      if ( (unsigned __int8)sub_CF5B00(*(_QWORD **)(v5 + 8), v132, v20) )
        goto LABEL_56;
      v39 = (unsigned int)v154;
      v40 = (unsigned int)v154 + 1LL;
      if ( v40 > HIDWORD(v154) )
      {
        sub_C8D5F0((__int64)&v153, v155, v40, 8u, v25, v26);
        v39 = (unsigned int)v154;
      }
      v153[v39] = v20;
      LODWORD(v154) = v154 + 1;
    }
    else
    {
      if ( (unsigned __int8)(v37 - 61) > 1u && (_BYTE)v37 != 89 )
        goto LABEL_56;
      sub_D66840(v12, v20);
      v91 = *(_QWORD **)(v5 + 8);
      v92 = _mm_loadu_si128(&v160);
      v149 = 1;
      v93 = _mm_loadu_si128(&v161);
      v94 = _mm_loadu_si128(&v162);
      v160 = (__m128i)(unsigned __int64)v91;
      v95 = v132;
      v96 = &v161.m128i_i64[1];
      v145[0] = v92;
      v161.m128i_i64[0] = 1;
      v145[1] = v93;
      v145[2] = v94;
      v146 = v92;
      v147 = v93;
      v148 = v94;
      do
      {
        *v96 = -4;
        v96 += 5;
        *(v96 - 4) = -3;
        *(v96 - 3) = -4;
        *(v96 - 2) = -3;
      }
      while ( v96 != (__int64 *)v4 );
      v165 = 0;
      v166 = 0;
      v164 = &v171;
      v172 = 0;
      v167 = v169;
      v168 = 0x400000000LL;
      v170 = 256;
      v173 = 1;
      v171 = &unk_49DDBE8;
      v97 = &v174;
      do
      {
        *v97 = -4096;
        v97 += 2;
      }
      while ( v97 != (__int64 *)v176 );
      v98 = sub_CF63E0(v91, v95, &v146, (__int64)v12);
      v171 = &unk_49DDBE8;
      if ( (v173 & 1) == 0 )
        sub_C7D6A0(v174, 16LL * v175, 8);
      nullsub_184();
      if ( v167 != v169 )
        _libc_free((unsigned __int64)v167);
      if ( (v161.m128i_i8[0] & 1) == 0 )
        sub_C7D6A0(v161.m128i_i64[1], 40LL * v162.m128i_u32[0], 8);
      if ( v98 )
        goto LABEL_56;
      v99 = v156.m128i_u32[2];
      v100 = v156.m128i_i64[0];
      v101 = (const __m128i *)v145;
      v25 = v156.m128i_u32[2] + 1LL;
      if ( v25 > v156.m128i_u32[3] )
      {
        if ( v156.m128i_i64[0] > (unsigned __int64)v145
          || (unsigned __int64)v145 >= v156.m128i_i64[0] + 48 * (unsigned __int64)v156.m128i_u32[2] )
        {
          sub_C8D5F0((__int64)&v156, &v157, v156.m128i_u32[2] + 1LL, 0x30u, v25, v26);
          v100 = v156.m128i_i64[0];
          v99 = v156.m128i_u32[2];
          v101 = (const __m128i *)v145;
        }
        else
        {
          v109 = (char *)v145 - v156.m128i_i64[0];
          sub_C8D5F0((__int64)&v156, &v157, v156.m128i_u32[2] + 1LL, 0x30u, v25, v26);
          v100 = v156.m128i_i64[0];
          v99 = v156.m128i_u32[2];
          v101 = (const __m128i *)&v109[v156.m128i_i64[0]];
        }
      }
      v102 = (__m128i *)(v100 + 48 * v99);
      *v102 = _mm_loadu_si128(v101);
      v102[1] = _mm_loadu_si128(v101 + 1);
      v102[2] = _mm_loadu_si128(v101 + 2);
      ++v156.m128i_i32[2];
    }
LABEL_48:
    v41 = (unsigned int)v151;
    v42 = (unsigned int)v151 + 1LL;
    if ( v42 > HIDWORD(v151) )
    {
      sub_C8D5F0((__int64)&v150, v152, v42, 8u, v25, v26);
      v41 = (unsigned int)v151;
    }
    v150[v41] = v20;
    LODWORD(v151) = v151 + 1;
    if ( (v20[7] & 0x40) != 0 )
    {
      v43 = (__int64 *)*((_QWORD *)v20 - 1);
      v20 = (unsigned __int8 *)&v43[4 * (*((_DWORD *)v20 + 1) & 0x7FFFFFF)];
    }
    else
    {
      v43 = (__int64 *)&v20[-32 * (*((_DWORD *)v20 + 1) & 0x7FFFFFF)];
    }
    if ( v20 != (unsigned __int8 *)v43 )
    {
      v127 = v5;
      v44 = v43;
      while ( (unsigned __int8)sub_28B1630((__int64)v134, *v44) )
      {
        v44 += 4;
        if ( v20 == (unsigned __int8 *)v44 )
        {
          v5 = v127;
          goto LABEL_65;
        }
      }
LABEL_56:
      v130 = 0;
      goto LABEL_57;
    }
LABEL_65:
    v129 = (unsigned __int8 *)(*(_QWORD *)v129 & 0xFFFFFFFFFFFFFFF8LL);
  }
  while ( v126 != v129 );
  v19 = v132;
LABEL_67:
  v45 = *(_QWORD *)(v5 + 40);
  v46 = *(_DWORD *)(v45 + 56);
  v47 = *(_QWORD *)(v45 + 40);
  if ( !v46 )
    goto LABEL_154;
  v48 = (v46 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v49 = (unsigned __int8 **)(v47 + 16LL * v48);
  v50 = *v49;
  if ( v19 != *v49 )
  {
    v119 = 1;
    while ( v50 != (unsigned __int8 *)-4096LL )
    {
      v120 = v119 + 1;
      v48 = (v46 - 1) & (v119 + v48);
      v49 = (unsigned __int8 **)(v47 + 16LL * v48);
      v50 = *v49;
      if ( v19 == *v49 )
        goto LABEL_69;
      v119 = v120;
    }
    goto LABEL_154;
  }
LABEL_69:
  v51 = v49[1];
  if ( !v51 )
  {
LABEL_154:
    v110 = *(_QWORD *)(a4 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v111 = (_QWORD *)(*((_QWORD *)v19 + 3) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v111 != (_QWORD *)v110 )
    {
      v112 = v46 - 1;
      do
      {
        v113 = v111 - 3;
        if ( !v111 )
          v113 = 0;
        if ( v46 )
        {
          v114 = v112 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
          v115 = (_QWORD *)(v47 + 16LL * v114);
          v116 = (_QWORD *)*v115;
          if ( v113 == (_QWORD *)*v115 )
          {
LABEL_156:
            v53 = (__int64 *)v115[1];
            if ( v53 )
              goto LABEL_72;
          }
          else
          {
            v117 = 1;
            while ( v116 != (_QWORD *)-4096LL )
            {
              v118 = v117 + 1;
              v114 = v112 & (v117 + v114);
              v115 = (_QWORD *)(v47 + 16LL * v114);
              v116 = (_QWORD *)*v115;
              if ( v113 == (_QWORD *)*v115 )
                goto LABEL_156;
              v117 = v118;
            }
          }
        }
        v111 = (_QWORD *)(*v111 & 0xFFFFFFFFFFFFFFF8LL);
      }
      while ( (_QWORD *)v110 != v111 );
    }
    v53 = 0;
    goto LABEL_72;
  }
  v52 = *((_QWORD *)v51 + 4) & 0xFFFFFFFFFFFFFFF8LL;
  v53 = (__int64 *)(v52 - 32);
  if ( !v52 )
    v53 = 0;
LABEL_72:
  v54 = &v150[(unsigned int)v151];
  if ( v150 != v54 )
  {
    v131 = (unsigned __int64)v150;
    v55 = v121;
    for ( i = (_QWORD **)(v54 - 1); ; --i )
    {
      v58 = (__int64)*i;
      LOWORD(v55) = 0;
      sub_B444E0(*i, (__int64)(v19 + 24), v55);
      v61 = *(_QWORD *)(v5 + 40);
      v62 = *(_DWORD *)(v61 + 56);
      v63 = *(_QWORD *)(v61 + 40);
      if ( v62 )
      {
        v64 = v62 - 1;
        v65 = v64 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v66 = (__int64 *)(v63 + 16LL * v65);
        v67 = *v66;
        if ( v58 == *v66 )
        {
LABEL_74:
          if ( v66[1] )
          {
            v57 = (__int64)v53;
            v53 = (__int64 *)v66[1];
            sub_D75390(*(__int64 **)(v5 + 48), v53, v57, v59, v67, v60);
          }
        }
        else
        {
          v68 = 1;
          while ( v67 != -4096 )
          {
            v59 = (unsigned int)(v68 + 1);
            v65 = v64 & (v68 + v65);
            v66 = (__int64 *)(v63 + 16LL * v65);
            v67 = *v66;
            if ( v58 == *v66 )
              goto LABEL_74;
            v68 = v59;
          }
        }
      }
      if ( (_QWORD **)v131 == i )
        break;
      v19 = v132;
    }
  }
LABEL_57:
  if ( v153 != (unsigned __int8 **)v155 )
    _libc_free((unsigned __int64)v153);
  if ( (__m128i *)v156.m128i_i64[0] != &v157 )
    _libc_free(v156.m128i_u64[0]);
  if ( v150 != v152 )
    _libc_free((unsigned __int64)v150);
LABEL_63:
  sub_C7D6A0(v136, 8LL * (unsigned int)v138, 8);
  return v130;
}
