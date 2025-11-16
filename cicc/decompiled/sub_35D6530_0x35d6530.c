// Function: sub_35D6530
// Address: 0x35d6530
//
void __fastcall sub_35D6530(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 (*v3)(); // rax
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // r12d
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // ecx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // r8
  __int64 *v27; // rax
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r12
  int v32; // r9d
  unsigned int i; // eax
  __int64 v34; // rsi
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rcx
  int v40; // r10d
  unsigned int k; // eax
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 *v44; // r9
  __int64 *v45; // rbx
  __int64 *v46; // r13
  __int64 v47; // r14
  char *v48; // rax
  __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  unsigned int v51; // eax
  unsigned __int64 v52; // r10
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rax
  int v56; // r10d
  unsigned int j; // esi
  unsigned int v58; // esi
  __int64 v59; // rax
  char *v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rax
  int v63; // edx
  char *v64; // rcx
  char *v65; // rax
  char v66; // bl
  __int64 v67; // rsi
  __int64 v68; // r14
  __int64 (*v69)(void); // rax
  __int64 v70; // r12
  __int64 *v71; // rax
  _QWORD *v72; // rsi
  __int32 v73; // eax
  __int64 v74; // rdx
  int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rbx
  __int64 v78; // rdi
  __int64 (__fastcall *v79)(__int64, __int64, unsigned int); // rax
  __int64 v80; // rsi
  int v81; // eax
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rsi
  __int64 v85; // r14
  __int64 *v86; // rax
  _QWORD *v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rbx
  __int64 v90; // r14
  __int64 v91; // r15
  __int32 v92; // edx
  __int64 v93; // r14
  __int64 v94; // rsi
  __m128i *v95; // rdx
  const __m128i *v96; // rax
  const __m128i *v97; // rsi
  __int64 v98; // rsi
  __m128i *v99; // rdx
  const __m128i *v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rbx
  _QWORD *v103; // rax
  _QWORD *v104; // r13
  __int64 v105; // rdx
  _QWORD *v106; // r12
  _QWORD *v107; // rax
  _QWORD *v108; // r15
  _QWORD *v109; // r13
  __int64 v110; // rax
  __int32 v111; // r14d
  __int64 v112; // rcx
  __int64 *v113; // rax
  __int64 v114; // rax
  __int64 v115; // [rsp+8h] [rbp-838h]
  unsigned __int64 v116; // [rsp+18h] [rbp-828h]
  unsigned __int64 v117; // [rsp+30h] [rbp-810h]
  _BYTE *v118; // [rsp+50h] [rbp-7F0h]
  __int64 v119; // [rsp+58h] [rbp-7E8h]
  __int64 *v120; // [rsp+68h] [rbp-7D8h]
  unsigned __int64 v121; // [rsp+78h] [rbp-7C8h]
  __int32 v122; // [rsp+80h] [rbp-7C0h]
  char v123; // [rsp+86h] [rbp-7BAh]
  bool v124; // [rsp+87h] [rbp-7B9h]
  __int64 v125; // [rsp+88h] [rbp-7B8h]
  __int64 (__fastcall *v126)(__int64, unsigned __int16); // [rsp+88h] [rbp-7B8h]
  _BYTE *v127; // [rsp+88h] [rbp-7B8h]
  unsigned int v128; // [rsp+88h] [rbp-7B8h]
  unsigned __int64 v129; // [rsp+90h] [rbp-7B0h]
  __int64 v130; // [rsp+90h] [rbp-7B0h]
  __int64 *v131; // [rsp+98h] [rbp-7A8h]
  __int64 v132; // [rsp+98h] [rbp-7A8h]
  _BYTE *v133; // [rsp+A0h] [rbp-7A0h] BYREF
  __int64 v134; // [rsp+A8h] [rbp-798h]
  _BYTE v135[64]; // [rsp+B0h] [rbp-790h] BYREF
  __int64 v136[38]; // [rsp+F0h] [rbp-750h] BYREF
  __int64 v137; // [rsp+220h] [rbp-620h] BYREF
  __int64 *v138; // [rsp+228h] [rbp-618h]
  int v139; // [rsp+230h] [rbp-610h]
  int v140; // [rsp+234h] [rbp-60Ch]
  int v141; // [rsp+238h] [rbp-608h]
  char v142; // [rsp+23Ch] [rbp-604h]
  __int64 v143; // [rsp+240h] [rbp-600h] BYREF
  __int64 *v144; // [rsp+280h] [rbp-5C0h]
  unsigned int v145; // [rsp+288h] [rbp-5B8h]
  int v146; // [rsp+28Ch] [rbp-5B4h]
  __int64 v147[24]; // [rsp+290h] [rbp-5B0h] BYREF
  unsigned __int8 *v148; // [rsp+350h] [rbp-4F0h] BYREF
  unsigned __int64 v149; // [rsp+358h] [rbp-4E8h]
  __int64 v150; // [rsp+360h] [rbp-4E0h]
  char v151; // [rsp+36Ch] [rbp-4D4h]
  char v152[64]; // [rsp+370h] [rbp-4D0h] BYREF
  __m128i *v153; // [rsp+3B0h] [rbp-490h] BYREF
  __int64 v154; // [rsp+3B8h] [rbp-488h]
  _BYTE v155[192]; // [rsp+3C0h] [rbp-480h] BYREF
  __m128i v156; // [rsp+480h] [rbp-3C0h] BYREF
  __int64 v157; // [rsp+490h] [rbp-3B0h]
  __int64 v158; // [rsp+498h] [rbp-3A8h]
  __int64 v159; // [rsp+4A0h] [rbp-3A0h]
  char *v160; // [rsp+4E0h] [rbp-360h]
  char v161; // [rsp+4F0h] [rbp-350h] BYREF
  _BYTE *v162; // [rsp+5B0h] [rbp-290h] BYREF
  unsigned __int64 v163; // [rsp+5B8h] [rbp-288h]
  _BYTE v164[16]; // [rsp+5C0h] [rbp-280h] BYREF
  char v165[64]; // [rsp+5D0h] [rbp-270h] BYREF
  __m128i *v166; // [rsp+610h] [rbp-230h] BYREF
  __int64 v167; // [rsp+618h] [rbp-228h]
  _BYTE v168[192]; // [rsp+620h] [rbp-220h] BYREF
  __int64 v169; // [rsp+6E0h] [rbp-160h] BYREF
  char *v170; // [rsp+6E8h] [rbp-158h]
  __int64 v171; // [rsp+6F0h] [rbp-150h]
  int v172; // [rsp+6F8h] [rbp-148h]
  char v173; // [rsp+6FCh] [rbp-144h]
  char v174; // [rsp+700h] [rbp-140h] BYREF
  char *v175; // [rsp+740h] [rbp-100h]
  char v176; // [rsp+750h] [rbp-F0h] BYREF

  v2 = a1;
  v3 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 2216LL);
  if ( v3 == sub_302E1B0 )
    return;
  v123 = v3();
  if ( !v123 || !*(_DWORD *)(a1 + 144) )
    return;
  v6 = *(_QWORD *)a1;
  v133 = v135;
  memset(v136, 0, sizeof(v136));
  LODWORD(v136[2]) = 8;
  v136[1] = (__int64)&v136[4];
  BYTE4(v136[3]) = 1;
  v136[12] = (__int64)&v136[14];
  HIDWORD(v136[13]) = 8;
  v7 = *(_QWORD *)(v6 + 328);
  v138 = &v143;
  v143 = v7;
  v139 = 8;
  v141 = 0;
  v142 = 1;
  v144 = v147;
  v146 = 8;
  v140 = 1;
  v137 = 1;
  v8 = *(_QWORD *)(v7 + 112);
  v9 = *(unsigned int *)(v7 + 120);
  v134 = 0x800000000LL;
  v147[0] = v8 + 8 * v9;
  v147[1] = v8;
  v147[2] = v7;
  v145 = 1;
  sub_2DACB60((__int64)&v137, a2, v8, v147[0], v4, v5);
  sub_C8CD80((__int64)&v162, (__int64)v165, (__int64)v136, v10, v11, v12);
  v167 = 0x800000000LL;
  v16 = v136[13];
  v166 = (__m128i *)v168;
  if ( LODWORD(v136[13]) )
  {
    v98 = LODWORD(v136[13]);
    v99 = (__m128i *)v168;
    if ( LODWORD(v136[13]) > 8 )
    {
      sub_2DACD40((__int64)&v166, LODWORD(v136[13]), (__int64)v168, v13, v14, v15);
      v99 = v166;
      v98 = LODWORD(v136[13]);
    }
    v100 = (const __m128i *)v136[12];
    v101 = v136[12] + 24 * v98;
    if ( v136[12] != v101 )
    {
      do
      {
        if ( v99 )
        {
          *v99 = _mm_loadu_si128(v100);
          v99[1].m128i_i64[0] = v100[1].m128i_i64[0];
        }
        v100 = (const __m128i *)((char *)v100 + 24);
        v99 = (__m128i *)((char *)v99 + 24);
      }
      while ( (const __m128i *)v101 != v100 );
    }
    LODWORD(v167) = v16;
  }
  sub_2DACDE0((__int64)&v169, (__int64)&v162);
  sub_C8CD80((__int64)&v148, (__int64)v152, (__int64)&v137, v17, v18, v19);
  v22 = v145;
  v153 = (__m128i *)v155;
  v154 = 0x800000000LL;
  if ( v145 )
  {
    v94 = v145;
    v95 = (__m128i *)v155;
    if ( v145 > 8 )
    {
      v128 = v145;
      sub_2DACD40((__int64)&v153, v145, (__int64)v155, v145, v20, v21);
      v95 = v153;
      v94 = v145;
      v22 = v128;
    }
    v96 = (const __m128i *)v144;
    v97 = (const __m128i *)&v144[3 * v94];
    if ( v144 != (__int64 *)v97 )
    {
      do
      {
        if ( v95 )
        {
          *v95 = _mm_loadu_si128(v96);
          v95[1].m128i_i64[0] = v96[1].m128i_i64[0];
        }
        v96 = (const __m128i *)((char *)v96 + 24);
        v95 = (__m128i *)((char *)v95 + 24);
      }
      while ( v97 != v96 );
    }
    LODWORD(v154) = v22;
  }
  sub_2DACDE0((__int64)&v156, (__int64)&v148);
  sub_2E564A0((__int64)&v156, (__int64)&v169, (__int64)&v133, v23, v24, v25);
  if ( v160 != &v161 )
    _libc_free((unsigned __int64)v160);
  if ( !BYTE4(v158) )
    _libc_free(v156.m128i_u64[1]);
  if ( v153 != (__m128i *)v155 )
    _libc_free((unsigned __int64)v153);
  if ( !v151 )
    _libc_free(v149);
  if ( v175 != &v176 )
    _libc_free((unsigned __int64)v175);
  if ( !v173 )
    _libc_free((unsigned __int64)v170);
  if ( v166 != (__m128i *)v168 )
    _libc_free((unsigned __int64)v166);
  if ( !v164[12] )
    _libc_free(v163);
  if ( v144 != v147 )
    _libc_free((unsigned __int64)v144);
  if ( !v142 )
    _libc_free((unsigned __int64)v138);
  if ( (__int64 *)v136[12] != &v136[14] )
    _libc_free(v136[12]);
  if ( !BYTE4(v136[3]) )
    _libc_free(v136[1]);
  v116 = (unsigned __int64)v133;
  v118 = &v133[8 * (unsigned int)v134];
  if ( v133 != v118 )
  {
    while ( 1 )
    {
      v27 = *(__int64 **)(v2 + 136);
      v120 = &v27[*(unsigned int *)(v2 + 144)];
      if ( v27 != v120 )
        break;
LABEL_98:
      v118 -= 8;
      if ( (_BYTE *)v116 == v118 )
        goto LABEL_99;
    }
    v131 = *(__int64 **)(v2 + 136);
    v28 = *((_QWORD *)v118 - 1);
    v121 = (unsigned __int64)(((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)) << 32;
    while ( 1 )
    {
      v29 = *(unsigned int *)(v2 + 88);
      v30 = *(_QWORD *)(v2 + 72);
      v31 = *v131;
      if ( (_DWORD)v29 )
      {
        v32 = 1;
        for ( i = (v29 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))) >> 31)
                 ^ (484763065 * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)))); ; i = (v29 - 1) & v35 )
        {
          v34 = v30 + 24LL * i;
          v26 = *(_QWORD **)v34;
          if ( v28 == *(_QWORD *)v34 && v31 == *(_QWORD *)(v34 + 8) )
            break;
          if ( v26 == (_QWORD *)-4096LL && *(_QWORD *)(v34 + 8) == -4096 )
            goto LABEL_40;
          v35 = v32 + i;
          ++v32;
        }
        v43 = 3 * v29;
        v37 = *(unsigned int *)(v2 + 56);
        v38 = *(_QWORD *)(v2 + 40);
        v39 = v30 + 8 * v43;
        v124 = v34 != v39;
        if ( !(_DWORD)v37 )
        {
LABEL_48:
          if ( v39 != v34 )
          {
LABEL_49:
            v122 = *(_DWORD *)(v34 + 16);
LABEL_50:
            v169 = 0;
            v162 = v164;
            v163 = 0x400000000LL;
            v170 = &v174;
            v171 = 8;
            v172 = 0;
            v173 = 1;
            v44 = *(__int64 **)(v28 + 64);
            v45 = &v44[*(unsigned int *)(v28 + 72)];
            if ( v44 != v45 )
            {
              v125 = v28;
              v46 = *(__int64 **)(v28 + 64);
              v47 = *v44;
LABEL_52:
              v48 = v170;
              v49 = HIDWORD(v171);
              v50 = (unsigned __int64)&v170[8 * HIDWORD(v171)];
              if ( v170 == (char *)v50 )
              {
LABEL_68:
                if ( HIDWORD(v171) >= (unsigned int)v171 )
                  goto LABEL_58;
                ++HIDWORD(v171);
                *(_QWORD *)v50 = v47;
                ++v169;
LABEL_59:
                v51 = sub_35D5240(v2, v47, v31);
                v49 = HIDWORD(v163);
                v52 = v51 | v129 & 0xFFFFFFFF00000000LL;
                v53 = (unsigned int)v163;
                v129 = v52;
                v50 = (unsigned int)v163 + 1LL;
                if ( v50 > HIDWORD(v163) )
                {
                  v117 = v52;
                  sub_C8D5F0((__int64)&v162, v164, v50, 0x10u, (__int64)v26, (__int64)v44);
                  v53 = (unsigned int)v163;
                  v52 = v117;
                }
                v54 = (__int64 *)&v162[16 * v53];
                *v54 = v47;
                v54[1] = v52;
                LODWORD(v163) = v163 + 1;
                if ( v125 == v47 && !v124 )
                {
                  v55 = *(unsigned int *)(v2 + 88);
                  v49 = *(_QWORD *)(v2 + 72);
                  if ( (_DWORD)v55 )
                  {
                    v56 = 1;
                    for ( j = (v55 - 1)
                            & (((0xBF58476D1CE4E5B9LL * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))) >> 31)
                             ^ (484763065 * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))));
                          ;
                          j = (v55 - 1) & v58 )
                    {
                      v50 = v49 + 24LL * j;
                      v26 = *(_QWORD **)v50;
                      if ( v125 == *(_QWORD *)v50 && v31 == *(_QWORD *)(v50 + 8) )
                        break;
                      if ( v26 == (_QWORD *)-4096LL && *(_QWORD *)(v50 + 8) == -4096 )
                        goto LABEL_144;
                      v58 = v56 + j;
                      ++v56;
                    }
                  }
                  else
                  {
LABEL_144:
                    v50 = v49 + 24 * v55;
                  }
                  v122 = *(_DWORD *)(v50 + 16);
                  v124 = v123;
                  goto LABEL_56;
                }
                goto LABEL_56;
              }
              while ( v47 != *(_QWORD *)v48 )
              {
                v48 += 8;
                if ( (char *)v50 == v48 )
                  goto LABEL_68;
              }
LABEL_56:
              while ( v45 != ++v46 )
              {
                v47 = *v46;
                if ( v173 )
                  goto LABEL_52;
LABEL_58:
                sub_C8CC70((__int64)&v169, v47, v50, v49, (__int64)v26, (__int64)v44);
                if ( (_BYTE)v50 )
                  goto LABEL_59;
              }
              v28 = v125;
              if ( (_DWORD)v163 )
              {
                v59 = 16LL * (unsigned int)v163;
                v60 = &v162[v59];
                v61 = v59 >> 4;
                v62 = v59 >> 6;
                if ( v62 )
                {
                  v63 = *((_DWORD *)v162 + 2);
                  v64 = &v162[64 * v62];
                  v65 = v162;
                  while ( 1 )
                  {
                    if ( v63 != *((_DWORD *)v65 + 6) )
                    {
                      v65 += 16;
                      goto LABEL_78;
                    }
                    if ( v63 != *((_DWORD *)v65 + 10) )
                      break;
                    if ( v63 != *((_DWORD *)v65 + 14) )
                    {
                      v65 += 48;
                      goto LABEL_78;
                    }
                    v65 += 64;
                    if ( v64 == v65 )
                    {
                      v61 = (v60 - v65) >> 4;
                      goto LABEL_103;
                    }
                    if ( v63 != *((_DWORD *)v65 + 2) )
                      goto LABEL_78;
                  }
                  v65 += 32;
LABEL_78:
                  v66 = v123;
                  if ( v60 != v65 )
                  {
LABEL_79:
                    if ( *(_BYTE *)v31 <= 0x1Cu )
                    {
                      v136[0] = 0;
                    }
                    else
                    {
                      v67 = *(_QWORD *)(v31 + 48);
                      v136[0] = v67;
                      if ( v67 )
                        sub_B96E90((__int64)v136, v67, 1);
                    }
                    v68 = 0;
                    v69 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)v2 + 16LL) + 128LL);
                    if ( v69 != sub_2DAC790 )
                      v68 = v69();
                    if ( !v66 )
                    {
                      v70 = *(_QWORD *)(v68 + 8) - 800LL;
                      v137 = v136[0];
                      if ( v136[0] )
                      {
                        sub_B96E90((__int64)&v137, v136[0], 1);
                        v148 = (unsigned __int8 *)v137;
                        if ( v137 )
                        {
                          sub_B976B0((__int64)&v137, (unsigned __int8 *)v137, (__int64)&v148);
                          v137 = 0;
                        }
                      }
                      else
                      {
                        v148 = 0;
                      }
                      v149 = 0;
                      v150 = 0;
                      v71 = (__int64 *)sub_2E311E0(v28);
                      v72 = sub_2F26260(v28, v71, (__int64 *)&v148, v70, v122);
                      v73 = *((_DWORD *)v162 + 2);
                      v156.m128i_i64[0] = 0;
                      v157 = 0;
                      v156.m128i_i32[2] = v73;
                      v158 = 0;
                      v159 = 0;
                      sub_2E8EAD0(v74, (__int64)v72, &v156);
                      if ( v148 )
                        sub_B91220((__int64)&v148, (__int64)v148);
                      if ( v137 )
                        sub_B91220((__int64)&v137, v137);
                      goto LABEL_92;
                    }
                    v76 = sub_2E79000(*(__int64 **)v2);
                    v77 = *(_QWORD *)(v2 + 16);
                    v78 = v76;
                    v79 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v77 + 32LL);
                    v126 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v77 + 552LL);
                    if ( v79 == sub_2D42F30 )
                    {
                      v80 = 2;
                      v81 = sub_AE2980(v78, 0)[1];
                      if ( v81 != 1 )
                      {
                        v80 = 3;
                        if ( v81 != 2 )
                        {
                          v80 = 4;
                          if ( v81 != 4 )
                          {
                            v80 = 5;
                            if ( v81 != 8 )
                            {
                              v80 = 6;
                              if ( v81 != 16 )
                              {
                                v80 = 7;
                                if ( v81 != 32 )
                                {
                                  v80 = 8;
                                  if ( v81 != 64 )
                                    v80 = 9 * (unsigned int)(v81 == 128);
                                }
                              }
                            }
                          }
                        }
                      }
                      if ( v126 == sub_2EC09E0 )
                      {
LABEL_127:
                        v84 = *(_QWORD *)(v77 + 8LL * (unsigned __int16)v80 + 112);
LABEL_128:
                        if ( !v124 )
                          v122 = sub_2EC06C0(*(_QWORD *)(*(_QWORD *)v2 + 32LL), v84, byte_3F871B3, 0, v82, v83);
                        v85 = *(_QWORD *)(v68 + 8);
                        v148 = (unsigned __int8 *)v136[0];
                        if ( v136[0] )
                        {
                          sub_B96E90((__int64)&v148, v136[0], 1);
                          v156.m128i_i64[0] = (__int64)v148;
                          if ( v148 )
                          {
                            sub_B976B0((__int64)&v148, v148, (__int64)&v156);
                            v148 = 0;
                          }
                        }
                        else
                        {
                          v156.m128i_i64[0] = 0;
                        }
                        v156.m128i_i64[1] = 0;
                        v157 = 0;
                        v86 = (__int64 *)sub_2E311E0(v28);
                        v87 = sub_2F26260(v28, v86, v156.m128i_i64, v85, v122);
                        v119 = v88;
                        v89 = (__int64)v87;
                        if ( v156.m128i_i64[0] )
                          sub_B91220((__int64)&v156, v156.m128i_i64[0]);
                        if ( v148 )
                          sub_B91220((__int64)&v148, (__int64)v148);
                        v90 = 16LL * (unsigned int)v163;
                        v127 = &v162[v90];
                        if ( &v162[v90] != v162 )
                        {
                          v115 = v2;
                          v91 = (__int64)v162;
                          do
                          {
                            v92 = *(_DWORD *)(v91 + 8);
                            v93 = *(_QWORD *)v91;
                            v156.m128i_i64[0] = 0;
                            v91 += 16;
                            v156.m128i_i32[2] = v92;
                            v157 = 0;
                            v158 = 0;
                            v159 = 0;
                            sub_2E8EAD0(v119, v89, &v156);
                            v156.m128i_i8[0] = 4;
                            v157 = 0;
                            v156.m128i_i32[0] &= 0xFFF000FF;
                            v158 = v93;
                            sub_2E8EAD0(v119, v89, &v156);
                          }
                          while ( v127 != (_BYTE *)v91 );
                          v2 = v115;
                        }
                        if ( !v124 )
                          sub_35D4CD0(v2, v28, v31, v122);
LABEL_92:
                        if ( v136[0] )
                          sub_B91220((__int64)v136, v136[0]);
                        if ( v173 )
                        {
LABEL_95:
                          if ( v162 != v164 )
                            _libc_free((unsigned __int64)v162);
                          goto LABEL_97;
                        }
LABEL_110:
                        _libc_free((unsigned __int64)v170);
                        goto LABEL_95;
                      }
                    }
                    else
                    {
                      v80 = (unsigned int)v79(v77, v78, 0);
                      if ( v126 == sub_2EC09E0 )
                        goto LABEL_127;
                    }
                    v84 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v126)(v77, v80, 0);
                    goto LABEL_128;
                  }
                }
                else
                {
                  v65 = v162;
LABEL_103:
                  switch ( v61 )
                  {
                    case 2LL:
                      v75 = *((_DWORD *)v162 + 2);
                      if ( *((_DWORD *)v65 + 2) != v75 )
                        goto LABEL_78;
                      v65 += 16;
                      break;
                    case 3LL:
                      v75 = *((_DWORD *)v162 + 2);
                      if ( *((_DWORD *)v65 + 2) != v75 )
                        goto LABEL_78;
                      v65 += 16;
                      if ( *((_DWORD *)v65 + 2) != v75 )
                        goto LABEL_78;
                      v65 += 16;
                      break;
                    case 1LL:
                      v75 = *((_DWORD *)v162 + 2);
                      break;
                    default:
                      goto LABEL_108;
                  }
                  if ( v75 != *((_DWORD *)v65 + 2) )
                    goto LABEL_78;
                }
              }
            }
LABEL_108:
            v66 = 0;
            if ( v124 )
              goto LABEL_79;
            sub_35D4CD0(v2, v28, v31, *((_DWORD *)v162 + 2));
            if ( v173 )
              goto LABEL_95;
            goto LABEL_110;
          }
LABEL_203:
          v122 = 0;
          goto LABEL_50;
        }
      }
      else
      {
LABEL_40:
        v36 = 3 * v29;
        v37 = *(unsigned int *)(v2 + 56);
        v38 = *(_QWORD *)(v2 + 40);
        v34 = v30 + 8 * v36;
        if ( !(_DWORD)v37 )
        {
          v124 = 0;
          goto LABEL_203;
        }
        v124 = 0;
        v39 = v30 + 8 * v36;
      }
      v40 = 1;
      for ( k = (v37 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4))) >> 31)
               ^ (484763065 * (v121 | ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)))); ; k = (v37 - 1) & v42 )
      {
        v26 = (_QWORD *)(v38 + 24LL * k);
        if ( v28 == *v26 && v31 == v26[1] )
          break;
        if ( *v26 == -4096 && v26[1] == -4096 )
          goto LABEL_48;
        v42 = v40 + k;
        ++v40;
      }
      if ( v39 != v34 )
        goto LABEL_49;
      if ( v26 == (_QWORD *)(v38 + 24 * v37) )
        goto LABEL_203;
LABEL_97:
      if ( v120 == ++v131 )
        goto LABEL_98;
    }
  }
LABEL_99:
  if ( !*(_DWORD *)(v2 + 80) )
    goto LABEL_100;
  v102 = *(_QWORD *)(*(_QWORD *)v2 + 32LL);
  v103 = *(_QWORD **)(v2 + 72);
  v104 = &v103[3 * *(unsigned int *)(v2 + 88)];
  if ( v103 == v104 )
    goto LABEL_100;
  while ( 1 )
  {
    v105 = *v103;
    v106 = v103;
    if ( *v103 != -4096 )
      break;
    if ( v103[1] != -4096 )
      goto LABEL_178;
LABEL_205:
    v103 += 3;
    if ( v104 == v103 )
      goto LABEL_100;
  }
  if ( v105 == -8192 && v103[1] == -8192 )
    goto LABEL_205;
LABEL_178:
  if ( v104 == v103 )
    goto LABEL_100;
  v107 = (_QWORD *)v2;
  v108 = v104;
  v109 = v107;
  while ( 2 )
  {
    v111 = *((_DWORD *)v106 + 4);
    if ( v111 >= 0 )
    {
      v110 = *(_QWORD *)(*(_QWORD *)(v102 + 304) + 8LL * (unsigned int)v111);
      if ( !v110 )
        goto LABEL_189;
    }
    else
    {
      v110 = *(_QWORD *)(*(_QWORD *)(v102 + 56) + 16LL * (v111 & 0x7FFFFFFF) + 8);
      if ( !v110 )
        goto LABEL_189;
    }
    if ( (*(_BYTE *)(v110 + 3) & 0x10) == 0 )
    {
      v114 = *(_QWORD *)(v110 + 32);
      if ( !v114 || (*(_BYTE *)(v114 + 3) & 0x10) == 0 )
      {
LABEL_189:
        v112 = *(_QWORD *)(v109[3] + 8LL);
        v132 = *(_QWORD *)(*(_QWORD *)(*v109 + 96LL) + 8LL * *(unsigned int *)(v105 + 24));
        v162 = 0;
        v169 = 0;
        v130 = v112 - 400;
        v170 = 0;
        v171 = 0;
        v113 = (__int64 *)sub_2E311E0(v132);
        sub_2F26260(v132, v113, &v169, v130, v111);
        if ( v169 )
          sub_B91220((__int64)&v169, v169);
        if ( v162 )
          sub_B91220((__int64)&v162, (__int64)v162);
      }
    }
    v106 += 3;
    if ( v106 == v108 )
      break;
    while ( 2 )
    {
      if ( *v106 == -4096 )
      {
        if ( v106[1] != -4096 )
          break;
        goto LABEL_194;
      }
      if ( *v106 == -8192 && v106[1] == -8192 )
      {
LABEL_194:
        v106 += 3;
        if ( v108 == v106 )
          goto LABEL_100;
        continue;
      }
      break;
    }
    if ( v106 != v108 )
    {
      v105 = *v106;
      continue;
    }
    break;
  }
LABEL_100:
  if ( v133 != v135 )
    _libc_free((unsigned __int64)v133);
}
