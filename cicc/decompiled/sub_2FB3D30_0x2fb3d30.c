// Function: sub_2FB3D30
// Address: 0x2fb3d30
//
void __fastcall sub_2FB3D30(__int64 a1, char a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  const __m128i *v9; // r14
  const __m128i *v10; // r15
  const __m128i *v11; // r12
  __int64 v12; // r14
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  __int64 i; // rdi
  __int16 v17; // dx
  __int64 v18; // rdi
  unsigned int v19; // esi
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  __int64 v22; // r11
  signed __int64 v23; // rax
  __int8 v24; // dl
  signed __int64 v25; // r13
  int v26; // eax
  __int64 v27; // r9
  int v28; // esi
  __int64 v29; // rax
  int v30; // edx
  unsigned __int64 v31; // rsi
  _QWORD *v32; // rcx
  __int64 v33; // r8
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r11
  __int64 v37; // r9
  __int8 v38; // al
  __int64 *v39; // r11
  unsigned __int64 v40; // rcx
  __int64 v41; // r14
  __int64 *v42; // rax
  unsigned __int64 v43; // r14
  __int64 v44; // r13
  __int64 v45; // r8
  unsigned __int64 v46; // rax
  unsigned int v47; // edx
  __int64 v48; // r12
  __int64 v49; // r15
  unsigned __int16 v50; // ax
  __int64 v51; // r12
  __int64 *v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // r15
  __int64 v56; // rax
  _QWORD *v57; // r12
  _QWORD *v58; // r13
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // r15
  int *v64; // r13
  __int64 v65; // r12
  __int64 v66; // r9
  __int64 v67; // r15
  unsigned __int64 v68; // rdx
  unsigned int v69; // eax
  __int64 v70; // r14
  unsigned int v71; // eax
  __int64 v72; // rcx
  __int64 *v73; // r14
  __int64 v74; // rax
  unsigned int v75; // ecx
  __int64 v76; // rcx
  __int64 v77; // r11
  __int64 *v78; // rax
  int v79; // edx
  _QWORD *v80; // r8
  __int64 v81; // rcx
  __int64 v82; // r9
  __int64 v83; // rdx
  unsigned int v84; // ecx
  __int64 v85; // rcx
  __int64 *v86; // r12
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 v90; // rcx
  unsigned __int64 v91; // rdx
  __int64 v92; // rax
  __m128i *v93; // rax
  __m128i *v94; // rdx
  __m128i v95; // xmm3
  __int64 v96; // r12
  unsigned __int64 v97; // r10
  _QWORD *v98; // rax
  _QWORD *v99; // rsi
  __int64 v100; // r15
  __int64 v101; // r9
  _QWORD *v102; // rdx
  _QWORD *v103; // rsi
  __int64 v104; // r11
  unsigned __int64 v105; // r10
  _QWORD *v106; // rdx
  _QWORD *v107; // rdi
  __int64 v108; // rcx
  unsigned int v109; // eax
  unsigned int v110; // eax
  int v111; // r8d
  __int8 *v112; // r12
  __int64 v113; // [rsp+8h] [rbp-478h]
  unsigned __int64 v114; // [rsp+10h] [rbp-470h]
  __int64 v115; // [rsp+18h] [rbp-468h]
  _BYTE *v116; // [rsp+28h] [rbp-458h]
  int v117; // [rsp+38h] [rbp-448h]
  __int64 v118; // [rsp+40h] [rbp-440h]
  __int64 *v119; // [rsp+48h] [rbp-438h]
  __int64 v120; // [rsp+48h] [rbp-438h]
  _QWORD *v121; // [rsp+48h] [rbp-438h]
  int v122; // [rsp+48h] [rbp-438h]
  int v123; // [rsp+48h] [rbp-438h]
  unsigned __int64 v125; // [rsp+50h] [rbp-430h]
  __int64 v126; // [rsp+50h] [rbp-430h]
  __int64 v127; // [rsp+58h] [rbp-428h]
  __int64 *v128; // [rsp+58h] [rbp-428h]
  __int64 v129; // [rsp+58h] [rbp-428h]
  __int64 v130; // [rsp+58h] [rbp-428h]
  __int64 *v131; // [rsp+58h] [rbp-428h]
  __int64 v132; // [rsp+58h] [rbp-428h]
  int v133; // [rsp+58h] [rbp-428h]
  __int64 *v134; // [rsp+60h] [rbp-420h] BYREF
  __int64 v135; // [rsp+68h] [rbp-418h]
  _BYTE v136[32]; // [rsp+70h] [rbp-410h] BYREF
  unsigned __int64 v137; // [rsp+90h] [rbp-3F0h] BYREF
  __int64 v138; // [rsp+98h] [rbp-3E8h]
  _BYTE v139[224]; // [rsp+A0h] [rbp-3E0h] BYREF
  __m128i v140; // [rsp+180h] [rbp-300h] BYREF
  __m128i v141; // [rsp+190h] [rbp-2F0h]
  __int64 v142; // [rsp+1A0h] [rbp-2E0h]
  _BYTE *v143; // [rsp+1A8h] [rbp-2D8h]
  __int64 v144; // [rsp+1B0h] [rbp-2D0h]
  _BYTE v145[48]; // [rsp+1B8h] [rbp-2C8h] BYREF
  int v146; // [rsp+1E8h] [rbp-298h]
  __int64 v147; // [rsp+1F0h] [rbp-290h]
  _QWORD *v148; // [rsp+1F8h] [rbp-288h]
  __int64 v149; // [rsp+200h] [rbp-280h]
  unsigned int v150; // [rsp+208h] [rbp-278h]
  _QWORD *v151; // [rsp+210h] [rbp-270h]
  __int64 v152; // [rsp+218h] [rbp-268h]
  _QWORD v153[3]; // [rsp+220h] [rbp-260h] BYREF
  _BYTE *v154; // [rsp+238h] [rbp-248h]
  __int64 v155; // [rsp+240h] [rbp-240h]
  _BYTE v156[568]; // [rsp+248h] [rbp-238h] BYREF

  v6 = *(_QWORD *)(a1 + 72);
  v137 = (unsigned __int64)v139;
  v7 = *(_QWORD *)(a1 + 24);
  v138 = 0x400000000LL;
  v8 = *(unsigned int *)(*(_QWORD *)(v6 + 8) + 112LL);
  if ( (int)v8 < 0 )
  {
    v9 = *(const __m128i **)(*(_QWORD *)(v7 + 56) + 16 * (v8 & 0x7FFFFFFF) + 8);
    if ( v9 )
      goto LABEL_3;
LABEL_120:
    v108 = *(_QWORD *)(v6 + 16);
    v63 = *(_QWORD *)v108 + 4LL * *(unsigned int *)(v6 + 64);
    v130 = *(_QWORD *)v108 + 4LL * *(unsigned int *)(v108 + 8);
    if ( v130 != v63 )
      goto LABEL_59;
    return;
  }
  v9 = *(const __m128i **)(*(_QWORD *)(v7 + 304) + 8 * v8);
  if ( !v9 )
    goto LABEL_120;
LABEL_3:
  v10 = v9;
  v118 = a1 + 192;
  do
  {
    while ( 1 )
    {
      v11 = v10;
      v10 = (const __m128i *)v10[2].m128i_i64[0];
      v12 = v11[1].m128i_i64[0];
      if ( (unsigned __int16)(*(_WORD *)(v12 + 68) - 14) <= 1u )
      {
        sub_2EAB0C0((__int64)v11, 0);
        goto LABEL_5;
      }
      v13 = v11[1].m128i_u64[0];
      v14 = v13;
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
      if ( (*(_DWORD *)(v12 + 44) & 4) != 0 )
      {
        do
          v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v14 + 44) & 4) != 0 );
      }
      if ( (*(_DWORD *)(v12 + 44) & 8) != 0 )
      {
        do
          v13 = *(_QWORD *)(v13 + 8);
        while ( (*(_BYTE *)(v13 + 44) & 8) != 0 );
      }
      for ( i = *(_QWORD *)(v13 + 8); i != v14; v14 = *(_QWORD *)(v14 + 8) )
      {
        v17 = *(_WORD *)(v14 + 68);
        if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
          break;
      }
      v18 = *(_QWORD *)(v15 + 128);
      v19 = *(_DWORD *)(v15 + 144);
      if ( !v19 )
        goto LABEL_73;
      v20 = (v19 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( v14 != *v21 )
      {
        v79 = 1;
        while ( v22 != -4096 )
        {
          v111 = v79 + 1;
          v20 = (v19 - 1) & (v79 + v20);
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( *v21 == v14 )
            goto LABEL_17;
          v79 = v111;
        }
LABEL_73:
        v21 = (__int64 *)(v18 + 16LL * v19);
      }
LABEL_17:
      v23 = v21[1];
      v24 = v11->m128i_i8[4];
      v25 = v23;
      if ( (v11->m128i_i8[3] & 0x10) != 0 || (v24 & 1) != 0 )
        v25 = v23 & 0xFFFFFFFFFFFFFFF8LL | ((v24 & 4) == 0 ? 4LL : 2LL);
      v127 = *(_QWORD *)(a1 + 8);
      v26 = sub_2FB3BF0(v118, v25, 0);
      v27 = v127;
      v28 = v26;
      v117 = v26;
      v29 = *(_QWORD *)(a1 + 72);
      v30 = v28;
      v31 = *(unsigned int *)(v127 + 160);
      v32 = *(_QWORD **)(v29 + 16);
      v33 = *(unsigned int *)(*v32 + 4LL * (unsigned int)(*(_DWORD *)(v29 + 64) + v30));
      v34 = *(_DWORD *)(*v32 + 4LL * (unsigned int)(*(_DWORD *)(v29 + 64) + v30)) & 0x7FFFFFFF;
      v35 = 8LL * v34;
      if ( v34 >= (unsigned int)v31 || (v35 = 8LL * v34, (v36 = *(_QWORD *)(*(_QWORD *)(v127 + 152) + v35)) == 0) )
      {
        v75 = v34 + 1;
        if ( (unsigned int)v31 < v34 + 1 && v75 != v31 )
        {
          if ( v75 >= v31 )
          {
            v104 = *(_QWORD *)(v127 + 168);
            v105 = v75 - v31;
            if ( v75 > (unsigned __int64)*(unsigned int *)(v127 + 164) )
            {
              v113 = v35;
              v114 = v75 - v31;
              v115 = *(_QWORD *)(v127 + 168);
              v123 = v33;
              sub_C8D5F0(v127 + 152, (const void *)(v127 + 168), v75, 8u, v33, v127);
              v27 = v127;
              v35 = v113;
              v105 = v114;
              v104 = v115;
              v31 = *(unsigned int *)(v127 + 160);
              LODWORD(v33) = v123;
            }
            v76 = *(_QWORD *)(v27 + 152);
            v106 = (_QWORD *)(v76 + 8 * v31);
            v107 = &v106[v105];
            if ( v106 != v107 )
            {
              do
                *v106++ = v104;
              while ( v107 != v106 );
              LODWORD(v31) = *(_DWORD *)(v27 + 160);
              v76 = *(_QWORD *)(v27 + 152);
            }
            *(_DWORD *)(v27 + 160) = v105 + v31;
            goto LABEL_70;
          }
          *(_DWORD *)(v127 + 160) = v75;
        }
        v76 = *(_QWORD *)(v127 + 152);
LABEL_70:
        v121 = (_QWORD *)v27;
        v131 = (__int64 *)(v76 + v35);
        v77 = sub_2E10F30(v33);
        v78 = v131;
        v132 = v77;
        *v78 = v77;
        sub_2E11E80(v121, v77);
        v36 = v132;
      }
      v128 = (__int64 *)v36;
      sub_2EAB0C0((__int64)v11, *(_DWORD *)(v36 + 112));
      if ( !a2 )
        goto LABEL_5;
      v38 = v11->m128i_i8[4];
      if ( (v38 & 1) != 0 )
        goto LABEL_5;
      v39 = v128;
      if ( (v11->m128i_i8[3] & 0x10) == 0 )
        break;
      if ( (v11->m128i_i32[0] & 0xFFF00) != 0 || (v38 & 4) != 0 )
      {
        v40 = v25 & 0xFFFFFFFFFFFFFFF8LL;
        v41 = ((v25 >> 1) & 3) != 0
            ? v40 | (2LL * (int)(((v25 >> 1) & 3) - 1))
            : *(_QWORD *)v40 & 0xFFFFFFFFFFFFFFF8LL | 6;
        v129 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
        v119 = v39;
        v42 = (__int64 *)sub_2E09D00((__int64 *)v129, v41);
        if ( v42 != (__int64 *)(*(_QWORD *)v129 + 24LL * *(unsigned int *)(v129 + 8)) )
        {
          v39 = v119;
          if ( (*(_DWORD *)((*v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v42 >> 1) & 3)) <= (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) )
          {
            if ( !v119[13] )
              goto LABEL_32;
LABEL_94:
            if ( (v11->m128i_i8[3] & 0x10) == 0 )
            {
              v90 = (unsigned int)v138;
              v91 = v137;
              v140 = _mm_loadu_si128(v11);
              a5 = (unsigned int)v138 + 1LL;
              v141 = _mm_loadu_si128(v11 + 1);
              v92 = v11[2].m128i_i64[0];
              v144 = v25;
              v142 = v92;
              LODWORD(v143) = v117;
              v93 = &v140;
              if ( a5 > HIDWORD(v138) )
              {
                if ( v137 > (unsigned __int64)&v140 || (unsigned __int64)&v140 >= v137 + 56LL * (unsigned int)v138 )
                {
                  sub_C8D5F0((__int64)&v137, v139, (unsigned int)v138 + 1LL, 0x38u, a5, v37);
                  v91 = v137;
                  v90 = (unsigned int)v138;
                  v93 = &v140;
                }
                else
                {
                  v112 = &v140.m128i_i8[-v137];
                  sub_C8D5F0((__int64)&v137, v139, (unsigned int)v138 + 1LL, 0x38u, a5, v37);
                  v91 = v137;
                  v90 = (unsigned int)v138;
                  v93 = (__m128i *)&v112[v137];
                }
              }
              v94 = (__m128i *)(v91 + 56 * v90);
              *v94 = _mm_loadu_si128(v93);
              v95 = _mm_loadu_si128(v93 + 1);
              LODWORD(v138) = v138 + 1;
              v94[1] = v95;
              v94[2] = _mm_loadu_si128(v93 + 2);
              v94[3].m128i_i64[0] = v93[3].m128i_i64[0];
            }
          }
        }
      }
LABEL_5:
      if ( !v10 )
        goto LABEL_33;
    }
    if ( (v11->m128i_i16[1] & 0xFF0) != 0
      && (v109 = sub_2EAB0A0((__int64)v11),
          v110 = sub_2E89F40(v12, v109),
          v39 = v128,
          (*(_BYTE *)(*(_QWORD *)(v12 + 32) + 40LL * v110 + 4) & 4) != 0) )
    {
      v89 = 2;
    }
    else
    {
      v89 = 4;
    }
    v25 = v89 | v25 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v39[13] )
      goto LABEL_94;
LABEL_32:
    sub_2E20270(
      (_QWORD *)(a1 + 712LL * ((*(_DWORD *)(a1 + 84) != 0) & (unsigned __int8)(v117 != 0)) + 424),
      v39,
      v25,
      0,
      0,
      0);
  }
  while ( v10 );
LABEL_33:
  v43 = v137;
  v116 = (_BYTE *)(v137 + 56LL * (unsigned int)v138);
  if ( v116 != (_BYTE *)v137 )
  {
    while ( 1 )
    {
      v44 = *(_QWORD *)(a1 + 8);
      v45 = *(unsigned int *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                            + 4LL * (unsigned int)(*(_DWORD *)(v43 + 40) + *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)));
      v46 = *(unsigned int *)(v44 + 160);
      v47 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                      + 4LL * (unsigned int)(*(_DWORD *)(v43 + 40) + *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)))
          & 0x7FFFFFFF;
      v48 = 8LL * v47;
      if ( v47 >= (unsigned int)v46 )
        break;
      v49 = *(_QWORD *)(*(_QWORD *)(v44 + 152) + 8LL * v47);
      if ( !v49 )
        break;
LABEL_36:
      v140 = 0u;
      v143 = v145;
      v144 = 0x600000000LL;
      v141 = 0u;
      v151 = v153;
      v154 = v156;
      v142 = 0;
      v146 = 0;
      v147 = 0;
      v148 = 0;
      v149 = 0;
      v150 = 0;
      v152 = 0;
      v153[0] = 0;
      v153[1] = 0;
      v155 = 0x1000000000LL;
      v50 = (*(_DWORD *)v43 >> 8) & 0xFFF;
      if ( !v50 )
      {
        v53 = sub_2EBF1E0(*(_QWORD *)(a1 + 24), *(_DWORD *)(v43 + 8));
        v51 = *(_QWORD *)(v49 + 104);
        v54 = v88;
        if ( v51 )
        {
LABEL_38:
          v120 = v49;
          v55 = v53;
          do
          {
            if ( v55 & *(_QWORD *)(v51 + 112) | v54 & *(_QWORD *)(v51 + 120) )
            {
              if ( *(_DWORD *)(v51 + 8) )
              {
                sub_2E1DCC0(
                  (__int64)&v140,
                  *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL),
                  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL),
                  *(_QWORD *)(a1 + 32),
                  *(_QWORD *)(a1 + 8) + 56LL,
                  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
                v80 = *(_QWORD **)(a1 + 24);
                v81 = *(_QWORD *)(v51 + 120);
                v82 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
                v83 = *(_QWORD *)(v51 + 112);
                v134 = (__int64 *)v136;
                v135 = 0x400000000LL;
                sub_2E0B070(v120, (__int64)&v134, v83, v81, v80, v82);
                sub_2E20270(&v140, (__int64 *)v51, *(_QWORD *)(v43 + 48), 0, v134, (unsigned int)v135);
                if ( v134 != (__int64 *)v136 )
                  _libc_free((unsigned __int64)v134);
              }
            }
            v51 = *(_QWORD *)(v51 + 104);
          }
          while ( v51 );
        }
        if ( v154 != v156 )
          _libc_free((unsigned __int64)v154);
        goto LABEL_43;
      }
      v51 = *(_QWORD *)(v49 + 104);
      v52 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 272LL) + 16LL * v50);
      v53 = *v52;
      v54 = v52[1];
      if ( v51 )
        goto LABEL_38;
LABEL_43:
      if ( v151 != v153 )
        _libc_free((unsigned __int64)v151);
      v56 = v150;
      if ( v150 )
      {
        v57 = v148;
        v58 = &v148[19 * v150];
        do
        {
          if ( *v57 != -8192 && *v57 != -4096 )
          {
            v59 = v57[10];
            if ( (_QWORD *)v59 != v57 + 12 )
              _libc_free(v59);
            v60 = v57[1];
            if ( (_QWORD *)v60 != v57 + 3 )
              _libc_free(v60);
          }
          v57 += 19;
        }
        while ( v58 != v57 );
        v56 = v150;
      }
      sub_C7D6A0((__int64)v148, 152 * v56, 8);
      if ( v143 != v145 )
        _libc_free((unsigned __int64)v143);
      v43 += 56LL;
      if ( (_BYTE *)v43 == v116 )
        goto LABEL_58;
    }
    v84 = v47 + 1;
    if ( (unsigned int)v46 >= v47 + 1 || v84 == v46 )
    {
LABEL_78:
      v85 = *(_QWORD *)(v44 + 152);
    }
    else
    {
      if ( v84 < v46 )
      {
        *(_DWORD *)(v44 + 160) = v84;
        goto LABEL_78;
      }
      v100 = *(_QWORD *)(v44 + 168);
      v101 = v84 - v46;
      if ( v84 > (unsigned __int64)*(unsigned int *)(v44 + 164) )
      {
        v126 = v84 - v46;
        v133 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                         + 4LL * (unsigned int)(*(_DWORD *)(v43 + 40) + *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL)));
        sub_C8D5F0(v44 + 152, (const void *)(v44 + 168), v84, 8u, v45, v101);
        v46 = *(unsigned int *)(v44 + 160);
        v101 = v126;
        LODWORD(v45) = v133;
      }
      v85 = *(_QWORD *)(v44 + 152);
      v102 = (_QWORD *)(v85 + 8 * v46);
      v103 = &v102[v101];
      if ( v102 != v103 )
      {
        do
          *v102++ = v100;
        while ( v103 != v102 );
        LODWORD(v46) = *(_DWORD *)(v44 + 160);
        v85 = *(_QWORD *)(v44 + 152);
      }
      *(_DWORD *)(v44 + 160) = v101 + v46;
    }
    v86 = (__int64 *)(v85 + v48);
    v87 = sub_2E10F30(v45);
    *v86 = v87;
    v49 = v87;
    sub_2E11E80((_QWORD *)v44, v87);
    goto LABEL_36;
  }
LABEL_58:
  v61 = *(_QWORD *)(a1 + 72);
  v62 = *(_QWORD *)(v61 + 16);
  v63 = *(_QWORD *)v62 + 4LL * *(unsigned int *)(v61 + 64);
  v130 = *(_QWORD *)v62 + 4LL * *(unsigned int *)(v62 + 8);
  if ( v130 != v63 )
  {
LABEL_59:
    v64 = (int *)v63;
    while ( 2 )
    {
      v66 = (unsigned int)*v64;
      v67 = *(_QWORD *)(a1 + 8);
      v68 = *(unsigned int *)(v67 + 160);
      v69 = *v64 & 0x7FFFFFFF;
      v70 = 8LL * v69;
      if ( v69 < (unsigned int)v68 )
      {
        v65 = *(_QWORD *)(*(_QWORD *)(v67 + 152) + 8LL * v69);
        if ( v65 )
          goto LABEL_61;
      }
      v71 = v69 + 1;
      if ( (unsigned int)v68 >= v71 || v71 == v68 )
      {
LABEL_66:
        v72 = *(_QWORD *)(v67 + 152);
      }
      else
      {
        if ( v71 < v68 )
        {
          *(_DWORD *)(v67 + 160) = v71;
          goto LABEL_66;
        }
        v96 = *(_QWORD *)(v67 + 168);
        v97 = v71 - v68;
        if ( v71 > (unsigned __int64)*(unsigned int *)(v67 + 164) )
        {
          v122 = *v64;
          v125 = v71 - v68;
          sub_C8D5F0(v67 + 152, (const void *)(v67 + 168), v71, 8u, a5, v66);
          v68 = *(unsigned int *)(v67 + 160);
          LODWORD(v66) = v122;
          v97 = v125;
        }
        v72 = *(_QWORD *)(v67 + 152);
        v98 = (_QWORD *)(v72 + 8 * v68);
        v99 = &v98[v97];
        if ( v98 != v99 )
        {
          do
            *v98++ = v96;
          while ( v99 != v98 );
          LODWORD(v68) = *(_DWORD *)(v67 + 160);
          v72 = *(_QWORD *)(v67 + 152);
        }
        *(_DWORD *)(v67 + 160) = v97 + v68;
      }
      v73 = (__int64 *)(v72 + v70);
      v74 = sub_2E10F30(v66);
      *v73 = v74;
      v65 = v74;
      sub_2E11E80((_QWORD *)v67, v74);
LABEL_61:
      if ( *(_QWORD *)(v65 + 104) )
      {
        *(_DWORD *)(v65 + 72) = 0;
        *(_DWORD *)(v65 + 8) = 0;
        sub_2E0AF60(v65);
        sub_2E15850(*(_QWORD **)(a1 + 8), v65);
      }
      if ( (int *)v130 == ++v64 )
        break;
      continue;
    }
  }
  if ( (_BYTE *)v137 != v139 )
    _libc_free(v137);
}
