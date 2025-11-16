// Function: sub_3766890
// Address: 0x3766890
//
void __fastcall sub_3766890(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __int64 a5, __int64 a6, int a7)
{
  int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rax
  bool v13; // r14
  const __m128i *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdx
  __m128i v17; // xmm1
  __int64 v18; // rax
  __int64 v19; // rax
  __int16 *v20; // rax
  __int16 v21; // bx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdi
  _QWORD *v25; // r8
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rbx
  unsigned __int16 v30; // ax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r11
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdi
  __m128i v37; // rax
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  __m128i v40; // xmm7
  __m128i v41; // xmm7
  unsigned __int8 *v42; // rax
  _QWORD *v43; // rdi
  __m128i v44; // xmm3
  __m128i v45; // xmm2
  __int32 v46; // edx
  unsigned __int8 *v47; // rbx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  unsigned int v51; // edx
  unsigned __int8 *v52; // r12
  unsigned __int64 v53; // rdx
  unsigned __int8 **v54; // rax
  unsigned __int64 v55; // rcx
  __int64 v56; // rax
  unsigned __int8 **v57; // rax
  unsigned int v58; // edx
  __int64 v59; // rdx
  unsigned __int8 *v60; // rax
  __int64 v61; // r9
  unsigned __int8 *v62; // rdx
  unsigned __int8 *v63; // r15
  __int64 v64; // rdx
  unsigned __int8 *v65; // r14
  unsigned __int8 **v66; // rdx
  __int64 v67; // rax
  __m128i v68; // xmm0
  __int64 v69; // rax
  __m128i v70; // xmm0
  __int64 v71; // rax
  __int64 v72; // rdi
  __m128i v73; // rax
  __int64 v74; // rsi
  __int128 v75; // rax
  __int64 v76; // rdi
  __m128i v77; // xmm0
  __m128i v78; // rax
  _QWORD *v79; // rdi
  __int64 v80; // r9
  __m128i v81; // rax
  _QWORD *v82; // rdi
  __int64 v83; // r9
  __m128i v84; // rax
  int v85; // r9d
  __int64 v86; // rdi
  __int128 v87; // rax
  __int64 v88; // r15
  __int64 v89; // rdx
  unsigned __int64 v90; // r15
  __int128 v91; // rax
  unsigned __int8 *v92; // rax
  __int64 v93; // r9
  unsigned __int8 *v94; // rdx
  unsigned __int8 *v95; // r15
  __int64 v96; // rdx
  unsigned __int8 *v97; // r14
  unsigned __int8 **v98; // rdx
  __int64 v99; // rsi
  int v100; // esi
  __int16 v101; // ax
  int v102; // r9d
  unsigned __int64 v103; // rcx
  _QWORD *v104; // rdi
  __m128i v105; // xmm3
  __m128i v106; // xmm2
  __m128i v107; // rax
  _QWORD *v108; // rdi
  __int64 v109; // r14
  __m128i v110; // xmm4
  __m128i v111; // xmm6
  __int64 v112; // rdx
  const __m128i *v113; // rax
  _QWORD *v114; // rdi
  __m128i v115; // xmm5
  unsigned __int8 *v116; // r14
  __int64 v117; // rdx
  __int64 v118; // r15
  __m128i v119; // rax
  _QWORD *v120; // rdi
  unsigned __int8 *v121; // rdx
  __int64 v122; // rdx
  __int64 v123; // r8
  __int64 v124; // r9
  unsigned __int8 *v125; // r12
  __int64 v126; // rax
  unsigned int v127; // edx
  unsigned __int8 *v128; // rbx
  unsigned __int8 **v129; // rax
  __int128 v130; // [rsp-20h] [rbp-1A0h]
  __int128 v131; // [rsp-20h] [rbp-1A0h]
  __int128 v132; // [rsp-20h] [rbp-1A0h]
  __int128 v133; // [rsp-10h] [rbp-190h]
  __int128 v134; // [rsp-10h] [rbp-190h]
  __int128 v135; // [rsp-10h] [rbp-190h]
  __int128 v136; // [rsp+0h] [rbp-180h]
  __int128 v137; // [rsp+0h] [rbp-180h]
  __int128 v138; // [rsp+0h] [rbp-180h]
  __int128 v139; // [rsp+0h] [rbp-180h]
  __m128i v140; // [rsp+10h] [rbp-170h] BYREF
  __m128i v141; // [rsp+20h] [rbp-160h] BYREF
  __int128 v142; // [rsp+30h] [rbp-150h]
  __m128i v143; // [rsp+40h] [rbp-140h] BYREF
  __m128i v144; // [rsp+50h] [rbp-130h] BYREF
  __int64 v145; // [rsp+68h] [rbp-118h]
  __m128i v146; // [rsp+70h] [rbp-110h] BYREF
  unsigned __int8 *v147; // [rsp+80h] [rbp-100h]
  __int64 v148; // [rsp+88h] [rbp-F8h]
  unsigned __int8 *v149; // [rsp+90h] [rbp-F0h]
  __int64 v150; // [rsp+98h] [rbp-E8h]
  unsigned __int8 *v151; // [rsp+A0h] [rbp-E0h]
  __int64 v152; // [rsp+A8h] [rbp-D8h]
  unsigned int v153; // [rsp+B0h] [rbp-D0h]
  __int64 v154; // [rsp+B8h] [rbp-C8h]
  __int64 v155; // [rsp+C0h] [rbp-C0h] BYREF
  int v156; // [rsp+C8h] [rbp-B8h]
  __m128i v157; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v158; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v159; // [rsp+F0h] [rbp-90h]
  __int64 v160; // [rsp+F8h] [rbp-88h]
  __int64 v161; // [rsp+100h] [rbp-80h] BYREF
  __int64 v162; // [rsp+108h] [rbp-78h]
  __int16 v163; // [rsp+110h] [rbp-70h]
  __int64 v164; // [rsp+118h] [rbp-68h]
  __m128i v165; // [rsp+120h] [rbp-60h] BYREF
  __m128i v166; // [rsp+130h] [rbp-50h]
  __m128i v167; // [rsp+140h] [rbp-40h]

  v10 = *(_DWORD *)(a2 + 24);
  if ( v10 > 239 )
  {
    v58 = v10 - 242;
    v12 = (unsigned int)(v10 - 242) < 2 ? 0x28 : 0;
    v13 = v58 < 2;
  }
  else if ( v10 > 237 )
  {
    v12 = 40;
    v13 = 1;
  }
  else
  {
    v11 = v10 - 101;
    v12 = (unsigned int)(v10 - 101) < 0x30 ? 0x28 : 0;
    v13 = v11 < 0x30;
  }
  v14 = (const __m128i *)(*(_QWORD *)(a2 + 40) + v12);
  v15 = *(_QWORD *)(a2 + 80);
  v16 = v14->m128i_i64[0];
  v17 = _mm_loadu_si128(v14);
  v18 = v14->m128i_u32[2];
  v144 = v17;
  v19 = *(_QWORD *)(v16 + 48) + 16 * v18;
  LOWORD(v16) = *(_WORD *)v19;
  v154 = *(_QWORD *)(v19 + 8);
  v20 = *(__int16 **)(a2 + 48);
  LOWORD(v153) = v16;
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v155 = v15;
  LOWORD(v145) = v21;
  v146.m128i_i64[0] = v22;
  if ( v15 )
    sub_B96E90((__int64)&v155, v15, 1);
  v23 = *(_DWORD *)(a2 + 72);
  v24 = a1[1];
  v25 = (_QWORD *)*a1;
  v157.m128i_i64[0] = 0;
  v156 = v23;
  v157.m128i_i32[2] = 0;
  v158.m128i_i64[0] = 0;
  v158.m128i_i32[2] = 0;
  if ( (unsigned __int8)sub_3451C50(v24, a2, (__int64)&v157, a4, (__int64)&v158, v25, a7) )
  {
    v67 = *(unsigned int *)(a3 + 8);
    v68 = _mm_load_si128(&v157);
    if ( v67 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v146 = v68;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v67 + 1, 0x10u, v27, v28);
      v67 = *(unsigned int *)(a3 + 8);
      v68 = _mm_load_si128(&v146);
    }
    *(__m128i *)(*(_QWORD *)a3 + 16 * v67) = v68;
    v69 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = v69;
    if ( v13 )
    {
      v70 = _mm_load_si128(&v158);
      if ( v69 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v146 = v70;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v69 + 1, 0x10u, v27, v28);
        v69 = *(unsigned int *)(a3 + 8);
        v70 = _mm_load_si128(&v146);
      }
      *(__m128i *)(*(_QWORD *)a3 + 16 * v69) = v70;
      ++*(_DWORD *)(a3 + 8);
    }
    goto LABEL_30;
  }
  v29 = a1[1];
  v30 = v153;
  if ( v13 )
  {
    if ( !(_WORD)v153
      || (v26 = (unsigned __int16)v153, v59 = v29 + 500LL * (unsigned __int16)v153, *(_BYTE *)(v59 + 6557) == 2)
      || *(_BYTE *)(v59 + 6606) == 2 )
    {
      sub_3763F80(a1, a2, a3, v26, v27, v28, a4);
LABEL_30:
      if ( v155 )
        sub_B91220((__int64)&v155, v155);
      return;
    }
  }
  else if ( !(_WORD)v153
         || (v26 = (unsigned __int16)v153, v31 = v29 + 500LL * (unsigned __int16)v153, *(_BYTE *)(v31 + 6634) == 2)
         || *(_BYTE *)(v31 + 6606) == 2 )
  {
    v60 = sub_3412A00((_QWORD *)*a1, a2, 0, v26, v27, v28, a4);
    v63 = v62;
    v64 = *(unsigned int *)(a3 + 8);
    v65 = v60;
    if ( v64 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v64 + 1, 0x10u, v64 + 1, v61);
      v64 = *(unsigned int *)(a3 + 8);
    }
    v66 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v64);
    *v66 = v65;
    v66[1] = v63;
    ++*(_DWORD *)(a3 + 8);
    goto LABEL_30;
  }
  v32 = v154;
  if ( (unsigned __int16)(v153 - 17) <= 0xD3u )
  {
    v32 = 0;
    v30 = word_4456580[(int)v26 - 1];
  }
  v165.m128i_i16[0] = v30;
  v165.m128i_i64[1] = v32;
  if ( v30 )
  {
    if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
      BUG();
    v33 = *(_QWORD *)&byte_444C4A0[16 * v30 - 16];
  }
  else
  {
    v159 = sub_3007260((__int64)&v165);
    LODWORD(v33) = v159;
    v160 = v34;
  }
  if ( v13 )
  {
    v71 = 1;
    if ( (_WORD)v145 != 1
      && (!(_WORD)v145 || (v71 = (unsigned __int16)v145, !*(_QWORD *)(v29 + 8LL * (unsigned __int16)v145 + 112)))
      || (*(_BYTE *)(v29 + 500 * v71 + 6517) & 0xFB) != 0 )
    {
LABEL_19:
      v36 = *a1;
      v141.m128i_i32[0] = ((_DWORD)v33 != 32) + 12;
      LODWORD(v142) = v141.m128i_i32[0];
      v37.m128i_i64[0] = (__int64)sub_3400D50(v36, 0, (__int64)&v155, 1u, a4);
      v140 = v37;
      v100 = word_4456340[(unsigned __int16)v153 - 1];
      if ( (unsigned __int16)(v153 - 176) > 0x34u )
        v101 = sub_2D43050(v141.m128i_i16[0], v100);
      else
        v101 = sub_2D43AD0(v141.m128i_i16[0], v100);
      v39 = v143.m128i_i64[0];
      v38 = (_QWORD *)*a1;
      LOWORD(v39) = v101;
      v143.m128i_i64[0] = v39;
      if ( !v13 )
      {
        v151 = sub_33FAF80((__int64)v38, 221, (__int64)&v155, v143.m128i_u32[0], 0, v102, a4);
        v152 = v122;
        *((_QWORD *)&v132 + 1) = (unsigned int)v122;
        *(_QWORD *)&v132 = v151;
        v125 = sub_3406EB0(
                 (_QWORD *)*a1,
                 0xE6u,
                 (__int64)&v155,
                 (unsigned __int16)v145,
                 v146.m128i_i64[0],
                 (unsigned __int16)v145,
                 v132,
                 *(_OWORD *)&v140);
        v126 = *(unsigned int *)(a3 + 8);
        v128 = (unsigned __int8 *)v127;
        if ( v126 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v126 + 1, 0x10u, v123, v124);
          v126 = *(unsigned int *)(a3 + 8);
        }
        v129 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v126);
        *v129 = v125;
        v129[1] = v128;
        ++*(_DWORD *)(a3 + 8);
        goto LABEL_30;
      }
      v40 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      *((_QWORD *)&v136 + 1) = 2;
      *(_QWORD *)&v136 = &v165;
      v161 = v39;
      v165 = v40;
      v41 = _mm_load_si128(&v144);
      v162 = 0;
      v163 = 1;
      v144.m128i_i64[0] = (__int64)&v161;
      v166 = v41;
      v164 = 0;
      v42 = sub_3411BE0(v38, 0x90u, (__int64)&v155, (unsigned __int16 *)&v161, 2, (__int64)&v161, v136);
      v43 = (_QWORD *)*a1;
      v44 = _mm_load_si128(&v140);
      v45 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      v166.m128i_i32[2] = v46;
      *((_QWORD *)&v133 + 1) = 3;
      LOWORD(v161) = v145;
      *(_QWORD *)&v133 = &v165;
      v162 = v146.m128i_i64[0];
      v163 = 1;
      v166.m128i_i64[0] = (__int64)v42;
      v164 = 0;
      v165 = v45;
      v167 = v44;
      v47 = sub_3411BE0(v43, 0x91u, (__int64)&v155, (unsigned __int16 *)v144.m128i_i64[0], 2, v144.m128i_i64[0], v133);
      v50 = *(unsigned int *)(a3 + 8);
      v52 = (unsigned __int8 *)v51;
      v53 = v50 + 1;
      if ( v50 + 1 <= (unsigned __int64)*(unsigned int *)(a3 + 12) )
        goto LABEL_22;
      goto LABEL_65;
    }
  }
  else
  {
    v35 = 1;
    if ( (_WORD)v145 != 1 )
    {
      if ( !(_WORD)v145 )
        goto LABEL_19;
      v35 = (unsigned __int16)v145;
      if ( !*(_QWORD *)(v29 + 8LL * (unsigned __int16)v145 + 112) )
        goto LABEL_19;
    }
    if ( (*(_BYTE *)(v29 + 500 * v35 + 6512) & 0xFB) != 0 )
      goto LABEL_19;
  }
  v72 = *a1;
  v141.m128i_i32[0] = (unsigned int)v33 >> 1;
  LODWORD(v142) = v33;
  v73.m128i_i64[0] = (__int64)sub_3400BD0(v72, (unsigned int)v33 >> 1, (__int64)&v155, v153, v154, 0, a4, 0);
  v74 = 0xFFFFFFFFLL;
  v143 = v73;
  if ( (_DWORD)v142 != 64 )
    v74 = 0xFFFF;
  *(_QWORD *)&v75 = sub_3400BD0(*a1, v74, (__int64)&v155, v153, v154, 0, a4, 0);
  v76 = *a1;
  v142 = v75;
  v77 = 0;
  if ( 1LL << v141.m128i_i8[0] < 0 )
  {
    v103 = (1LL << v141.m128i_i8[0]) & 1 | ((unsigned __int64)(1LL << v141.m128i_i8[0]) >> 1);
    *(double *)v77.m128i_i64 = (double)(int)v103 + (double)(int)v103;
  }
  else
  {
    *(double *)v77.m128i_i64 = (double)(int)(1LL << v141.m128i_i8[0]);
  }
  v140.m128i_i64[0] = (unsigned __int16)v145;
  v78.m128i_i64[0] = sub_33FE730(v76, (__int64)&v155, (unsigned __int16)v145, v146.m128i_i64[0], 0, v77);
  v79 = (_QWORD *)*a1;
  v141 = v78;
  v81.m128i_i64[0] = (__int64)sub_3406EB0(
                                v79,
                                0xC0u,
                                (__int64)&v155,
                                v153,
                                v154,
                                v80,
                                *(_OWORD *)&v144,
                                *(_OWORD *)&v143);
  v82 = (_QWORD *)*a1;
  v143 = v81;
  v84.m128i_i64[0] = (__int64)sub_3406EB0(v82, 0xBAu, (__int64)&v155, v153, v154, v83, *(_OWORD *)&v144, v142);
  v85 = (unsigned __int16)v145;
  v144 = v84;
  if ( v13 )
  {
    v104 = (_QWORD *)*a1;
    v105 = _mm_load_si128(&v143);
    v106 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    *((_QWORD *)&v137 + 1) = 2;
    *(_QWORD *)&v137 = &v165;
    LOWORD(v161) = v145;
    v163 = 1;
    v165 = v106;
    v166 = v105;
    v162 = v146.m128i_i64[0];
    *(_QWORD *)&v142 = &v161;
    v143.m128i_i64[0] = (__int64)&v165;
    v164 = 0;
    v107.m128i_i64[0] = (__int64)sub_3411BE0(
                                   v104,
                                   0x8Fu,
                                   (__int64)&v155,
                                   (unsigned __int16 *)&v161,
                                   2,
                                   (__int64)&v161,
                                   v137);
    v108 = (_QWORD *)*a1;
    v109 = v107.m128i_i64[1];
    v165.m128i_i64[0] = v107.m128i_i64[0];
    v166 = v107;
    v110 = _mm_load_si128(&v141);
    *((_QWORD *)&v134 + 1) = 3;
    *(_QWORD *)&v134 = v143.m128i_i64[0];
    LOWORD(v161) = v145;
    v163 = 1;
    v167 = v110;
    v162 = v146.m128i_i64[0];
    v140.m128i_i64[0] = (__int64)&v161;
    v165.m128i_i32[2] = 1;
    v164 = 0;
    v149 = sub_3411BE0(v108, 0x67u, (__int64)&v155, (unsigned __int16 *)v142, 2, v142, v134);
    *(_QWORD *)&v142 = v149;
    v111 = _mm_load_si128(&v144);
    v150 = v112;
    v113 = *(const __m128i **)(a2 + 40);
    v114 = (_QWORD *)*a1;
    v141.m128i_i64[0] = (unsigned int)v112 | v109 & 0xFFFFFFFF00000000LL;
    v115 = _mm_loadu_si128(v113);
    *((_QWORD *)&v138 + 1) = 2;
    LOWORD(v161) = v145;
    *(_QWORD *)&v138 = v143.m128i_i64[0];
    v163 = 1;
    v165 = v115;
    v166 = v111;
    v162 = v146.m128i_i64[0];
    v144.m128i_i64[0] = (__int64)&v161;
    v164 = 0;
    v116 = sub_3411BE0(v114, 0x8Fu, (__int64)&v155, (unsigned __int16 *)v140.m128i_i64[0], 2, v140.m128i_i64[0], v138);
    v118 = v117;
    *((_QWORD *)&v135 + 1) = 1;
    *(_QWORD *)&v135 = v116;
    *((_QWORD *)&v131 + 1) = 1;
    *(_QWORD *)&v131 = v142;
    v119.m128i_i64[0] = (__int64)sub_3406EB0((_QWORD *)*a1, 2u, (__int64)&v155, 1, 0, 1, v131, v135);
    v120 = (_QWORD *)*a1;
    v165 = v119;
    v166.m128i_i64[0] = v142;
    v166.m128i_i64[1] = v141.m128i_i64[0];
    *((_QWORD *)&v139 + 1) = 3;
    *(_QWORD *)&v139 = v143.m128i_i64[0];
    v163 = 1;
    LOWORD(v161) = v145;
    v162 = v146.m128i_i64[0];
    v167.m128i_i64[0] = (__int64)v116;
    v167.m128i_i64[1] = v118;
    v164 = 0;
    v47 = sub_3411BE0(v120, 0x65u, (__int64)&v155, (unsigned __int16 *)v144.m128i_i64[0], 2, v144.m128i_i64[0], v139);
    v50 = *(unsigned int *)(a3 + 8);
    v52 = v121;
    v53 = v50 + 1;
    if ( v50 + 1 <= (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
LABEL_22:
      v54 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v50);
      *v54 = v47;
      v54[1] = v52;
      v55 = *(unsigned int *)(a3 + 12);
      v56 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v56;
      if ( v56 + 1 > v55 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v56 + 1, 0x10u, v48, v49);
        v56 = *(unsigned int *)(a3 + 8);
      }
      v57 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v56);
      *v57 = v47;
      v57[1] = (unsigned __int8 *)1;
      ++*(_DWORD *)(a3 + 8);
      goto LABEL_30;
    }
LABEL_65:
    sub_C8D5F0(a3, (const void *)(a3 + 16), v53, 0x10u, v48, v49);
    v50 = *(unsigned int *)(a3 + 8);
    goto LABEL_22;
  }
  v86 = *a1;
  v145 = v140.m128i_i64[0];
  *(_QWORD *)&v87 = sub_33FAF80(v86, 220, (__int64)&v155, v140.m128i_u32[0], v146.m128i_i64[0], v85, v77);
  v88 = *((_QWORD *)&v87 + 1);
  v147 = sub_3406EB0(
           (_QWORD *)*a1,
           0x62u,
           (__int64)&v155,
           (unsigned int)v145,
           v146.m128i_i64[0],
           v145,
           v87,
           *(_OWORD *)&v141);
  v148 = v89;
  v90 = (unsigned int)v89 | v88 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v91 = sub_33FAF80(*a1, 220, (__int64)&v155, (unsigned int)v145, v146.m128i_i64[0], v145, v77);
  *((_QWORD *)&v130 + 1) = v90;
  *(_QWORD *)&v130 = v147;
  v92 = sub_3406EB0((_QWORD *)*a1, 0x60u, (__int64)&v155, (unsigned int)v145, v146.m128i_i64[0], v145, v130, v91);
  v95 = v94;
  v96 = *(unsigned int *)(a3 + 8);
  v97 = v92;
  if ( v96 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v96 + 1, 0x10u, v96 + 1, v93);
    v96 = *(unsigned int *)(a3 + 8);
  }
  v98 = (unsigned __int8 **)(*(_QWORD *)a3 + 16 * v96);
  *v98 = v97;
  v99 = v155;
  v98[1] = v95;
  ++*(_DWORD *)(a3 + 8);
  if ( v99 )
    sub_B91220((__int64)&v155, v99);
}
