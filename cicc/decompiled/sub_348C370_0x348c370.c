// Function: sub_348C370
// Address: 0x348c370
//
__int64 __fastcall sub_348C370(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  unsigned __int16 *v12; // r8
  int v13; // r14d
  __int64 v14; // r15
  _QWORD *v15; // rbx
  __int64 *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r9
  __int16 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // r12d
  const __m128i *v31; // rax
  _QWORD *v32; // rsi
  __int128 v33; // xmm0
  __m128i *v34; // rax
  __int64 v35; // r9
  char v36; // r14
  __int128 v37; // rdi
  int v38; // eax
  __int64 v39; // r14
  __int64 v40; // r15
  unsigned __int8 *v41; // r14
  __int64 v42; // rdx
  unsigned __int64 v43; // r15
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r9
  unsigned int v48; // r11d
  __int64 v49; // r10
  char v50; // al
  unsigned int v51; // edx
  __int64 v52; // r8
  char v53; // al
  __int64 v54; // r10
  unsigned int v55; // r11d
  unsigned int v56; // edx
  __int64 v57; // r8
  __int64 v58; // r14
  __int128 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r15
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // rdx
  unsigned int v69; // edx
  __int64 v70; // r9
  unsigned int v71; // edx
  __int64 v72; // r9
  unsigned int v73; // edx
  __int64 v74; // r9
  unsigned int v75; // edx
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  unsigned int v79; // eax
  __int128 v80; // rax
  unsigned int v81; // eax
  __int128 v82; // rax
  unsigned int v83; // eax
  __int64 v84; // rdx
  __int64 v85; // r12
  __int64 v86; // rdx
  __int64 v87; // r13
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // r9
  unsigned __int8 *v92; // rax
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // r9
  unsigned int v102; // edx
  unsigned int v103; // edx
  unsigned int v104; // edx
  unsigned int v105; // edx
  unsigned __int8 *v106; // rax
  __int128 v107; // rdi
  __int32 v108; // edx
  unsigned __int8 *v109; // rax
  __int32 v110; // edx
  __int128 v111; // [rsp-20h] [rbp-620h]
  __int128 v112; // [rsp-20h] [rbp-620h]
  __int128 v113; // [rsp-10h] [rbp-610h]
  __int128 v114; // [rsp-10h] [rbp-610h]
  __int128 v115; // [rsp-10h] [rbp-610h]
  __int128 v116; // [rsp-10h] [rbp-610h]
  __int128 v117; // [rsp-10h] [rbp-610h]
  __int128 v118; // [rsp+0h] [rbp-600h]
  __int128 v119; // [rsp+0h] [rbp-600h]
  __int128 v120; // [rsp+0h] [rbp-600h]
  __int128 v121; // [rsp+0h] [rbp-600h]
  unsigned int v122; // [rsp+10h] [rbp-5F0h]
  __int64 v123; // [rsp+10h] [rbp-5F0h]
  __int64 v124; // [rsp+18h] [rbp-5E8h]
  __int128 v125; // [rsp+20h] [rbp-5E0h]
  __int128 v126; // [rsp+20h] [rbp-5E0h]
  __int64 v127; // [rsp+30h] [rbp-5D0h]
  __int64 v128; // [rsp+30h] [rbp-5D0h]
  unsigned int v129; // [rsp+30h] [rbp-5D0h]
  __int128 v130; // [rsp+30h] [rbp-5D0h]
  __int128 v131; // [rsp+50h] [rbp-5B0h]
  __int64 v132; // [rsp+50h] [rbp-5B0h]
  __int64 v133; // [rsp+50h] [rbp-5B0h]
  __int128 v134; // [rsp+50h] [rbp-5B0h]
  unsigned int v135; // [rsp+80h] [rbp-580h]
  unsigned int v137; // [rsp+88h] [rbp-578h]
  __int64 v140; // [rsp+98h] [rbp-568h]
  __int64 v141; // [rsp+98h] [rbp-568h]
  __int64 v142; // [rsp+A0h] [rbp-560h]
  __int64 v143; // [rsp+A0h] [rbp-560h]
  __int128 v144; // [rsp+A0h] [rbp-560h]
  unsigned __int8 *v145; // [rsp+B0h] [rbp-550h]
  unsigned __int8 *v146; // [rsp+C0h] [rbp-540h]
  char v147; // [rsp+10Ah] [rbp-4F6h] BYREF
  char v148; // [rsp+10Bh] [rbp-4F5h] BYREF
  char v149; // [rsp+10Ch] [rbp-4F4h] BYREF
  char v150; // [rsp+10Dh] [rbp-4F3h] BYREF
  char v151; // [rsp+10Eh] [rbp-4F2h] BYREF
  char v152; // [rsp+10Fh] [rbp-4F1h] BYREF
  __int64 v153; // [rsp+110h] [rbp-4F0h] BYREF
  __int64 v154; // [rsp+118h] [rbp-4E8h]
  unsigned __int16 v155[4]; // [rsp+120h] [rbp-4E0h] BYREF
  __int64 v156; // [rsp+128h] [rbp-4D8h]
  __int64 v157; // [rsp+130h] [rbp-4D0h] BYREF
  __int64 v158; // [rsp+138h] [rbp-4C8h]
  unsigned int v159; // [rsp+140h] [rbp-4C0h] BYREF
  __int64 v160; // [rsp+148h] [rbp-4B8h]
  const __m128i *v161[2]; // [rsp+150h] [rbp-4B0h] BYREF
  __int64 (__fastcall *v162)(unsigned __int64 *, const __m128i **, int); // [rsp+160h] [rbp-4A0h]
  __int64 (__fastcall *v163)(__int64 *, __int64, __m128i); // [rsp+168h] [rbp-498h]
  __m128i v164; // [rsp+170h] [rbp-490h] BYREF
  __int64 (__fastcall *v165)(__m128i *, __m128i *, int); // [rsp+180h] [rbp-480h]
  __int64 (__fastcall *v166)(__int64 (__fastcall **)(_QWORD, _QWORD), _QWORD *); // [rsp+188h] [rbp-478h]
  __int64 *v167; // [rsp+190h] [rbp-470h] BYREF
  __int64 v168; // [rsp+198h] [rbp-468h]
  _BYTE v169[256]; // [rsp+1A0h] [rbp-460h] BYREF
  __int64 *v170; // [rsp+2A0h] [rbp-360h] BYREF
  __int64 v171; // [rsp+2A8h] [rbp-358h]
  _BYTE v172[256]; // [rsp+2B0h] [rbp-350h] BYREF
  __int64 *v173; // [rsp+3B0h] [rbp-250h] BYREF
  __int64 v174; // [rsp+3B8h] [rbp-248h]
  _BYTE v175[256]; // [rsp+3C0h] [rbp-240h] BYREF
  __int64 *v176; // [rsp+4C0h] [rbp-140h] BYREF
  __int64 v177; // [rsp+4C8h] [rbp-138h]
  _BYTE v178[304]; // [rsp+4D0h] [rbp-130h] BYREF

  v12 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * a5);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v137 = a2;
  v15 = *(_QWORD **)(a9 + 16);
  LOWORD(v153) = v13;
  v154 = v14;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v14 = 0;
      LOWORD(v13) = word_4456580[v13 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v153) )
  {
    LOWORD(v13) = sub_3009970((__int64)&v153, a2, v65, v66, v67);
    v14 = v68;
  }
  v16 = (__int64 *)v15[5];
  v155[0] = v13;
  v156 = v14;
  v17 = sub_2E79000(v16);
  v18 = (unsigned int)v153;
  v19 = sub_2FE6750(a1, (unsigned int)v153, v154, v17);
  LODWORD(v157) = v19;
  v22 = v19;
  v158 = v23;
  if ( (_WORD)v19 )
  {
    if ( (unsigned __int16)(v19 - 17) > 0xD3u )
    {
LABEL_6:
      v24 = v158;
      goto LABEL_7;
    }
    v24 = 0;
    v22 = word_4456580[(unsigned __int16)v19 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v157) )
      goto LABEL_6;
    v22 = sub_3009970((__int64)&v157, v18, v63, v64, v20);
  }
LABEL_7:
  LOWORD(v159) = v22;
  v160 = v24;
  if ( *(int *)(a9 + 8) > 1 )
  {
    v25 = 1;
    if ( (_WORD)v153 != 1 )
    {
      if ( !(_WORD)v153 )
        return 0;
      v25 = (unsigned __int16)v153;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v153 + 112) )
        return 0;
    }
    if ( (*(_BYTE *)(a1 + 500 * v25 + 6472) & 0xFB) != 0 )
      return 0;
  }
  v28 = sub_33DFBC0(a7, a8, 0, 0, v20, v21);
  if ( !v28 )
    return 0;
  v29 = *(_QWORD *)(v28 + 96);
  v30 = *(_DWORD *)(v29 + 32);
  if ( v30 > 0x40 )
  {
    if ( v30 == (unsigned int)sub_C444A0(v29 + 24) )
      goto LABEL_15;
    return 0;
  }
  if ( *(_QWORD *)(v29 + 24) )
    return 0;
LABEL_15:
  v147 = 0;
  v167 = (__int64 *)v169;
  v168 = 0x1000000000LL;
  v171 = 0x1000000000LL;
  v174 = 0x1000000000LL;
  v177 = 0x1000000000LL;
  v170 = (__int64 *)v172;
  v31 = *(const __m128i **)(a4 + 40);
  v173 = (__int64 *)v175;
  v176 = (__int64 *)v178;
  v32 = (_QWORD *)v31[2].m128i_i64[1];
  v148 = 0;
  v149 = 1;
  v150 = 0;
  v151 = 0;
  v152 = 1;
  v33 = (__int128)_mm_loadu_si128(v31);
  v142 = (__int64)v32;
  v140 = v31[3].m128i_i64[0];
  v127 = v31[2].m128i_i64[1];
  v162 = 0;
  v34 = (__m128i *)sub_22077B0(0x70u);
  if ( v34 )
  {
    v34[3].m128i_i64[0] = (__int64)&v167;
    v34->m128i_i64[0] = (__int64)&v147;
    v34->m128i_i64[1] = (__int64)&v148;
    v34[1].m128i_i64[0] = (__int64)&v149;
    v34[1].m128i_i64[1] = (__int64)&v150;
    v34[2].m128i_i64[0] = (__int64)&v152;
    v34[2].m128i_i64[1] = (__int64)&v151;
    v34[4].m128i_i64[1] = (__int64)v155;
    v34[3].m128i_i64[1] = (__int64)v15;
    v34[4].m128i_i64[0] = a10;
    v34[5].m128i_i64[0] = (__int64)&v170;
    v34[5].m128i_i64[1] = (__int64)&v173;
    v34[6].m128i_i64[0] = (__int64)&v159;
    v34[6].m128i_i64[1] = (__int64)&v176;
  }
  v161[0] = v34;
  v163 = sub_3443A20;
  v162 = sub_343F980;
  v165 = 0;
  sub_343F980((unsigned __int64 *)&v164, v161, 2);
  v166 = (__int64 (__fastcall *)(__int64 (__fastcall **)(_QWORD, _QWORD), _QWORD *))v163;
  v165 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))v162;
  v36 = sub_33CA8D0(v32, v140, (__int64)&v164, 0, 0);
  if ( v165 )
    v165(&v164, &v164, 3);
  if ( v162 )
    v162((unsigned __int64 *)v161, v161, 3);
  if ( !v36 || v149 || v152 )
    goto LABEL_45;
  *(_QWORD *)&v37 = v167;
  v38 = *(_DWORD *)(v127 + 24);
  if ( v38 == 156 )
  {
    if ( v148 )
    {
      *((_QWORD *)&v37 + 1) = (unsigned int)v168;
      v164.m128i_i64[0] = (__int64)sub_33CF170;
      v166 = sub_343F120;
      v165 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_343F130;
      sub_344D9B0(v37, &v164, 0, 0);
      sub_A17130((__int64)&v164);
      v106 = sub_3400BD0((__int64)v15, 0, a10, *(unsigned int *)v155, v156, 0, (__m128i)v33, 0);
      *((_QWORD *)&v107 + 1) = (unsigned int)v171;
      *(_QWORD *)&v107 = v170;
      v164.m128i_i64[0] = (__int64)sub_33CF460;
      v166 = sub_343F120;
      v165 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_343F130;
      sub_344D9B0(v107, &v164, (__int64)v106, v108);
      sub_A17130((__int64)&v164);
      v109 = sub_3400BD0((__int64)v15, 0, a10, v159, v160, 0, (__m128i)v33, 0);
      *((_QWORD *)&v107 + 1) = (unsigned int)v174;
      *(_QWORD *)&v107 = v173;
      v164.m128i_i64[0] = (__int64)sub_33CF460;
      v166 = sub_343F120;
      v165 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_343F130;
      sub_344D9B0(v107, &v164, (__int64)v109, v110);
      sub_A17130((__int64)&v164);
    }
    *((_QWORD *)&v119 + 1) = (unsigned int)v168;
    *(_QWORD *)&v119 = v167;
    v39 = (__int64)sub_33FC220(v15, 156, a10, v153, v154, v35, v119);
    v40 = v69;
    *((_QWORD *)&v116 + 1) = (unsigned int)v171;
    *(_QWORD *)&v116 = v170;
    *(_QWORD *)&v131 = sub_33FC220(v15, 156, a10, v153, v154, v70, v116);
    *((_QWORD *)&v131 + 1) = v71;
    *((_QWORD *)&v120 + 1) = (unsigned int)v174;
    *(_QWORD *)&v120 = v173;
    *(_QWORD *)&v125 = sub_33FC220(v15, 156, a10, v157, v158, v72, v120);
    *((_QWORD *)&v125 + 1) = v73;
    *((_QWORD *)&v117 + 1) = (unsigned int)v177;
    *(_QWORD *)&v117 = v176;
    v124 = (__int64)sub_33FC220(v15, 156, a10, v153, v154, v74, v117);
    v135 = v75;
  }
  else if ( v38 == 168 )
  {
    v39 = sub_3288900((__int64)v15, v153, v154, a10, *v167, v167[1]);
    v40 = v102;
    *(_QWORD *)&v131 = sub_3288900((__int64)v15, v153, v154, a10, *v170, v170[1]);
    *((_QWORD *)&v131 + 1) = v103;
    *(_QWORD *)&v125 = sub_3288900((__int64)v15, v157, v158, a10, *v173, v173[1]);
    *((_QWORD *)&v125 + 1) = v104;
    v124 = sub_3288900((__int64)v15, v153, v154, a10, *v176, v176[1]);
    v135 = v105;
  }
  else
  {
    v39 = *v167;
    v40 = *((unsigned int *)v167 + 2);
    *(_QWORD *)&v131 = *v170;
    *((_QWORD *)&v131 + 1) = *((unsigned int *)v170 + 2);
    *(_QWORD *)&v125 = *v173;
    *((_QWORD *)&v125 + 1) = *((unsigned int *)v173 + 2);
    v124 = *v176;
    v135 = *((_DWORD *)v176 + 2);
  }
  *((_QWORD *)&v118 + 1) = v40;
  *(_QWORD *)&v118 = v39;
  v41 = sub_3406EB0(v15, 0x3Au, a10, (unsigned int)v153, v154, v35, v33, v118);
  v43 = v42;
  sub_3489D20(a11, (__int64)v41, v42, v44, v45, v46);
  v47 = (__int64)v41;
  if ( v151 )
  {
    v48 = v153;
    v49 = v154;
    if ( *(int *)(a9 + 8) > 1 )
    {
      v122 = v153;
      v128 = v154;
      v50 = sub_328A020(a1, 0x38u, v153, v154, 0);
      v49 = v128;
      v48 = v122;
      if ( !v50 )
        goto LABEL_45;
    }
    *((_QWORD *)&v113 + 1) = v43;
    *(_QWORD *)&v113 = v41;
    v146 = sub_3406EB0(v15, 0x38u, a10, v48, v49, v47, v113, v131);
    v43 = v51 | v43 & 0xFFFFFFFF00000000LL;
    sub_3489D20(a11, (__int64)v146, v51, 0xFFFFFFFF00000000LL, v52, (__int64)v146);
    v47 = (__int64)v146;
  }
  if ( v150 )
  {
    if ( *(int *)(a9 + 8) <= 1 )
    {
      v55 = v153;
      v54 = v154;
    }
    else
    {
      v123 = v47;
      v129 = v153;
      v132 = v154;
      v53 = sub_328A020(a1, 0xC2u, v153, v154, 0);
      v54 = v132;
      v55 = v129;
      v47 = v123;
      if ( !v53 )
        goto LABEL_45;
    }
    *((_QWORD *)&v114 + 1) = v43;
    *(_QWORD *)&v114 = v47;
    v145 = sub_3406EB0(v15, 0xC2u, a10, v55, v54, v47, v114, v125);
    v43 = v56 | v43 & 0xFFFFFFFF00000000LL;
    sub_3489D20(a11, (__int64)v145, v56, 0xFFFFFFFF00000000LL, v57, (__int64)v145);
    v47 = (__int64)v145;
  }
  v58 = v47;
  *(_QWORD *)&v59 = sub_33ED040(v15, 3 * (unsigned int)(a6 == 17) + 10);
  *((_QWORD *)&v115 + 1) = v135;
  *(_QWORD *)&v115 = v124;
  *((_QWORD *)&v111 + 1) = v43;
  *(_QWORD *)&v111 = v58;
  v60 = sub_340F900(v15, 0xD0u, a10, v137, a3, v135, v111, v115, v59);
  v26 = v60;
  v62 = v61;
  if ( !v147 )
    goto LABEL_46;
  v133 = v60;
  if ( (unsigned __int8)sub_328A020(a1, 0xD0u, v137, a3, 0)
    && (unsigned __int8)sub_328A020(a1, 0xBAu, v153, v154, 0)
    && ((*(_DWORD *)(a1 + 4 * (((unsigned __int16)v153 >> 3) + 36LL * (int)a6 - (int)a6) + 521536) >> (4 * (v153 & 7)))
      & 0xB) == 0
    && (unsigned __int8)sub_328A020(a1, 0xCEu, v137, a3, 0) )
  {
    sub_3489D20(a11, v133, v76, v77, v78, v133);
    v79 = sub_32844A0(v155, v133);
    sub_986680((__int64)&v164, v79);
    *(_QWORD *)&v80 = sub_34007B0((__int64)v15, (__int64)&v164, a10, v153, v154, 0, (__m128i)v33, 0);
    v126 = v80;
    sub_969240(v164.m128i_i64);
    v81 = sub_32844A0(v155, (__int64)&v164);
    sub_9865E0((__int64)&v164, v81);
    *(_QWORD *)&v82 = sub_34007B0((__int64)v15, (__int64)&v164, a10, v153, v154, 0, (__m128i)v33, 0);
    v134 = v82;
    sub_969240(v164.m128i_i64);
    v83 = sub_32844A0(v155, (__int64)&v164);
    sub_9691E0((__int64)&v164, v83, 0, 0, 0);
    *(_QWORD *)&v130 = sub_34007B0((__int64)v15, (__int64)&v164, a10, v153, v154, 0, (__m128i)v33, 0);
    *((_QWORD *)&v130 + 1) = v84;
    sub_969240(v164.m128i_i64);
    v85 = sub_32889F0((__int64)v15, a10, v137, a3, v142, v140, v126, 0x11u, 0);
    v87 = v86;
    sub_3489D20(a11, v85, v86, v88, v89, v90);
    v92 = sub_3406EB0(v15, 0xBAu, a10, (unsigned int)v153, v154, v91, v33, v134);
    v141 = v93;
    v143 = (__int64)v92;
    sub_3489D20(a11, (__int64)v92, v93, v94, v95, v96);
    *(_QWORD *)&v144 = sub_32889F0((__int64)v15, a10, v137, a3, v143, v141, v130, a6, 0);
    *((_QWORD *)&v144 + 1) = v97;
    sub_3489D20(a11, v144, v97, v98, v99, v100);
    *((_QWORD *)&v121 + 1) = v62;
    *(_QWORD *)&v121 = v26;
    *((_QWORD *)&v112 + 1) = v87;
    *(_QWORD *)&v112 = v85;
    v26 = sub_340F900(v15, 0xCEu, a10, v137, a3, v101, v112, v144, v121);
    goto LABEL_46;
  }
LABEL_45:
  v26 = 0;
LABEL_46:
  if ( v176 != (__int64 *)v178 )
    _libc_free((unsigned __int64)v176);
  if ( v173 != (__int64 *)v175 )
    _libc_free((unsigned __int64)v173);
  if ( v170 != (__int64 *)v172 )
    _libc_free((unsigned __int64)v170);
  if ( v167 != (__int64 *)v169 )
    _libc_free((unsigned __int64)v167);
  return v26;
}
