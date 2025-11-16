// Function: sub_3780EC0
// Address: 0x3780ec0
//
void __fastcall sub_3780EC0(__int64 *a1, unsigned __int64 a2, __m128i **a3, __m128i **a4, char a5, __m128i a6)
{
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int16 *v11; // rax
  __int16 v12; // dx
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rcx
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int16 *v22; // rax
  _QWORD *v23; // r13
  __int64 v24; // rax
  __int16 v25; // dx
  __int64 v26; // rax
  __m128i v27; // xmm3
  _QWORD *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r9
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rsi
  __m128i v34; // xmm5
  unsigned __int16 *v35; // rax
  _QWORD *v36; // r13
  __int64 v37; // rax
  __int16 v38; // dx
  __int64 v39; // rax
  __m128i v40; // xmm1
  unsigned __int8 v41; // al
  __m128i v42; // xmm6
  __m128i v43; // xmm7
  __int64 *v44; // rdi
  __m128i v45; // xmm3
  __m128i v46; // xmm4
  unsigned __int16 v47; // r13
  __int16 v48; // r13
  unsigned __int64 v49; // rax
  __int32 v50; // edx
  __m128i *v51; // rax
  __m128i **v52; // r11
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // rcx
  __m128i v57; // xmm0
  __m128i v58; // xmm1
  __m128i v59; // xmm2
  __m128i v60; // xmm5
  __int64 *v61; // rdi
  unsigned __int64 v62; // rax
  __int32 v63; // edx
  __m128i **v64; // rbx
  __int64 v65; // rdx
  unsigned __int8 *v66; // rax
  __int64 v67; // rdx
  bool v68; // zf
  __int64 v69; // rax
  __m128i v70; // xmm3
  __m128i v71; // xmm4
  __m128i v72; // xmm7
  __int64 *v73; // rdi
  unsigned __int16 v74; // r13
  unsigned __int64 v75; // rax
  __int32 v76; // edx
  __m128i *v77; // rax
  __int32 v78; // esi
  __int64 v79; // rdi
  __int64 v80; // rdx
  __m128i **v81; // rdx
  __m128i v82; // xmm0
  __int64 v83; // rcx
  __m128i v84; // xmm6
  __m128i v85; // xmm7
  unsigned __int16 v86; // r13
  __int64 *v87; // rdi
  unsigned __int64 v88; // rax
  __int32 v89; // edx
  __m128i *v90; // rax
  __m128i **v91; // rbx
  __int64 v92; // rdx
  __m128i v93; // xmm4
  __int64 v94; // rax
  __int128 v95; // [rsp-20h] [rbp-2F0h]
  __int128 v96; // [rsp-10h] [rbp-2E0h]
  unsigned __int64 v97; // [rsp+8h] [rbp-2C8h]
  unsigned __int64 *v98; // [rsp+10h] [rbp-2C0h]
  __int64 v99; // [rsp+18h] [rbp-2B8h]
  __int64 v100; // [rsp+20h] [rbp-2B0h]
  unsigned __int64 v101; // [rsp+28h] [rbp-2A8h]
  __int64 v102; // [rsp+30h] [rbp-2A0h]
  unsigned __int64 v103; // [rsp+38h] [rbp-298h]
  __int64 v104; // [rsp+40h] [rbp-290h]
  __int64 v105; // [rsp+48h] [rbp-288h]
  __int64 v106; // [rsp+50h] [rbp-280h]
  __int64 v107; // [rsp+58h] [rbp-278h]
  __int64 v108; // [rsp+60h] [rbp-270h]
  __int64 v109; // [rsp+68h] [rbp-268h]
  const __m128i *v110; // [rsp+70h] [rbp-260h]
  __int64 v111; // [rsp+78h] [rbp-258h]
  __m128i *v112; // [rsp+80h] [rbp-250h]
  __int64 v113; // [rsp+88h] [rbp-248h]
  __m128i **v114; // [rsp+90h] [rbp-240h]
  __m128i **v115; // [rsp+98h] [rbp-238h]
  __int64 *v116; // [rsp+A0h] [rbp-230h]
  __int64 v117; // [rsp+A8h] [rbp-228h]
  unsigned __int8 *v118; // [rsp+B0h] [rbp-220h]
  __int64 v119; // [rsp+B8h] [rbp-218h]
  __m128i *v120; // [rsp+C0h] [rbp-210h]
  __int64 v121; // [rsp+C8h] [rbp-208h]
  __m128i *v122; // [rsp+D0h] [rbp-200h]
  __int64 v123; // [rsp+D8h] [rbp-1F8h]
  __m128i *v124; // [rsp+E0h] [rbp-1F0h]
  __int64 v125; // [rsp+E8h] [rbp-1E8h]
  __m128i *v126; // [rsp+F0h] [rbp-1E0h]
  __int64 v127; // [rsp+F8h] [rbp-1D8h]
  __int64 v128; // [rsp+100h] [rbp-1D0h] BYREF
  int v129; // [rsp+108h] [rbp-1C8h]
  __int64 v130; // [rsp+110h] [rbp-1C0h] BYREF
  __int64 v131; // [rsp+118h] [rbp-1B8h]
  __m128i v132; // [rsp+120h] [rbp-1B0h] BYREF
  __m128i v133; // [rsp+130h] [rbp-1A0h] BYREF
  __m128i v134; // [rsp+140h] [rbp-190h] BYREF
  __m128i v135; // [rsp+150h] [rbp-180h] BYREF
  __m128i v136; // [rsp+160h] [rbp-170h] BYREF
  __m128i v137; // [rsp+170h] [rbp-160h] BYREF
  __m128i v138; // [rsp+180h] [rbp-150h] BYREF
  __m128i v139; // [rsp+190h] [rbp-140h] BYREF
  __m128i v140; // [rsp+1A0h] [rbp-130h] BYREF
  __m128i v141; // [rsp+1B0h] [rbp-120h]
  __m128i v142; // [rsp+1C0h] [rbp-110h] BYREF
  __m128i v143; // [rsp+1D0h] [rbp-100h] BYREF
  __int64 v144; // [rsp+1E0h] [rbp-F0h] BYREF
  __int64 v145; // [rsp+1E8h] [rbp-E8h]
  __m128i v146; // [rsp+1F0h] [rbp-E0h]
  __m128i v147; // [rsp+200h] [rbp-D0h]
  __m128i v148; // [rsp+210h] [rbp-C0h]
  __m128i v149; // [rsp+220h] [rbp-B0h]
  __m128i v150; // [rsp+230h] [rbp-A0h]
  __m128i v151; // [rsp+240h] [rbp-90h] BYREF
  __m128i v152; // [rsp+250h] [rbp-80h] BYREF
  __m128i v153; // [rsp+260h] [rbp-70h]
  __m128i v154; // [rsp+270h] [rbp-60h]
  __m128i v155; // [rsp+280h] [rbp-50h]
  __m128i v156; // [rsp+290h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 80);
  v115 = a3;
  v114 = a4;
  v128 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v128, v9, 1);
  v10 = a1[1];
  v129 = *(_DWORD *)(a2 + 72);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *v11;
  v145 = *((_QWORD *)v11 + 1);
  LOWORD(v144) = v12;
  v116 = &v144;
  sub_33D0340((__int64)&v151, v10, &v144);
  v13 = *(_QWORD *)(a2 + 40);
  v107 = v151.m128i_i64[1];
  v108 = v151.m128i_i64[0];
  v105 = v152.m128i_i64[1];
  v106 = v152.m128i_i64[0];
  v109 = *(_QWORD *)v13;
  v117 = *(_QWORD *)(v13 + 8);
  v14 = *(_DWORD *)(a2 + 24);
  if ( v14 <= 365 )
  {
    if ( v14 <= 363 )
    {
      if ( v14 != 339 && (v14 & 0xFFFFFFBF) != 0x12B )
        goto LABEL_7;
      goto LABEL_26;
    }
LABEL_24:
    v15 = v13 + 120;
    goto LABEL_8;
  }
  if ( v14 > 467 )
  {
    if ( v14 == 497 )
      goto LABEL_24;
LABEL_7:
    v15 = v13 + 40;
    goto LABEL_8;
  }
  if ( v14 <= 464 )
    goto LABEL_7;
LABEL_26:
  v15 = v13 + 80;
LABEL_8:
  v111 = *(_QWORD *)v15;
  LODWORD(v112) = *(_DWORD *)(v15 + 8);
  if ( v14 == 364 )
  {
    v16 = _mm_loadu_si128((const __m128i *)(v13 + 160));
    v17 = _mm_loadu_si128((const __m128i *)(v13 + 200));
    v141 = _mm_loadu_si128((const __m128i *)(v13 + 80));
    v142 = v16;
    v143 = v17;
  }
  else
  {
    if ( v14 == 470 )
    {
      v94 = 120;
      v141 = _mm_loadu_si128((const __m128i *)(v13 + 160));
      v142 = _mm_loadu_si128((const __m128i *)(v13 + 80));
    }
    else
    {
      v93 = _mm_loadu_si128((const __m128i *)(v13 + 120));
      v94 = 160;
      v141 = _mm_loadu_si128((const __m128i *)(v13 + 200));
      v142 = v93;
    }
    v143 = _mm_loadu_si128((const __m128i *)(v13 + v94));
  }
  v18 = *(_QWORD *)(a2 + 104);
  v19 = *(_WORD *)(a2 + 96);
  v132.m128i_i64[0] = 0;
  v132.m128i_i32[2] = 0;
  v131 = v18;
  v20 = *(_QWORD *)(a2 + 112);
  LOWORD(v130) = v19;
  v133.m128i_i64[0] = 0;
  LOBYTE(v20) = *(_BYTE *)(v20 + 34);
  v133.m128i_i32[2] = 0;
  LOBYTE(v110) = v20;
  if ( a5 && *(_DWORD *)(v141.m128i_i64[0] + 24) == 208 )
  {
    sub_377EF80(a1, v141.m128i_i64[0], (__int64)&v132, (__int64)&v133, a6);
  }
  else
  {
    sub_3777810(&v151, a1, v141.m128i_u64[0], v141.m128i_u64[1], (__int64)&v128, a6);
    v132.m128i_i64[0] = v151.m128i_i64[0];
    v132.m128i_i32[2] = v151.m128i_i32[2];
    v133.m128i_i64[0] = v152.m128i_i64[0];
    v133.m128i_i32[2] = v152.m128i_i32[2];
  }
  sub_33D0340((__int64)&v151, a1[1], &v130);
  v21 = *a1;
  v134.m128i_i32[2] = 0;
  v135.m128i_i32[2] = 0;
  v104 = v151.m128i_i64[0];
  v134.m128i_i64[0] = 0;
  v103 = v151.m128i_u64[1];
  v135.m128i_i64[0] = 0;
  v101 = v152.m128i_u64[1];
  v102 = v152.m128i_i64[0];
  v22 = (unsigned __int16 *)(*(_QWORD *)(v142.m128i_i64[0] + 48) + 16LL * v142.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v151, v21, *(_QWORD *)(a1[1] + 64), *v22, *((_QWORD *)v22 + 1));
  if ( v151.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v142.m128i_u64[0], v142.m128i_i64[1], (__int64)&v135, (__int64)&v134);
  }
  else
  {
    v23 = (_QWORD *)a1[1];
    v139.m128i_i16[0] = 0;
    v140.m128i_i16[0] = 0;
    v139.m128i_i64[1] = 0;
    v140.m128i_i64[1] = 0;
    v24 = *(_QWORD *)(v142.m128i_i64[0] + 48) + 16LL * v142.m128i_u32[2];
    v25 = *(_WORD *)v24;
    v26 = *(_QWORD *)(v24 + 8);
    LOWORD(v144) = v25;
    v145 = v26;
    sub_33D0340((__int64)&v151, (__int64)v23, v116);
    v27 = _mm_loadu_si128(&v151);
    v140 = _mm_loadu_si128(&v152);
    v139 = v27;
    sub_3408290(
      (__int64)&v151,
      v23,
      (__int128 *)v142.m128i_i8,
      (__int64)&v128,
      (unsigned int *)&v139,
      (unsigned int *)&v140,
      a6);
    v135.m128i_i64[0] = v151.m128i_i64[0];
    v135.m128i_i32[2] = v151.m128i_i32[2];
    v134.m128i_i64[0] = v152.m128i_i64[0];
    v134.m128i_i32[2] = v152.m128i_i32[2];
  }
  v28 = *(_QWORD **)(a1[1] + 40);
  v29 = *(_QWORD *)(a2 + 112);
  v30 = *(_QWORD *)(v29 + 72);
  v151 = _mm_loadu_si128((const __m128i *)(v29 + 40));
  v152 = _mm_loadu_si128((const __m128i *)(v29 + 56));
  v110 = (const __m128i *)sub_2E7BD70(
                            v28,
                            1u,
                            -1,
                            (unsigned __int8)v110,
                            (int)&v151,
                            v30,
                            *(_OWORD *)v29,
                            *(_QWORD *)(v29 + 16),
                            1u,
                            0,
                            0);
  v31 = *(_DWORD *)(a2 + 24);
  if ( v31 == 364 )
  {
    v32 = *(_QWORD *)(a2 + 40);
    v33 = *a1;
    v137.m128i_i32[2] = 0;
    v138.m128i_i32[2] = 0;
    v34 = _mm_loadu_si128((const __m128i *)(v32 + 40));
    v137.m128i_i64[0] = 0;
    v136 = v34;
    v138.m128i_i64[0] = 0;
    v35 = (unsigned __int16 *)(*(_QWORD *)(v34.m128i_i64[0] + 48) + 16LL * v34.m128i_u32[2]);
    sub_2FE6CC0((__int64)&v151, v33, *(_QWORD *)(a1[1] + 64), *v35, *((_QWORD *)v35 + 1));
    if ( v151.m128i_i8[0] == 6 )
    {
      sub_375E8D0((__int64)a1, v136.m128i_u64[0], v136.m128i_i64[1], (__int64)&v137, (__int64)&v138);
    }
    else
    {
      v36 = (_QWORD *)a1[1];
      v139.m128i_i16[0] = 0;
      v140.m128i_i16[0] = 0;
      v139.m128i_i64[1] = 0;
      v140.m128i_i64[1] = 0;
      v37 = *(_QWORD *)(v136.m128i_i64[0] + 48) + 16LL * v136.m128i_u32[2];
      v38 = *(_WORD *)v37;
      v39 = *(_QWORD *)(v37 + 8);
      LOWORD(v144) = v38;
      v145 = v39;
      sub_33D0340((__int64)&v151, (__int64)v36, v116);
      v40 = _mm_loadu_si128(&v151);
      v140 = _mm_loadu_si128(&v152);
      v139 = v40;
      sub_3408290(
        (__int64)&v151,
        v36,
        (__int128 *)v136.m128i_i8,
        (__int64)&v128,
        (unsigned int *)&v139,
        (unsigned int *)&v140,
        a6);
      v137.m128i_i64[0] = v151.m128i_i64[0];
      v137.m128i_i32[2] = v151.m128i_i32[2];
      v138.m128i_i64[0] = v152.m128i_i64[0];
      v138.m128i_i32[2] = v152.m128i_i32[2];
    }
    v41 = *(_BYTE *)(a2 + 33);
    v99 = 6;
    v42 = _mm_loadu_si128(&v137);
    v144 = v109;
    v43 = _mm_loadu_si128(&v132);
    v44 = (__int64 *)a1[1];
    v148.m128i_i64[0] = v111;
    v148.m128i_i32[2] = (int)v112;
    v45 = _mm_loadu_si128(&v135);
    v46 = _mm_loadu_si128(&v143);
    v145 = v117;
    v146 = v42;
    v47 = *(_WORD *)(a2 + 32);
    v147 = v43;
    v149 = v45;
    v150 = v46;
    v98 = (unsigned __int64 *)v116;
    v48 = (v47 >> 7) & 7;
    v116 = v44;
    LODWORD(v100) = (v41 >> 2) & 3;
    v49 = sub_33E5110(v44, v108, v107, 1, 0);
    v51 = sub_33E8420(v44, v49, v50, v104, v103, (__int64)&v128, v98, 6, v110, v48, v100);
    v52 = v115;
    v53 = v109;
    v127 = v54;
    LODWORD(v54) = (_DWORD)v112;
    v126 = v51;
    v55 = v111;
    *v115 = v51;
    v56 = v117;
    v57 = _mm_loadu_si128(&v138);
    v151.m128i_i64[0] = v53;
    *((_DWORD *)v52 + 2) = v127;
    v58 = _mm_loadu_si128(&v133);
    v59 = _mm_loadu_si128(&v134);
    v60 = _mm_loadu_si128(&v143);
    v154.m128i_i64[0] = v55;
    v61 = (__int64 *)a1[1];
    v154.m128i_i32[2] = v54;
    v151.m128i_i64[1] = v56;
    v112 = &v151;
    v113 = 6;
    v116 = v61;
    v152 = v57;
    v153 = v58;
    v155 = v59;
    v156 = v60;
    v62 = sub_33E5110(v61, v106, v105, 1, 0);
    v64 = v114;
    v124 = sub_33E8420(v61, v62, v63, v102, v101, (__int64)&v128, (unsigned __int64 *)&v151, 6, v110, v48, v100);
    v125 = v65;
    *v114 = v124;
    *((_DWORD *)v64 + 2) = v125;
  }
  else
  {
    v68 = v31 == 470;
    v69 = 200;
    if ( !v68 )
      v69 = 240;
    sub_3408380(
      &v151,
      (_QWORD *)a1[1],
      *(_QWORD *)(v69 + *(_QWORD *)(a2 + 40)),
      *(_QWORD *)(v69 + *(_QWORD *)(a2 + 40) + 8),
      (unsigned int)v130,
      v131,
      a6,
      (__int64)&v128);
    v99 = 6;
    v100 = v152.m128i_i64[0];
    v145 = v117;
    v70 = _mm_loadu_si128(&v143);
    v71 = _mm_loadu_si128(&v132);
    v146.m128i_i32[2] = (int)v112;
    v72 = _mm_loadu_si128(&v135);
    v144 = v109;
    v73 = (__int64 *)a1[1];
    v150.m128i_i64[1] = v151.m128i_u32[2];
    v74 = *(_WORD *)(a2 + 32);
    v146.m128i_i64[0] = v111;
    v148 = v70;
    v149 = v71;
    v97 = _mm_cvtsi32_si128(v152.m128i_u32[2]).m128i_u64[0];
    v147 = v72;
    v98 = (unsigned __int64 *)v116;
    v116 = v73;
    v150.m128i_i64[0] = v151.m128i_i64[0];
    v75 = sub_33E5110(v73, v108, v107, 1, 0);
    v77 = sub_33E79D0(v73, v75, v76, v104, v103, (__int64)&v128, v98, 6, v110, (v74 >> 7) & 7);
    v78 = (int)v112;
    v79 = v117;
    v123 = v80;
    v81 = v115;
    v122 = v77;
    v82 = _mm_loadu_si128(&v134);
    *v115 = v77;
    v83 = v111;
    v84 = _mm_loadu_si128(&v143);
    v152.m128i_i32[2] = v78;
    *((_DWORD *)v81 + 2) = v123;
    v85 = _mm_loadu_si128(&v133);
    v151.m128i_i64[1] = v79;
    v86 = *(_WORD *)(a2 + 32);
    v87 = (__int64 *)a1[1];
    v151.m128i_i64[0] = v109;
    v153 = v82;
    v152.m128i_i64[0] = v83;
    v154 = v84;
    v112 = &v151;
    v113 = 6;
    v116 = v87;
    v155 = v85;
    v156.m128i_i64[1] = v97;
    v156.m128i_i64[0] = v100;
    v88 = sub_33E5110(v87, v106, v105, 1, 0);
    v90 = sub_33E79D0(v116, v88, v89, v102, v101, (__int64)&v128, (unsigned __int64 *)v112, v113, v110, (v86 >> 7) & 7);
    v91 = v114;
    v120 = v90;
    v121 = v92;
    *v114 = v90;
    *((_DWORD *)v91 + 2) = v121;
  }
  *((_QWORD *)&v96 + 1) = 1;
  *(_QWORD *)&v96 = *v114;
  *((_QWORD *)&v95 + 1) = 1;
  *(_QWORD *)&v95 = *v115;
  v66 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v128, 1, 0, a1[1], v95, v96);
  v119 = v67;
  v118 = v66;
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v66, v117 & 0xFFFFFFFF00000000LL | (unsigned int)v67);
  if ( v128 )
    sub_B91220((__int64)&v128, v128);
}
