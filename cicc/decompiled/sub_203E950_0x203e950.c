// Function: sub_203E950
// Address: 0x203e950
//
unsigned __int64 __fastcall sub_203E950(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  __int64 *v6; // r15
  const __m128i *v9; // rax
  __m128i v10; // xmm0
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned __int16 v13; // bx
  __m128i v14; // xmm1
  __int64 v15; // rax
  __int128 v16; // rax
  __int64 v17; // rsi
  char v18; // di
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // di
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdx
  unsigned __int64 result; // rax
  unsigned int v26; // eax
  unsigned int v27; // r14d
  int v28; // eax
  const void **v29; // rdx
  unsigned __int8 v30; // cl
  __int64 v31; // r14
  __int64 v32; // r15
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 *v39; // rdx
  __int64 v40; // rsi
  __int128 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rcx
  unsigned int v44; // edx
  __int64 *v45; // rbx
  __int64 v46; // rax
  unsigned int v47; // edx
  char v48; // al
  __int128 v49; // rax
  __int64 *v50; // r8
  __int64 v51; // rdx
  __int64 v52; // r9
  __int64 v53; // r11
  _QWORD *v54; // rdi
  unsigned __int64 v55; // rsi
  char v56; // r10
  __int64 v57; // rdx
  bool v58; // zf
  int v59; // edx
  unsigned int v60; // ebx
  __int64 v61; // rax
  const void **v62; // r8
  __int64 v63; // rbx
  unsigned __int64 v64; // rdx
  __int64 v65; // r14
  __int64 v66; // r15
  signed int v67; // eax
  int v68; // edx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // r9
  __int64 v72; // r8
  __int64 v73; // rdx
  __int64 *v74; // rdx
  __int64 v75; // rcx
  __int128 v76; // rax
  unsigned int v77; // esi
  __int64 v78; // rbx
  unsigned int v79; // edx
  __int64 *v80; // rbx
  __int64 v81; // rax
  unsigned int v82; // edx
  char v83; // al
  __int128 v84; // rax
  __int64 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r9
  __int64 v88; // r11
  _QWORD *v89; // rdi
  unsigned __int64 v90; // rsi
  char v91; // r10
  __int64 v92; // rdx
  int v93; // edx
  const void **v94; // rdx
  __int64 v95; // [rsp+0h] [rbp-220h]
  __int64 v96; // [rsp+8h] [rbp-218h]
  __int64 v97; // [rsp+10h] [rbp-210h]
  unsigned int v98; // [rsp+18h] [rbp-208h]
  unsigned int v99; // [rsp+1Ch] [rbp-204h]
  __int64 v100; // [rsp+20h] [rbp-200h]
  __int64 v101; // [rsp+28h] [rbp-1F8h]
  __int64 v102; // [rsp+30h] [rbp-1F0h]
  __int64 v103; // [rsp+40h] [rbp-1E0h]
  unsigned __int64 v104; // [rsp+48h] [rbp-1D8h]
  __int128 v105; // [rsp+50h] [rbp-1D0h]
  unsigned int v106; // [rsp+60h] [rbp-1C0h]
  signed int v107; // [rsp+70h] [rbp-1B0h]
  int v108; // [rsp+78h] [rbp-1A8h]
  int v109; // [rsp+7Ch] [rbp-1A4h]
  int v110; // [rsp+7Ch] [rbp-1A4h]
  unsigned int v111; // [rsp+88h] [rbp-198h]
  __int64 v112; // [rsp+90h] [rbp-190h]
  unsigned int v114; // [rsp+A0h] [rbp-180h]
  unsigned int v115; // [rsp+A4h] [rbp-17Ch]
  __int64 *v116; // [rsp+B0h] [rbp-170h]
  __int64 (__fastcall *v117)(__int64, __int64); // [rsp+B0h] [rbp-170h]
  __int64 v118; // [rsp+B0h] [rbp-170h]
  const void **v119; // [rsp+B0h] [rbp-170h]
  __int64 *v120; // [rsp+B0h] [rbp-170h]
  __int64 (__fastcall *v121)(__int64, __int64); // [rsp+B0h] [rbp-170h]
  __int64 v122; // [rsp+B0h] [rbp-170h]
  __int64 v123; // [rsp+B8h] [rbp-168h]
  __int64 v124; // [rsp+B8h] [rbp-168h]
  __int64 v125; // [rsp+C0h] [rbp-160h]
  __int64 v126; // [rsp+C0h] [rbp-160h]
  __int64 v127; // [rsp+C8h] [rbp-158h]
  signed int v128; // [rsp+D0h] [rbp-150h]
  __int64 v129; // [rsp+D0h] [rbp-150h]
  unsigned int v130; // [rsp+D8h] [rbp-148h]
  _QWORD *v131; // [rsp+D8h] [rbp-148h]
  unsigned int v132; // [rsp+D8h] [rbp-148h]
  unsigned __int64 v133; // [rsp+E8h] [rbp-138h]
  __int64 *v134; // [rsp+F0h] [rbp-130h]
  __int64 *v135; // [rsp+F0h] [rbp-130h]
  unsigned int v136; // [rsp+F8h] [rbp-128h]
  unsigned int v137; // [rsp+FCh] [rbp-124h]
  unsigned int v138; // [rsp+FCh] [rbp-124h]
  unsigned __int64 v139; // [rsp+108h] [rbp-118h]
  __m128i v140; // [rsp+110h] [rbp-110h]
  __int64 v141; // [rsp+140h] [rbp-E0h] BYREF
  int v142; // [rsp+148h] [rbp-D8h]
  char v143[8]; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v144; // [rsp+158h] [rbp-C8h]
  unsigned int v145; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v146; // [rsp+168h] [rbp-B8h]
  int v147; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v148; // [rsp+178h] [rbp-A8h]
  __int64 v149; // [rsp+180h] [rbp-A0h] BYREF
  const void **v150; // [rsp+188h] [rbp-98h]
  __m128i v151; // [rsp+190h] [rbp-90h] BYREF
  __int64 v152; // [rsp+1A0h] [rbp-80h]
  __int128 v153; // [rsp+1B0h] [rbp-70h]
  __int64 v154; // [rsp+1C0h] [rbp-60h]
  __int128 v155; // [rsp+1D0h] [rbp-50h]
  __int64 v156; // [rsp+1E0h] [rbp-40h]

  v6 = (__int64 *)a1;
  v9 = *(const __m128i **)(a3 + 32);
  v10 = _mm_loadu_si128(v9 + 5);
  v102 = v9->m128i_i64[0];
  v140 = v10;
  v112 = v9->m128i_i64[1];
  v11 = sub_1E34390(*(_QWORD *)(a3 + 104));
  v12 = *(_QWORD *)(a3 + 104);
  v13 = *(_WORD *)(v12 + 32);
  v14 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v15 = *(_QWORD *)(v12 + 56);
  v151 = v14;
  v152 = v15;
  *(_QWORD *)&v16 = sub_20363F0(a1, *(_QWORD *)(*(_QWORD *)(a3 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a3 + 32) + 48LL));
  v17 = *(_QWORD *)(a3 + 72);
  v105 = v16;
  v141 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v141, v17, 2);
  v142 = *(_DWORD *)(a3 + 64);
  v18 = *(_BYTE *)(a3 + 88);
  v19 = *(_QWORD *)(a3 + 96);
  v143[0] = v18;
  v144 = v19;
  if ( v18 )
    v137 = sub_2021900(v18);
  else
    v137 = sub_1F58D40((__int64)v143);
  v20 = *(_QWORD *)(v105 + 40) + 16LL * DWORD2(v105);
  v21 = *(_BYTE *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  LOBYTE(v145) = v21;
  v146 = v22;
  if ( v21 )
    v98 = sub_2021900(v21);
  else
    v98 = sub_1F58D40((__int64)&v145);
  LOBYTE(v23) = sub_1F7E0F0((__int64)&v145);
  v147 = v23;
  v148 = v24;
  if ( (_BYTE)v23 )
    result = sub_2021900(v23);
  else
    result = sub_1F58D40((__int64)&v147);
  v99 = result;
  if ( !v137 )
    goto LABEL_41;
  v136 = 0;
  v114 = v13;
  v26 = v11;
  v27 = v137;
  v111 = v26;
  v128 = 0;
  while ( 1 )
  {
    v28 = sub_20219E0(v6[1], *v6, v27, v145, v146, 2, 0, 0);
    LODWORD(v149) = v28;
    v150 = v29;
    if ( !(_BYTE)v28 )
      break;
    v138 = sub_2021900(v28);
    v115 = v138 >> 3;
    if ( v30 <= 0x5Fu )
    {
      v109 = word_4305480[v30];
      goto LABEL_14;
    }
LABEL_44:
    v60 = v149;
    v131 = *(_QWORD **)(v6[1] + 48);
    v119 = v150;
    LOBYTE(v61) = sub_1D15020(v149, v98 / v138);
    v62 = 0;
    if ( !(_BYTE)v61 )
    {
      v61 = sub_1F593D0(v131, v60, (__int64)v119, v98 / v138);
      v97 = v61;
      v62 = v94;
    }
    v63 = v97;
    LOBYTE(v63) = v61;
    v97 = v63;
    v132 = v27;
    v103 = sub_1D309E0(
             (__int64 *)v6[1],
             158,
             (__int64)&v141,
             (unsigned int)v63,
             v62,
             0,
             *(double *)v10.m128i_i64,
             *(double *)v14.m128i_i64,
             *(double *)a6.m128i_i64,
             v105);
    v104 = v64;
    v65 = v96;
    v135 = v6;
    v66 = v95;
    v67 = v99 * v128 / v138;
    v108 = v67 + 1;
    v107 = v67;
    v129 = v67;
    do
    {
      v80 = (__int64 *)v135[1];
      v126 = *v135;
      v121 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*v135 + 48LL);
      v81 = sub_1E0A0C0(v80[4]);
      if ( v121 == sub_1D13A20 )
      {
        v82 = 8 * sub_15A9520(v81, 0);
        if ( v82 == 32 )
        {
          v83 = 5;
        }
        else if ( v82 > 0x20 )
        {
          v83 = 6;
          if ( v82 != 64 )
          {
            v83 = 0;
            if ( v82 == 128 )
              v83 = 7;
          }
        }
        else
        {
          v83 = 3;
          if ( v82 != 8 )
            v83 = 4 * (v82 == 16);
        }
      }
      else
      {
        v83 = v121(v126, v81);
      }
      LOBYTE(v65) = v83;
      v110 = v129 + v108 - v107;
      *(_QWORD *)&v84 = sub_1D38BB0(
                          (__int64)v80,
                          v129,
                          (__int64)&v141,
                          (unsigned int)v65,
                          0,
                          0,
                          v10,
                          *(double *)v14.m128i_i64,
                          a6,
                          0);
      v85 = sub_1D332F0(
              v80,
              106,
              (__int64)&v141,
              (unsigned int)v149,
              v150,
              0,
              *(double *)v10.m128i_i64,
              *(double *)v14.m128i_i64,
              a6,
              v103,
              v104,
              v84);
      v87 = v86;
      v88 = *(_QWORD *)(a3 + 104);
      v89 = (_QWORD *)v135[1];
      v90 = *(_QWORD *)v88 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v90 )
      {
        v91 = *(_BYTE *)(v88 + 16);
        if ( (*(_QWORD *)v88 & 4) != 0 )
        {
          *((_QWORD *)&v155 + 1) = *(_QWORD *)(v88 + 8) + v136;
          v93 = *(_DWORD *)(v90 + 12);
          LOBYTE(v156) = v91;
          *(_QWORD *)&v155 = v90 | 4;
          HIDWORD(v156) = v93;
        }
        else
        {
          *((_QWORD *)&v155 + 1) = *(_QWORD *)(v88 + 8) + v136;
          v92 = *(_QWORD *)v90;
          *(_QWORD *)&v155 = v90;
          v58 = *(_BYTE *)(v92 + 8) == 16;
          LOBYTE(v156) = v91;
          if ( v58 )
            v92 = **(_QWORD **)(v92 + 16);
          HIDWORD(v156) = *(_DWORD *)(v92 + 8) >> 8;
        }
      }
      else
      {
        v68 = *(_DWORD *)(v88 + 20);
        LODWORD(v156) = 0;
        v155 = 0u;
        HIDWORD(v156) = v68;
      }
      v69 = sub_1D2BF40(
              v89,
              v102,
              v112,
              (__int64)&v141,
              (__int64)v85,
              v87,
              v140.m128i_i64[0],
              v140.m128i_i64[1],
              v155,
              v156,
              -(v136 | v111) & (v136 | v111),
              v114,
              (__int64)&v151);
      v71 = v70;
      v72 = v69;
      v73 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v73 >= *(_DWORD *)(a2 + 12) )
      {
        v122 = v69;
        v124 = v71;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v69, v71);
        v73 = *(unsigned int *)(a2 + 8);
        v72 = v122;
        v71 = v124;
      }
      v74 = (__int64 *)(*(_QWORD *)a2 + 16 * v73);
      v74[1] = v71;
      *v74 = v72;
      ++*(_DWORD *)(a2 + 8);
      v132 -= v138;
      v136 += v115;
      v75 = 16LL * v140.m128i_u32[2] + *(_QWORD *)(v140.m128i_i64[0] + 40);
      LOBYTE(v66) = *(_BYTE *)v75;
      v120 = (__int64 *)v135[1];
      *(_QWORD *)&v76 = sub_1D38BB0(
                          (__int64)v120,
                          v115,
                          (__int64)&v141,
                          (unsigned int)v66,
                          *(const void ***)(v75 + 8),
                          0,
                          v10,
                          *(double *)v14.m128i_i64,
                          a6,
                          0);
      v77 = v106;
      v78 = *(_QWORD *)(v140.m128i_i64[0] + 40) + 16LL * v140.m128i_u32[2];
      LOBYTE(v77) = *(_BYTE *)v78;
      v133 = v140.m128i_u32[2] | v133 & 0xFFFFFFFF00000000LL;
      ++v129;
      v140.m128i_i64[0] = (__int64)sub_1D332F0(
                                     v120,
                                     52,
                                     (__int64)&v141,
                                     v77,
                                     *(const void ***)(v78 + 8),
                                     3u,
                                     *(double *)v10.m128i_i64,
                                     *(double *)v14.m128i_i64,
                                     a6,
                                     v140.m128i_i64[0],
                                     v133,
                                     v76);
      v140.m128i_i64[1] = v79 | v140.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    while ( v132 && v132 >= v138 );
    v96 = v65;
    v95 = v66;
    v27 = v132;
    v6 = v135;
    result = v110 * v138 / v99;
    v128 = v110 * v138 / v99;
LABEL_40:
    if ( !v27 )
      goto LABEL_41;
  }
  v138 = sub_1F58D40((__int64)&v149);
  v115 = v138 >> 3;
  if ( !sub_1F58D20((__int64)&v149) )
    goto LABEL_44;
  v109 = sub_1F58D30((__int64)&v149);
LABEL_14:
  v130 = v27;
  v134 = v6;
  v31 = v101;
  v32 = v100;
  while ( 1 )
  {
    v45 = (__int64 *)v134[1];
    v125 = *v134;
    v117 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*v134 + 48LL);
    v46 = sub_1E0A0C0(v45[4]);
    if ( v117 == sub_1D13A20 )
    {
      v47 = 8 * sub_15A9520(v46, 0);
      if ( v47 == 32 )
      {
        v48 = 5;
      }
      else if ( v47 > 0x20 )
      {
        v48 = 6;
        if ( v47 != 64 )
        {
          v48 = 0;
          if ( v47 == 128 )
            v48 = 7;
        }
      }
      else
      {
        v48 = 3;
        if ( v47 != 8 )
          v48 = 4 * (v47 == 16);
      }
    }
    else
    {
      v48 = v117(v125, v46);
    }
    LOBYTE(v31) = v48;
    *(_QWORD *)&v49 = sub_1D38BB0(
                        (__int64)v45,
                        v128,
                        (__int64)&v141,
                        (unsigned int)v31,
                        0,
                        0,
                        v10,
                        *(double *)v14.m128i_i64,
                        a6,
                        0);
    v50 = sub_1D332F0(
            v45,
            109,
            (__int64)&v141,
            (unsigned int)v149,
            v150,
            0,
            *(double *)v10.m128i_i64,
            *(double *)v14.m128i_i64,
            a6,
            v105,
            *((unsigned __int64 *)&v105 + 1),
            v49);
    v52 = v51;
    v53 = *(_QWORD *)(a3 + 104);
    v54 = (_QWORD *)v134[1];
    v55 = *(_QWORD *)v53 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v55 )
    {
      v56 = *(_BYTE *)(v53 + 16);
      if ( (*(_QWORD *)v53 & 4) != 0 )
      {
        *((_QWORD *)&v153 + 1) = *(_QWORD *)(v53 + 8) + v136;
        v59 = *(_DWORD *)(v55 + 12);
        LOBYTE(v154) = v56;
        *(_QWORD *)&v153 = v55 | 4;
        HIDWORD(v154) = v59;
      }
      else
      {
        *((_QWORD *)&v153 + 1) = *(_QWORD *)(v53 + 8) + v136;
        v57 = *(_QWORD *)v55;
        *(_QWORD *)&v153 = v55;
        v58 = *(_BYTE *)(v57 + 8) == 16;
        LOBYTE(v154) = v56;
        if ( v58 )
          v57 = **(_QWORD **)(v57 + 16);
        HIDWORD(v154) = *(_DWORD *)(v57 + 8) >> 8;
      }
    }
    else
    {
      v33 = *(_DWORD *)(v53 + 20);
      LODWORD(v154) = 0;
      v153 = 0u;
      HIDWORD(v154) = v33;
    }
    v34 = sub_1D2BF40(
            v54,
            v102,
            v112,
            (__int64)&v141,
            (__int64)v50,
            v52,
            v140.m128i_i64[0],
            v140.m128i_i64[1],
            v153,
            v154,
            -(v136 | v111) & (v136 | v111),
            v114,
            (__int64)&v151);
    v36 = v35;
    v37 = v34;
    v38 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v38 >= *(_DWORD *)(a2 + 12) )
    {
      v118 = v34;
      v123 = v36;
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v34, v36);
      v38 = *(unsigned int *)(a2 + 8);
      v37 = v118;
      v36 = v123;
    }
    v39 = (__int64 *)(*(_QWORD *)a2 + 16 * v38);
    v39[1] = v36;
    *v39 = v37;
    ++*(_DWORD *)(a2 + 8);
    v130 -= v138;
    v136 += v115;
    v128 += v109;
    v40 = 16LL * v140.m128i_u32[2] + *(_QWORD *)(v140.m128i_i64[0] + 40);
    v116 = (__int64 *)v134[1];
    LOBYTE(v32) = *(_BYTE *)v40;
    *(_QWORD *)&v41 = sub_1D38BB0(
                        (__int64)v116,
                        v115,
                        (__int64)&v141,
                        (unsigned int)v32,
                        *(const void ***)(v40 + 8),
                        0,
                        v10,
                        *(double *)v14.m128i_i64,
                        a6,
                        0);
    v42 = *(_QWORD *)(v140.m128i_i64[0] + 40) + 16LL * v140.m128i_u32[2];
    v139 = v140.m128i_u32[2] | v139 & 0xFFFFFFFF00000000LL;
    v43 = v127;
    LOBYTE(v43) = *(_BYTE *)v42;
    v140.m128i_i64[0] = (__int64)sub_1D332F0(
                                   v116,
                                   52,
                                   (__int64)&v141,
                                   v43,
                                   *(const void ***)(v42 + 8),
                                   3u,
                                   *(double *)v10.m128i_i64,
                                   *(double *)v14.m128i_i64,
                                   a6,
                                   v140.m128i_i64[0],
                                   v139,
                                   v41);
    result = v130;
    v140.m128i_i64[1] = v44 | v140.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( !v130 )
      break;
    if ( v130 < v138 )
    {
      v100 = v32;
      v6 = v134;
      v101 = v31;
      v27 = v130;
      goto LABEL_40;
    }
  }
LABEL_41:
  if ( v141 )
    return sub_161E7C0((__int64)&v141, v141);
  return result;
}
