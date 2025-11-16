// Function: sub_207F710
// Address: 0x207f710
//
void __fastcall sub_207F710(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned __int16 *v8; // rsi
  int v9; // edx
  unsigned int v10; // eax
  unsigned __int16 *v11; // r12
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int16 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rdx
  int v18; // edx
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // r10
  unsigned __int8 v23; // r9
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // r13
  int v36; // eax
  unsigned int *v37; // rax
  __int64 **v38; // rax
  __int64 v39; // rax
  __int64 v40; // rsi
  unsigned int v41; // edx
  int v42; // r9d
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 *v46; // rax
  unsigned __int16 *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned int v50; // edx
  int v51; // r9d
  __int64 v52; // r14
  __int64 v53; // r8
  __int64 v54; // rax
  __int64 *v55; // rax
  __int64 v56; // rax
  __m128i v57; // xmm2
  __int64 v58; // rdx
  int v59; // esi
  __int64 v60; // rdi
  unsigned int v61; // esi
  unsigned int v62; // edx
  int v63; // r9d
  __int64 v64; // r14
  __int64 v65; // r8
  __int64 v66; // rax
  __int64 *v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // rdi
  unsigned int v71; // edx
  __int64 v72; // r14
  __int64 v73; // r8
  __int64 v74; // rax
  __int64 *v75; // rax
  unsigned int v76; // edx
  const __m128i *v77; // rax
  __int64 v78; // rcx
  __int64 v79; // r8
  const __m128i *v80; // rax
  unsigned __int64 v81; // r9
  __int64 v82; // rcx
  __m128 *v83; // rdx
  int v84; // r8d
  int v85; // r9d
  __int64 v86; // r14
  __int64 v87; // rax
  const __m128i *v88; // r14
  __int64 v89; // rax
  __m128i v90; // xmm0
  __int64 v91; // rcx
  __int64 v92; // rdx
  __int64 v93; // r14
  __m128i v94; // xmm0
  __int64 v95; // rdi
  unsigned __int64 v96; // r10
  __int64 v97; // rdx
  __int64 v98; // rdi
  __int64 v99; // rax
  int v100; // r8d
  int v101; // r9d
  __int64 v102; // rax
  unsigned __int8 *v103; // rax
  __int64 v104; // rax
  unsigned __int8 *v105; // rax
  __int64 v106; // rdi
  __int64 v107; // rax
  unsigned __int64 v108; // rdx
  __int64 v109; // r9
  __int64 v110; // rcx
  int v111; // r8d
  __int64 v112; // rax
  unsigned __int8 *v113; // r15
  __int64 *v114; // rax
  __int64 v115; // rdi
  unsigned int v116; // edx
  __int64 *v117; // rdi
  __int64 v118; // rax
  int v119; // edx
  __int64 v120; // r9
  __int64 v121; // r14
  __m128i v122; // xmm0
  __int64 *v123; // rax
  __int64 v124; // rax
  unsigned __int16 *v125; // r13
  unsigned int v126; // r12d
  unsigned int v127; // r14d
  __int64 *v128; // rax
  __int64 v129; // rdx
  __int64 v130; // r9
  __int64 *v131; // r8
  __int64 v132; // rdx
  __int64 *v133; // rdx
  const __m128i *v134; // r14
  __int64 v135; // rax
  int v136; // [rsp-8h] [rbp-10F8h]
  const __m128i *v137; // [rsp+8h] [rbp-10E8h]
  __int64 v138; // [rsp+8h] [rbp-10E8h]
  __int64 v139; // [rsp+8h] [rbp-10E8h]
  bool v140; // [rsp+35h] [rbp-10BBh]
  bool v141; // [rsp+36h] [rbp-10BAh]
  char v142; // [rsp+37h] [rbp-10B9h]
  unsigned __int8 v144; // [rsp+40h] [rbp-10B0h]
  unsigned int v145; // [rsp+40h] [rbp-10B0h]
  __int64 v146; // [rsp+40h] [rbp-10B0h]
  __int64 *v147; // [rsp+40h] [rbp-10B0h]
  __int64 v148; // [rsp+48h] [rbp-10A8h]
  _QWORD *v149; // [rsp+50h] [rbp-10A0h]
  __int64 v150; // [rsp+58h] [rbp-1098h]
  __m128i v151; // [rsp+60h] [rbp-1090h] BYREF
  unsigned __int64 v152; // [rsp+70h] [rbp-1080h]
  __int64 *v153; // [rsp+78h] [rbp-1078h]
  __m128i v154; // [rsp+80h] [rbp-1070h]
  __int64 *v155; // [rsp+90h] [rbp-1060h]
  __int64 v156; // [rsp+98h] [rbp-1058h]
  __int64 v157; // [rsp+A0h] [rbp-1050h]
  __int64 v158; // [rsp+A8h] [rbp-1048h]
  __int64 v159; // [rsp+B0h] [rbp-1040h] BYREF
  int v160; // [rsp+B8h] [rbp-1038h]
  __m128i v161; // [rsp+C0h] [rbp-1030h] BYREF
  __int64 v162; // [rsp+D0h] [rbp-1020h]
  __int64 v163; // [rsp+E0h] [rbp-1010h] BYREF
  int v164; // [rsp+E8h] [rbp-1008h]
  __int64 v165; // [rsp+F0h] [rbp-1000h]
  int v166; // [rsp+F8h] [rbp-FF8h]
  unsigned __int8 *v167; // [rsp+100h] [rbp-FF0h] BYREF
  __int64 v168; // [rsp+108h] [rbp-FE8h]
  unsigned __int8 *v169; // [rsp+110h] [rbp-FE0h] BYREF
  int v170; // [rsp+118h] [rbp-FD8h]
  __int64 *v171; // [rsp+140h] [rbp-FB0h] BYREF
  __int64 v172; // [rsp+148h] [rbp-FA8h]
  _BYTE v173[128]; // [rsp+150h] [rbp-FA0h] BYREF
  __m128i v174; // [rsp+1D0h] [rbp-F20h] BYREF
  __int64 v175; // [rsp+1E0h] [rbp-F10h]
  unsigned __int64 v176; // [rsp+1E8h] [rbp-F08h]
  __int64 v177; // [rsp+1F0h] [rbp-F00h]
  __int64 v178; // [rsp+1F8h] [rbp-EF8h]
  __int64 v179; // [rsp+200h] [rbp-EF0h]
  __int64 v180; // [rsp+208h] [rbp-EE8h]
  __int64 v181; // [rsp+210h] [rbp-EE0h]
  __int64 v182; // [rsp+218h] [rbp-ED8h]
  __int64 v183; // [rsp+220h] [rbp-ED0h]
  __int64 v184; // [rsp+228h] [rbp-EC8h] BYREF
  int v185; // [rsp+230h] [rbp-EC0h]
  __int64 v186; // [rsp+238h] [rbp-EB8h]
  _BYTE *v187; // [rsp+240h] [rbp-EB0h]
  __int64 v188; // [rsp+248h] [rbp-EA8h]
  _BYTE v189[1536]; // [rsp+250h] [rbp-EA0h] BYREF
  _BYTE *v190; // [rsp+850h] [rbp-8A0h]
  __int64 v191; // [rsp+858h] [rbp-898h]
  _BYTE v192[512]; // [rsp+860h] [rbp-890h] BYREF
  _BYTE *v193; // [rsp+A60h] [rbp-690h]
  __int64 v194; // [rsp+A68h] [rbp-688h]
  _BYTE v195[1536]; // [rsp+A70h] [rbp-680h] BYREF
  _BYTE *v196; // [rsp+1070h] [rbp-80h]
  __int64 v197; // [rsp+1078h] [rbp-78h]
  _BYTE v198[112]; // [rsp+1080h] [rbp-70h] BYREF

  v8 = (unsigned __int16 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v9 = *(_DWORD *)(a1 + 536);
  v10 = v8[9];
  v11 = v8;
  v159 = 0;
  LODWORD(v152) = (v10 >> 2) & 0x3FFFDFFF;
  v141 = (_DWORD)v152 == 13;
  LOBYTE(v10) = *(_BYTE *)(*(_QWORD *)v8 + 8LL);
  v160 = v9;
  v12 = (_BYTE)v10 == 0;
  v142 = v10;
  v13 = *(_QWORD *)a1;
  v140 = !v12;
  if ( *(_QWORD *)a1 )
  {
    v153 = &v159;
    if ( &v159 != (__int64 *)(v13 + 48) )
    {
      v14 = *(_QWORD *)(v13 + 48);
      v159 = v14;
      if ( v14 )
        sub_1623A60((__int64)&v159, v14, 2);
    }
  }
  else
  {
    v153 = &v159;
  }
  if ( (*((_BYTE *)v11 + 23) & 0x40) != 0 )
    v15 = (unsigned __int16 *)*((_QWORD *)v11 - 1);
  else
    v15 = &v11[-12 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF)];
  v16 = sub_20685E0(a1, *((__int64 **)v15 + 6), a4, a5, a6);
  v151.m128i_i64[1] = v17;
  v18 = *((unsigned __int16 *)v16 + 12);
  v19 = v16;
  v151.m128i_i64[0] = (__int64)v16;
  if ( (_WORD)v18 == 10 || v18 == 32 )
  {
    v26 = v16[11];
    v27 = *(_QWORD **)(v26 + 24);
    if ( *(_DWORD *)(v26 + 32) > 0x40u )
      v27 = (_QWORD *)*v27;
    v157 = sub_1D38E70(*(_QWORD *)(a1 + 552), (__int64)v27, (__int64)v153, 1u, a4, *(double *)a5.m128i_i64, a6);
    v151.m128i_i64[0] = v157;
    v158 = v28;
    v151.m128i_i64[1] = (unsigned int)v28 | v151.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else if ( (unsigned __int16)(v18 - 34) <= 1u || (unsigned __int16)(v18 - 12) <= 1u )
  {
    v20 = v16[5];
    v21 = v19[9];
    v22 = *(_QWORD **)(a1 + 552);
    v23 = *(_BYTE *)v20;
    v24 = *(_QWORD *)(v20 + 8);
    v174.m128i_i64[0] = v21;
    if ( v21 )
    {
      v144 = v23;
      v149 = v22;
      v150 = v24;
      sub_1623A60((__int64)&v174, v21, 2);
      v23 = v144;
      v22 = v149;
      v24 = v150;
    }
    v174.m128i_i32[2] = *((_DWORD *)v19 + 16);
    v155 = sub_1D29600(v22, v19[11], (__int64)&v174, v23, v24, 0, 1, 0);
    v151.m128i_i64[0] = (__int64)v155;
    v156 = v25;
    v151.m128i_i64[1] = (unsigned int)v25 | v151.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v174.m128i_i64[0] )
      sub_161E7C0((__int64)&v174, v174.m128i_i64[0]);
  }
  v29 = sub_20685E0(a1, *(__int64 **)&v11[-12 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF) + 36], a4, a5, a6)[11];
  if ( *(_DWORD *)(v29 + 32) <= 0x40u )
  {
    v30 = *(_QWORD *)(v29 + 24);
    v145 = v30;
    if ( (_DWORD)v152 != 13 )
      goto LABEL_20;
LABEL_99:
    v124 = sub_1643270(*(_QWORD **)(*(_QWORD *)(a1 + 552) + 48LL));
    LODWORD(v30) = 0;
    v31 = v124;
    goto LABEL_21;
  }
  v30 = **(_QWORD **)(v29 + 24);
  v145 = v30;
  if ( (_DWORD)v152 == 13 )
    goto LABEL_99;
LABEL_20:
  v31 = *(_QWORD *)v11;
LABEL_21:
  v32 = *(_QWORD *)(a1 + 552);
  v174 = 0u;
  v176 = 0xFFFFFFFF00000020LL;
  v183 = v32;
  v187 = v189;
  v188 = 0x2000000000LL;
  v191 = 0x2000000000LL;
  v194 = 0x2000000000LL;
  v190 = v192;
  v196 = v198;
  v197 = 0x400000000LL;
  v193 = v195;
  v175 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  sub_207E8C0(a1, (__int64)&v174, a2, 4u, v30, v31, a4, a5, a6, *(_OWORD *)&v151, 1u);
  sub_2061DC0(&v161, a1, &v174, a3, a4, a5, a6);
  v33 = v162;
  if ( v142 && *(_WORD *)(v162 + 24) == 47 )
    v33 = **(_QWORD **)(v162 + 32);
  v34 = 0;
  v35 = **(_QWORD **)(v33 + 32);
  v36 = *(_DWORD *)(v35 + 56);
  if ( v36 )
  {
    v37 = (unsigned int *)(*(_QWORD *)(v35 + 32) + 40LL * (unsigned int)(v36 - 1));
    v34 = *(_QWORD *)v37;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v37 + 40LL) + 16LL * v37[2]) != 111 )
      v34 = 0;
  }
  v171 = (__int64 *)v173;
  v172 = 0x800000000LL;
  if ( (*((_BYTE *)v11 + 23) & 0x40) != 0 )
    v38 = (__int64 **)*((_QWORD *)v11 - 1);
  else
    v38 = (__int64 **)&v11[-12 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF)];
  v39 = sub_20685E0(a1, *v38, a4, a5, a6)[11];
  if ( *(_DWORD *)(v39 + 32) <= 0x40u )
    v40 = *(_QWORD *)(v39 + 24);
  else
    v40 = **(_QWORD **)(v39 + 24);
  v43 = sub_1D38BB0(*(_QWORD *)(a1 + 552), v40, (__int64)v153, 6, 0, 1, a4, *(double *)a5.m128i_i64, a6, 0);
  v44 = (unsigned int)v172;
  v45 = v41;
  if ( (unsigned int)v172 >= HIDWORD(v172) )
  {
    v139 = v41;
    sub_16CD150((__int64)&v171, v173, 0, 16, v41, v42);
    v44 = (unsigned int)v172;
    v45 = v139;
  }
  v46 = &v171[2 * v44];
  *v46 = v43;
  v46[1] = v45;
  LODWORD(v172) = v172 + 1;
  if ( (*((_BYTE *)v11 + 23) & 0x40) != 0 )
    v47 = (unsigned __int16 *)*((_QWORD *)v11 - 1);
  else
    v47 = &v11[-12 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF)];
  v48 = sub_20685E0(a1, *((__int64 **)v47 + 3), a4, a5, a6)[11];
  if ( *(_DWORD *)(v48 + 32) <= 0x40u )
    v49 = *(_QWORD *)(v48 + 24);
  else
    v49 = **(_QWORD **)(v48 + 24);
  v52 = sub_1D38BB0(*(_QWORD *)(a1 + 552), v49, (__int64)v153, 5, 0, 1, a4, *(double *)a5.m128i_i64, a6, 0);
  v53 = v50;
  v54 = (unsigned int)v172;
  if ( (unsigned int)v172 >= HIDWORD(v172) )
  {
    v138 = v50;
    sub_16CD150((__int64)&v171, v173, 0, 16, v50, v51);
    v54 = (unsigned int)v172;
    v53 = v138;
  }
  v55 = &v171[2 * v54];
  *v55 = v52;
  v55[1] = v53;
  v56 = (unsigned int)(v172 + 1);
  LODWORD(v172) = v56;
  if ( HIDWORD(v172) <= (unsigned int)v56 )
  {
    sub_16CD150((__int64)&v171, v173, 0, 16, v53, v51);
    v56 = (unsigned int)v172;
  }
  v57 = _mm_load_si128(&v151);
  v58 = (__int64)v153;
  *(__m128i *)&v171[2 * v56] = v57;
  v59 = *(_DWORD *)(v35 + 56);
  v60 = *(_QWORD *)(a1 + 552);
  LODWORD(v172) = v172 + 1;
  v61 = (v34 == 0) + v59 - 4;
  if ( (_DWORD)v152 == 13 )
    v61 = v145;
  v64 = sub_1D38BB0(v60, v61, v58, 5, 0, 1, a4, *(double *)a5.m128i_i64, v57, 0);
  v65 = v62;
  v66 = (unsigned int)v172;
  if ( (unsigned int)v172 >= HIDWORD(v172) )
  {
    v151.m128i_i64[0] = v62;
    sub_16CD150((__int64)&v171, v173, 0, 16, v62, v63);
    v66 = (unsigned int)v172;
    v65 = v151.m128i_i64[0];
  }
  v67 = &v171[2 * v66];
  v68 = (unsigned int)v152;
  *v67 = v64;
  v69 = (__int64)v153;
  v67[1] = v65;
  v70 = *(_QWORD *)(a1 + 552);
  LODWORD(v172) = v172 + 1;
  v72 = sub_1D38BB0(v70, v68, v69, 5, 0, 1, a4, *(double *)a5.m128i_i64, v57, 0);
  v73 = v71;
  v74 = (unsigned int)v172;
  if ( (unsigned int)v172 >= HIDWORD(v172) )
  {
    v151.m128i_i64[0] = v71;
    sub_16CD150((__int64)&v171, v173, 0, 16, v71, v136);
    v74 = (unsigned int)v172;
    v73 = v151.m128i_i64[0];
  }
  v75 = &v171[2 * v74];
  *v75 = v72;
  v75[1] = v73;
  v76 = v172 + 1;
  LODWORD(v172) = v172 + 1;
  v151.m128i_i32[0] = v145 + 4;
  if ( (_DWORD)v152 == 13 && v145 )
  {
    v152 = v35;
    v125 = v11;
    v126 = 4;
    v127 = v145 + 4;
    do
    {
      v128 = sub_20685E0(
               a1,
               *(__int64 **)&v125[12 * (v126 - (unsigned __int64)(*((_DWORD *)v125 + 5) & 0xFFFFFFF))],
               a4,
               a5,
               v57);
      v130 = v129;
      v131 = v128;
      v132 = (unsigned int)v172;
      if ( (unsigned int)v172 >= HIDWORD(v172) )
      {
        v147 = v128;
        v148 = v130;
        sub_16CD150((__int64)&v171, v173, 0, 16, (int)v128, v130);
        v132 = (unsigned int)v172;
        v131 = v147;
        v130 = v148;
      }
      v133 = &v171[2 * v132];
      ++v126;
      *v133 = (__int64)v131;
      v133[1] = v130;
      v76 = v172 + 1;
      LODWORD(v172) = v172 + 1;
    }
    while ( v126 != v127 );
    v11 = v125;
    v35 = v152;
  }
  v77 = *(const __m128i **)(v35 + 32);
  v78 = 5LL * *(unsigned int *)(v35 + 56);
  v79 = (__int64)&v77[-2] + v78 * 8 - 8;
  if ( v34 )
    v79 = (__int64)&v77[-5].m128i_i64[v78];
  v80 = v77 + 5;
  v81 = 0xCCCCCCCCCCCCCCCDLL * ((v79 - (__int64)v80) >> 3);
  v82 = v76;
  if ( v81 > HIDWORD(v172) - (unsigned __int64)v76 )
  {
    v137 = v80;
    v146 = v79;
    v152 = 0xCCCCCCCCCCCCCCCDLL * ((v79 - (__int64)v80) >> 3);
    sub_16CD150((__int64)&v171, v173, v81 + v76, 16, v79, v81);
    v82 = (unsigned int)v172;
    v80 = v137;
    v79 = v146;
    LODWORD(v81) = v152;
  }
  v83 = (__m128 *)&v171[2 * v82];
  if ( v80 != (const __m128i *)v79 )
  {
    do
    {
      if ( v83 )
      {
        a5 = _mm_loadu_si128(v80);
        *v83 = (__m128)a5;
      }
      v80 = (const __m128i *)((char *)v80 + 40);
      ++v83;
    }
    while ( (const __m128i *)v79 != v80 );
    LODWORD(v82) = v172;
  }
  LODWORD(v172) = v81 + v82;
  sub_207EC30(a2, v151.m128i_u32[0], (__int64)v153, (__int64)&v171, a1, a4, a5, v57);
  v86 = *(_QWORD *)(v35 + 32) + 40LL * *(unsigned int *)(v35 + 56);
  v87 = (unsigned int)v172;
  if ( v34 )
  {
    if ( (unsigned int)v172 >= HIDWORD(v172) )
    {
      sub_16CD150((__int64)&v171, v173, 0, 16, v84, v85);
      v87 = (unsigned int)v172;
    }
    *(__m128i *)&v171[2 * v87] = _mm_loadu_si128((const __m128i *)(v86 - 80));
    v88 = *(const __m128i **)(v35 + 32);
    v89 = (unsigned int)(v172 + 1);
    LODWORD(v172) = v89;
    if ( (unsigned int)v89 >= HIDWORD(v172) )
    {
      sub_16CD150((__int64)&v171, v173, 0, 16, v84, v85);
      v90 = _mm_loadu_si128(v88);
      v89 = (unsigned int)v172;
    }
    else
    {
      v90 = _mm_loadu_si128(v88);
    }
    *(__m128i *)&v171[2 * v89] = v90;
    v91 = 5LL * *(unsigned int *)(v35 + 56);
    v92 = *(_QWORD *)(v35 + 32);
    LODWORD(v172) = v172 + 1;
    v93 = v92 + 8 * v91;
    if ( (unsigned int)v172 >= HIDWORD(v172) )
      sub_16CD150((__int64)&v171, v173, 0, 16, v84, v85);
    v94 = _mm_loadu_si128((const __m128i *)(v93 - 40));
  }
  else
  {
    if ( (unsigned int)v172 >= HIDWORD(v172) )
    {
      sub_16CD150((__int64)&v171, v173, 0, 16, v84, v85);
      v87 = (unsigned int)v172;
    }
    *(__m128i *)&v171[2 * v87] = _mm_loadu_si128((const __m128i *)(v86 - 40));
    v134 = *(const __m128i **)(v35 + 32);
    v135 = (unsigned int)(v172 + 1);
    LODWORD(v172) = v135;
    if ( (unsigned int)v135 < HIDWORD(v172) )
    {
      *(__m128i *)&v171[2 * v135] = _mm_loadu_si128(v134);
      LODWORD(v172) = v172 + 1;
      goto LABEL_66;
    }
    sub_16CD150((__int64)&v171, v173, 0, 16, v84, v85);
    v94 = _mm_loadu_si128(v134);
  }
  *(__m128i *)&v171[2 * (unsigned int)v172] = v94;
  LODWORD(v172) = v172 + 1;
LABEL_66:
  v95 = *(_QWORD *)(a1 + 552);
  if ( v141 && v140 )
  {
    v96 = *(_QWORD *)(v95 + 16);
    v97 = *(_QWORD *)v11;
    v167 = (unsigned __int8 *)&v169;
    v168 = 0x300000000LL;
    v98 = *(_QWORD *)(v95 + 32);
    v152 = v96;
    v151.m128i_i64[0] = v97;
    v99 = sub_1E0A0C0(v98);
    sub_20C7CE0(v152, v99, v151.m128i_i64[0], &v167, 0, 0);
    v102 = (unsigned int)v168;
    if ( (unsigned int)v168 >= HIDWORD(v168) )
    {
      sub_16CD150((__int64)&v167, &v169, 0, 16, v100, v101);
      v102 = (unsigned int)v168;
    }
    v103 = &v167[16 * v102];
    *(_QWORD *)v103 = 1;
    *((_QWORD *)v103 + 1) = 0;
    v104 = (unsigned int)(v168 + 1);
    LODWORD(v168) = v104;
    if ( HIDWORD(v168) <= (unsigned int)v104 )
    {
      sub_16CD150((__int64)&v167, &v169, 0, 16, v100, v101);
      v104 = (unsigned int)v168;
    }
    v105 = &v167[16 * v104];
    *(_QWORD *)v105 = 111;
    *((_QWORD *)v105 + 1) = 0;
    v106 = *(_QWORD *)(a1 + 552);
    LODWORD(v168) = v168 + 1;
    v107 = sub_1D25C30(v106, v167, (unsigned int)v168);
    v110 = v107;
    v111 = v108;
    if ( v167 != (unsigned __int8 *)&v169 )
    {
      v152 = v108;
      v151.m128i_i64[0] = v107;
      _libc_free((unsigned __int64)v167);
      v111 = v152;
      v110 = v151.m128i_i64[0];
    }
    v112 = sub_1D23DE0(*(_QWORD **)(a1 + 552), 21, (__int64)v153, v110, v111, v109, v171, (unsigned int)v172);
    v167 = (unsigned __int8 *)v11;
    v113 = (unsigned __int8 *)v112;
    v114 = sub_205F5C0(a1 + 8, (__int64 *)&v167);
    v114[1] = (__int64)v113;
    *((_DWORD *)v114 + 4) = 0;
    v115 = *(_QWORD *)(a1 + 552);
    v163 = v35;
    v164 = 0;
    v165 = v35;
    v166 = 1;
    v167 = v113;
    LODWORD(v168) = 1;
    v169 = v113;
    v170 = 2;
    sub_1D451D0(v115, &v163, (__int64 *)&v167, 2);
  }
  else
  {
    v118 = sub_1D252B0(v95, 1, 0, 111, 0);
    v121 = sub_1D23DE0(*(_QWORD **)(a1 + 552), 21, (__int64)v153, v118, v119, v120, v171, (unsigned int)v172);
    if ( v142 )
    {
      v122 = _mm_load_si128(&v161);
      v167 = (unsigned __int8 *)v11;
      v151 = v122;
      v123 = sub_205F5C0(a1 + 8, (__int64 *)&v167);
      v154 = _mm_load_si128(&v151);
      v123[1] = v154.m128i_i64[0];
      *((_DWORD *)v123 + 4) = v154.m128i_i32[2];
    }
    sub_1D444E0(*(_QWORD *)(a1 + 552), v35, v121);
  }
  sub_1D2DE10(*(_QWORD *)(a1 + 552), v35, v116);
  v117 = v171;
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 56LL) + 40LL) = 1;
  if ( v117 != (__int64 *)v173 )
    _libc_free((unsigned __int64)v117);
  if ( v196 != v198 )
    _libc_free((unsigned __int64)v196);
  if ( v193 != v195 )
    _libc_free((unsigned __int64)v193);
  if ( v190 != v192 )
    _libc_free((unsigned __int64)v190);
  if ( v187 != v189 )
    _libc_free((unsigned __int64)v187);
  if ( v184 )
    sub_161E7C0((__int64)&v184, v184);
  if ( v180 )
    j_j___libc_free_0(v180, v182 - v180);
  if ( v159 )
    sub_161E7C0((__int64)v153, v159);
}
