// Function: sub_212B970
// Address: 0x212b970
//
__int64 __fastcall sub_212B970(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  __int64 v8; // rax
  __int64 v9; // r14
  __m128 v10; // xmm0
  __int64 v11; // r12
  __int64 v12; // r12
  const void **v13; // r14
  __int64 v14; // rsi
  int v15; // eax
  unsigned __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // r15d
  __int128 v19; // rax
  __int128 v20; // rax
  __int128 v21; // rax
  __int64 *v22; // r10
  __int64 v23; // rax
  unsigned __int8 v24; // al
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int16 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int8 v35; // al
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rdx
  __int16 v41; // ax
  __int16 *v42; // rdx
  __int64 *v43; // r13
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r15
  __int64 *v48; // r14
  __int64 *v49; // rax
  unsigned __int64 v50; // rdx
  __int64 *v51; // r14
  int v52; // edx
  unsigned int v53; // edx
  __int64 *v54; // rax
  __int64 *v55; // r15
  unsigned int v56; // edx
  __int64 *v57; // rax
  unsigned __int64 v58; // rcx
  __int64 v59; // r13
  __int64 v60; // rdx
  __int64 v61; // r14
  __int128 v62; // kr00_16
  const void **v63; // r8
  __int64 v64; // rdx
  char v65; // al
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 *v68; // rax
  int v69; // edx
  unsigned __int64 v70; // rdx
  int v71; // edx
  __int64 *v72; // rax
  __int64 *v73; // r13
  unsigned int v74; // edx
  __int64 *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r15
  __int64 *v78; // r14
  __int64 *v79; // rax
  unsigned __int64 v80; // rdx
  unsigned int v81; // edx
  __int64 v82; // rax
  unsigned int v83; // edx
  __int64 v84; // r13
  const void **v85; // r8
  __int64 *v86; // r14
  unsigned int v87; // edx
  __int64 v88; // r10
  unsigned __int64 v89; // rcx
  __int64 v90; // r11
  __int64 v91; // rax
  char v92; // dl
  __int64 v93; // rax
  bool v94; // al
  unsigned int v95; // esi
  __int64 *v96; // rax
  int v97; // edx
  const void **v98; // rcx
  unsigned __int64 v99; // rdx
  __int64 *v100; // r13
  __int64 *v101; // rax
  __int64 v102; // rdx
  int v103; // edx
  __int64 *v104; // r13
  unsigned int v105; // edx
  __int128 v106; // rax
  __int64 *v107; // rax
  unsigned __int64 v108; // rdx
  __int64 *v109; // rax
  unsigned int v110; // esi
  int v111; // edx
  __int128 v112; // rax
  unsigned int v113; // edx
  __int64 *v114; // rax
  __int64 *v115; // r14
  unsigned int v116; // edx
  __int64 *v117; // rax
  __int64 v118; // rdx
  __int64 *v119; // rax
  int v120; // edx
  unsigned __int64 v121; // rdx
  const void **v122; // rcx
  int v123; // edx
  bool v125; // al
  __int128 v126; // [rsp-20h] [rbp-230h]
  __int128 v127; // [rsp-20h] [rbp-230h]
  __int128 v128; // [rsp-20h] [rbp-230h]
  __int128 v129; // [rsp-10h] [rbp-220h]
  __int128 v130; // [rsp-10h] [rbp-220h]
  unsigned int v131; // [rsp+8h] [rbp-208h]
  __int128 v133; // [rsp+20h] [rbp-1F0h]
  __int128 v134; // [rsp+30h] [rbp-1E0h]
  const void **v135; // [rsp+30h] [rbp-1E0h]
  __int64 *v137; // [rsp+48h] [rbp-1C8h]
  __int64 *v138; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v139; // [rsp+48h] [rbp-1C8h]
  __int64 v140; // [rsp+50h] [rbp-1C0h]
  __int64 v141; // [rsp+50h] [rbp-1C0h]
  __int128 v142; // [rsp+50h] [rbp-1C0h]
  __int64 v143; // [rsp+60h] [rbp-1B0h]
  __int64 v144; // [rsp+60h] [rbp-1B0h]
  const void **v145; // [rsp+60h] [rbp-1B0h]
  __int16 *v146; // [rsp+60h] [rbp-1B0h]
  const void **v147; // [rsp+60h] [rbp-1B0h]
  __int64 (__fastcall *v148)(__int64, __int64, __int64, _QWORD, const void **); // [rsp+68h] [rbp-1A8h]
  const void **v149; // [rsp+68h] [rbp-1A8h]
  __int64 (__fastcall *v150)(__int64, __int64, __int64, __int64, const void **); // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v151; // [rsp+68h] [rbp-1A8h]
  __int64 *v152; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v153; // [rsp+68h] [rbp-1A8h]
  unsigned int v154; // [rsp+70h] [rbp-1A0h]
  __int64 *v155; // [rsp+70h] [rbp-1A0h]
  __int64 *v156; // [rsp+70h] [rbp-1A0h]
  __int128 v157; // [rsp+70h] [rbp-1A0h]
  __int64 v158; // [rsp+80h] [rbp-190h]
  unsigned __int64 v159; // [rsp+80h] [rbp-190h]
  __int16 *v160; // [rsp+80h] [rbp-190h]
  __int64 *v161; // [rsp+88h] [rbp-188h]
  unsigned __int64 v162; // [rsp+88h] [rbp-188h]
  __int128 v163; // [rsp+90h] [rbp-180h]
  __int64 v164; // [rsp+90h] [rbp-180h]
  __int64 *v165; // [rsp+90h] [rbp-180h]
  __int64 *v166; // [rsp+90h] [rbp-180h]
  __int64 v167; // [rsp+98h] [rbp-178h]
  __int64 v168; // [rsp+98h] [rbp-178h]
  __int64 v169; // [rsp+98h] [rbp-178h]
  unsigned __int8 v170; // [rsp+A0h] [rbp-170h]
  __int64 v171; // [rsp+A0h] [rbp-170h]
  __int128 v172; // [rsp+A0h] [rbp-170h]
  __int128 v173; // [rsp+A0h] [rbp-170h]
  __int128 v174; // [rsp+A0h] [rbp-170h]
  unsigned __int64 v175; // [rsp+B0h] [rbp-160h]
  __int64 *v176; // [rsp+B0h] [rbp-160h]
  __int16 *v177; // [rsp+B8h] [rbp-158h]
  int v178; // [rsp+D8h] [rbp-138h]
  int v179; // [rsp+118h] [rbp-F8h]
  int v180; // [rsp+158h] [rbp-B8h]
  unsigned __int64 v181; // [rsp+180h] [rbp-90h] BYREF
  const void **v182; // [rsp+188h] [rbp-88h]
  __int64 v183; // [rsp+190h] [rbp-80h] BYREF
  int v184; // [rsp+198h] [rbp-78h]
  __int128 v185; // [rsp+1A0h] [rbp-70h] BYREF
  __int128 v186; // [rsp+1B0h] [rbp-60h] BYREF
  char v187[8]; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v188; // [rsp+1C8h] [rbp-48h]
  const void **v189; // [rsp+1D0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(v8 + 40);
  v10 = (__m128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = 16LL * *(unsigned int *)(v8 + 48);
  sub_1F40D10(
    (__int64)v187,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v12 = *(_QWORD *)(v9 + 40) + v11;
  v182 = v189;
  v13 = *(const void ***)(v12 + 8);
  LOBYTE(v181) = v188;
  v170 = *(_BYTE *)v12;
  if ( (_BYTE)v188 )
    v154 = sub_2127930(v188);
  else
    v154 = sub_1F58D40((__int64)&v181);
  v14 = *(_QWORD *)(a2 + 72);
  v183 = v14;
  if ( v14 )
    sub_1623A60((__int64)&v183, v14, 2);
  v15 = *(_DWORD *)(a2 + 64);
  DWORD2(v185) = 0;
  DWORD2(v186) = 0;
  v184 = v15;
  v16 = *(unsigned __int64 **)(a2 + 32);
  *(_QWORD *)&v185 = 0;
  v17 = v16[1];
  *(_QWORD *)&v186 = 0;
  sub_20174B0((__int64)a1, *v16, v17, &v185, &v186);
  v18 = v170;
  *(_QWORD *)&v19 = sub_1D38BB0(a1[1], v154, (__int64)&v183, v170, v13, 0, (__m128i)v10, a6, a7, 0);
  v163 = v19;
  *(_QWORD *)&v20 = sub_1D332F0(
                      (__int64 *)a1[1],
                      53,
                      (__int64)&v183,
                      v170,
                      v13,
                      0,
                      *(double *)v10.m128_u64,
                      a6,
                      a7,
                      v10.m128_i64[0],
                      v10.m128_u64[1],
                      v19);
  v134 = v20;
  *(_QWORD *)&v21 = sub_1D332F0(
                      (__int64 *)a1[1],
                      53,
                      (__int64)&v183,
                      v170,
                      v13,
                      0,
                      *(double *)v10.m128_u64,
                      a6,
                      a7,
                      v163,
                      *((unsigned __int64 *)&v163 + 1),
                      *(_OWORD *)&v10);
  v22 = (__int64 *)a1[1];
  v133 = v21;
  v161 = v22;
  v143 = *a1;
  v148 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, const void **))(*(_QWORD *)*a1 + 264LL);
  v158 = v22[6];
  v23 = sub_1E0A0C0(v22[4]);
  v24 = v148(v143, v23, v158, v170, v13);
  v149 = (const void **)v25;
  v159 = v24;
  v28 = sub_1D28D50(v161, 0xCu, v25, v24, v26, v27);
  v30 = sub_1D3A900(
          v161,
          0x89u,
          (__int64)&v183,
          v159,
          v149,
          0,
          v10,
          a6,
          a7,
          v10.m128_u64[0],
          (__int16 *)v10.m128_u64[1],
          v163,
          v28,
          v29);
  v160 = v31;
  v137 = (__int64 *)a1[1];
  v162 = (unsigned __int64)v30;
  *(_QWORD *)&v163 = sub_1D38BB0((__int64)v137, 0, (__int64)&v183, v170, v13, 0, (__m128i)v10, a6, a7, 0);
  v32 = a1[1];
  *((_QWORD *)&v163 + 1) = v33;
  v140 = v170;
  v144 = *a1;
  v150 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(*(_QWORD *)*a1 + 264LL);
  v171 = *(_QWORD *)(v32 + 48);
  v34 = sub_1E0A0C0(*(_QWORD *)(v32 + 32));
  v35 = v150(v144, v34, v171, v140, v13);
  v145 = (const void **)v36;
  v151 = v35;
  v39 = sub_1D28D50(v137, 0x11u, v36, v35, v37, v38);
  v152 = sub_1D3A900(
           v137,
           0x89u,
           (__int64)&v183,
           v151,
           v145,
           0,
           v10,
           a6,
           a7,
           v10.m128_u64[0],
           (__int16 *)v10.m128_u64[1],
           v163,
           v39,
           v40);
  v41 = *(_WORD *)(a2 + 24);
  v146 = v42;
  if ( v41 == 123 )
  {
    *(_QWORD *)&v174 = sub_1D332F0(
                         (__int64 *)a1[1],
                         123,
                         (__int64)&v183,
                         (unsigned int)v181,
                         v182,
                         0,
                         *(double *)v10.m128_u64,
                         a6,
                         a7,
                         v186,
                         *((unsigned __int64 *)&v186 + 1),
                         *(_OWORD *)&v10);
    v104 = (__int64 *)a1[1];
    *((_QWORD *)&v174 + 1) = v105;
    *(_QWORD *)&v106 = sub_1D332F0(
                         v104,
                         122,
                         (__int64)&v183,
                         (unsigned int)v181,
                         v182,
                         0,
                         *(double *)v10.m128_u64,
                         a6,
                         a7,
                         v186,
                         *((unsigned __int64 *)&v186 + 1),
                         v133);
    v142 = v106;
    v107 = sub_1D332F0(
             (__int64 *)a1[1],
             124,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             0,
             *(double *)v10.m128_u64,
             a6,
             a7,
             v185,
             *((unsigned __int64 *)&v185 + 1),
             *(_OWORD *)&v10);
    v109 = sub_1D332F0(
             v104,
             119,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             0,
             *(double *)v10.m128_u64,
             a6,
             a7,
             (__int64)v107,
             v108,
             v142);
    v110 = v154;
    v156 = (__int64 *)a1[1];
    v176 = v109;
    LODWORD(v104) = v111;
    *(_QWORD *)&v112 = sub_1D38BB0((__int64)v156, v110 - 1, (__int64)&v183, v18, v13, 0, (__m128i)v10, a6, a7, 0);
    v166 = sub_1D332F0(
             v156,
             123,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             0,
             *(double *)v10.m128_u64,
             a6,
             a7,
             v186,
             *((unsigned __int64 *)&v186 + 1),
             v112);
    v169 = v113;
    v114 = sub_1D332F0(
             (__int64 *)a1[1],
             123,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             0,
             *(double *)v10.m128_u64,
             a6,
             a7,
             v186,
             *((unsigned __int64 *)&v186 + 1),
             v134);
    v115 = (__int64 *)a1[1];
    *((_QWORD *)&v128 + 1) = (unsigned int)v104;
    *(_QWORD *)&v128 = v176;
    v117 = sub_1F810E0(
             v115,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             v162,
             v160,
             v10,
             a6,
             a7,
             v128,
             (__int64)v114,
             v116);
    v119 = sub_1F810E0(
             v115,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             (unsigned __int64)v152,
             v146,
             v10,
             a6,
             a7,
             v185,
             (__int64)v117,
             v118);
    v178 = v120;
    v121 = (unsigned int)v181;
    *(_QWORD *)a3 = v119;
    v122 = v182;
    *(_DWORD *)(a3 + 8) = v178;
    *(_QWORD *)a4 = sub_1F810E0(
                      (__int64 *)a1[1],
                      (__int64)&v183,
                      v121,
                      v122,
                      v162,
                      v160,
                      v10,
                      a6,
                      a7,
                      v174,
                      (__int64)v166,
                      v169);
    *(_DWORD *)(a4 + 8) = v123;
  }
  else if ( v41 == 124 )
  {
    *(_QWORD *)&v172 = sub_1D332F0(
                         (__int64 *)a1[1],
                         124,
                         (__int64)&v183,
                         (unsigned int)v181,
                         v182,
                         0,
                         *(double *)v10.m128_u64,
                         a6,
                         a7,
                         v186,
                         *((unsigned __int64 *)&v186 + 1),
                         *(_OWORD *)&v10);
    v43 = (__int64 *)a1[1];
    *((_QWORD *)&v172 + 1) = v44;
    v45 = sub_1D332F0(
            v43,
            122,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v186,
            *((unsigned __int64 *)&v186 + 1),
            v133);
    v47 = v46;
    v48 = v45;
    v49 = sub_1D332F0(
            (__int64 *)a1[1],
            124,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v185,
            *((unsigned __int64 *)&v185 + 1),
            *(_OWORD *)&v10);
    *((_QWORD *)&v129 + 1) = v47;
    *(_QWORD *)&v129 = v48;
    v51 = sub_1D332F0(
            v43,
            119,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            (__int64)v49,
            v50,
            v129);
    LODWORD(v43) = v52;
    v164 = sub_1D38BB0(a1[1], 0, (__int64)&v183, (unsigned int)v181, v182, 0, (__m128i)v10, a6, a7, 0);
    v167 = v53;
    v54 = sub_1D332F0(
            (__int64 *)a1[1],
            124,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v186,
            *((unsigned __int64 *)&v186 + 1),
            v134);
    v55 = (__int64 *)a1[1];
    *((_QWORD *)&v126 + 1) = (unsigned int)v43;
    *(_QWORD *)&v126 = v51;
    v57 = sub_1F810E0(v55, (__int64)&v183, (unsigned int)v181, v182, v162, v160, v10, a6, a7, v126, (__int64)v54, v56);
    v58 = v181;
    v59 = (__int64)v57;
    v61 = v60;
    v62 = v185;
    v177 = v146;
    v63 = v182;
    v64 = v152[5] + 16LL * (unsigned int)v146;
    v175 = (unsigned __int64)v152;
    v65 = *(_BYTE *)v64;
    v66 = *(_QWORD *)(v64 + 8);
    v187[0] = v65;
    v188 = v66;
    if ( v65 )
    {
      v67 = ((unsigned __int8)(v65 - 14) < 0x60u) + 134;
    }
    else
    {
      v147 = v182;
      v153 = v181;
      v157 = v185;
      v125 = sub_1F58D20((__int64)v187);
      v63 = v147;
      v58 = v153;
      v67 = 134 - (!v125 - 1);
      v62 = v157;
    }
    v68 = sub_1D3A900(v55, v67, (__int64)&v183, v58, v63, 0, v10, a6, a7, v175, v177, v62, v59, v61);
    v179 = v69;
    v70 = (unsigned int)v181;
    *(_QWORD *)a3 = v68;
    *(_DWORD *)(a3 + 8) = v179;
    *(_QWORD *)a4 = sub_1F810E0((__int64 *)a1[1], (__int64)&v183, v70, v182, v162, v160, v10, a6, a7, v172, v164, v167);
    *(_DWORD *)(a4 + 8) = v71;
  }
  else
  {
    v72 = sub_1D332F0(
            (__int64 *)a1[1],
            122,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v185,
            *((unsigned __int64 *)&v185 + 1),
            *(_OWORD *)&v10);
    v73 = (__int64 *)a1[1];
    v131 = v74;
    v138 = v72;
    v75 = sub_1D332F0(
            v73,
            124,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v185,
            *((unsigned __int64 *)&v185 + 1),
            v133);
    v77 = v76;
    v78 = v75;
    v79 = sub_1D332F0(
            (__int64 *)a1[1],
            122,
            (__int64)&v183,
            (unsigned int)v181,
            v182,
            0,
            *(double *)v10.m128_u64,
            a6,
            a7,
            v186,
            *((unsigned __int64 *)&v186 + 1),
            *(_OWORD *)&v10);
    *((_QWORD *)&v130 + 1) = v77;
    *(_QWORD *)&v130 = v78;
    *(_QWORD *)&v173 = sub_1D332F0(
                         v73,
                         119,
                         (__int64)&v183,
                         (unsigned int)v181,
                         v182,
                         0,
                         *(double *)v10.m128_u64,
                         a6,
                         a7,
                         (__int64)v79,
                         v80,
                         v130);
    *((_QWORD *)&v173 + 1) = v81;
    v82 = sub_1D38BB0(a1[1], 0, (__int64)&v183, (unsigned int)v181, v182, 0, (__m128i)v10, a6, a7, 0);
    v84 = v83;
    v141 = v82;
    v165 = sub_1D332F0(
             (__int64 *)a1[1],
             122,
             (__int64)&v183,
             (unsigned int)v181,
             v182,
             0,
             *(double *)v10.m128_u64,
             a6,
             a7,
             v185,
             *((unsigned __int64 *)&v185 + 1),
             v134);
    v85 = v182;
    v86 = v138;
    v168 = v87;
    v88 = v141;
    v155 = (__int64 *)a1[1];
    v89 = v181;
    v90 = (unsigned int)v84;
    v91 = *(_QWORD *)(v162 + 40) + 16LL * (unsigned int)v160;
    v92 = *(_BYTE *)v91;
    v93 = *(_QWORD *)(v91 + 8);
    v187[0] = v92;
    v188 = v93;
    if ( v92 )
    {
      v95 = ((unsigned __int8)(v92 - 14) < 0x60u) + 134;
    }
    else
    {
      v135 = v182;
      v139 = v181;
      v94 = sub_1F58D20((__int64)v187);
      v85 = v135;
      v89 = v139;
      v88 = v141;
      v90 = v84;
      v95 = 134 - (!v94 - 1);
    }
    *((_QWORD *)&v127 + 1) = v131;
    *(_QWORD *)&v127 = v86;
    v96 = sub_1D3A900(v155, v95, (__int64)&v183, v89, v85, 0, v10, a6, a7, v162, v160, v127, v88, v90);
    v180 = v97;
    v98 = v182;
    *(_QWORD *)a3 = v96;
    v99 = (unsigned int)v181;
    *(_DWORD *)(a3 + 8) = v180;
    v100 = (__int64 *)a1[1];
    v101 = sub_1F810E0(v100, (__int64)&v183, v99, v98, v162, v160, v10, a6, a7, v173, (__int64)v165, v168);
    *(_QWORD *)a4 = sub_1F810E0(
                      v100,
                      (__int64)&v183,
                      (unsigned int)v181,
                      v182,
                      (unsigned __int64)v152,
                      v146,
                      v10,
                      a6,
                      a7,
                      v186,
                      (__int64)v101,
                      v102);
    *(_DWORD *)(a4 + 8) = v103;
  }
  if ( v183 )
    sub_161E7C0((__int64)&v183, v183);
  return 1;
}
