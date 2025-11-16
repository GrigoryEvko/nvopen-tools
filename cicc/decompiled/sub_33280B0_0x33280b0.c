// Function: sub_33280B0
// Address: 0x33280b0
//
__int64 __fastcall sub_33280B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  const __m128i *v4; // rax
  __int32 v5; // ecx
  __m128i v6; // xmm0
  __int64 v7; // r8
  __int64 v8; // rbx
  __int64 v9; // rcx
  unsigned __int16 *v10; // rax
  __int64 v11; // r13
  __int64 v12; // r15
  __int128 v13; // rax
  int v14; // r9d
  __int128 v15; // rax
  int v16; // r9d
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int16 v19; // bx
  __int64 v20; // rbx
  __int64 v21; // r10
  __int64 v22; // rax
  unsigned __int16 v23; // cx
  unsigned int v24; // eax
  int v25; // r15d
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  __int128 v28; // rax
  int v29; // r9d
  int v30; // r9d
  __int64 v31; // rdx
  int v32; // r10d
  unsigned __int16 *v33; // rdx
  int v34; // eax
  __int64 v35; // r13
  bool v36; // al
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // rax
  unsigned __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned __int16 *v44; // rdx
  int v45; // eax
  __int64 v46; // rdx
  int v47; // eax
  bool v48; // zf
  unsigned __int16 *v49; // rdx
  int v50; // eax
  __int64 v51; // r13
  unsigned __int64 v52; // r13
  unsigned __int16 *v53; // rdx
  int v54; // eax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  int v58; // r9d
  __int64 v59; // r12
  char v61; // dl
  __int64 v62; // rdx
  __int128 v63; // rax
  __int128 v64; // rax
  __int128 v65; // rax
  __int64 v66; // r15
  __int64 v67; // rax
  unsigned __int16 v68; // ax
  int v69; // edx
  int v70; // r13d
  __int128 v71; // rax
  int v72; // r9d
  __int64 v73; // r8
  __int64 v74; // r12
  int v75; // r11d
  unsigned int v76; // edx
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int16 v79; // ax
  __int64 v80; // rdx
  __int64 v81; // r9
  bool v82; // al
  int v83; // esi
  char v84; // si
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // rdx
  unsigned int v88; // edx
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // rdx
  bool v93; // al
  __int64 v94; // rcx
  __int64 v95; // r8
  unsigned __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rax
  unsigned int v99; // edx
  __int128 v100; // rax
  __int64 v101; // rax
  int v102; // edx
  __int64 v103; // rdx
  __int64 v104; // rdx
  __int64 v105; // rdx
  __int64 v106; // rdx
  __int128 v107; // rax
  int v108; // edx
  __int128 v109; // [rsp-30h] [rbp-330h]
  __int128 v110; // [rsp-30h] [rbp-330h]
  __int128 v111; // [rsp-10h] [rbp-310h]
  __int128 v112; // [rsp-10h] [rbp-310h]
  int v113; // [rsp+8h] [rbp-2F8h]
  __int64 v114; // [rsp+10h] [rbp-2F0h]
  int v115; // [rsp+1Ch] [rbp-2E4h]
  int v116; // [rsp+20h] [rbp-2E0h]
  unsigned __int16 v117; // [rsp+26h] [rbp-2DAh]
  int v118; // [rsp+28h] [rbp-2D8h]
  int v119; // [rsp+38h] [rbp-2C8h]
  int v120; // [rsp+40h] [rbp-2C0h]
  __int64 v121; // [rsp+40h] [rbp-2C0h]
  __int64 v122; // [rsp+40h] [rbp-2C0h]
  int v123; // [rsp+48h] [rbp-2B8h]
  __int64 v124; // [rsp+48h] [rbp-2B8h]
  int v125; // [rsp+50h] [rbp-2B0h]
  __int64 (__fastcall *v126)(__int64, __int64, __int64, __int64, __int64); // [rsp+50h] [rbp-2B0h]
  __int64 v127; // [rsp+50h] [rbp-2B0h]
  int v128; // [rsp+50h] [rbp-2B0h]
  int v129; // [rsp+50h] [rbp-2B0h]
  int v130; // [rsp+50h] [rbp-2B0h]
  unsigned __int16 v131; // [rsp+58h] [rbp-2A8h]
  int v132; // [rsp+58h] [rbp-2A8h]
  __int64 v133; // [rsp+58h] [rbp-2A8h]
  __int16 v134; // [rsp+5Ah] [rbp-2A6h]
  __int128 v135; // [rsp+60h] [rbp-2A0h]
  __int128 v136; // [rsp+60h] [rbp-2A0h]
  __int128 v137; // [rsp+60h] [rbp-2A0h]
  __int128 v138; // [rsp+60h] [rbp-2A0h]
  __int64 v139; // [rsp+70h] [rbp-290h]
  __int128 v140; // [rsp+70h] [rbp-290h]
  unsigned __int32 v141; // [rsp+80h] [rbp-280h]
  __int64 v142; // [rsp+80h] [rbp-280h]
  __int64 v143; // [rsp+80h] [rbp-280h]
  __int128 v144; // [rsp+80h] [rbp-280h]
  __int64 v145; // [rsp+80h] [rbp-280h]
  __int64 v146; // [rsp+88h] [rbp-278h]
  __int128 v147; // [rsp+90h] [rbp-270h]
  unsigned __int64 v148; // [rsp+98h] [rbp-268h]
  __int64 v149; // [rsp+D0h] [rbp-230h] BYREF
  int v150; // [rsp+D8h] [rbp-228h]
  __int16 v151; // [rsp+E0h] [rbp-220h] BYREF
  __int64 v152; // [rsp+E8h] [rbp-218h]
  __int16 v153; // [rsp+F0h] [rbp-210h] BYREF
  __int64 v154; // [rsp+F8h] [rbp-208h]
  __int16 v155; // [rsp+100h] [rbp-200h] BYREF
  __int64 v156; // [rsp+108h] [rbp-1F8h]
  __int16 v157; // [rsp+110h] [rbp-1F0h] BYREF
  __int64 v158; // [rsp+118h] [rbp-1E8h]
  __int64 v159; // [rsp+120h] [rbp-1E0h]
  __int64 v160; // [rsp+128h] [rbp-1D8h]
  unsigned __int64 v161; // [rsp+130h] [rbp-1D0h]
  __int64 v162; // [rsp+138h] [rbp-1C8h]
  __int16 v163; // [rsp+140h] [rbp-1C0h] BYREF
  __int64 v164; // [rsp+148h] [rbp-1B8h]
  __int16 v165; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v166; // [rsp+158h] [rbp-1A8h]
  __int16 v167; // [rsp+160h] [rbp-1A0h] BYREF
  __int64 v168; // [rsp+168h] [rbp-198h]
  unsigned __int64 v169; // [rsp+170h] [rbp-190h] BYREF
  __int64 v170; // [rsp+178h] [rbp-188h]
  unsigned __int64 v171; // [rsp+180h] [rbp-180h] BYREF
  __int64 v172; // [rsp+188h] [rbp-178h]
  __int16 v173; // [rsp+190h] [rbp-170h] BYREF
  __int64 v174; // [rsp+198h] [rbp-168h]
  __int64 v175; // [rsp+1A0h] [rbp-160h]
  int v176; // [rsp+1A8h] [rbp-158h]
  __int64 v177; // [rsp+1B0h] [rbp-150h]
  int v178; // [rsp+1B8h] [rbp-148h]
  __int64 v179; // [rsp+1C0h] [rbp-140h]
  int v180; // [rsp+1C8h] [rbp-138h]
  __int64 v181; // [rsp+1D0h] [rbp-130h]
  __int64 v182; // [rsp+1D8h] [rbp-128h]
  int v183; // [rsp+1E0h] [rbp-120h]
  char v184; // [rsp+1E4h] [rbp-11Ch]
  __int64 v185; // [rsp+1E8h] [rbp-118h]
  __int64 v186; // [rsp+1F0h] [rbp-110h]
  int v187; // [rsp+1F8h] [rbp-108h]
  char v188; // [rsp+1FCh] [rbp-104h]
  __int128 v189; // [rsp+200h] [rbp-100h]
  unsigned __int64 v190; // [rsp+210h] [rbp-F0h] BYREF
  unsigned int v191; // [rsp+218h] [rbp-E8h]
  unsigned __int8 v192; // [rsp+220h] [rbp-E0h]
  int v193; // [rsp+230h] [rbp-D0h] BYREF
  __int64 v194; // [rsp+238h] [rbp-C8h]
  __int64 v195; // [rsp+240h] [rbp-C0h]
  int v196; // [rsp+248h] [rbp-B8h]
  __int64 v197; // [rsp+250h] [rbp-B0h]
  int v198; // [rsp+258h] [rbp-A8h]
  __int64 v199; // [rsp+260h] [rbp-A0h]
  int v200; // [rsp+268h] [rbp-98h]
  __int64 v201; // [rsp+270h] [rbp-90h]
  __int64 v202; // [rsp+278h] [rbp-88h]
  int v203; // [rsp+280h] [rbp-80h]
  char v204; // [rsp+284h] [rbp-7Ch]
  __int64 v205; // [rsp+288h] [rbp-78h]
  __int64 v206; // [rsp+290h] [rbp-70h]
  int v207; // [rsp+298h] [rbp-68h]
  char v208; // [rsp+29Ch] [rbp-64h]
  __int128 v209; // [rsp+2A0h] [rbp-60h]
  unsigned __int64 v210; // [rsp+2B0h] [rbp-50h] BYREF
  unsigned int v211; // [rsp+2B8h] [rbp-48h]
  unsigned __int8 v212; // [rsp+2C0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v149 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v149, v3, 1);
  v150 = *(_DWORD *)(a2 + 72);
  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4->m128i_i32[2];
  v6 = _mm_loadu_si128(v4);
  v7 = v4[3].m128i_i64[0];
  v8 = v4->m128i_i64[0];
  v173 = 0;
  v141 = v5;
  v9 = v4[2].m128i_i64[1];
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  *(_QWORD *)&v189 = 0;
  DWORD2(v189) = 0;
  v191 = 1;
  v190 = 0;
  sub_3326BF0(a1, (__int64)&v173, (int)&v149, v9, v7, 0);
  v10 = (unsigned __int16 *)(*(_QWORD *)(v189 + 48) + 16LL * DWORD2(v189));
  v11 = *((_QWORD *)v10 + 1);
  v131 = *v10;
  v12 = *v10;
  *(_QWORD *)&v13 = sub_34007B0(*(_QWORD *)(a1 + 16), (unsigned int)&v190, (unsigned int)&v149, v12, v11, 0, 0);
  *(_QWORD *)&v15 = sub_3406EB0(*(_QWORD *)(a1 + 16), 186, (unsigned int)&v149, v12, v11, v14, v189, v13);
  v17 = *(_QWORD *)(a1 + 8);
  v147 = v15;
  v18 = *(_QWORD *)(v8 + 48) + 16LL * v141;
  v19 = *(_WORD *)v18;
  if ( *(_WORD *)v18 == 1 )
  {
    v61 = *(_BYTE *)(v17 + 7159);
    if ( v61 && v61 != 4 )
      goto LABEL_5;
    v62 = 1;
  }
  else
  {
    if ( !v19 )
      goto LABEL_5;
    v62 = v19;
    if ( !*(_QWORD *)(v17 + 8LL * v19 + 112) )
      goto LABEL_5;
    v84 = *(_BYTE *)(v17 + 500LL * v19 + 6659);
    if ( v84 )
    {
      if ( v84 != 4 || !*(_QWORD *)(v17 + 8 * (v19 + 14LL)) )
        goto LABEL_5;
    }
  }
  if ( (*(_BYTE *)(v17 + 500 * v62 + 6658) & 0xFB) == 0 )
  {
    v143 = *(_QWORD *)(v18 + 8);
    *(_QWORD *)&v63 = sub_33FAF80(*(_QWORD *)(a1 + 16), 245, (unsigned int)&v149, v19, v143, v19, *(_OWORD *)&v6);
    v119 = v143;
    v140 = v63;
    *(_QWORD *)&v64 = sub_33FAF80(*(_QWORD *)(a1 + 16), 244, (unsigned int)&v149, v19, v143, v19, v63);
    v122 = *(_QWORD *)(a1 + 16);
    v138 = v64;
    *(_QWORD *)&v65 = sub_3400BD0(v122, 0, (unsigned int)&v149, v12, v11, 0, 0);
    v66 = *(_QWORD *)(a1 + 8);
    v144 = v65;
    *(_QWORD *)&v65 = *(_QWORD *)(a1 + 16);
    v124 = v131;
    v126 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v66 + 528LL);
    v133 = *(_QWORD *)(v65 + 64);
    v67 = sub_2E79000(*(__int64 **)(v65 + 40));
    v68 = v126(v66, v67, v133, v124, v11);
    v70 = v69;
    LODWORD(v66) = v68;
    *(_QWORD *)&v71 = sub_33ED040(v122, 22);
    v73 = sub_340F900(v122, 208, (unsigned int)&v149, v66, v70, v72, v147, v144, v71);
    v74 = *(_QWORD *)(a1 + 16);
    v75 = v119;
    v77 = v76;
    v78 = *(_QWORD *)(v73 + 48) + 16LL * v76;
    v79 = *(_WORD *)v78;
    v80 = *(_QWORD *)(v78 + 8);
    v81 = v77;
    LOWORD(v193) = v79;
    v194 = v80;
    if ( v79 )
    {
      v83 = ((unsigned __int16)(v79 - 17) < 0xD4u) + 205;
    }
    else
    {
      v145 = v73;
      v146 = v77;
      v82 = sub_30070B0((__int64)&v193);
      v73 = v145;
      v81 = v146;
      v75 = v119;
      v83 = 205 - (!v82 - 1);
    }
    v59 = sub_340EC60(v74, v83, (unsigned int)&v149, v19, v75, 0, v73, v81, v138, v140);
    goto LABEL_43;
  }
LABEL_5:
  v204 = 0;
  v208 = 0;
  v142 = v147;
  v20 = DWORD2(v147);
  LOWORD(v193) = 0;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v205 = 0;
  v206 = 0;
  v207 = 0;
  *(_QWORD *)&v209 = 0;
  DWORD2(v209) = 0;
  v211 = 1;
  v210 = 0;
  sub_3326BF0(a1, (__int64)&v193, (int)&v149, v6.m128i_i64[0], v6.m128i_i64[1], v16);
  v21 = *(_QWORD *)(a1 + 16);
  v22 = *(_QWORD *)(v209 + 48) + 16LL * DWORD2(v209);
  v23 = *(_WORD *)v22;
  v139 = *(_QWORD *)(v22 + 8);
  v24 = v211;
  v25 = v23;
  v117 = v23;
  LODWORD(v170) = v211;
  if ( v211 > 0x40 )
  {
    v128 = v21;
    sub_C43780((__int64)&v169, (const void **)&v210);
    v24 = v170;
    LODWORD(v21) = v128;
    if ( (unsigned int)v170 > 0x40 )
    {
      sub_C43D10((__int64)&v169);
      v24 = v170;
      v27 = v169;
      LODWORD(v21) = v128;
      goto LABEL_10;
    }
    v26 = v169;
  }
  else
  {
    v26 = v210;
  }
  v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v24) & ~v26;
  if ( !v24 )
    v27 = 0;
  v169 = v27;
LABEL_10:
  LODWORD(v172) = v24;
  v171 = v27;
  LODWORD(v170) = 0;
  *(_QWORD *)&v28 = sub_34007B0(v21, (unsigned int)&v171, (unsigned int)&v149, v25, v139, 0, 0);
  if ( (unsigned int)v172 > 0x40 && v171 )
  {
    v135 = v28;
    j_j___libc_free_0_0(v171);
    v28 = v135;
  }
  if ( (unsigned int)v170 > 0x40 && v169 )
  {
    v136 = v28;
    j_j___libc_free_0_0(v169);
    v28 = v136;
  }
  *(_QWORD *)&v137 = sub_3406EB0(*(_QWORD *)(a1 + 16), 186, (unsigned int)&v149, v25, v139, v29, v209, v28);
  *((_QWORD *)&v137 + 1) = v31;
  v32 = v131;
  v116 = v192;
  v125 = v192 - v212;
  v132 = v11;
  v33 = (unsigned __int16 *)(*(_QWORD *)(v147 + 48) + 16LL * DWORD2(v147));
  v115 = v212;
  v34 = *v33;
  v35 = *((_QWORD *)v33 + 1);
  v157 = v34;
  v158 = v35;
  if ( (_WORD)v34 )
  {
    if ( (unsigned __int16)(v34 - 17) > 0xD3u )
    {
      v155 = v34;
      v156 = v35;
      goto LABEL_58;
    }
    LOWORD(v34) = word_4456580[v34 - 1];
    v103 = 0;
  }
  else
  {
    v120 = v32;
    v36 = sub_30070B0((__int64)&v157);
    v32 = v120;
    if ( !v36 )
    {
      v156 = v35;
      v155 = 0;
LABEL_19:
      v123 = v32;
      v40 = sub_3007260((__int64)&v155);
      v32 = v123;
      v159 = v40;
      v41 = v40;
      v160 = v42;
      goto LABEL_20;
    }
    LOWORD(v34) = sub_3009970((__int64)&v157, 186, v37, v38, v39);
    v32 = v120;
  }
  v155 = v34;
  v156 = v103;
  if ( !(_WORD)v34 )
    goto LABEL_19;
LABEL_58:
  if ( (_WORD)v34 == 1 || (unsigned __int16)(v34 - 504) <= 7u )
    goto LABEL_100;
  v41 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v34 - 16];
LABEL_20:
  v43 = v137;
  v121 = 16LL * DWORD2(v137);
  v44 = (unsigned __int16 *)(*(_QWORD *)(v137 + 48) + v121);
  v45 = *v44;
  v46 = *((_QWORD *)v44 + 1);
  v153 = v45;
  v154 = v46;
  if ( !(_WORD)v45 )
  {
    v113 = v32;
    v114 = v46;
    v93 = sub_30070B0((__int64)&v153);
    v32 = v113;
    if ( !v93 )
    {
      v152 = v114;
      v151 = 0;
      goto LABEL_77;
    }
    LOWORD(v45) = sub_3009970((__int64)&v153, v137, v114, v94, v95);
    v32 = v113;
LABEL_86:
    v151 = v45;
    v152 = v104;
    if ( (_WORD)v45 )
      goto LABEL_23;
LABEL_77:
    v118 = v32;
    v96 = sub_3007260((__int64)&v151);
    v32 = v118;
    v161 = v96;
    v162 = v97;
    if ( v96 <= v41 )
      goto LABEL_26;
LABEL_78:
    v134 = HIWORD(v32);
    v98 = sub_33FAF80(*(_QWORD *)(a1 + 16), 214, (unsigned int)&v149, v25, v139, v30, v147);
    HIWORD(v32) = v134;
    v43 = 0xFFFFFFFF00000000LL;
    LOWORD(v32) = v117;
    v20 = v99;
    v142 = v98;
    v47 = v125;
    v132 = v139;
    *((_QWORD *)&v147 + 1) = v99 | *((_QWORD *)&v147 + 1) & 0xFFFFFFFF00000000LL;
    v48 = v125 == 0;
    if ( v125 <= 0 )
      goto LABEL_27;
LABEL_79:
    v129 = v32;
    *(_QWORD *)&v100 = sub_3400BD0(*(_QWORD *)(a1 + 16), v47, (unsigned int)&v149, v32, v132, 0, 0);
    v43 = 192;
    v148 = v20 | *((_QWORD *)&v147 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v109 + 1) = v148;
    *(_QWORD *)&v109 = v142;
    v101 = sub_3406EB0(*(_QWORD *)(a1 + 16), 192, (unsigned int)&v149, v129, v132, DWORD2(v100), v109, v100);
    LODWORD(v20) = v102;
    goto LABEL_80;
  }
  if ( (unsigned __int16)(v45 - 17) <= 0xD3u )
  {
    LOWORD(v45) = word_4456580[v45 - 1];
    v104 = 0;
    goto LABEL_86;
  }
  v151 = v45;
  v152 = v46;
LABEL_23:
  if ( (_WORD)v45 == 1 || (unsigned __int16)(v45 - 504) <= 7u )
    goto LABEL_100;
  if ( *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v45 - 16] > v41 )
    goto LABEL_78;
LABEL_26:
  v47 = v125;
  v48 = v125 == 0;
  if ( v125 > 0 )
    goto LABEL_79;
LABEL_27:
  if ( v48 )
    goto LABEL_28;
  v130 = v32;
  *(_QWORD *)&v107 = sub_3400BD0(*(_QWORD *)(a1 + 16), v115 - v116, (unsigned int)&v149, v32, v132, 0, 0);
  v43 = 190;
  v148 = v20 | *((_QWORD *)&v147 + 1) & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v110 + 1) = v148;
  *(_QWORD *)&v110 = v142;
  v101 = sub_3406EB0(*(_QWORD *)(a1 + 16), 190, (unsigned int)&v149, v130, v132, DWORD2(v107), v110, v107);
  LODWORD(v20) = v108;
LABEL_80:
  v20 = (unsigned int)v20;
  v142 = v101;
  *((_QWORD *)&v147 + 1) = (unsigned int)v20 | v148 & 0xFFFFFFFF00000000LL;
LABEL_28:
  v49 = (unsigned __int16 *)(*(_QWORD *)(v142 + 48) + 16 * v20);
  v50 = *v49;
  v51 = *((_QWORD *)v49 + 1);
  LOWORD(v171) = v50;
  v172 = v51;
  if ( (_WORD)v50 )
  {
    if ( (unsigned __int16)(v50 - 17) > 0xD3u )
    {
      v167 = v50;
      v168 = v51;
LABEL_31:
      if ( (_WORD)v50 != 1 && (unsigned __int16)(v50 - 504) > 7u )
      {
        v52 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v50 - 16];
        goto LABEL_34;
      }
LABEL_100:
      BUG();
    }
    LOWORD(v50) = word_4456580[v50 - 1];
    v106 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v171) )
    {
      v43 = 0;
      v168 = v51;
      v167 = 0;
      goto LABEL_72;
    }
    LOWORD(v50) = sub_3009970((__int64)&v171, v43, v89, v90, v91);
  }
  v167 = v50;
  v168 = v106;
  if ( (_WORD)v50 )
    goto LABEL_31;
LABEL_72:
  v169 = sub_3007260((__int64)&v167);
  v52 = v169;
  v170 = v92;
LABEL_34:
  v53 = (unsigned __int16 *)(*(_QWORD *)(v137 + 48) + v121);
  v54 = *v53;
  v55 = *((_QWORD *)v53 + 1);
  v165 = v54;
  v166 = v55;
  if ( !(_WORD)v54 )
  {
    v127 = v55;
    if ( !sub_30070B0((__int64)&v165) )
    {
      v164 = v127;
      v163 = 0;
      goto LABEL_68;
    }
    LOWORD(v54) = sub_3009970((__int64)&v165, v43, v127, v85, v86);
LABEL_89:
    v163 = v54;
    v164 = v105;
    if ( (_WORD)v54 )
      goto LABEL_37;
LABEL_68:
    v171 = sub_3007260((__int64)&v163);
    v172 = v87;
    if ( v52 <= v171 )
      goto LABEL_40;
LABEL_69:
    *((_QWORD *)&v147 + 1) = v20 | *((_QWORD *)&v147 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v112 + 1) = *((_QWORD *)&v147 + 1);
    *(_QWORD *)&v112 = v142;
    v142 = sub_33FAF80(*(_QWORD *)(a1 + 16), 216, (unsigned int)&v149, v25, v139, v30, v112);
    v20 = v88;
    goto LABEL_40;
  }
  if ( (unsigned __int16)(v54 - 17) <= 0xD3u )
  {
    LOWORD(v54) = word_4456580[v54 - 1];
    v105 = 0;
    goto LABEL_89;
  }
  v163 = v54;
  v164 = v55;
LABEL_37:
  if ( (_WORD)v54 == 1 || (unsigned __int16)(v54 - 504) <= 7u )
    goto LABEL_100;
  if ( v52 > *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v54 - 16] )
    goto LABEL_69;
LABEL_40:
  *((_QWORD *)&v111 + 1) = v20 | *((_QWORD *)&v147 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v111 = v142;
  v56 = sub_3405C90(*(_QWORD *)(a1 + 16), 187, (unsigned int)&v149, v25, v139, 8, v137, v111);
  v59 = sub_3325820(a1, &v193, (int)&v149, v56, v57, v58);
  if ( v211 > 0x40 && v210 )
    j_j___libc_free_0_0(v210);
LABEL_43:
  if ( v191 > 0x40 && v190 )
    j_j___libc_free_0_0(v190);
  if ( v149 )
    sub_B91220((__int64)&v149, v149);
  return v59;
}
