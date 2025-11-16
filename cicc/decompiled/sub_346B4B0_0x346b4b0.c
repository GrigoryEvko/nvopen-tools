// Function: sub_346B4B0
// Address: 0x346b4b0
//
unsigned __int8 *__fastcall sub_346B4B0(_BYTE *a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 v6; // rsi
  const __m128i *v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int16 v10; // cx
  __int16 *v11; // rdx
  __int16 v12; // cx
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // r13
  int v16; // r14d
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rsi
  int v30; // r9d
  unsigned int v31; // edx
  __int64 v32; // rax
  __int16 v33; // cx
  void *v34; // r15
  void *v35; // rax
  const void *v36; // rbx
  unsigned __int8 v37; // al
  __int128 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // rax
  __int128 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // r15
  __int64 (__fastcall *v46)(_BYTE *, __int64, __int64, _QWORD, __int64, __int64, _QWORD, _QWORD); // rbx
  __int64 v47; // rax
  __int64 v48; // r9
  __int64 v49; // rdx
  __int128 v50; // rax
  __int64 v51; // r9
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  __int64 v54; // r10
  unsigned int v55; // esi
  __int64 v56; // r8
  unsigned int v57; // edx
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned __int8 *v60; // rdx
  __int16 v61; // di
  __int64 v62; // rax
  __int64 v63; // r11
  unsigned int v64; // eax
  __int64 v65; // rax
  int v66; // edx
  int v67; // ebx
  __int64 v68; // rdx
  __int64 v69; // rbx
  unsigned __int64 v70; // r15
  __int128 v71; // rax
  __int64 v72; // r9
  __int64 v73; // r10
  unsigned int v74; // edx
  __int64 v75; // r9
  __int64 v76; // rdx
  __int16 v77; // ax
  __int64 v78; // rdx
  __int64 v79; // r11
  unsigned int v80; // ecx
  __int64 v81; // r8
  unsigned int v82; // esi
  __int64 v83; // rax
  int v84; // edx
  int v85; // ebx
  unsigned __int8 *v86; // rdx
  unsigned __int8 *v87; // rbx
  unsigned __int64 v88; // r15
  unsigned __int8 *v89; // r12
  unsigned int v91; // edx
  __int64 v92; // r9
  int v93; // r9d
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  unsigned int v97; // ebx
  __int64 v98; // r8
  unsigned __int64 v99; // rax
  bool v100; // al
  bool v101; // al
  __int128 v102; // rax
  __int64 v103; // rdx
  __int64 (__fastcall *v104)(_BYTE *, __int64, __int64, _QWORD, __int64); // r15
  __int64 v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rdx
  __int64 v108; // r15
  __int128 v109; // rax
  __int64 v110; // r9
  __int64 v111; // rax
  unsigned int v112; // r15d
  __int64 v113; // r8
  __int64 v114; // rsi
  unsigned int v115; // edx
  __int64 v116; // rbx
  __int64 v117; // rdx
  __int16 v118; // ax
  __int64 v119; // rdx
  __int64 v120; // r9
  __int128 v121; // kr40_16
  unsigned int v122; // eax
  __int64 v123; // rax
  __int64 v124; // rdx
  __int128 v125; // rax
  __int64 v126; // r9
  __int64 v127; // rax
  unsigned int v128; // ecx
  __int64 v129; // r8
  unsigned int v130; // edx
  __int64 v131; // r10
  __int64 v132; // rdx
  __int16 v133; // ax
  __int64 v134; // rdx
  __int128 v135; // kr50_16
  bool v136; // al
  bool v137; // al
  __int128 v138; // [rsp-10h] [rbp-210h]
  unsigned __int128 v139; // [rsp+0h] [rbp-200h]
  __int64 v140; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v141; // [rsp+20h] [rbp-1E0h]
  __int128 v142; // [rsp+20h] [rbp-1E0h]
  unsigned __int8 v143; // [rsp+30h] [rbp-1D0h]
  __int128 v144; // [rsp+30h] [rbp-1D0h]
  __int64 v145; // [rsp+30h] [rbp-1D0h]
  __int128 v146; // [rsp+40h] [rbp-1C0h]
  __int64 v147; // [rsp+40h] [rbp-1C0h]
  __int64 v148; // [rsp+40h] [rbp-1C0h]
  unsigned int v149; // [rsp+40h] [rbp-1C0h]
  bool v150; // [rsp+50h] [rbp-1B0h]
  __int128 v151; // [rsp+50h] [rbp-1B0h]
  __int64 v152; // [rsp+50h] [rbp-1B0h]
  __int64 v153; // [rsp+50h] [rbp-1B0h]
  __int64 v154; // [rsp+58h] [rbp-1A8h]
  __int64 v155; // [rsp+58h] [rbp-1A8h]
  __int64 v156; // [rsp+60h] [rbp-1A0h]
  unsigned int v157; // [rsp+60h] [rbp-1A0h]
  __int128 v158; // [rsp+60h] [rbp-1A0h]
  __int64 v159; // [rsp+60h] [rbp-1A0h]
  unsigned __int8 *v160; // [rsp+70h] [rbp-190h]
  __int64 v161; // [rsp+70h] [rbp-190h]
  __int64 v162; // [rsp+70h] [rbp-190h]
  __int64 v163; // [rsp+70h] [rbp-190h]
  __int64 v164; // [rsp+70h] [rbp-190h]
  unsigned int v165; // [rsp+70h] [rbp-190h]
  __int128 v166; // [rsp+80h] [rbp-180h]
  unsigned __int8 *v167; // [rsp+80h] [rbp-180h]
  unsigned int v168; // [rsp+80h] [rbp-180h]
  __int64 v169; // [rsp+80h] [rbp-180h]
  __int128 v170; // [rsp+80h] [rbp-180h]
  int v171; // [rsp+90h] [rbp-170h]
  __int128 v172; // [rsp+90h] [rbp-170h]
  __int128 v173; // [rsp+90h] [rbp-170h]
  unsigned __int8 *v174; // [rsp+A0h] [rbp-160h]
  unsigned int v175; // [rsp+A0h] [rbp-160h]
  __int128 v176; // [rsp+B0h] [rbp-150h]
  __int128 v177; // [rsp+B0h] [rbp-150h]
  __int64 v178; // [rsp+B0h] [rbp-150h]
  __int64 v179; // [rsp+B8h] [rbp-148h]
  __int64 v180; // [rsp+100h] [rbp-100h] BYREF
  int v181; // [rsp+108h] [rbp-F8h]
  unsigned int v182; // [rsp+110h] [rbp-F0h] BYREF
  __int64 v183; // [rsp+118h] [rbp-E8h]
  __int64 v184; // [rsp+120h] [rbp-E0h] BYREF
  __int64 v185; // [rsp+128h] [rbp-D8h]
  __int16 v186; // [rsp+130h] [rbp-D0h] BYREF
  __int64 v187; // [rsp+138h] [rbp-C8h]
  unsigned __int64 v188; // [rsp+140h] [rbp-C0h] BYREF
  unsigned int v189; // [rsp+148h] [rbp-B8h]
  unsigned __int64 v190; // [rsp+150h] [rbp-B0h] BYREF
  unsigned int v191; // [rsp+158h] [rbp-A8h]
  __int64 v192; // [rsp+160h] [rbp-A0h]
  __int64 v193; // [rsp+168h] [rbp-98h]
  __int64 v194; // [rsp+170h] [rbp-90h]
  __int64 v195; // [rsp+178h] [rbp-88h]
  __int16 v196; // [rsp+180h] [rbp-80h] BYREF
  __int64 v197; // [rsp+188h] [rbp-78h]
  unsigned __int64 v198; // [rsp+190h] [rbp-70h] BYREF
  __int64 v199; // [rsp+198h] [rbp-68h]
  unsigned __int64 v200; // [rsp+1B0h] [rbp-50h] BYREF
  __int64 v201; // [rsp+1B8h] [rbp-48h]

  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 80);
  v171 = v5;
  v180 = v6;
  v150 = v5 == 228;
  if ( v6 )
    sub_B96E90((__int64)&v180, v6, 1);
  v181 = *(_DWORD *)(a2 + 72);
  v7 = *(const __m128i **)(a2 + 40);
  v8 = _mm_loadu_si128(v7);
  v156 = v7->m128i_u32[2];
  v9 = *(_QWORD *)(v7->m128i_i64[0] + 48) + 16 * v156;
  v160 = (unsigned __int8 *)v7->m128i_i64[0];
  v10 = *(_WORD *)v9;
  v183 = *(_QWORD *)(v9 + 8);
  v11 = *(__int16 **)(a2 + 48);
  LOWORD(v182) = v10;
  v12 = *v11;
  v185 = *((_QWORD *)v11 + 1);
  v13 = v7[2].m128i_i64[1];
  LOWORD(v184) = v12;
  v14 = *(unsigned __int16 *)(v13 + 96);
  v15 = *(_QWORD *)(v13 + 104);
  v186 = v14;
  v187 = v15;
  if ( !(_WORD)v14 )
  {
    if ( !sub_30070B0((__int64)&v186) )
    {
      v201 = v15;
      LOWORD(v200) = 0;
      goto LABEL_16;
    }
    LOWORD(v14) = sub_3009970((__int64)&v186, v6, v94, v95, v96);
LABEL_15:
    LOWORD(v200) = v14;
    v201 = v23;
    if ( (_WORD)v14 )
      goto LABEL_6;
LABEL_16:
    v24 = sub_3007260((__int64)&v200);
    v16 = (unsigned __int16)v184;
    v6 = v25;
    v192 = v24;
    LODWORD(v17) = v24;
    v193 = v25;
    if ( !(_WORD)v184 )
      goto LABEL_9;
LABEL_17:
    if ( (unsigned __int16)(v16 - 17) <= 0xD3u )
    {
      v26 = 0;
      LOWORD(v16) = word_4456580[v16 - 1];
LABEL_19:
      LOWORD(v198) = v16;
      v199 = v26;
      if ( (_WORD)v16 )
        goto LABEL_11;
      goto LABEL_20;
    }
LABEL_18:
    v26 = v185;
    goto LABEL_19;
  }
  if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
  {
    LOWORD(v14) = word_4456580[v14 - 1];
    v23 = 0;
    goto LABEL_15;
  }
  LOWORD(v200) = v14;
  v201 = v15;
LABEL_6:
  if ( (_WORD)v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    goto LABEL_113;
  v16 = (unsigned __int16)v184;
  v17 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v14 - 16];
  if ( (_WORD)v184 )
    goto LABEL_17;
LABEL_9:
  if ( !sub_30070B0((__int64)&v184) )
    goto LABEL_18;
  LOWORD(v16) = sub_3009970((__int64)&v184, v6, v18, v19, v20);
  v199 = v21;
  LOWORD(v198) = v16;
  if ( (_WORD)v16 )
  {
LABEL_11:
    if ( (_WORD)v16 != 1 && (unsigned __int16)(v16 - 504) > 7u )
    {
      v22 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v16 - 16];
      goto LABEL_21;
    }
LABEL_113:
    BUG();
  }
LABEL_20:
  v194 = sub_3007260((__int64)&v198);
  LODWORD(v22) = v194;
  v195 = v27;
LABEL_21:
  LODWORD(v199) = v17;
  v189 = 1;
  v188 = 0;
  v191 = 1;
  v190 = 0;
  if ( v171 == 228 )
  {
    v97 = v17 - 1;
    v147 = 1LL << ((unsigned __int8)v17 - 1);
    if ( (unsigned int)v17 > 0x40 )
    {
      sub_C43690((__int64)&v198, 0, 0);
      if ( (unsigned int)v199 > 0x40 )
      {
        *(_QWORD *)(v198 + 8LL * (v97 >> 6)) |= v147;
        goto LABEL_78;
      }
    }
    else
    {
      v198 = 0;
    }
    v198 |= v147;
LABEL_78:
    sub_C44830((__int64)&v200, &v198, v22);
    if ( v189 > 0x40 && v188 )
      j_j___libc_free_0_0(v188);
    v188 = v200;
    v189 = v201;
    if ( (unsigned int)v199 > 0x40 && v198 )
      j_j___libc_free_0_0(v198);
    LODWORD(v199) = v17;
    v98 = ~v147;
    if ( (unsigned int)v17 > 0x40 )
    {
      v148 = ~v147;
      sub_C43690((__int64)&v198, -1, 1);
      v98 = v148;
      if ( (unsigned int)v199 > 0x40 )
      {
        *(_QWORD *)(v198 + 8LL * (v97 >> 6)) &= v148;
LABEL_87:
        v29 = (__int64)&v198;
        sub_C44830((__int64)&v200, &v198, v22);
        if ( v191 <= 0x40 )
          goto LABEL_37;
        goto LABEL_35;
      }
    }
    else
    {
      v99 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v17 - 1) & 0x3F));
      if ( !(_DWORD)v17 )
        v99 = 0;
      v198 = v99;
    }
    v198 &= v98;
    goto LABEL_87;
  }
  if ( (unsigned int)v17 > 0x40 )
    sub_C43690((__int64)&v198, 0, 0);
  else
    v198 = 0;
  sub_C449B0((__int64)&v200, (const void **)&v198, v22);
  if ( v189 > 0x40 && v188 )
    j_j___libc_free_0_0(v188);
  v188 = v200;
  v189 = v201;
  if ( (unsigned int)v199 > 0x40 && v198 )
    j_j___libc_free_0_0(v198);
  LODWORD(v199) = v17;
  if ( (unsigned int)v17 > 0x40 )
  {
    sub_C43690((__int64)&v198, -1, 1);
  }
  else
  {
    v28 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    if ( !(_DWORD)v17 )
      v28 = 0;
    v198 = v28;
  }
  v29 = (__int64)&v198;
  sub_C449B0((__int64)&v200, (const void **)&v198, v22);
  if ( v191 > 0x40 )
  {
LABEL_35:
    if ( v190 )
      j_j___libc_free_0_0(v190);
  }
LABEL_37:
  v190 = v200;
  v191 = v201;
  if ( (unsigned int)v199 > 0x40 && v198 )
    j_j___libc_free_0_0(v198);
  if ( (unsigned __int16)(v182 - 10) <= 1u )
  {
    v29 = 233;
    v160 = sub_33FAF80(a3, 233, (__int64)&v180, 12, 0, v30, v8);
    v32 = *((_QWORD *)v160 + 6) + 16LL * v31;
    v33 = *(_WORD *)v32;
    v183 = *(_QWORD *)(v32 + 8);
    LOWORD(v182) = v33;
    v156 = v31;
  }
  v34 = sub_300AC80((unsigned __int16 *)&v182, v29);
  v35 = sub_C33340();
  v36 = v35;
  if ( v34 == v35 )
  {
    sub_C3C460(&v198, (__int64)v35);
    sub_C3C460(&v200, (__int64)v36);
  }
  else
  {
    sub_C37380(&v198, (__int64)v34);
    sub_C37380(&v200, (__int64)v34);
  }
  if ( (const void *)v198 == v36 )
    v37 = sub_C400C0(&v198, (__int64)&v188, v150, 0);
  else
    v37 = sub_C36910((__int64)&v198, (__int64)&v188, v150, 0);
  v141 = v37;
  if ( (const void *)v200 == v36 )
    v143 = sub_C400C0(&v200, (__int64)&v190, v150, 0);
  else
    v143 = sub_C36910((__int64)&v200, (__int64)&v190, v150, 0);
  *(_QWORD *)&v38 = sub_33FE6E0(a3, (__int64 *)&v198, (__int64)&v180, v182, v183, 0, v8);
  v151 = v38;
  *(_QWORD *)&v146 = sub_33FE6E0(a3, (__int64 *)&v200, (__int64)&v180, v182, v183, 0, v8);
  *((_QWORD *)&v146 + 1) = v40;
  if ( (_WORD)v182 == 1 )
  {
    if ( a1[7193] )
      goto LABEL_52;
    v41 = 1;
  }
  else
  {
    if ( !(_WORD)v182 )
      goto LABEL_52;
    if ( !*(_QWORD *)&a1[8 * (unsigned __int16)v182 + 112] )
      goto LABEL_52;
    v41 = (unsigned __int16)v182;
    if ( a1[500 * (unsigned __int16)v182 + 6693] )
      goto LABEL_52;
  }
  if ( !a1[500 * v41 + 6694] && ((v141 | v143) & 0x10) == 0 )
  {
    *(_QWORD *)&v177 = v160;
    *((_QWORD *)&v177 + 1) = v156 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v138 = sub_3406EB0(
                         (_QWORD *)a3,
                         0x118u,
                         (__int64)&v180,
                         v182,
                         v183,
                         v39,
                         __PAIR128__(*((unsigned __int64 *)&v177 + 1), (unsigned __int64)v160),
                         v151);
    *((_QWORD *)&v138 + 1) = v91 | v156 & 0xFFFFFFFF00000000LL | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    sub_3406EB0((_QWORD *)a3, 0x117u, (__int64)&v180, v182, v183, v92, v138, v146);
    if ( v171 != 228 )
    {
      v89 = sub_33FAF80(a3, 227, (__int64)&v180, (unsigned int)v184, v185, v93, v8);
      goto LABEL_58;
    }
    *(_QWORD *)&v102 = sub_33FAF80(a3, 226, (__int64)&v180, (unsigned int)v184, v185, v93, v8);
    v158 = v102;
    *(_QWORD *)&v172 = sub_3400BD0(a3, 0, (__int64)&v180, (unsigned int)v184, v185, 0, v8, 0);
    *((_QWORD *)&v172 + 1) = v103;
    v163 = *(_QWORD *)(a3 + 64);
    v104 = *(__int64 (__fastcall **)(_BYTE *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
    v105 = sub_2E79000(*(__int64 **)(a3 + 40));
    v106 = v104(a1, v105, v163, v182, v183);
    v108 = v107;
    v168 = v106;
    *(_QWORD *)&v109 = sub_33ED040((_QWORD *)a3, 8u);
    v111 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v180, v168, v108, v110, v177, v177, v109);
    v112 = v184;
    v113 = v111;
    v114 = v185;
    v116 = v115;
    v117 = *(_QWORD *)(v111 + 48) + 16LL * v115;
    v118 = *(_WORD *)v117;
    v119 = *(_QWORD *)(v117 + 8);
    v120 = v116;
    v196 = v118;
    v197 = v119;
    v121 = v158;
    if ( v118 )
    {
      v122 = ((unsigned __int16)(v118 - 17) < 0xD4u) + 205;
    }
    else
    {
      v164 = v185;
      v169 = v113;
      v137 = sub_30070B0((__int64)&v196);
      v114 = v164;
      v120 = v116;
      v113 = v169;
      v121 = v158;
      v122 = 205 - (!v137 - 1);
    }
    v123 = sub_340EC60((_QWORD *)a3, v122, (__int64)&v180, v112, v114, 0, v113, v120, v172, v121);
LABEL_107:
    v89 = (unsigned __int8 *)v123;
    goto LABEL_58;
  }
LABEL_52:
  *(_QWORD *)&v42 = sub_34007B0(a3, (__int64)&v188, (__int64)&v180, v184, v185, 0, v8, 0);
  v144 = v42;
  *(_QWORD *)&v142 = sub_34007B0(a3, (__int64)&v190, (__int64)&v180, v184, v185, 0, v8, 0);
  *((_QWORD *)&v142 + 1) = v43;
  *(_QWORD *)&v176 = v160;
  *((_QWORD *)&v176 + 1) = v156 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v139 = __PAIR128__(*((unsigned __int64 *)&v176 + 1), (unsigned __int64)v160);
  v174 = sub_33FAF80(a3, (unsigned int)(v171 != 228) + 226, (__int64)&v180, (unsigned int)v184, v185, 0, v8);
  v45 = v44;
  v161 = *(_QWORD *)(a3 + 64);
  v46 = *(__int64 (__fastcall **)(_BYTE *, __int64, __int64, _QWORD, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 528LL);
  v47 = sub_2E79000(*(__int64 **)(a3 + 40));
  LODWORD(v46) = v46(a1, v47, v161, v182, v183, v48, v139, *((_QWORD *)&v139 + 1));
  v162 = v49;
  v157 = (unsigned int)v46;
  *(_QWORD *)&v50 = sub_33ED040((_QWORD *)a3, 0xCu);
  v52 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v180, (unsigned int)v46, v162, v51, v176, v151, v50);
  v53 = v45;
  v54 = v52;
  v55 = v184;
  v56 = v185;
  v58 = v57;
  v59 = *(_QWORD *)(v52 + 48) + 16LL * v57;
  v60 = v174;
  v61 = *(_WORD *)v59;
  v62 = *(_QWORD *)(v59 + 8);
  v63 = v58;
  v196 = v61;
  v197 = v62;
  if ( v61 )
  {
    v64 = ((unsigned __int16)(v61 - 17) < 0xD4u) + 205;
  }
  else
  {
    v140 = v185;
    v175 = v184;
    v153 = v54;
    v167 = v60;
    v155 = v58;
    v101 = sub_30070B0((__int64)&v196);
    v56 = v140;
    v55 = v175;
    v63 = v155;
    v54 = v153;
    v53 = v45;
    v60 = v167;
    v64 = 205 - (!v101 - 1);
  }
  v65 = sub_340EC60(
          (_QWORD *)a3,
          v64,
          (__int64)&v180,
          v55,
          v56,
          0,
          v54,
          v63,
          v144,
          __PAIR128__(v53, (unsigned __int64)v60));
  v67 = v66;
  v68 = v65;
  LODWORD(v65) = v67;
  v69 = v68;
  v70 = (unsigned int)v65 | v45 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v71 = sub_33ED040((_QWORD *)a3, 2u);
  v73 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v180, v157, v162, v72, v176, v146, v71);
  *(_QWORD *)&v166 = v69;
  v75 = v74;
  v76 = *(_QWORD *)(v73 + 48) + 16LL * v74;
  *((_QWORD *)&v166 + 1) = v70;
  v77 = *(_WORD *)v76;
  v78 = *(_QWORD *)(v76 + 8);
  v79 = v75;
  v80 = v184;
  v81 = v185;
  v196 = v77;
  v197 = v78;
  if ( v77 )
  {
    v82 = ((unsigned __int16)(v77 - 17) < 0xD4u) + 205;
  }
  else
  {
    v145 = v185;
    v149 = v184;
    v152 = v73;
    v154 = v75;
    v100 = sub_30070B0((__int64)&v196);
    v81 = v145;
    v80 = v149;
    v79 = v154;
    v73 = v152;
    v82 = 205 - (!v100 - 1);
  }
  v83 = sub_340EC60((_QWORD *)a3, v82, (__int64)&v180, v80, v81, 0, v73, v79, v142, v166);
  v85 = v84;
  v86 = (unsigned __int8 *)v83;
  LODWORD(v83) = v85;
  v87 = v86;
  v88 = (unsigned int)v83 | v70 & 0xFFFFFFFF00000000LL;
  if ( v171 == 228 )
  {
    *(_QWORD *)&v173 = sub_3400BD0(a3, 0, (__int64)&v180, (unsigned int)v184, v185, 0, v8, 0);
    *((_QWORD *)&v173 + 1) = v124;
    *(_QWORD *)&v125 = sub_33ED040((_QWORD *)a3, 8u);
    v127 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v180, v157, v162, v126, v176, v176, v125);
    v128 = v184;
    v178 = v127;
    v129 = v185;
    v131 = v130;
    v132 = *(_QWORD *)(v127 + 48) + 16LL * v130;
    v133 = *(_WORD *)v132;
    v134 = *(_QWORD *)(v132 + 8);
    v179 = v131;
    v196 = v133;
    v197 = v134;
    v135 = __PAIR128__(v88, (unsigned __int64)v87);
    if ( v133 )
    {
      v136 = (unsigned __int16)(v133 - 17) <= 0xD3u;
    }
    else
    {
      v159 = v185;
      v165 = v184;
      *(_QWORD *)&v170 = v87;
      *((_QWORD *)&v170 + 1) = v88;
      v136 = sub_30070B0((__int64)&v196);
      v129 = v159;
      v128 = v165;
      v135 = v170;
    }
    v123 = sub_340EC60(
             (_QWORD *)a3,
             205 - ((unsigned int)!v136 - 1),
             (__int64)&v180,
             v128,
             v129,
             0,
             v178,
             v179,
             v173,
             v135);
    goto LABEL_107;
  }
  v89 = v86;
LABEL_58:
  sub_91D830(&v200);
  sub_91D830(&v198);
  if ( v191 > 0x40 && v190 )
    j_j___libc_free_0_0(v190);
  if ( v189 > 0x40 && v188 )
    j_j___libc_free_0_0(v188);
  if ( v180 )
    sub_B91220((__int64)&v180, v180);
  return v89;
}
