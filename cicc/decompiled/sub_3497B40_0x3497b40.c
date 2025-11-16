// Function: sub_3497B40
// Address: 0x3497b40
//
__int64 __fastcall sub_3497B40(_WORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  __int64 v5; // r12
  __int64 v9; // rsi
  __int64 *v10; // rdi
  __int16 *v11; // rax
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 (__fastcall *v15)(_WORD *, __int64, __int64, _QWORD, __int64); // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __m128i v19; // xmm0
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // r15d
  unsigned __int64 v25; // rax
  bool v26; // bl
  __int64 v27; // rsi
  unsigned __int16 v28; // dx
  bool v29; // al
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int16 v38; // cx
  __int64 v39; // rdi
  unsigned int v40; // r15d
  __int128 v41; // rax
  __int64 v42; // r15
  __int64 v43; // r14
  __int64 v44; // r9
  __int64 v45; // r8
  int v46; // edx
  __int64 v47; // r9
  unsigned __int8 *v48; // r14
  __int64 v49; // rdx
  __int64 v50; // r15
  __int128 v51; // rax
  unsigned int v52; // r14d
  __int64 v53; // r9
  int v54; // edx
  unsigned __int16 v55; // ax
  __int64 v56; // r9
  _BOOL8 v57; // r10
  unsigned int *v58; // r8
  unsigned int v59; // r15d
  __int64 v60; // rsi
  int v61; // edx
  __int64 v62; // r9
  int v63; // edx
  __int64 v64; // rdx
  __int64 v65; // rax
  unsigned __int8 *v66; // r14
  __int64 v67; // rdx
  __int64 v68; // r15
  __int128 v69; // rax
  int v70; // r9d
  int v71; // edx
  __int64 v72; // rax
  unsigned __int16 v73; // bx
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rax
  unsigned __int16 v78; // dx
  __int64 v79; // rax
  bool v81; // al
  unsigned int v82; // edx
  unsigned int v83; // edx
  __int128 v84; // rax
  int v85; // r9d
  int v86; // edx
  __int64 v87; // rax
  __int128 v88; // rax
  int v89; // r9d
  int v90; // edx
  unsigned __int64 v91; // r15
  __int64 v92; // rdx
  char v93; // cl
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rax
  unsigned __int64 v98; // rdx
  int v99; // edx
  unsigned __int16 v100; // r14
  unsigned __int16 *v101; // rax
  __int64 v102; // r15
  unsigned int v103; // ecx
  bool v104; // al
  __int64 v105; // rdx
  __int64 v106; // r8
  __int64 v107; // rdx
  __int64 v108; // rax
  __int64 v109; // rsi
  __int64 v110; // rdx
  __int128 v111; // rax
  __int64 v112; // r9
  unsigned __int8 *v113; // r14
  __int64 v114; // rdx
  __int64 v115; // r15
  __int128 v116; // rax
  int v117; // edx
  unsigned int v118; // r15d
  char v119; // dl
  unsigned __int64 v120; // rsi
  unsigned __int64 v121; // rax
  __int64 *v122; // r15
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rax
  int v126; // eax
  unsigned int *v127; // rax
  __int64 v128; // rdx
  __int64 v129; // r9
  unsigned __int8 *v130; // rax
  int v131; // edx
  __int64 v132; // rdx
  unsigned __int16 v133; // ax
  __int64 v134; // rdx
  __int64 v135; // r8
  unsigned __int16 v136; // ax
  __int128 v137; // [rsp-30h] [rbp-250h]
  __int128 v138; // [rsp-30h] [rbp-250h]
  __int128 v139; // [rsp-20h] [rbp-240h]
  __int128 v140; // [rsp-10h] [rbp-230h]
  __int128 v141; // [rsp-10h] [rbp-230h]
  __int64 v142; // [rsp+8h] [rbp-218h]
  unsigned int v143; // [rsp+10h] [rbp-210h]
  __int64 v144; // [rsp+18h] [rbp-208h]
  unsigned __int16 v145; // [rsp+18h] [rbp-208h]
  __int64 v146; // [rsp+18h] [rbp-208h]
  __int64 v147; // [rsp+20h] [rbp-200h]
  __int64 v148; // [rsp+20h] [rbp-200h]
  bool v149; // [rsp+20h] [rbp-200h]
  unsigned int v150; // [rsp+28h] [rbp-1F8h]
  __int64 v151; // [rsp+30h] [rbp-1F0h]
  int v153; // [rsp+44h] [rbp-1DCh]
  bool v154; // [rsp+44h] [rbp-1DCh]
  __int128 v156; // [rsp+50h] [rbp-1D0h]
  __int128 v157; // [rsp+50h] [rbp-1D0h]
  __int64 v158; // [rsp+50h] [rbp-1D0h]
  __int64 v159; // [rsp+60h] [rbp-1C0h]
  __m128i v160; // [rsp+60h] [rbp-1C0h]
  __int128 v161; // [rsp+60h] [rbp-1C0h]
  char v162; // [rsp+60h] [rbp-1C0h]
  unsigned int v163; // [rsp+60h] [rbp-1C0h]
  __int128 v164; // [rsp+60h] [rbp-1C0h]
  unsigned __int8 *v165; // [rsp+C0h] [rbp-160h]
  unsigned __int8 *v166; // [rsp+120h] [rbp-100h]
  __int64 v167; // [rsp+140h] [rbp-E0h] BYREF
  int v168; // [rsp+148h] [rbp-D8h]
  unsigned int v169; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v170; // [rsp+158h] [rbp-C8h]
  __int128 v171; // [rsp+160h] [rbp-C0h] BYREF
  __int128 v172; // [rsp+170h] [rbp-B0h] BYREF
  unsigned int v173; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v174; // [rsp+188h] [rbp-98h]
  unsigned __int16 v175; // [rsp+190h] [rbp-90h] BYREF
  __int64 v176; // [rsp+198h] [rbp-88h]
  __int64 v177; // [rsp+1A0h] [rbp-80h]
  __int64 v178; // [rsp+1A8h] [rbp-78h]
  __int64 v179; // [rsp+1B0h] [rbp-70h]
  __int64 v180; // [rsp+1B8h] [rbp-68h]
  unsigned __int16 v181; // [rsp+1C0h] [rbp-60h] BYREF
  __int64 v182; // [rsp+1C8h] [rbp-58h]
  __int64 v183; // [rsp+1D0h] [rbp-50h] BYREF
  __int64 v184; // [rsp+1D8h] [rbp-48h]
  __int64 v185; // [rsp+1E0h] [rbp-40h]
  __int64 v186; // [rsp+1E8h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v167 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v167, v9, 1);
  v10 = (__int64 *)a5[5];
  v168 = *(_DWORD *)(a2 + 72);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v169) = v12;
  v14 = a5[8];
  v170 = v13;
  v159 = v14;
  v15 = *(__int64 (__fastcall **)(_WORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v16 = sub_2E79000(v10);
  v150 = v15(a1, v16, v159, v169, v170);
  v17 = *(_QWORD *)(a2 + 40);
  v151 = v18;
  v19 = _mm_loadu_si128((const __m128i *)v17);
  v160 = _mm_loadu_si128((const __m128i *)(v17 + 40));
  v153 = *(_DWORD *)(a2 + 24);
  v22 = sub_33DFBC0(v160.m128i_i64[0], v160.m128i_u32[2], 0, 0, v20, v21);
  if ( v22 )
  {
    v23 = *(_QWORD *)(v22 + 96);
    v24 = *(_DWORD *)(v23 + 32);
    if ( v24 > 0x40 )
    {
      v144 = *(_QWORD *)(v22 + 96);
      v148 = v23 + 24;
      v39 = v23 + 24;
      if ( (unsigned int)sub_C44630(v23 + 24) == 1 )
      {
        v40 = v24 - 1;
        v26 = 0;
        if ( v153 == 80 )
        {
          v26 = 1;
          if ( (*(_QWORD *)(*(_QWORD *)(v144 + 24) + 8LL * (v40 >> 6)) & (1LL << v40)) != 0 )
          {
            v126 = sub_C44590(v39);
            v39 = v148;
            v26 = v126 != v40;
          }
        }
        v27 = v40 - (unsigned int)sub_C444A0(v39);
        goto LABEL_26;
      }
    }
    else
    {
      v25 = *(_QWORD *)(v23 + 24);
      if ( v25 && (v25 & (v25 - 1)) == 0 )
      {
        v26 = 0;
        if ( v153 == 80 )
          v26 = 1LL << ((unsigned __int8)v24 - 1) != v25;
        _BitScanReverse64(&v25, v25);
        v27 = 63 - ((unsigned int)v25 ^ 0x3F);
LABEL_26:
        *(_QWORD *)&v41 = sub_3400E40((__int64)a5, v27, v169, v170, (__int64)&v167, v19);
        v42 = *((_QWORD *)&v41 + 1);
        v43 = v41;
        v166 = sub_3406EB0(a5, 0xBEu, (__int64)&v167, v169, v170, v44, *(_OWORD *)&v19, v41);
        v45 = v170;
        *(_QWORD *)a3 = v166;
        *(_DWORD *)(a3 + 8) = v46;
        *((_QWORD *)&v140 + 1) = v42;
        *(_QWORD *)&v140 = v43;
        v48 = sub_3406EB0(a5, (unsigned int)!v26 + 191, (__int64)&v167, v169, v45, v47, *(_OWORD *)a3, v140);
        v50 = v49;
        *(_QWORD *)&v51 = sub_33ED040(a5, 0x16u);
        *((_QWORD *)&v137 + 1) = v50;
        *(_QWORD *)&v137 = v48;
        v52 = 1;
        *a4 = sub_340F900(a5, 0xD0u, (__int64)&v167, v150, v151, v53, v137, *(_OWORD *)&v19, v51);
        *((_DWORD *)a4 + 2) = v54;
        goto LABEL_42;
      }
    }
  }
  v28 = v169;
  if ( (_WORD)v169 )
  {
    if ( (unsigned __int16)(v169 - 17) > 0xD3u )
      goto LABEL_12;
    v28 = word_4456580[(unsigned __int16)v169 - 1];
    v32 = 0;
  }
  else
  {
    v29 = sub_30070B0((__int64)&v169);
    v28 = 0;
    if ( !v29 )
    {
LABEL_12:
      v32 = v170;
      goto LABEL_13;
    }
    v133 = sub_3009970((__int64)&v169, v160.m128i_i64[1], 0, v30, v31);
    v135 = v134;
    v28 = v133;
    v32 = v135;
  }
LABEL_13:
  v175 = v28;
  v176 = v32;
  if ( v28 )
  {
    if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
      goto LABEL_117;
    v33 = *(_QWORD *)&byte_444C4A0[16 * v28 - 16];
  }
  else
  {
    v177 = sub_3007260((__int64)&v175);
    LODWORD(v33) = v177;
    v178 = v34;
  }
  v35 = 2 * v33;
  switch ( v35 )
  {
    case 2u:
      v38 = 3;
      break;
    case 4u:
      v38 = 4;
      break;
    case 8u:
      v38 = 5;
      break;
    case 0x10u:
      v38 = 6;
      break;
    case 0x20u:
      v38 = 7;
      break;
    case 0x40u:
      v38 = 8;
      break;
    case 0x80u:
      v38 = 9;
      break;
    default:
      v36 = sub_3007020((_QWORD *)a5[8], v35);
      v147 = v37;
      v5 = v36;
      v38 = v36;
      goto LABEL_31;
  }
  v147 = 0;
LABEL_31:
  v55 = v169;
  LOWORD(v5) = v38;
  v56 = v5;
  if ( (_WORD)v169 )
  {
    if ( (unsigned __int16)(v169 - 17) > 0xD3u )
    {
      *(_QWORD *)&v171 = 0;
      DWORD2(v171) = 0;
      *(_QWORD *)&v172 = 0;
      v57 = v153 == 80;
      DWORD2(v172) = 0;
      v154 = v153 == 80;
      if ( (_WORD)v169 == 1 )
        goto LABEL_34;
      goto LABEL_88;
    }
    v119 = (unsigned __int16)(v169 - 176) <= 0x34u;
    LODWORD(v120) = word_4456340[(unsigned __int16)v169 - 1];
    LOBYTE(v121) = v119;
  }
  else
  {
    v145 = v38;
    v81 = sub_30070B0((__int64)&v169);
    v38 = v145;
    v56 = v5;
    if ( !v81 )
    {
      *(_QWORD *)&v171 = 0;
      DWORD2(v171) = 0;
      *(_QWORD *)&v172 = 0;
      v57 = v153 == 80;
      DWORD2(v172) = 0;
      v154 = v153 == 80;
      goto LABEL_47;
    }
    v120 = sub_3007240((__int64)&v169);
    v121 = HIDWORD(v120);
    v119 = BYTE4(v120);
  }
  LODWORD(v185) = v120;
  v122 = (__int64 *)a5[8];
  BYTE4(v185) = v121;
  if ( v119 )
    v38 = sub_2D43AD0(v5, v120);
  else
    v38 = sub_2D43050(v5, v120);
  if ( v38 )
  {
    v147 = 0;
  }
  else
  {
    v142 = sub_3009450(v122, (unsigned int)v5, v147, v185, v123, v124);
    v38 = v142;
    v147 = v132;
  }
  v125 = v142;
  *(_QWORD *)&v171 = 0;
  DWORD2(v171) = 0;
  LOWORD(v125) = v38;
  v154 = v153 == 80;
  v57 = v154;
  *(_QWORD *)&v172 = 0;
  v56 = v125;
  v55 = v169;
  DWORD2(v172) = 0;
  if ( (_WORD)v169 == 1 )
    goto LABEL_34;
  if ( !(_WORD)v169 )
  {
LABEL_47:
    if ( v38 && *(_QWORD *)&a1[4 * v38 + 56] )
      goto LABEL_49;
    v149 = v57;
    if ( !sub_30070B0((__int64)&v169) )
    {
      LOBYTE(v57) = v149;
LABEL_94:
      v60 = (__int64)a5;
      sub_3495B70(
        a1,
        (__int64)a5,
        (__int64)&v167,
        v57,
        v19.m128i_i64[0],
        v19.m128i_i64[1],
        v19,
        *(_OWORD *)&v160,
        &v171,
        &v172);
      goto LABEL_37;
    }
LABEL_101:
    v52 = 0;
    goto LABEL_42;
  }
LABEL_88:
  if ( !*(_QWORD *)&a1[4 * v55 + 56] )
    goto LABEL_70;
LABEL_34:
  v58 = &dword_44E2180[3 * v57];
  v59 = *v58;
  if ( *v58 > 0x1F3 || (*((_BYTE *)&a1[250 * v55 + 3207] + v59) & 0xFB) == 0 )
  {
    v60 = v59;
    *(_QWORD *)&v171 = sub_3406EB0(a5, 0x3Au, (__int64)&v167, v169, v170, v56, *(_OWORD *)&v19, *(_OWORD *)&v160);
    DWORD2(v171) = v61;
    *(_QWORD *)&v172 = sub_3406EB0(a5, v59, (__int64)&v167, v169, v170, v62, *(_OWORD *)&v19, *(_OWORD *)&v160);
    DWORD2(v172) = v63;
    goto LABEL_37;
  }
  v118 = v58[1];
  if ( v55 != 1 && !*(_QWORD *)&a1[4 * v55 + 56]
    || v118 <= 0x1F3 && (*((_BYTE *)&a1[250 * v55 + 3207] + v118) & 0xFB) != 0 )
  {
LABEL_70:
    if ( v38 && *(_QWORD *)&a1[4 * v38 + 56] )
    {
LABEL_49:
      LOWORD(v56) = v38;
      v146 = v56;
      v143 = dword_44E2180[3 * v57 + 2];
      *(_QWORD *)&v156 = sub_33FAF80((__int64)a5, v143, (__int64)&v167, (unsigned int)v56, v147, v56, v19);
      *((_QWORD *)&v156 + 1) = v82 | v19.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v165 = sub_33FAF80((__int64)a5, v143, (__int64)&v167, (unsigned int)v146, v147, v146, v19);
      *((_QWORD *)&v141 + 1) = v83 | v160.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v141 = v165;
      *(_QWORD *)&v84 = sub_3406EB0(a5, 0x3Au, (__int64)&v167, (unsigned int)v146, v147, v146, v156, v141);
      v157 = v84;
      *(_QWORD *)&v171 = sub_33FAF80((__int64)a5, 216, (__int64)&v167, v169, v170, v85, v19);
      DWORD2(v171) = v86;
      v87 = sub_32844A0((unsigned __int16 *)&v169, 216);
      *(_QWORD *)&v88 = sub_3400E40((__int64)a5, v87, v146, v147, (__int64)&v167, v19);
      sub_3406EB0(a5, 0xC0u, (__int64)&v167, (unsigned int)v146, v147, v146, v157, v88);
      v60 = 216;
      *(_QWORD *)&v172 = sub_33FAF80((__int64)a5, 216, (__int64)&v167, v169, v170, v89, v19);
      DWORD2(v172) = v90;
      goto LABEL_37;
    }
    if ( (unsigned __int16)(v55 - 17) > 0xD3u )
      goto LABEL_94;
    goto LABEL_101;
  }
  v127 = (unsigned int *)sub_33E5110(a5, v169, v170, v169, v170);
  v60 = v118;
  v130 = sub_3411F20(a5, v118, (__int64)&v167, v127, v128, v129, *(_OWORD *)&v19, *(_OWORD *)&v160);
  DWORD2(v172) = 1;
  *(_QWORD *)&v171 = v130;
  DWORD2(v171) = v131;
  *(_QWORD *)&v172 = v130;
LABEL_37:
  v64 = v171;
  v65 = DWORD2(v171);
  *(_QWORD *)a3 = v171;
  *(_DWORD *)(a3 + 8) = v65;
  if ( !v154 )
  {
    v66 = sub_3400BD0((__int64)a5, 0, (__int64)&v167, v169, v170, 0, v19, 0);
    v68 = v67;
    v161 = v172;
    *(_QWORD *)&v69 = sub_33ED040(a5, 0x16u);
    *((_QWORD *)&v138 + 1) = v68;
    *(_QWORD *)&v138 = v66;
    *a4 = sub_340F900(a5, 0xD0u, (__int64)&v167, v150, v151, *((__int64 *)&v161 + 1), v161, v138, v69);
    *((_DWORD *)a4 + 2) = v71;
    goto LABEL_39;
  }
  v100 = v169;
  v101 = (unsigned __int16 *)(*(_QWORD *)(v64 + 48) + 16 * v65);
  v102 = *((_QWORD *)v101 + 1);
  v103 = *v101;
  if ( (_WORD)v169 )
  {
    if ( (unsigned __int16)(v169 - 17) <= 0xD3u )
    {
      v100 = word_4456580[(unsigned __int16)v169 - 1];
      v107 = 0;
      goto LABEL_62;
    }
  }
  else
  {
    v158 = *v101;
    v104 = sub_30070B0((__int64)&v169);
    v103 = v158;
    if ( v104 )
    {
      v136 = sub_3009970((__int64)&v169, v60, v105, v158, v106);
      v103 = v158;
      v100 = v136;
      goto LABEL_62;
    }
  }
  v107 = v170;
LABEL_62:
  LOWORD(v183) = v100;
  v184 = v107;
  if ( v100 )
  {
    if ( v100 == 1 || (unsigned __int16)(v100 - 504) <= 7u )
      goto LABEL_117;
    v109 = *(_QWORD *)&byte_444C4A0[16 * v100 - 16];
  }
  else
  {
    v163 = v103;
    v108 = sub_3007260((__int64)&v183);
    v103 = v163;
    v179 = v108;
    v109 = v108;
    v180 = v110;
  }
  *(_QWORD *)&v111 = sub_3400E40((__int64)a5, v109 - 1, v103, v102, (__int64)&v167, v19);
  v113 = sub_3406EB0(a5, 0xBFu, (__int64)&v167, v169, v170, v112, v171, v111);
  v115 = v114;
  v164 = v172;
  *(_QWORD *)&v116 = sub_33ED040(a5, 0x16u);
  *((_QWORD *)&v139 + 1) = v115;
  *(_QWORD *)&v139 = v113;
  *a4 = sub_340F900(a5, 0xD0u, (__int64)&v167, v150, v151, *((__int64 *)&v164 + 1), v164, v139, v116);
  *((_DWORD *)a4 + 2) = v117;
LABEL_39:
  v72 = *(_QWORD *)(a2 + 48);
  v73 = *(_WORD *)(v72 + 16);
  v74 = *(_QWORD *)(v72 + 24);
  LOWORD(v173) = v73;
  v75 = *a4;
  v76 = *((unsigned int *)a4 + 2);
  v174 = v74;
  v77 = *(_QWORD *)(v75 + 48) + 16 * v76;
  v78 = *(_WORD *)v77;
  v79 = *(_QWORD *)(v77 + 8);
  if ( v78 == v73 )
  {
    if ( v73 || v74 == v79 )
    {
LABEL_41:
      v52 = 1;
      goto LABEL_42;
    }
    v182 = v79;
    v181 = 0;
  }
  else
  {
    v181 = v78;
    v182 = v79;
    if ( v78 )
    {
      if ( v78 == 1 || (unsigned __int16)(v78 - 504) <= 7u )
        goto LABEL_117;
      v91 = *(_QWORD *)&byte_444C4A0[16 * v78 - 16];
      v93 = byte_444C4A0[16 * v78 - 8];
      if ( !v73 )
        goto LABEL_52;
      goto LABEL_76;
    }
  }
  v185 = sub_3007260((__int64)&v181);
  v91 = v185;
  v186 = v92;
  v93 = v92;
  if ( v73 )
  {
LABEL_76:
    if ( v73 != 1 && (unsigned __int16)(v73 - 504) > 7u )
    {
      v98 = *(_QWORD *)&byte_444C4A0[16 * v73 - 16];
      v52 = (unsigned __int8)byte_444C4A0[16 * v73 - 8];
      goto LABEL_53;
    }
LABEL_117:
    BUG();
  }
LABEL_52:
  v162 = v93;
  v94 = sub_3007260((__int64)&v173);
  v93 = v162;
  v95 = v94;
  v97 = v96;
  v183 = v95;
  v98 = v95;
  v184 = v97;
  v52 = (unsigned __int8)v97;
LABEL_53:
  if ( !(_BYTE)v52 || v93 )
  {
    if ( v91 > v98 )
    {
      *a4 = (__int64)sub_33FAF80((__int64)a5, 216, (__int64)&v167, v173, v174, v70, v19);
      *((_DWORD *)a4 + 2) = v99;
    }
    goto LABEL_41;
  }
LABEL_42:
  if ( v167 )
    sub_B91220((__int64)&v167, v167);
  return v52;
}
