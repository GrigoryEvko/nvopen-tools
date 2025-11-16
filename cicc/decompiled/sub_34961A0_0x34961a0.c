// Function: sub_34961A0
// Address: 0x34961a0
//
__int64 __fastcall sub_34961A0(_WORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v6; // rsi
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __int64 v9; // rax
  __int16 v10; // cx
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // r14
  __int64 (__fastcall *v15)(_WORD *, __int64, __int64, _QWORD, __int64); // r13
  __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 v18; // r9
  unsigned __int16 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned int v23; // r14d
  __int64 v24; // rdx
  __int64 v25; // r13
  unsigned int v26; // r13d
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int16 v30; // cx
  __int64 v31; // rdx
  unsigned int v32; // r12d
  unsigned int *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r9
  unsigned __int8 *v36; // rax
  __int64 v37; // rdx
  __int128 v38; // rax
  __int64 v39; // r13
  __int64 v40; // r8
  unsigned __int64 v41; // rax
  __int128 v42; // rax
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r13
  unsigned __int8 *v46; // r12
  __int64 v47; // r9
  __int64 v48; // rdx
  __int128 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned int v52; // edx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  unsigned __int16 v56; // ax
  __int64 v57; // r9
  __int64 v58; // rdx
  unsigned int *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r9
  unsigned __int8 *v62; // rax
  int v64; // edx
  __int64 v65; // r9
  __int32 v66; // edx
  __int128 v67; // rax
  __int64 v68; // r9
  __int128 v69; // rax
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rax
  unsigned __int8 *v72; // r12
  __int64 v73; // rdx
  __int64 v74; // r13
  __int128 v75; // rax
  __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rdi
  unsigned __int16 *v80; // rdx
  __int64 v81; // r9
  unsigned __int64 v82; // rdi
  bool v83; // al
  __int64 v84; // rdx
  unsigned __int8 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r13
  unsigned __int8 *v88; // r12
  __int128 v89; // rax
  unsigned __int8 *v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r13
  unsigned __int8 *v93; // r12
  int v94; // r9d
  int v95; // edx
  __int128 v96; // rax
  int v97; // r9d
  __int32 v98; // edx
  unsigned __int64 v99; // rsi
  unsigned __int64 v100; // rax
  char v101; // dl
  __int64 *v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  unsigned int v106; // r13d
  __int64 v107; // rdx
  __int64 v108; // r12
  __int64 v109; // rdx
  __int64 v110; // r12
  unsigned __int64 v111; // rax
  __int64 v112; // rdx
  unsigned int v113; // r12d
  unsigned __int64 v114; // rax
  __int128 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rdi
  unsigned __int16 *v119; // rdx
  __int64 v120; // r9
  unsigned __int8 *v121; // rax
  unsigned int v122; // edx
  unsigned int v123; // edx
  unsigned int v124; // esi
  unsigned __int8 *v125; // r12
  __int64 v126; // rdx
  __int64 v127; // r13
  __int128 v128; // rax
  __int64 v129; // r9
  __int64 v130; // rdx
  __int128 v131; // rax
  __int64 v132; // r9
  unsigned __int8 *v133; // r12
  __int64 v134; // rdx
  __int64 v135; // r13
  __int128 v136; // rax
  __int128 v137; // rax
  __int64 v138; // rdx
  __int128 v139; // rax
  __int64 v140; // r9
  __int128 v141; // rax
  unsigned int *v142; // rax
  __int64 v143; // rdx
  __int64 v144; // r9
  unsigned __int8 *v145; // rbx
  __int128 v146; // rax
  __int64 v147; // r8
  __int128 v148; // [rsp-40h] [rbp-250h]
  __int128 v149; // [rsp-30h] [rbp-240h]
  __int128 v150; // [rsp-30h] [rbp-240h]
  __int128 v151; // [rsp-20h] [rbp-230h]
  __int128 v152; // [rsp-20h] [rbp-230h]
  __int128 v153; // [rsp-20h] [rbp-230h]
  __int128 v154; // [rsp-10h] [rbp-220h]
  __int128 v155; // [rsp-10h] [rbp-220h]
  __int128 v156; // [rsp-10h] [rbp-220h]
  __int128 v157; // [rsp-10h] [rbp-220h]
  __int128 v158; // [rsp+0h] [rbp-210h]
  __int64 v159; // [rsp+10h] [rbp-200h]
  unsigned __int16 v160; // [rsp+24h] [rbp-1ECh]
  __int64 *v161; // [rsp+28h] [rbp-1E8h]
  unsigned int v162; // [rsp+30h] [rbp-1E0h]
  __int128 v163; // [rsp+30h] [rbp-1E0h]
  __int64 v164; // [rsp+40h] [rbp-1D0h]
  __int128 v165; // [rsp+40h] [rbp-1D0h]
  __int128 v166; // [rsp+40h] [rbp-1D0h]
  unsigned int v167; // [rsp+50h] [rbp-1C0h]
  __int64 v168; // [rsp+50h] [rbp-1C0h]
  __int128 v169; // [rsp+50h] [rbp-1C0h]
  bool v170; // [rsp+60h] [rbp-1B0h]
  __int128 v171; // [rsp+60h] [rbp-1B0h]
  unsigned __int8 *v172; // [rsp+60h] [rbp-1B0h]
  __int128 v173; // [rsp+70h] [rbp-1A0h]
  __int128 v174; // [rsp+80h] [rbp-190h]
  __int64 v175; // [rsp+90h] [rbp-180h]
  __int128 v176; // [rsp+90h] [rbp-180h]
  __int64 v177; // [rsp+90h] [rbp-180h]
  __int64 v178; // [rsp+90h] [rbp-180h]
  __int64 v179; // [rsp+90h] [rbp-180h]
  __int64 v180; // [rsp+A0h] [rbp-170h]
  __int64 v181; // [rsp+A0h] [rbp-170h]
  __int128 v182; // [rsp+A0h] [rbp-170h]
  unsigned int v183; // [rsp+A0h] [rbp-170h]
  __int128 v184; // [rsp+A0h] [rbp-170h]
  __int64 v185; // [rsp+B0h] [rbp-160h]
  __int128 v186; // [rsp+B0h] [rbp-160h]
  __int128 v187; // [rsp+B0h] [rbp-160h]
  unsigned __int8 v189; // [rsp+C0h] [rbp-150h]
  unsigned int v190; // [rsp+C0h] [rbp-150h]
  __int64 v191; // [rsp+C0h] [rbp-150h]
  __int64 v192; // [rsp+C8h] [rbp-148h]
  unsigned int v193; // [rsp+D0h] [rbp-140h]
  __int128 v194; // [rsp+D0h] [rbp-140h]
  __int128 v195; // [rsp+D0h] [rbp-140h]
  __int128 v196; // [rsp+D0h] [rbp-140h]
  __int128 v197; // [rsp+D0h] [rbp-140h]
  __int64 v198; // [rsp+150h] [rbp-C0h]
  __int64 v199; // [rsp+170h] [rbp-A0h] BYREF
  int v200; // [rsp+178h] [rbp-98h]
  unsigned int v201; // [rsp+180h] [rbp-90h] BYREF
  __int64 v202; // [rsp+188h] [rbp-88h]
  __int128 v203; // [rsp+190h] [rbp-80h] BYREF
  __int128 v204; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v205; // [rsp+1B0h] [rbp-60h] BYREF
  unsigned int v206; // [rsp+1B8h] [rbp-58h]
  unsigned __int64 v207; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v208; // [rsp+1C8h] [rbp-48h]
  __int64 v209; // [rsp+1D0h] [rbp-40h]
  __int64 v210; // [rsp+1D8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v199 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v199, v6, 1);
  v7 = *(_QWORD *)(a2 + 40);
  v200 = *(_DWORD *)(a2 + 72);
  v8 = _mm_loadu_si128((const __m128i *)v7);
  v9 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * *(unsigned int *)(v7 + 8);
  v10 = *(_WORD *)v9;
  v174 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v202 = *(_QWORD *)(v9 + 8);
  v11 = *(_QWORD *)(v7 + 80);
  LOWORD(v201) = v10;
  v12 = *(_QWORD *)(v11 + 96);
  if ( *(_DWORD *)(v12 + 32) <= 0x40u )
    v185 = *(_QWORD *)(v12 + 24);
  else
    v185 = **(_QWORD **)(v12 + 24);
  v193 = v185;
  v13 = *(_DWORD *)(a2 + 24);
  if ( v13 == 90 )
  {
    v170 = 1;
  }
  else
  {
    v189 = 1;
    v170 = v13 == 91;
    if ( v13 == 88 )
      goto LABEL_7;
  }
  v189 = v13 == 90;
LABEL_7:
  v14 = *(_QWORD *)(a3 + 64);
  v15 = *(__int64 (__fastcall **)(_WORD *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v16 = sub_2E79000(*(__int64 **)(a3 + 40));
  v17 = v15(a1, v16, v14, v201, v202);
  v19 = v201;
  v164 = v20;
  v162 = v17;
  if ( (_WORD)v201 )
  {
    if ( (unsigned __int16)(v201 - 17) <= 0xD3u )
    {
      v208 = 0;
      v19 = word_4456580[(unsigned __int16)v201 - 1];
      LOWORD(v207) = v19;
      if ( !v19 )
        goto LABEL_11;
      goto LABEL_44;
    }
    goto LABEL_9;
  }
  if ( !sub_30070B0((__int64)&v201) )
  {
LABEL_9:
    v21 = v202;
    goto LABEL_10;
  }
  v19 = sub_3009970((__int64)&v201, v16, v53, v54, v55);
LABEL_10:
  LOWORD(v207) = v19;
  v208 = v21;
  if ( !v19 )
  {
LABEL_11:
    v209 = sub_3007260((__int64)&v207);
    v210 = v22;
    LODWORD(v180) = v209;
    goto LABEL_12;
  }
LABEL_44:
  if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
    BUG();
  v180 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
LABEL_12:
  v23 = v180;
  if ( (_DWORD)v185 )
  {
LABEL_21:
    *(_QWORD *)&v203 = 0;
    DWORD2(v203) = 0;
    *(_QWORD *)&v204 = 0;
    DWORD2(v204) = 0;
    v26 = (v189 << 31 >> 31) + 64;
    v167 = 173 - (v189 == 0);
    goto LABEL_22;
  }
  if ( !v170 )
  {
    v24 = 1;
    if ( (_WORD)v201 == 1
      || (_WORD)v201 && (v24 = (unsigned __int16)v201, *(_QWORD *)&a1[4 * (unsigned __int16)v201 + 56]) )
    {
      if ( (a1[250 * v24 + 3236] & 0xFB) == 0 )
      {
        v25 = (__int64)sub_3406EB0((_QWORD *)a3, 0x3Au, (__int64)&v199, v201, v202, v18, *(_OWORD *)&v8, v174);
        goto LABEL_59;
      }
    }
    goto LABEL_21;
  }
  if ( !v189 )
  {
    v106 = v201;
    v177 = v202;
    if ( (unsigned __int8)sub_328A020((__int64)a1, 0x51u, v201, v202, 0) )
    {
      v142 = (unsigned int *)sub_33E5110((__int64 *)a3, v106, v177, v162, v164);
      v145 = sub_3411F20((_QWORD *)a3, 81, (__int64)&v199, v142, v143, v144, *(_OWORD *)&v8, v174);
      sub_9691E0((__int64)&v207, v180, -1, 1u, 0);
      *(_QWORD *)&v146 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
      v25 = sub_3288B20(a3, (int)&v199, v201, v202, (__int64)v145, 1, v146, (unsigned __int64)v145, 0);
      if ( (unsigned int)v208 > 0x40 )
      {
        v82 = v207;
        if ( v207 )
          goto LABEL_80;
      }
      goto LABEL_59;
    }
    *(_QWORD *)&v203 = 0;
    v26 = 64;
    DWORD2(v203) = 0;
    *(_QWORD *)&v204 = 0;
    DWORD2(v204) = 0;
    v167 = 172;
LABEL_22:
    v27 = 2 * v180;
    if ( 2 * (_DWORD)v180 == 2 )
    {
      v30 = 3;
    }
    else
    {
      switch ( v27 )
      {
        case 4u:
          v30 = 4;
          break;
        case 8u:
          v30 = 5;
          break;
        case 0x10u:
          v30 = 6;
          break;
        case 0x20u:
          v30 = 7;
          break;
        case 0x40u:
          v30 = 8;
          break;
        case 0x80u:
          v30 = 9;
          break;
        default:
          v28 = sub_3007020(*(_QWORD **)(a3 + 64), v27);
          v175 = v29;
          v3 = v28;
          v30 = v28;
          goto LABEL_52;
      }
    }
    v175 = 0;
LABEL_52:
    v56 = v201;
    LOWORD(v3) = v30;
    v57 = v3;
    if ( (_WORD)v201 )
    {
      if ( (unsigned __int16)(v201 - 17) > 0xD3u )
      {
        v58 = 1;
        if ( (_WORD)v201 == 1 )
          goto LABEL_55;
        goto LABEL_93;
      }
      v101 = (unsigned __int16)(v201 - 176) <= 0x34u;
      LODWORD(v99) = word_4456340[(unsigned __int16)v201 - 1];
      LOBYTE(v100) = v101;
    }
    else
    {
      v160 = v30;
      v83 = sub_30070B0((__int64)&v201);
      v30 = v160;
      v57 = v3;
      if ( !v83 )
        goto LABEL_82;
      v99 = sub_3007240((__int64)&v201);
      v100 = HIDWORD(v99);
      v101 = BYTE4(v99);
    }
    v102 = *(__int64 **)(a3 + 64);
    LODWORD(v207) = v99;
    BYTE4(v207) = v100;
    v161 = v102;
    if ( v101 )
      v30 = sub_2D43AD0(v3, v99);
    else
      v30 = sub_2D43050(v3, v99);
    if ( v30 )
    {
      v175 = 0;
    }
    else
    {
      v159 = sub_3009450(v161, (unsigned int)v3, v175, v207, v103, v104);
      v30 = v159;
      v175 = v107;
    }
    v105 = v159;
    v58 = 1;
    LOWORD(v105) = v30;
    v57 = v105;
    v56 = v201;
    if ( (_WORD)v201 == 1 )
      goto LABEL_55;
    if ( (_WORD)v201 )
    {
LABEL_93:
      v58 = v56;
      if ( !*(_QWORD *)&a1[4 * v56 + 56] )
        goto LABEL_83;
LABEL_55:
      if ( (*((_BYTE *)&a1[250 * (unsigned int)v58 + 3207] + v26) & 0xFB) == 0 )
      {
        v59 = (unsigned int *)sub_33E5110((__int64 *)a3, v201, v202, v201, v202);
        v62 = sub_3411F20((_QWORD *)a3, v26, (__int64)&v199, v59, v60, v61, *(_OWORD *)&v8, v174);
        DWORD2(v203) = 0;
        *(_QWORD *)&v203 = v62;
        *(_QWORD *)&v204 = v62;
        DWORD2(v204) = 1;
        goto LABEL_57;
      }
      if ( v56 == 1 )
      {
        v58 = 1;
      }
      else if ( !*(_QWORD *)&a1[4 * (int)v58 + 56] )
      {
        goto LABEL_83;
      }
      if ( (*((_BYTE *)&a1[250 * v58 + 3207] + v167) & 0xFB) == 0 )
      {
        *(_QWORD *)&v203 = sub_3406EB0((_QWORD *)a3, 0x3Au, (__int64)&v199, v201, v202, v57, *(_OWORD *)&v8, v174);
        DWORD2(v203) = v64;
        *(_QWORD *)&v204 = sub_3406EB0((_QWORD *)a3, v167, (__int64)&v199, v201, v202, v65, *(_OWORD *)&v8, v174);
        DWORD2(v204) = v66;
        if ( (_DWORD)v185 == (_DWORD)v180 )
          goto LABEL_58;
        goto LABEL_66;
      }
LABEL_83:
      LOWORD(v57) = v30;
      v84 = 1;
      if ( v30 == 1 || v30 && (v84 = v30, *(_QWORD *)&a1[4 * v30 + 56]) )
      {
        if ( (a1[250 * v84 + 3236] & 0xFB) == 0 )
        {
          v168 = v57;
          v85 = sub_33FAF80(a3, (unsigned int)(v189 == 0) + 213, (__int64)&v199, (unsigned int)v57, v175, v57, v8);
          v87 = v86;
          v88 = v85;
          *(_QWORD *)&v89 = sub_33FAF80(
                              a3,
                              (unsigned int)(v189 == 0) + 213,
                              (__int64)&v199,
                              (unsigned int)v168,
                              v175,
                              v168,
                              v8);
          *((_QWORD *)&v156 + 1) = v87;
          *(_QWORD *)&v156 = v88;
          v90 = sub_3406EB0((_QWORD *)a3, 0x3Au, (__int64)&v199, (unsigned int)v168, v175, v168, v156, v89);
          v92 = v91;
          v93 = v90;
          *(_QWORD *)&v203 = sub_33FAF80(a3, 216, (__int64)&v199, v201, v202, v94, v8);
          DWORD2(v203) = v95;
          *(_QWORD *)&v96 = sub_3400E40(a3, (unsigned int)v180, v168, v175, (__int64)&v199, v8);
          *((_QWORD *)&v152 + 1) = v92;
          *(_QWORD *)&v152 = v93;
          sub_3406EB0((_QWORD *)a3, 0xBFu, (__int64)&v199, (unsigned int)v168, v175, v168, v152, v96);
          *(_QWORD *)&v204 = sub_33FAF80(a3, 216, (__int64)&v199, v201, v202, v97, v8);
          DWORD2(v204) = v98;
          goto LABEL_57;
        }
      }
      if ( v56 )
      {
        if ( (unsigned __int16)(v56 - 17) <= 0xD3u )
        {
LABEL_100:
          v25 = 0;
          goto LABEL_59;
        }
      }
      else if ( sub_30070B0((__int64)&v201) )
      {
        goto LABEL_100;
      }
      sub_3495B70(a1, a3, (__int64)&v199, v189, v8.m128i_i64[0], v8.m128i_i64[1], v8, v174, &v203, &v204);
LABEL_57:
      if ( (_DWORD)v185 == (_DWORD)v180 )
      {
LABEL_58:
        v25 = v204;
        goto LABEL_59;
      }
LABEL_66:
      *(_QWORD *)&v67 = sub_3400E40(a3, (unsigned int)v185, v201, v202, (__int64)&v199, v8);
      *(_QWORD *)&v69 = sub_340F900((_QWORD *)a3, 0xC4u, (__int64)&v199, v201, v202, v68, v204, v203, v67);
      v187 = v69;
      if ( !v170 )
      {
        v25 = v69;
        goto LABEL_59;
      }
      if ( !v189 )
      {
        v206 = v180;
        if ( (unsigned int)v180 > 0x40 )
        {
          sub_C43690((__int64)&v205, -1, 1);
          LODWORD(v208) = v180;
          sub_C43690((__int64)&v207, 0, 0);
        }
        else
        {
          LODWORD(v208) = v180;
          v207 = 0;
          v70 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v180;
          if ( !(_DWORD)v180 )
            v70 = 0;
          v205 = v70;
        }
        if ( v193 )
        {
          if ( v193 > 0x40 )
          {
            sub_C43C90(&v207, 0, v193);
          }
          else
          {
            v71 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v193);
            if ( (unsigned int)v208 > 0x40 )
              *(_QWORD *)v207 |= v71;
            else
              v207 |= v71;
          }
        }
        v72 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
        v74 = v73;
        if ( (unsigned int)v208 > 0x40 && v207 )
          j_j___libc_free_0_0(v207);
        *(_QWORD *)&v75 = sub_34007B0(a3, (__int64)&v205, (__int64)&v199, v201, v202, 0, v8, 0);
        v76 = v75;
        v195 = v75;
        v182 = v204;
        v190 = DWORD2(v75);
        v77 = sub_33ED040((_QWORD *)a3, 0xAu);
        v79 = v78;
        v80 = (unsigned __int16 *)(*(_QWORD *)(v76 + 48) + 16LL * v190);
        *((_QWORD *)&v155 + 1) = v79;
        *(_QWORD *)&v155 = v77;
        *((_QWORD *)&v148 + 1) = v74;
        *(_QWORD *)&v148 = v72;
        v25 = (__int64)sub_33FC1D0(
                         (_QWORD *)a3,
                         207,
                         (__int64)&v199,
                         *v80,
                         *((_QWORD *)v80 + 1),
                         v81,
                         v182,
                         v148,
                         v195,
                         v187,
                         v155);
        goto LABEL_78;
      }
      LODWORD(v208) = v180;
      v183 = v180 - 1;
      v108 = 1LL << ((unsigned __int8)v23 - 1);
      if ( v23 > 0x40 )
      {
        sub_C43690((__int64)&v207, 0, 0);
        if ( (unsigned int)v208 > 0x40 )
        {
          *(_QWORD *)(v207 + 8LL * (v183 >> 6)) |= v108;
          v172 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
          v191 = v130;
          if ( (unsigned int)v208 <= 0x40 )
          {
            LODWORD(v208) = v23;
            v110 = ~v108;
            goto LABEL_148;
          }
LABEL_156:
          if ( v207 )
            j_j___libc_free_0_0(v207);
LABEL_117:
          LODWORD(v208) = v23;
          v110 = ~v108;
          if ( v23 <= 0x40 )
          {
            v111 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v23 - 1) & 0x3F));
            if ( !v23 )
              v111 = 0;
            v207 = v111;
            goto LABEL_121;
          }
LABEL_148:
          sub_C43690((__int64)&v207, -1, 1);
          if ( (unsigned int)v208 > 0x40 )
          {
            *(_QWORD *)(v207 + 8LL * (v183 >> 6)) &= v110;
LABEL_122:
            *(_QWORD *)&v173 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
            v178 = v112;
            if ( (unsigned int)v208 > 0x40 && v207 )
              j_j___libc_free_0_0(v207);
            if ( v193 )
            {
              LODWORD(v208) = v23;
              v113 = v193 - 1;
              if ( v23 > 0x40 )
                sub_C43690((__int64)&v207, 0, 0);
              else
                v207 = 0;
              if ( v193 != 1 )
              {
                if ( v113 > 0x40 )
                {
                  sub_C43C90(&v207, 0, v113);
                }
                else
                {
                  v114 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v193);
                  if ( (unsigned int)v208 > 0x40 )
                    *(_QWORD *)v207 |= v114;
                  else
                    v207 |= v114;
                }
              }
              *(_QWORD *)&v115 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
              v165 = v115;
              if ( (unsigned int)v208 > 0x40 && v207 )
                j_j___libc_free_0_0(v207);
              v163 = (__int128)_mm_loadu_si128((const __m128i *)&v204);
              *((_QWORD *)&v173 + 1) = v178;
              v116 = sub_33ED040((_QWORD *)a3, 0x12u);
              v118 = v117;
              v119 = (unsigned __int16 *)(*(_QWORD *)(v173 + 48) + 16LL * (unsigned int)v178);
              *((_QWORD *)&v158 + 1) = v118;
              *(_QWORD *)&v158 = v116;
              v121 = sub_33FC1D0(
                       (_QWORD *)a3,
                       207,
                       (__int64)&v199,
                       *v119,
                       *((_QWORD *)v119 + 1),
                       v120,
                       v163,
                       v165,
                       v173,
                       v187,
                       v158);
              LODWORD(v208) = v23;
              *(_QWORD *)&v187 = v121;
              *((_QWORD *)&v187 + 1) = v122 | *((_QWORD *)&v187 + 1) & 0xFFFFFFFF00000000LL;
              if ( v23 > 0x40 )
              {
                sub_C43690((__int64)&v207, 0, 0);
                v123 = v208;
                v183 = v208 - 1;
              }
              else
              {
                v207 = 0;
                v123 = v23;
              }
              v124 = v183 + v193 - v23;
              if ( v124 != v123 )
              {
                if ( v124 > 0x3F || v123 > 0x40 )
                  sub_C43C90(&v207, v124, v123);
                else
                  v207 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v193 + 63 - (unsigned __int8)v23) << v124;
              }
              v125 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
              v127 = v126;
              if ( (unsigned int)v208 > 0x40 && v207 )
                j_j___libc_free_0_0(v207);
              v196 = (__int128)_mm_loadu_si128((const __m128i *)&v204);
              *(_QWORD *)&v128 = sub_33ED040((_QWORD *)a3, 0x14u);
              *((_QWORD *)&v153 + 1) = v191;
              *(_QWORD *)&v153 = v172;
              *((_QWORD *)&v149 + 1) = v127;
              *(_QWORD *)&v149 = v125;
              v25 = (__int64)sub_33FC1D0(
                               (_QWORD *)a3,
                               207,
                               (__int64)&v199,
                               *(unsigned __int16 *)(*((_QWORD *)v172 + 6) + 16LL * (unsigned int)v191),
                               *(_QWORD *)(*((_QWORD *)v172 + 6) + 16LL * (unsigned int)v191 + 8),
                               v129,
                               v196,
                               v149,
                               v153,
                               v187,
                               v128);
            }
            else
            {
              *(_QWORD *)&v131 = sub_3400E40(a3, v183, v201, v202, (__int64)&v199, v8);
              v133 = sub_3406EB0((_QWORD *)a3, 0xBFu, (__int64)&v199, v201, v202, v132, v203, v131);
              v135 = v134;
              v197 = v204;
              *(_QWORD *)&v136 = sub_33ED040((_QWORD *)a3, 0x16u);
              *((_QWORD *)&v157 + 1) = v135;
              *(_QWORD *)&v157 = v133;
              *(_QWORD *)&v137 = sub_340F900(
                                   (_QWORD *)a3,
                                   0xD0u,
                                   (__int64)&v199,
                                   v162,
                                   v164,
                                   *((__int64 *)&v197 + 1),
                                   v197,
                                   v157,
                                   v136);
              v184 = v137;
              *(_QWORD *)&v169 = sub_3400BD0(a3, 0, (__int64)&v199, v201, v202, 0, v8, 0);
              *(_QWORD *)&v197 = v173;
              v166 = (__int128)_mm_loadu_si128((const __m128i *)&v204);
              *((_QWORD *)&v197 + 1) = v178;
              *((_QWORD *)&v169 + 1) = v138;
              *(_QWORD *)&v139 = sub_33ED040((_QWORD *)a3, 0x14u);
              *((_QWORD *)&v150 + 1) = v191;
              *(_QWORD *)&v150 = v172;
              *(_QWORD *)&v141 = sub_33FC1D0(
                                   (_QWORD *)a3,
                                   207,
                                   (__int64)&v199,
                                   *(unsigned __int16 *)(*((_QWORD *)v172 + 6) + 16LL * (unsigned int)v191),
                                   *(_QWORD *)(*((_QWORD *)v172 + 6) + 16LL * (unsigned int)v191 + 8),
                                   v140,
                                   v166,
                                   v169,
                                   v150,
                                   v197,
                                   v139);
              v25 = sub_3288B20(a3, (int)&v199, v201, v202, v184, *((__int64 *)&v184 + 1), v141, v187, 0);
            }
            goto LABEL_59;
          }
LABEL_121:
          v207 &= v110;
          goto LABEL_122;
        }
      }
      else
      {
        v207 = 0;
      }
      v207 |= v108;
      v172 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
      v191 = v109;
      if ( (unsigned int)v208 <= 0x40 )
        goto LABEL_117;
      goto LABEL_156;
    }
LABEL_82:
    v56 = 0;
    goto LABEL_83;
  }
  v31 = 1;
  if ( (_WORD)v201 != 1
    && (!(_WORD)v201 || (v31 = (unsigned __int16)v201, !*(_QWORD *)&a1[4 * (unsigned __int16)v201 + 56]))
    || (a1[250 * v31 + 3247] & 0xFB) != 0 )
  {
    *(_QWORD *)&v203 = 0;
    v26 = 63;
    DWORD2(v203) = 0;
    *(_QWORD *)&v204 = 0;
    DWORD2(v204) = 0;
    v167 = 173;
    goto LABEL_22;
  }
  v32 = v180 - 1;
  v33 = (unsigned int *)sub_33E5110((__int64 *)a3, v201, v202, v162, v164);
  v36 = sub_3411F20((_QWORD *)a3, 80, (__int64)&v199, v33, v34, v35, *(_OWORD *)&v8, v174);
  v192 = v37;
  v194 = (unsigned __int64)v36;
  v181 = (__int64)v36;
  *(_QWORD *)&v38 = sub_3400BD0(a3, 0, (__int64)&v199, v201, v202, 0, v8, 0);
  v206 = v23;
  v186 = v38;
  v39 = 1LL << v32;
  if ( v23 > 0x40 )
  {
    sub_C43690((__int64)&v205, 0, 0);
    if ( v206 <= 0x40 )
    {
      v205 |= v39;
      LODWORD(v208) = v23;
      v147 = ~v39;
    }
    else
    {
      v147 = ~v39;
      *(_QWORD *)(v205 + 8LL * (v32 >> 6)) |= v39;
      LODWORD(v208) = v23;
    }
    v179 = v147;
    sub_C43690((__int64)&v207, -1, 1);
    v40 = v179;
    if ( (unsigned int)v208 > 0x40 )
    {
      *(_QWORD *)(v207 + 8LL * (v32 >> 6)) &= v179;
      goto LABEL_40;
    }
  }
  else
  {
    LODWORD(v208) = v23;
    v205 = 1LL << v32;
    v40 = ~v39;
    v41 = 0xFFFFFFFFFFFFFFFFLL >> (63 - (v32 & 0x3F));
    if ( !v23 )
      v41 = 0;
    v207 = v41;
  }
  v207 &= v40;
LABEL_40:
  *(_QWORD *)&v42 = sub_34007B0(a3, (__int64)&v205, (__int64)&v199, v201, v202, 0, v8, 0);
  v171 = v42;
  v43 = sub_34007B0(a3, (__int64)&v207, (__int64)&v199, v201, v202, 0, v8, 0);
  v45 = v44;
  v46 = v43;
  *(_QWORD *)&v176 = sub_3406EB0((_QWORD *)a3, 0xBCu, (__int64)&v199, v201, v202, v47, *(_OWORD *)&v8, v174);
  *((_QWORD *)&v176 + 1) = v48;
  *(_QWORD *)&v49 = sub_33ED040((_QWORD *)a3, 0x14u);
  v50 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v199, v162, v164, *((__int64 *)&v176 + 1), v176, v186, v49);
  *((_QWORD *)&v154 + 1) = v45;
  *(_QWORD *)&v154 = v46;
  v198 = sub_3288B20(a3, (int)&v199, v201, v202, v50, v51, v171, v154, 0);
  *((_QWORD *)&v151 + 1) = v52 | v192 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v151 = v198;
  v25 = sub_3288B20(a3, (int)&v199, v201, v202, v181, 1, v151, v194, 0);
  if ( (unsigned int)v208 > 0x40 && v207 )
    j_j___libc_free_0_0(v207);
LABEL_78:
  if ( v206 > 0x40 )
  {
    v82 = v205;
    if ( v205 )
LABEL_80:
      j_j___libc_free_0_0(v82);
  }
LABEL_59:
  if ( v199 )
    sub_B91220((__int64)&v199, v199);
  return v25;
}
