// Function: sub_32D1280
// Address: 0x32d1280
//
__int64 __fastcall sub_32D1280(const __m128i **a1, __int64 a2)
{
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // r14
  unsigned int v7; // ecx
  __m128i v8; // xmm1
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned __int16 *v11; // rdx
  __int64 v12; // r13
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // eax
  const __m128i *v18; // r10
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  const __m128i *v22; // r10
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  const __m128i *v26; // rdi
  __m128i v27; // xmm3
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v31; // r8
  __int64 v32; // rsi
  unsigned __int16 *v33; // rax
  __int64 v34; // r11
  unsigned int v35; // ecx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  int v40; // eax
  char v41; // r11
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int16 *v44; // rax
  __int64 v45; // r14
  __int64 v46; // r15
  unsigned int v47; // ebx
  __int64 v48; // rax
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int128 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rax
  unsigned int v54; // esi
  bool v55; // al
  __int64 v56; // r9
  __int64 v57; // rax
  bool v58; // al
  unsigned int v59; // edx
  __int64 v61; // rax
  unsigned __int64 v62; // rdi
  int v63; // ecx
  unsigned __int64 v64; // rax
  unsigned __int64 v66; // rdi
  unsigned __int64 v67; // rax
  unsigned int v68; // ebx
  unsigned __int64 v69; // r14
  int v70; // eax
  unsigned __int64 v71; // r14
  unsigned int v72; // ebx
  __int128 v73; // rax
  __int64 v74; // r9
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r15
  __int64 v78; // r14
  __int128 v79; // rax
  __int64 v80; // r9
  __int64 v81; // rax
  __int64 v82; // rax
  const __m128i *v83; // rdi
  __int64 v84; // rdx
  __int128 v85; // rax
  int v86; // r9d
  __int64 v87; // rax
  __int64 v88; // rax
  int v89; // edx
  __int64 v90; // rax
  int v91; // edx
  const __m128i *v92; // rdi
  __int64 v93; // rsi
  __m128i *v94; // rdi
  unsigned int v95; // eax
  bool v96; // zf
  __int64 v97; // rax
  __int128 v98; // rax
  __int64 v99; // r9
  int v100; // eax
  unsigned int v101; // eax
  bool v102; // al
  __int64 v103; // r9
  const __m128i *v104; // rbx
  __int128 v105; // rax
  __int64 v106; // r9
  unsigned __int64 v107; // rax
  __int8 v108; // cl
  unsigned int *v109; // rax
  __int64 v110; // rdx
  unsigned __int16 *v111; // rax
  unsigned int v112; // r14d
  const __m128i *v113; // rdi
  __int64 v114; // rax
  const __m128i *v115; // rdi
  __int64 v116; // rdx
  __int64 v117; // rbx
  __int64 v118; // rax
  __int64 v119; // r14
  __int64 v120; // rdx
  __int64 v121; // r8
  __int64 v122; // r9
  int v123; // r9d
  __int64 v124; // rdx
  __int64 v125; // r10
  unsigned __int64 v126; // rax
  __int64 v127; // rax
  const __m128i *v128; // r12
  __int128 v129; // rax
  int v130; // r9d
  __int64 v131; // rbx
  __int64 v132; // r9
  __int128 v133; // rax
  __int128 *v134; // r12
  const __m128i *v135; // r14
  __int64 v136; // r9
  __int128 v137; // rax
  __int64 v138; // r9
  __int64 v139; // rax
  __int64 v140; // r8
  __int64 v141; // r9
  __int64 v142; // r14
  unsigned int v143; // eax
  __int128 v144; // [rsp-20h] [rbp-2C0h]
  __int128 v145; // [rsp-10h] [rbp-2B0h]
  unsigned int v146; // [rsp+8h] [rbp-298h]
  __int64 v147; // [rsp+8h] [rbp-298h]
  __int64 v148; // [rsp+10h] [rbp-290h]
  __int64 v149; // [rsp+10h] [rbp-290h]
  __int64 v150; // [rsp+18h] [rbp-288h]
  char v151; // [rsp+18h] [rbp-288h]
  unsigned int v152; // [rsp+18h] [rbp-288h]
  char v153; // [rsp+18h] [rbp-288h]
  __int64 v154; // [rsp+20h] [rbp-280h]
  __int128 v155; // [rsp+20h] [rbp-280h]
  __int64 v156; // [rsp+30h] [rbp-270h]
  unsigned int v157; // [rsp+3Ch] [rbp-264h]
  __m128i v158; // [rsp+40h] [rbp-260h] BYREF
  __m128i v159; // [rsp+50h] [rbp-250h] BYREF
  __int128 v160; // [rsp+60h] [rbp-240h]
  unsigned __int64 **v161; // [rsp+70h] [rbp-230h]
  __m128i *v162; // [rsp+78h] [rbp-228h]
  __int64 v163; // [rsp+80h] [rbp-220h]
  __int64 v164; // [rsp+88h] [rbp-218h]
  __int64 v165; // [rsp+90h] [rbp-210h]
  __int64 v166; // [rsp+98h] [rbp-208h]
  unsigned __int64 v167; // [rsp+A8h] [rbp-1F8h] BYREF
  __int64 v168; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v169; // [rsp+B8h] [rbp-1E8h]
  __int64 v170; // [rsp+C0h] [rbp-1E0h] BYREF
  int v171; // [rsp+C8h] [rbp-1D8h]
  unsigned __int64 v172; // [rsp+D0h] [rbp-1D0h] BYREF
  unsigned int v173; // [rsp+D8h] [rbp-1C8h]
  __int64 v174; // [rsp+E0h] [rbp-1C0h] BYREF
  int v175; // [rsp+E8h] [rbp-1B8h]
  __int64 v176; // [rsp+F0h] [rbp-1B0h]
  __int64 v177; // [rsp+F8h] [rbp-1A8h]
  unsigned __int64 *v178; // [rsp+100h] [rbp-1A0h] BYREF
  unsigned int v179; // [rsp+108h] [rbp-198h]
  __int128 v180; // [rsp+110h] [rbp-190h]
  __m128i v181; // [rsp+120h] [rbp-180h] BYREF
  __int64 v182; // [rsp+130h] [rbp-170h]
  int v183; // [rsp+138h] [rbp-168h]
  __int64 v184; // [rsp+140h] [rbp-160h]
  int v185; // [rsp+148h] [rbp-158h]
  __int64 v186; // [rsp+150h] [rbp-150h]
  __m128i v187; // [rsp+160h] [rbp-140h] BYREF
  _OWORD v188[19]; // [rsp+170h] [rbp-130h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128((const __m128i *)v4);
  v6 = *(_QWORD *)v4;
  v7 = *(_DWORD *)(v4 + 8);
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v9 = *(_QWORD *)(v4 + 40);
  LODWORD(v4) = *(_DWORD *)(v4 + 48);
  v158 = v5;
  v10 = *(_QWORD *)(v6 + 48);
  v157 = v7;
  LODWORD(v161) = v4;
  v156 = v7;
  v11 = (unsigned __int16 *)(16LL * v7 + v10);
  v154 = 16LL * v7;
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  *(_QWORD *)&v160 = v9;
  v159 = v8;
  LOWORD(v168) = v13;
  v169 = v12;
  if ( !(_WORD)v13 )
  {
    v162 = (__m128i *)&v168;
    if ( !sub_30070B0((__int64)&v168) )
    {
      v187.m128i_i64[1] = v12;
      v187.m128i_i16[0] = 0;
      goto LABEL_9;
    }
    LOWORD(v13) = sub_3009970((__int64)&v168, v9, v37, v38, v39);
LABEL_8:
    v187.m128i_i16[0] = v13;
    v187.m128i_i64[1] = v14;
    if ( (_WORD)v13 )
      goto LABEL_4;
LABEL_9:
    v176 = sub_3007260((__int64)&v187);
    v177 = v15;
    goto LABEL_10;
  }
  if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
  {
    LOWORD(v13) = word_4456580[v13 - 1];
    v14 = 0;
    goto LABEL_8;
  }
  v187.m128i_i16[0] = v13;
  v187.m128i_i64[1] = v12;
LABEL_4:
  if ( (_WORD)v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
    BUG();
LABEL_10:
  v16 = *(_QWORD *)(a2 + 80);
  v170 = v16;
  if ( v16 )
    sub_B96E90((__int64)&v170, v16, 1);
  v17 = *(_DWORD *)(a2 + 72);
  v18 = *a1;
  v182 = 0;
  v19 = *(unsigned int *)(a2 + 24);
  v183 = 0;
  v171 = v17;
  v20 = (__int64)a1[1];
  v181.m128i_i64[0] = (__int64)v18;
  v162 = (__m128i *)v18;
  v181.m128i_i64[1] = v20;
  v184 = 0;
  v185 = 0;
  v186 = a2;
  v21 = sub_33CB160(v19);
  v22 = v162;
  v178 = (unsigned __int64 *)v21;
  if ( BYTE4(v21) )
  {
    v23 = *(_QWORD *)(v186 + 40) + 40LL * (unsigned int)v178;
    v182 = *(_QWORD *)v23;
    v183 = *(_DWORD *)(v23 + 8);
    v24 = *(unsigned int *)(v186 + 24);
  }
  else
  {
    v31 = v186;
    v24 = *(unsigned int *)(v186 + 24);
    if ( (_DWORD)v24 == 488 )
    {
      v32 = *(_QWORD *)(v186 + 80);
      v33 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v186 + 40) + 48LL)
                               + 16LL * *(unsigned int *)(*(_QWORD *)(v186 + 40) + 8LL));
      v34 = *((_QWORD *)v33 + 1);
      v35 = *v33;
      v187.m128i_i64[0] = v32;
      if ( v32 )
      {
        v146 = v35;
        v148 = v34;
        v150 = v186;
        sub_B96E90((__int64)&v187, v32, 1);
        v35 = v146;
        v34 = v148;
        v31 = v150;
        v22 = v162;
      }
      v187.m128i_i32[2] = *(_DWORD *)(v31 + 72);
      v165 = sub_34015B0(v22, &v187, v35, v34, 0, 0);
      v166 = v36;
      v182 = v165;
      v183 = v36;
      if ( v187.m128i_i64[0] )
        sub_B91220((__int64)&v187, v187.m128i_i64[0]);
      v24 = *(unsigned int *)(v186 + 24);
    }
  }
  v187.m128i_i64[0] = sub_33CB1F0(v24);
  if ( v187.m128i_i8[4] )
  {
    v25 = *(_QWORD *)(v186 + 40) + 40LL * v187.m128i_u32[0];
    v184 = *(_QWORD *)v25;
    v185 = *(_DWORD *)(v25 + 8);
  }
  v26 = *a1;
  if ( *(_DWORD *)(v6 + 24) == 51 || *(_DWORD *)(v160 + 24) == 51 )
  {
    v29 = sub_3400BD0((_DWORD)v26, 0, (unsigned int)&v170, v168, v169, 0, 0);
    goto LABEL_20;
  }
  v27 = _mm_load_si128(&v159);
  v187 = _mm_load_si128(&v158);
  v188[0] = v27;
  v28 = sub_3402EA0((_DWORD)v26, 58, (unsigned int)&v170, v168, v169, 0, (__int64)&v187, 2);
  if ( v28 )
  {
    v29 = v28;
    goto LABEL_20;
  }
  if ( (unsigned __int8)sub_33E2390(*a1, v158.m128i_i64[0], v158.m128i_i64[1], 1)
    && !(unsigned __int8)sub_33E2390(*a1, v159.m128i_i64[0], v159.m128i_i64[1], 1) )
  {
    v29 = sub_328FC10(&v181, 0x3Au, (int)&v170, v168, v169, v56, *(_OWORD *)&v159, *(_OWORD *)&v158);
    goto LABEL_20;
  }
  v173 = 1;
  v172 = 0;
  if ( (_WORD)v168 )
  {
    if ( (unsigned __int16)(v168 - 17) > 0xD3u )
    {
LABEL_36:
      v40 = *(_DWORD *)(v160 + 24);
      if ( v40 != 35 && v40 != 11 )
        goto LABEL_38;
      v57 = *(_QWORD *)(v160 + 96);
      v54 = *(_DWORD *)(v57 + 32);
      if ( v54 <= 0x40 )
      {
        v107 = *(_QWORD *)(v57 + 24);
        v173 = v54;
        v172 = v107;
      }
      else
      {
        sub_C43990((__int64)&v172, v57 + 24);
        v54 = v173;
      }
      LOBYTE(v162) = (*(_BYTE *)(v160 + 32) & 8) != 0;
      goto LABEL_52;
    }
  }
  else if ( !sub_30070B0((__int64)&v168) )
  {
    goto LABEL_36;
  }
  if ( !(unsigned __int8)sub_33D1410(v160, &v172) )
  {
LABEL_38:
    if ( !(unsigned __int8)sub_326A930(v159.m128i_i64[0], v159.m128i_u32[2], 1u) )
      goto LABEL_90;
    LOBYTE(v162) = 0;
    v41 = 0;
    goto LABEL_40;
  }
  LOBYTE(v162) = 0;
  v54 = v173;
LABEL_52:
  if ( v54 <= 0x40 )
    v55 = v172 == 0;
  else
    v55 = v54 == (unsigned int)sub_C444A0((__int64)&v172);
  if ( v55 )
  {
    v29 = v159.m128i_i64[0];
    goto LABEL_46;
  }
  if ( v54 <= 0x40 )
    v58 = v172 == 1;
  else
    v58 = v54 - 1 == (unsigned int)sub_C444A0((__int64)&v172);
  if ( v58 )
  {
    v29 = v158.m128i_i64[0];
    goto LABEL_46;
  }
  if ( !v54 )
    goto LABEL_129;
  if ( v54 > 0x40 )
  {
    if ( v54 != (unsigned int)sub_C445E0((__int64)&v172) )
      goto LABEL_68;
LABEL_129:
    *(_QWORD *)&v98 = sub_3400BD0((unsigned int)*a1, 0, (unsigned int)&v170, v168, v169, 0, 0);
    v81 = sub_328FC10(&v181, 0x39u, (int)&v170, v168, v169, v99, v98, *(_OWORD *)&v158);
LABEL_89:
    v54 = v173;
    v29 = v81;
    goto LABEL_46;
  }
  if ( v172 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v54) )
    goto LABEL_129;
LABEL_68:
  v41 = sub_326A930(v159.m128i_i64[0], v159.m128i_u32[2], 1u);
  if ( !v41 )
  {
    v41 = 1;
    goto LABEL_70;
  }
LABEL_40:
  if ( (_WORD)v168 )
  {
    if ( (unsigned __int16)(v168 - 17) > 0xD3u )
      goto LABEL_43;
  }
  else
  {
    v153 = v41;
    v102 = sub_30070B0((__int64)&v168);
    v41 = v153;
    if ( !v102 )
      goto LABEL_43;
  }
  if ( *((int *)a1 + 6) > 2 )
  {
LABEL_70:
    if ( !(_BYTE)v162 && v41 )
    {
      v59 = v173;
      _RSI = v172;
      v61 = 1LL << ((unsigned __int8)v173 - 1);
      if ( v173 > 0x40 )
      {
        if ( (*(_QWORD *)(v172 + 8LL * ((v173 - 1) >> 6)) & v61) != 0 )
        {
          v152 = v173;
          LODWORD(v162) = sub_C44500((__int64)&v172);
          v100 = sub_C44590((__int64)&v172);
          v101 = (_DWORD)v162 + v100;
          if ( v101 == v152 )
          {
            v179 = v101;
            sub_C43780((__int64)&v178, (const void **)&v172);
            v59 = v179;
            if ( v179 > 0x40 )
            {
              sub_C43D10((__int64)&v178);
LABEL_84:
              sub_C46250((__int64)&v178);
              v68 = v179;
              v69 = (unsigned __int64)v178;
              v179 = 0;
              v187.m128i_i32[2] = v68;
              v187.m128i_i64[0] = (__int64)v178;
              if ( v68 > 0x40 )
              {
                v72 = v68 - 1 - sub_C444A0((__int64)&v187);
                if ( v69 )
                {
                  j_j___libc_free_0_0(v69);
                  if ( v179 > 0x40 )
                  {
                    if ( v178 )
                      j_j___libc_free_0_0((unsigned __int64)v178);
                  }
                }
              }
              else
              {
                v70 = 64;
                if ( v178 )
                {
                  _BitScanReverse64(&v71, (unsigned __int64)v178);
                  v70 = v71 ^ 0x3F;
                }
                v72 = 63 - v70;
              }
              *(_QWORD *)&v73 = sub_3400E40(*a1, v72, (unsigned int)v168, v169, &v170);
              v75 = sub_328FC10(&v181, 0xBEu, (int)&v170, v168, v169, v74, *(_OWORD *)&v158, v73);
              v77 = v76;
              v78 = v75;
              *(_QWORD *)&v79 = sub_3400BD0((unsigned int)*a1, 0, (unsigned int)&v170, v168, v169, 0, 0);
              *((_QWORD *)&v144 + 1) = v77;
              *(_QWORD *)&v144 = v78;
              v81 = sub_328FC10(&v181, 0x39u, (int)&v170, v168, v169, v80, v79, v144);
              goto LABEL_89;
            }
            v62 = (unsigned __int64)v178;
LABEL_81:
            v66 = ~v62;
            v67 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v59;
            if ( !v59 )
              v67 = 0;
            v178 = (unsigned __int64 *)(v67 & v66);
            goto LABEL_84;
          }
        }
      }
      else
      {
        v62 = v172;
        if ( (v61 & v172) != 0 )
        {
          if ( !v173 )
            goto LABEL_80;
          v63 = 64;
          if ( v172 << (64 - (unsigned __int8)v173) != -1 )
          {
            _BitScanReverse64(&v64, ~(v172 << (64 - (unsigned __int8)v173)));
            v63 = v64 ^ 0x3F;
          }
          __asm { tzcnt   rsi, rsi }
          if ( (unsigned int)_RSI > v173 )
            LODWORD(_RSI) = v173;
          if ( v173 == (_DWORD)_RSI + v63 )
          {
LABEL_80:
            v179 = v173;
            goto LABEL_81;
          }
        }
      }
    }
LABEL_90:
    v162 = &v181;
    if ( (unsigned __int8)sub_325F380((__int64)&v181, v6, 190) )
    {
      v82 = *(_QWORD *)(v6 + 40);
      v83 = *a1;
      v84 = *(_QWORD *)(v82 + 40);
      LODWORD(v82) = *(_DWORD *)(v82 + 48);
      v187 = _mm_load_si128(&v159);
      *(_QWORD *)&v188[0] = v84;
      DWORD2(v188[0]) = v82;
      *(_QWORD *)&v85 = sub_3402EA0((_DWORD)v83, 190, (unsigned int)&v170, v168, v169, 0, (__int64)&v187, 2);
      if ( (_QWORD)v85 )
      {
        v87 = sub_3406EB0(
                (unsigned int)*a1,
                58,
                (unsigned int)&v170,
                v168,
                v169,
                v86,
                *(_OWORD *)*(_QWORD *)(v6 + 40),
                v85);
        v54 = v173;
        v29 = v87;
        goto LABEL_46;
      }
    }
    if ( (unsigned __int8)sub_325F380((__int64)v162, v6, 190) )
    {
      v88 = *(_QWORD *)(v6 + 56);
      if ( v88 )
      {
        v89 = 1;
        do
        {
          if ( *(_DWORD *)(v88 + 8) == v157 )
          {
            if ( !v89 )
              goto LABEL_104;
            v88 = *(_QWORD *)(v88 + 32);
            if ( !v88 )
              goto LABEL_136;
            if ( *(_DWORD *)(v88 + 8) == v157 )
              goto LABEL_104;
            v89 = 0;
          }
          v88 = *(_QWORD *)(v88 + 32);
        }
        while ( v88 );
        if ( v89 == 1 )
          goto LABEL_104;
LABEL_136:
        if ( (unsigned __int8)sub_326A930(
                                *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL),
                                0) )
        {
          *(_QWORD *)&v155 = v160;
          *((_QWORD *)&v155 + 1) = (unsigned int)v161;
LABEL_138:
          v104 = v162;
          *(_QWORD *)&v105 = sub_328FC10(
                               v162,
                               0x3Au,
                               (int)&v170,
                               v168,
                               v169,
                               v103,
                               *(_OWORD *)*(_QWORD *)(v6 + 40),
                               v155);
          v53 = sub_328FC10(v104, 0xBEu, (int)&v170, v168, v169, v106, v105, *(_OWORD *)(*(_QWORD *)(v6 + 40) + 40LL));
          goto LABEL_45;
        }
      }
    }
LABEL_104:
    if ( (unsigned __int8)sub_325F380((__int64)v162, v160, 190) )
    {
      v90 = *(_QWORD *)(v160 + 56);
      if ( v90 )
      {
        v91 = 1;
        do
        {
          if ( (_DWORD)v161 == *(_DWORD *)(v90 + 8) )
          {
            if ( !v91 )
              goto LABEL_114;
            v90 = *(_QWORD *)(v90 + 32);
            if ( !v90 )
              goto LABEL_140;
            if ( *(_DWORD *)(v90 + 8) == (_DWORD)v161 )
              goto LABEL_114;
            v91 = 0;
          }
          v90 = *(_QWORD *)(v90 + 32);
        }
        while ( v90 );
        if ( v91 == 1 )
          goto LABEL_114;
LABEL_140:
        if ( (unsigned __int8)sub_326A930(
                                *(_QWORD *)(*(_QWORD *)(v160 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v160 + 40) + 48LL),
                                0) )
        {
          *(_QWORD *)&v155 = v6;
          v6 = v160;
          *((_QWORD *)&v155 + 1) = v156;
          goto LABEL_138;
        }
      }
    }
LABEL_114:
    if ( (unsigned __int8)sub_325F380((__int64)v162, v6, 56) )
    {
      v92 = *a1;
      v161 = (unsigned __int64 **)v159.m128i_i64[1];
      if ( (unsigned __int8)sub_33E2390(v92, v159.m128i_i64[0], v159.m128i_i64[1], 1) )
      {
        if ( (unsigned __int8)sub_33E2390(
                                *a1,
                                *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL),
                                1)
          && (unsigned __int8)sub_3260FE0(
                                a1,
                                a2,
                                v158.m128i_i64[0],
                                v158.m128i_i64[1],
                                v159.m128i_i64[0],
                                v159.m128i_i64[1]) )
        {
          v131 = *(_QWORD *)(v6 + 40);
          sub_3285E70((__int64)&v187, v159.m128i_i64[0]);
          *(_QWORD *)&v133 = sub_328FC10(
                               v162,
                               0x3Au,
                               (int)&v187,
                               v168,
                               v169,
                               v132,
                               *(_OWORD *)(v131 + 40),
                               *(_OWORD *)&v159);
          v134 = *(__int128 **)(v6 + 40);
          v160 = v133;
          sub_3285E70((__int64)&v178, v158.m128i_i64[0]);
          v135 = v162;
          *(_QWORD *)&v137 = sub_328FC10(v162, 0x3Au, (int)&v178, v168, v169, v136, *v134, *(_OWORD *)&v159);
          v29 = sub_328FC10(v135, 0x38u, (int)&v170, v168, v169, v138, v137, v160);
          sub_9C6650(&v178);
          sub_9C6650(&v187);
          v54 = v173;
          goto LABEL_46;
        }
      }
    }
    v93 = v159.m128i_i64[1];
    sub_33DFBC0(v159.m128i_i64[0], v159.m128i_i64[1], 0, 0);
    v175 = 1;
    v174 = 0;
    if ( (_WORD)v168 )
    {
      v94 = (__m128i *)&v168;
      if ( (unsigned __int16)(v168 - 17) > 0x9Eu )
        goto LABEL_123;
    }
    else
    {
      v162 = (__m128i *)&v168;
      if ( !sub_30070D0((__int64)&v168) )
        goto LABEL_123;
      v94 = v162;
    }
    v95 = sub_3281500(v94, v93);
    v167 = 1;
    LODWORD(v162) = v95;
    if ( v95 > 0x39 )
    {
      v139 = sub_22077B0(0x48u);
      v142 = v139;
      if ( v139 )
      {
        *(_DWORD *)(v139 + 64) = 0;
        *(_QWORD *)v139 = v139 + 16;
        *(_QWORD *)(v139 + 8) = 0x600000000LL;
      }
      v143 = (unsigned int)((_DWORD)v162 + 63) >> 6;
      if ( *(_DWORD *)(v142 + 12) < v143 )
        sub_C8D5F0(v142, (const void *)(v142 + 16), v143, 8u, v140, v141);
      v167 = v142;
    }
    if ( !*((_BYTE *)a1 + 33) || (unsigned __int8)sub_328A020((__int64)a1[1], 0xBAu, v168, v169, 0) )
    {
      *(_QWORD *)&v188[0] = 0;
      v161 = (unsigned __int64 **)&v167;
      v178 = &v167;
      *((_QWORD *)&v180 + 1) = sub_328F8B0;
      *(_QWORD *)&v180 = sub_325DDA0;
      sub_325DDA0(&v187, &v178, 2);
      v188[0] = v180;
      v108 = sub_33CA8D0(v159.m128i_i64[0], v159.m128i_i64[1], &v187);
      if ( *(_QWORD *)&v188[0] )
      {
        v159.m128i_i8[0] = v108;
        (*(void (__fastcall **)(__m128i *, __m128i *, __int64))&v188[0])(&v187, &v187, 3);
        v108 = v159.m128i_i8[0];
      }
      if ( v108 )
      {
        sub_A17130((__int64)&v178);
        v109 = *(unsigned int **)(v160 + 40);
        v110 = *(_QWORD *)v109;
        v111 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v109 + 48LL) + 16LL * v109[2]);
        v112 = *v111;
        v113 = *a1;
        *(_QWORD *)&v160 = *((_QWORD *)v111 + 1);
        v114 = sub_3400BD0((_DWORD)v113, 0, (unsigned int)&v170, v112, v160, 0, 0, v110);
        v115 = *a1;
        v159.m128i_i64[0] = v116;
        v117 = v114;
        v118 = sub_34015B0(v115, &v170, v112, v160, 0, 0);
        v119 = (unsigned int)v162;
        *(_QWORD *)&v160 = v188;
        v187.m128i_i64[0] = (__int64)v188;
        v187.m128i_i64[1] = 0x1000000000LL;
        sub_32982C0((__int64)&v187, (unsigned int)v162, v118, v120, v121, v122);
        v124 = 0;
        v125 = v159.m128i_i64[0];
        if ( (_DWORD)v162 )
        {
          do
          {
            if ( (v167 & 1) != 0 )
              v126 = (((v167 >> 1) & ~(-1LL << (v167 >> 58))) >> v124) & 1;
            else
              v126 = (*(_QWORD *)(*(_QWORD *)v167 + 8LL * ((unsigned int)v124 >> 6)) >> v124) & 1LL;
            if ( (_BYTE)v126 )
            {
              v164 = v125;
              v127 = v187.m128i_i64[0] + 16 * v124;
              v163 = v117;
              *(_QWORD *)v127 = v117;
              *(_DWORD *)(v127 + 8) = v164;
            }
            ++v124;
          }
          while ( v124 != v119 );
        }
        v128 = *a1;
        *((_QWORD *)&v145 + 1) = v187.m128i_u32[2];
        *(_QWORD *)&v145 = v187.m128i_i64[0];
        *(_QWORD *)&v129 = sub_33FC220((_DWORD)v128, 156, (unsigned int)&v170, v168, v169, v123, v145);
        v29 = sub_3406EB0((_DWORD)v128, 186, (unsigned int)&v170, v168, v169, v130, *(_OWORD *)&v158, v129);
        if ( v187.m128i_i64[0] != (_QWORD)v160 )
          _libc_free(v187.m128i_u64[0]);
        sub_228BF40(v161);
        goto LABEL_126;
      }
      sub_A17130((__int64)&v178);
    }
    else
    {
      v161 = (unsigned __int64 **)&v167;
    }
    sub_228BF40(v161);
LABEL_123:
    v96 = (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) == 0;
    v97 = 0;
    if ( !v96 )
      v97 = a2;
    v29 = v97;
LABEL_126:
    sub_969240(&v174);
    v54 = v173;
    goto LABEL_46;
  }
LABEL_43:
  v151 = v41;
  v42 = sub_3289780((__int64 *)a1, v159.m128i_i64[0], v159.m128i_i64[1], (__int64)&v170, 0, 0, *(_OWORD *)&v187, 0);
  v41 = v151;
  v149 = v43;
  v147 = v42;
  if ( !v42 )
    goto LABEL_70;
  v44 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + v154);
  v45 = (__int64)a1[1];
  v46 = *((_QWORD *)v44 + 1);
  v47 = *v44;
  v48 = sub_2E79000((__int64 *)(*a1)[2].m128i_i64[1]);
  v49 = sub_2FE6750(v45, v47, v46, v48);
  *(_QWORD *)&v51 = sub_33FB310(*a1, v147, v149, &v170, v49, v50);
  v53 = sub_328FC10(&v181, 0xBEu, (int)&v170, v168, v169, v52, *(_OWORD *)&v158, v51);
LABEL_45:
  v54 = v173;
  v29 = v53;
LABEL_46:
  if ( v54 > 0x40 && v172 )
    j_j___libc_free_0_0(v172);
LABEL_20:
  if ( v170 )
    sub_B91220((__int64)&v170, v170);
  return v29;
}
