// Function: sub_2135330
// Address: 0x2135330
//
__int64 __fastcall sub_2135330(__int64 a1, __int64 a2)
{
  unsigned __int8 *v5; // rax
  __int64 v6; // rdi
  const __m128i *v7; // rax
  __m128i v8; // xmm0
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // ebx
  __m128i v13; // xmm1
  __int64 v14; // rax
  int v15; // eax
  char v16; // di
  __m128i v17; // xmm2
  __int64 v18; // rax
  char v19; // r8
  unsigned int v20; // eax
  bool v21; // zf
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rdx
  char v25; // di
  int v26; // edx
  int v27; // eax
  char v28; // r14
  int v29; // ebx
  int v30; // eax
  unsigned int v31; // ebx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  unsigned int v35; // eax
  __int64 v36; // r14
  __int64 v37; // rbx
  __int128 v38; // rax
  unsigned int v39; // edx
  _QWORD *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v43; // rsi
  int v44; // edx
  int v45; // edx
  __int64 v46; // r12
  unsigned int v47; // eax
  char v48; // di
  __int64 v49; // rax
  int v50; // r14d
  unsigned int v51; // ebx
  char v52; // r8
  unsigned int v53; // r11d
  unsigned int v54; // r11d
  int v55; // esi
  __int64 v56; // r14
  unsigned int v57; // ebx
  unsigned int v58; // esi
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rbx
  unsigned int v62; // eax
  __int64 v63; // rbx
  __int64 *v64; // rdi
  int v65; // edx
  __int128 v66; // rax
  __int64 *v67; // rax
  __int64 v68; // r11
  unsigned int v69; // edx
  __int64 v70; // r14
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rsi
  unsigned __int64 v77; // rdi
  int v78; // ecx
  int v79; // edx
  int v80; // eax
  __int64 v81; // rdx
  __int64 v82; // r14
  char v83; // r8
  __int64 v84; // rdx
  __int64 v85; // r11
  char v86; // r8
  __int64 v87; // rcx
  unsigned int v88; // eax
  __int64 v89; // rax
  unsigned int v90; // edx
  unsigned __int8 v91; // al
  unsigned int v92; // ebx
  int v93; // esi
  __int128 v94; // rax
  __int64 *v95; // rax
  __int64 *v96; // r14
  __int64 v97; // rdi
  int v98; // edx
  __int64 v99; // rax
  unsigned int v100; // edx
  unsigned __int8 v101; // al
  __int128 v102; // rax
  __int128 v103; // rax
  int v104; // edx
  int v105; // edx
  __int64 v106; // [rsp+8h] [rbp-1D8h]
  __int64 v107; // [rsp+20h] [rbp-1C0h]
  unsigned int v108; // [rsp+20h] [rbp-1C0h]
  __int64 v109; // [rsp+30h] [rbp-1B0h]
  __int64 v110; // [rsp+30h] [rbp-1B0h]
  __int64 v111; // [rsp+30h] [rbp-1B0h]
  __int64 v112; // [rsp+38h] [rbp-1A8h]
  unsigned int v113; // [rsp+40h] [rbp-1A0h]
  __int64 v114; // [rsp+40h] [rbp-1A0h]
  unsigned int v115; // [rsp+40h] [rbp-1A0h]
  __int64 v116; // [rsp+48h] [rbp-198h]
  __int64 v117; // [rsp+50h] [rbp-190h]
  __int64 v118; // [rsp+58h] [rbp-188h]
  __int16 v119; // [rsp+64h] [rbp-17Ch]
  __int8 v120; // [rsp+64h] [rbp-17Ch]
  unsigned int v121; // [rsp+68h] [rbp-178h]
  __int64 *v122; // [rsp+70h] [rbp-170h]
  __int64 v123; // [rsp+70h] [rbp-170h]
  unsigned __int64 v124; // [rsp+78h] [rbp-168h]
  unsigned __int64 v125; // [rsp+78h] [rbp-168h]
  __int64 *v126; // [rsp+B0h] [rbp-130h]
  __int64 v127; // [rsp+F0h] [rbp-F0h]
  __m128i v128; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v129; // [rsp+110h] [rbp-D0h] BYREF
  int v130; // [rsp+118h] [rbp-C8h]
  __int64 v131; // [rsp+120h] [rbp-C0h] BYREF
  unsigned __int64 v132; // [rsp+128h] [rbp-B8h]
  __int128 v133; // [rsp+130h] [rbp-B0h] BYREF
  char v134[8]; // [rsp+140h] [rbp-A0h] BYREF
  __int64 v135; // [rsp+148h] [rbp-98h]
  __m128i v136; // [rsp+150h] [rbp-90h] BYREF
  __int64 v137; // [rsp+160h] [rbp-80h]
  __int128 v138; // [rsp+170h] [rbp-70h] BYREF
  __int64 v139; // [rsp+180h] [rbp-60h]
  __int128 v140; // [rsp+190h] [rbp-50h] BYREF
  __int64 v141; // [rsp+1A0h] [rbp-40h]

  if ( *(_WORD *)(a2 + 24) == 186 && (*(_BYTE *)(a2 + 27) & 4) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 )
    return sub_2146690();
  v5 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  sub_1F40D10((__int64)&v140, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v5, *((_QWORD *)v5 + 1));
  v6 = *(_QWORD *)(a2 + 104);
  v128.m128i_i8[0] = BYTE8(v140);
  v128.m128i_i64[1] = v141;
  v7 = *(const __m128i **)(a2 + 32);
  v8 = _mm_loadu_si128(v7 + 5);
  v118 = v7->m128i_i64[0];
  v117 = v7->m128i_i64[1];
  v9 = sub_1E34390(v6);
  v10 = *(_QWORD *)(a2 + 72);
  v121 = v9;
  v11 = *(_QWORD *)(a2 + 104);
  v129 = v10;
  v12 = *(unsigned __int16 *)(v11 + 32);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(_QWORD *)(v11 + 56);
  v136 = v13;
  v137 = v14;
  if ( v10 )
    sub_1623A60((__int64)&v129, v10, 2);
  v15 = *(_DWORD *)(a2 + 64);
  v16 = *(_BYTE *)(a2 + 88);
  v131 = 0;
  v17 = _mm_loadu_si128(&v128);
  LODWORD(v132) = 0;
  v130 = v15;
  v18 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)&v133 = 0;
  DWORD2(v133) = 0;
  LOBYTE(v138) = v16;
  *((_QWORD *)&v138 + 1) = v18;
  v140 = (__int128)v17;
  if ( v16 == v128.m128i_i8[0] )
  {
    if ( v16 || v18 == *((_QWORD *)&v140 + 1) )
    {
      v119 = v12;
LABEL_30:
      sub_20174B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v131, &v133);
      v46 = sub_1D2C750(
              *(_QWORD **)(a1 + 8),
              v118,
              v117,
              (__int64)&v129,
              v131,
              v132,
              v8.m128i_i64[0],
              v8.m128i_i64[1],
              *(_OWORD *)*(_QWORD *)(a2 + 104),
              *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
              *(unsigned __int8 *)(a2 + 88),
              *(_QWORD *)(a2 + 96),
              v121,
              v119,
              (__int64)&v136);
      goto LABEL_31;
    }
  }
  else if ( v16 )
  {
    v113 = sub_2127930(v16);
    goto LABEL_10;
  }
  v120 = v128.m128i_i8[0];
  v47 = sub_1F58D40((__int64)&v138);
  v19 = v120;
  v113 = v47;
LABEL_10:
  if ( v19 )
    v20 = sub_2127930(v19);
  else
    v20 = sub_1F58D40((__int64)&v140);
  v119 = v12;
  if ( v20 >= v113 )
    goto LABEL_30;
  v21 = *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) == 0;
  v22 = *(_QWORD *)(a2 + 32);
  v23 = *(_QWORD *)(v22 + 40);
  v24 = *(_QWORD *)(v22 + 48);
  if ( !v21 )
  {
    sub_20174B0(a1, v23, v24, &v131, &v133);
    v48 = *(_BYTE *)(a2 + 88);
    v49 = *(_QWORD *)(a2 + 96);
    v134[0] = v48;
    v135 = v49;
    if ( v48 )
      v50 = sub_2127930(v48);
    else
      v50 = sub_1F58D40((__int64)v134);
    v51 = (unsigned int)(v50 + 7) >> 3;
    if ( v128.m128i_i8[0] )
    {
      v53 = sub_2127930(v128.m128i_i8[0]);
    }
    else
    {
      v88 = sub_1F58D40((__int64)&v128);
      v52 = 0;
      v53 = v88;
    }
    v54 = v53 >> 3;
    v55 = v50;
    v56 = *(_QWORD *)(a1 + 8);
    v57 = v51 - v54;
    v108 = v54;
    v58 = v55 - 8 * v57;
    v115 = 8 * v57;
    if ( v58 == 32 )
    {
      LOBYTE(v59) = 5;
    }
    else if ( v58 > 0x20 )
    {
      if ( v58 == 64 )
      {
        LOBYTE(v59) = 6;
      }
      else
      {
        if ( v58 != 128 )
        {
LABEL_61:
          v59 = sub_1F58CC0(*(_QWORD **)(v56 + 48), v58);
          v56 = *(_QWORD *)(a1 + 8);
          v52 = v128.m128i_i8[0];
          v109 = v59;
LABEL_46:
          v61 = v109;
          v112 = v60;
          LOBYTE(v61) = v59;
          v111 = v61;
          if ( v52 )
            v62 = sub_2127930(v52);
          else
            v62 = sub_1F58D40((__int64)&v128);
          if ( v115 < v62 )
          {
            v89 = sub_1E0A0C0(*(_QWORD *)(v56 + 32));
            v90 = 8 * sub_15A9520(v89, 0);
            if ( v90 == 32 )
            {
              v91 = 5;
            }
            else if ( v90 > 0x20 )
            {
              v91 = 6;
              if ( v90 != 64 )
              {
                v91 = 0;
                if ( v90 == 128 )
                  v91 = 7;
              }
            }
            else
            {
              v91 = 3;
              if ( v90 != 8 )
                v91 = 4 * (v90 == 16);
            }
            v92 = v91;
            if ( v128.m128i_i8[0] )
              v93 = sub_2127930(v128.m128i_i8[0]);
            else
              v93 = sub_1F58D40((__int64)&v128);
            *(_QWORD *)&v94 = sub_1D38BB0(
                                v56,
                                v93 - v115,
                                (__int64)&v129,
                                v92,
                                0,
                                0,
                                v8,
                                *(double *)v13.m128i_i64,
                                v17,
                                0);
            v95 = sub_1D332F0(
                    (__int64 *)v56,
                    122,
                    (__int64)&v129,
                    v128.m128i_u32[0],
                    (const void **)v128.m128i_i64[1],
                    0,
                    *(double *)v8.m128i_i64,
                    *(double *)v13.m128i_i64,
                    v17,
                    v133,
                    *((unsigned __int64 *)&v133 + 1),
                    v94);
            v96 = *(__int64 **)(a1 + 8);
            *(_QWORD *)&v133 = v95;
            v97 = v96[4];
            DWORD2(v133) = v98;
            v99 = sub_1E0A0C0(v97);
            v100 = 8 * sub_15A9520(v99, 0);
            if ( v100 == 32 )
            {
              v101 = 5;
            }
            else if ( v100 > 0x20 )
            {
              v101 = 6;
              if ( v100 != 64 )
              {
                v101 = 0;
                if ( v100 == 128 )
                  v101 = 7;
              }
            }
            else
            {
              v101 = 3;
              if ( v100 != 8 )
                v101 = 4 * (v100 == 16);
            }
            *(_QWORD *)&v102 = sub_1D38BB0(
                                 (__int64)v96,
                                 v115,
                                 (__int64)&v129,
                                 v101,
                                 0,
                                 0,
                                 v8,
                                 *(double *)v13.m128i_i64,
                                 v17,
                                 0);
            *(_QWORD *)&v103 = sub_1D332F0(
                                 v96,
                                 124,
                                 (__int64)&v129,
                                 v128.m128i_u32[0],
                                 (const void **)v128.m128i_i64[1],
                                 0,
                                 *(double *)v8.m128i_i64,
                                 *(double *)v13.m128i_i64,
                                 v17,
                                 v131,
                                 v132,
                                 v102);
            v126 = sub_1D332F0(
                     v96,
                     119,
                     (__int64)&v129,
                     v128.m128i_u32[0],
                     (const void **)v128.m128i_i64[1],
                     0,
                     *(double *)v8.m128i_i64,
                     *(double *)v13.m128i_i64,
                     v17,
                     v133,
                     *((unsigned __int64 *)&v133 + 1),
                     v103);
            v56 = *(_QWORD *)(a1 + 8);
            *(_QWORD *)&v133 = v126;
            DWORD2(v133) = v104;
          }
          *(_QWORD *)&v133 = sub_1D2C750(
                               (_QWORD *)v56,
                               v118,
                               v117,
                               (__int64)&v129,
                               v133,
                               *((__int64 *)&v133 + 1),
                               v8.m128i_i64[0],
                               v8.m128i_i64[1],
                               *(_OWORD *)*(_QWORD *)(a2 + 104),
                               *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                               v111,
                               v112,
                               v121,
                               v119,
                               (__int64)&v136);
          v63 = 16LL * v8.m128i_u32[2];
          v64 = *(__int64 **)(a1 + 8);
          DWORD2(v133) = v65;
          *(_QWORD *)&v66 = sub_1D38BB0(
                              (__int64)v64,
                              v108,
                              (__int64)&v129,
                              *(unsigned __int8 *)(v63 + *(_QWORD *)(v8.m128i_i64[0] + 40)),
                              *(const void ***)(v63 + *(_QWORD *)(v8.m128i_i64[0] + 40) + 8),
                              0,
                              v8,
                              *(double *)v13.m128i_i64,
                              v17,
                              0);
          v67 = sub_1D332F0(
                  v64,
                  52,
                  (__int64)&v129,
                  *(unsigned __int8 *)(*(_QWORD *)(v8.m128i_i64[0] + 40) + v63),
                  *(const void ***)(*(_QWORD *)(v8.m128i_i64[0] + 40) + v63 + 8),
                  3u,
                  *(double *)v8.m128i_i64,
                  *(double *)v13.m128i_i64,
                  v17,
                  v8.m128i_i64[0],
                  v8.m128i_u32[2],
                  v66);
          v68 = v108;
          v123 = (__int64)v67;
          v70 = *(_QWORD *)(a1 + 8);
          v125 = v69 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v71 = (v68 | v121) & -(v68 | v121);
          if ( v115 == 32 )
          {
            LOBYTE(v72) = 5;
            goto LABEL_53;
          }
          if ( v115 > 0x20 )
          {
            if ( v115 == 64 )
            {
              LOBYTE(v72) = 6;
              goto LABEL_53;
            }
            if ( v115 == 128 )
            {
              LOBYTE(v72) = 7;
              goto LABEL_53;
            }
          }
          else
          {
            if ( v115 == 8 )
            {
              LOBYTE(v72) = 3;
              goto LABEL_53;
            }
            LOBYTE(v72) = 4;
            if ( v115 == 16 )
            {
LABEL_53:
              v73 = 0;
LABEL_54:
              v76 = v106;
              v74 = v73;
              v75 = *(_QWORD *)(a2 + 104);
              LOBYTE(v76) = v72;
              v77 = *(_QWORD *)v75 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v77 )
              {
                v85 = *(_QWORD *)(v75 + 8) + v68;
                v86 = *(_BYTE *)(v75 + 16);
                if ( (*(_QWORD *)v75 & 4) != 0 )
                {
                  *((_QWORD *)&v140 + 1) = v85;
                  LOBYTE(v141) = v86;
                  *(_QWORD *)&v140 = v77 | 4;
                  HIDWORD(v141) = *(_DWORD *)(v77 + 12);
                }
                else
                {
                  *(_QWORD *)&v140 = *(_QWORD *)v75 & 0xFFFFFFFFFFFFFFF8LL;
                  *((_QWORD *)&v140 + 1) = v85;
                  LOBYTE(v141) = v86;
                  v87 = *(_QWORD *)v77;
                  if ( *(_BYTE *)(*(_QWORD *)v77 + 8LL) == 16 )
                    v87 = **(_QWORD **)(v87 + 16);
                  HIDWORD(v141) = *(_DWORD *)(v87 + 8) >> 8;
                }
              }
              else
              {
                v78 = *(_DWORD *)(v75 + 20);
                LODWORD(v141) = 0;
                v140 = 0u;
                HIDWORD(v141) = v78;
              }
              v131 = sub_1D2C750(
                       (_QWORD *)v70,
                       v118,
                       v117,
                       (__int64)&v129,
                       v131,
                       v132,
                       v123,
                       v125,
                       v140,
                       v141,
                       v76,
                       v74,
                       v71,
                       v119,
                       (__int64)&v136);
              LODWORD(v132) = v79;
              goto LABEL_27;
            }
          }
          v72 = sub_1F58CC0(*(_QWORD **)(v70 + 48), v115);
          v68 = v108;
          v106 = v72;
          v73 = v81;
          goto LABEL_54;
        }
        LOBYTE(v59) = 7;
      }
    }
    else if ( v58 == 8 )
    {
      LOBYTE(v59) = 3;
    }
    else
    {
      LOBYTE(v59) = 4;
      if ( v58 != 16 )
      {
        LOBYTE(v59) = 2;
        if ( v58 != 1 )
          goto LABEL_61;
      }
    }
    v60 = 0;
    goto LABEL_46;
  }
  sub_20174B0(a1, v23, v24, &v131, &v133);
  v127 = sub_1D2BF40(
           *(_QWORD **)(a1 + 8),
           v118,
           v117,
           (__int64)&v129,
           v131,
           v132,
           v8.m128i_i64[0],
           v8.m128i_i64[1],
           *(_OWORD *)*(_QWORD *)(a2 + 104),
           *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
           v121,
           v12,
           (__int64)&v136);
  v25 = *(_BYTE *)(a2 + 88);
  v131 = v127;
  LOBYTE(v140) = v25;
  LODWORD(v132) = v26;
  *((_QWORD *)&v140 + 1) = *(_QWORD *)(a2 + 96);
  if ( v25 )
  {
    v80 = sub_2127930(v25);
    v28 = v128.m128i_i8[0];
    v29 = v80;
    if ( v128.m128i_i8[0] )
      goto LABEL_16;
  }
  else
  {
    v27 = sub_1F58D40((__int64)&v140);
    v28 = v128.m128i_i8[0];
    v29 = v27;
    if ( v128.m128i_i8[0] )
    {
LABEL_16:
      v30 = sub_2127930(v28);
      goto LABEL_17;
    }
  }
  v30 = sub_1F58D40((__int64)&v128);
LABEL_17:
  v31 = v29 - v30;
  v110 = *(_QWORD *)(a1 + 8);
  if ( v31 == 32 )
  {
    LOBYTE(v32) = 5;
  }
  else if ( v31 > 0x20 )
  {
    if ( v31 == 64 )
    {
      LOBYTE(v32) = 6;
    }
    else
    {
      if ( v31 != 128 )
      {
LABEL_63:
        v32 = sub_1F58CC0(*(_QWORD **)(v110 + 48), v31);
        v28 = v128.m128i_i8[0];
        v107 = v32;
        v110 = *(_QWORD *)(a1 + 8);
        goto LABEL_22;
      }
      LOBYTE(v32) = 7;
    }
  }
  else if ( v31 == 8 )
  {
    LOBYTE(v32) = 3;
  }
  else
  {
    LOBYTE(v32) = 4;
    if ( v31 != 16 )
    {
      LOBYTE(v32) = 2;
      if ( v31 != 1 )
        goto LABEL_63;
    }
  }
  v33 = 0;
LABEL_22:
  v34 = v107;
  v116 = v33;
  LOBYTE(v34) = v32;
  v114 = v34;
  if ( v28 )
    v35 = sub_2127930(v28);
  else
    v35 = sub_1F58D40((__int64)&v128);
  v36 = v35 >> 3;
  v37 = 16LL * v8.m128i_u32[2];
  *(_QWORD *)&v38 = sub_1D38BB0(
                      v110,
                      v36,
                      (__int64)&v129,
                      *(unsigned __int8 *)(v37 + *(_QWORD *)(v8.m128i_i64[0] + 40)),
                      *(const void ***)(v37 + *(_QWORD *)(v8.m128i_i64[0] + 40) + 8),
                      0,
                      v8,
                      *(double *)v13.m128i_i64,
                      v17,
                      0);
  v122 = sub_1D332F0(
           (__int64 *)v110,
           52,
           (__int64)&v129,
           *(unsigned __int8 *)(*(_QWORD *)(v8.m128i_i64[0] + 40) + v37),
           *(const void ***)(*(_QWORD *)(v8.m128i_i64[0] + 40) + v37 + 8),
           3u,
           *(double *)v8.m128i_i64,
           *(double *)v13.m128i_i64,
           v17,
           v8.m128i_i64[0],
           v8.m128i_u32[2],
           v38);
  v40 = *(_QWORD **)(a1 + 8);
  v124 = v39 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v41 = -(v36 | v121) & (v36 | v121);
  v42 = *(_QWORD *)(a2 + 104);
  v43 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v43 )
  {
    v82 = *(_QWORD *)(v42 + 8) + v36;
    v83 = *(_BYTE *)(v42 + 16);
    if ( (*(_QWORD *)v42 & 4) != 0 )
    {
      v105 = *(_DWORD *)(v43 + 12);
      *((_QWORD *)&v138 + 1) = v82;
      LOBYTE(v139) = v83;
      *(_QWORD *)&v138 = v43 | 4;
      HIDWORD(v139) = v105;
    }
    else
    {
      v84 = *(_QWORD *)v43;
      *(_QWORD *)&v138 = v43;
      *((_QWORD *)&v138 + 1) = v82;
      v21 = *(_BYTE *)(v84 + 8) == 16;
      LOBYTE(v139) = v83;
      if ( v21 )
        v84 = **(_QWORD **)(v84 + 16);
      HIDWORD(v139) = *(_DWORD *)(v84 + 8) >> 8;
    }
  }
  else
  {
    v44 = *(_DWORD *)(v42 + 20);
    LODWORD(v139) = 0;
    v138 = 0u;
    HIDWORD(v139) = v44;
  }
  *(_QWORD *)&v133 = sub_1D2C750(
                       v40,
                       v118,
                       v117,
                       (__int64)&v129,
                       v133,
                       *((__int64 *)&v133 + 1),
                       (__int64)v122,
                       v124,
                       v138,
                       v139,
                       v114,
                       v116,
                       v41,
                       v119,
                       (__int64)&v136);
  DWORD2(v133) = v45;
LABEL_27:
  v46 = (__int64)sub_1D332F0(
                   *(__int64 **)(a1 + 8),
                   2,
                   (__int64)&v129,
                   1,
                   0,
                   0,
                   *(double *)v8.m128i_i64,
                   *(double *)v13.m128i_i64,
                   v17,
                   v131,
                   v132,
                   v133);
LABEL_31:
  if ( v129 )
    sub_161E7C0((__int64)&v129, v129);
  return v46;
}
