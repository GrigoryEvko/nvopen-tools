// Function: sub_3460140
// Address: 0x3460140
//
__int64 __fastcall sub_3460140(__int64 a1, __m128i a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // ebx
  __int64 v7; // rsi
  unsigned __int16 v8; // r13
  __int64 v9; // rdx
  __int128 v10; // xmm2
  __m128i v11; // xmm3
  unsigned __int16 *v12; // rax
  unsigned __int16 v13; // r14
  __int64 v14; // rax
  bool v15; // al
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int128 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // r13d
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // r14
  unsigned __int16 v28; // ax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int8 v31; // al
  __int64 v32; // rdx
  __int8 v33; // cl
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // r9
  unsigned __int64 v39; // rax
  unsigned __int8 *v40; // r14
  unsigned __int64 v41; // r15
  int v42; // ecx
  __int64 v43; // rdx
  __m128i *v44; // rax
  unsigned __int64 v45; // r8
  unsigned int v46; // edx
  __m128i *v47; // r10
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __m128i **v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // r8
  unsigned __int64 v53; // rdx
  __m128i **v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __m128i v57; // xmm0
  __int16 v58; // si
  __int16 v59; // bx
  unsigned __int64 v60; // rdi
  __int64 v61; // rdx
  char v62; // r8
  __int64 v63; // rdx
  _OWORD *v64; // rdx
  unsigned __int8 *v65; // r14
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // r9
  unsigned __int8 *v69; // rax
  _OWORD *v70; // rdi
  int v71; // edx
  _OWORD *v72; // rdi
  __int16 v74; // ax
  __int64 v75; // rdx
  unsigned int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rdx
  unsigned int v82; // eax
  unsigned int v83; // r15d
  unsigned __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int16 v87; // dx
  __int64 v88; // r9
  __int64 v89; // rdx
  unsigned int v90; // r15d
  __int64 v91; // rax
  __int64 v92; // rcx
  _BYTE *v93; // rsi
  __int64 v94; // rdx
  __int64 v95; // rax
  _BYTE *v96; // rdx
  __int64 v97; // rax
  __int128 v98; // rax
  __int64 v99; // r9
  __int128 v100; // rax
  __int64 v101; // r9
  int v102; // r9d
  unsigned __int64 v103; // rdx
  __int64 v104; // r8
  unsigned __int8 *v105; // r10
  unsigned __int64 v106; // r11
  __int64 v107; // rax
  unsigned __int64 v108; // rdx
  unsigned __int8 **v109; // rax
  __int64 v110; // rax
  _BYTE *v111; // rax
  unsigned int v112; // eax
  int v113; // r9d
  unsigned int v114; // edx
  _OWORD *v115; // rdx
  unsigned __int8 *v116; // rax
  int v117; // edx
  unsigned __int16 v118; // ax
  unsigned __int16 v119; // ax
  __int64 v120; // rdx
  __int128 v121; // [rsp-20h] [rbp-320h]
  __int128 v122; // [rsp-10h] [rbp-310h]
  __int128 v123; // [rsp-10h] [rbp-310h]
  unsigned __int64 v124; // [rsp+0h] [rbp-300h]
  unsigned __int64 v125; // [rsp+0h] [rbp-300h]
  __int64 v127; // [rsp+28h] [rbp-2D8h]
  __int64 v128; // [rsp+38h] [rbp-2C8h]
  int v129; // [rsp+38h] [rbp-2C8h]
  __int64 v130; // [rsp+50h] [rbp-2B0h]
  unsigned int v131; // [rsp+58h] [rbp-2A8h]
  __m128i *v132; // [rsp+60h] [rbp-2A0h]
  __m128i *v133; // [rsp+60h] [rbp-2A0h]
  __int64 v135; // [rsp+78h] [rbp-288h]
  unsigned __int64 v136; // [rsp+80h] [rbp-280h]
  __int128 v137; // [rsp+80h] [rbp-280h]
  __int64 v138; // [rsp+90h] [rbp-270h]
  unsigned int v139; // [rsp+90h] [rbp-270h]
  __int128 v140; // [rsp+90h] [rbp-270h]
  int v141; // [rsp+A4h] [rbp-25Ch]
  int v142; // [rsp+A8h] [rbp-258h]
  __int64 v143; // [rsp+A8h] [rbp-258h]
  int v144; // [rsp+B0h] [rbp-250h]
  unsigned int v145; // [rsp+B0h] [rbp-250h]
  unsigned __int8 *v146; // [rsp+B0h] [rbp-250h]
  __int64 v147; // [rsp+B8h] [rbp-248h]
  unsigned __int64 v148; // [rsp+B8h] [rbp-248h]
  unsigned __int64 v149; // [rsp+B8h] [rbp-248h]
  __int64 v150; // [rsp+E0h] [rbp-220h] BYREF
  int v151; // [rsp+E8h] [rbp-218h]
  unsigned __int16 v152; // [rsp+F0h] [rbp-210h] BYREF
  __int64 v153; // [rsp+F8h] [rbp-208h]
  __int64 v154; // [rsp+100h] [rbp-200h] BYREF
  __int64 v155; // [rsp+108h] [rbp-1F8h]
  __int64 v156; // [rsp+110h] [rbp-1F0h] BYREF
  __int64 v157; // [rsp+118h] [rbp-1E8h]
  __int64 v158; // [rsp+120h] [rbp-1E0h]
  __int64 v159; // [rsp+128h] [rbp-1D8h]
  __int64 v160; // [rsp+130h] [rbp-1D0h]
  __int64 v161; // [rsp+138h] [rbp-1C8h]
  __int64 v162; // [rsp+140h] [rbp-1C0h]
  __int64 v163; // [rsp+148h] [rbp-1B8h]
  __int64 v164; // [rsp+150h] [rbp-1B0h]
  __int64 v165; // [rsp+158h] [rbp-1A8h]
  __int64 v166; // [rsp+160h] [rbp-1A0h]
  __int64 v167; // [rsp+168h] [rbp-198h]
  __int128 v168; // [rsp+170h] [rbp-190h] BYREF
  __int64 v169; // [rsp+180h] [rbp-180h]
  _OWORD v170[2]; // [rsp+190h] [rbp-170h] BYREF
  _BYTE *v171; // [rsp+1B0h] [rbp-150h] BYREF
  __int64 v172; // [rsp+1B8h] [rbp-148h]
  _BYTE v173[128]; // [rsp+1C0h] [rbp-140h] BYREF
  __m128i v174; // [rsp+240h] [rbp-C0h] BYREF
  _OWORD v175[11]; // [rsp+250h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a4 + 80);
  v150 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v150, v7, 1);
  v8 = *(_WORD *)(a4 + 96);
  v9 = *(_QWORD *)(a4 + 104);
  v151 = *(_DWORD *)(a4 + 72);
  v10 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a4 + 40));
  v11 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 40LL));
  v153 = v9;
  v12 = *(unsigned __int16 **)(a4 + 48);
  v152 = v8;
  v13 = *v12;
  v155 = *((_QWORD *)v12 + 1);
  LOBYTE(v12) = *(_BYTE *)(a4 + 33);
  LOWORD(v154) = v13;
  v141 = ((unsigned __int8)v12 >> 2) & 3;
  if ( v8 )
  {
    if ( (unsigned __int16)(v8 - 176) > 0x34u )
    {
      v14 = v8 - 1;
      v142 = word_4456340[v14];
      if ( (unsigned __int16)(v8 - 17) <= 0xD3u )
      {
        v8 = word_4456580[v14];
        v9 = 0;
      }
      goto LABEL_10;
    }
LABEL_128:
    sub_C64ED0("Cannot scalarize scalable vector loads", 1u);
  }
  v143 = v9;
  if ( sub_3007100((__int64)&v152) )
    goto LABEL_128;
  v138 = v143;
  v142 = sub_3007130((__int64)&v152, a4);
  v15 = sub_30070B0((__int64)&v152);
  v9 = v138;
  if ( v15 )
  {
    v118 = sub_3009970((__int64)&v152, a4, v138, v16, v17);
    v13 = v154;
    v8 = v118;
  }
LABEL_10:
  LOWORD(v156) = v8;
  v157 = v9;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
      goto LABEL_12;
    v130 = 0;
    v13 = word_4456580[v13 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v154) )
    {
LABEL_12:
      v130 = v155;
      goto LABEL_13;
    }
    v119 = sub_3009970((__int64)&v154, a4, v18, v19, v20);
    v8 = v156;
    v130 = v120;
    v13 = v119;
  }
LABEL_13:
  v131 = v13;
  if ( v8 )
  {
    if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
      goto LABEL_129;
    v32 = *(_QWORD *)&byte_444C4A0[16 * v8 - 16];
    v33 = byte_444C4A0[16 * v8 - 8];
    if ( v32 && (v32 & 7) == 0 )
    {
      v34 = *(_QWORD *)&byte_444C4A0[16 * v8 - 16];
      goto LABEL_38;
    }
LABEL_15:
    if ( v152 )
    {
      if ( v152 == 1 || (unsigned __int16)(v152 - 504) <= 7u )
        goto LABEL_129;
      *(_QWORD *)&v22 = *(_QWORD *)&byte_444C4A0[16 * v152 - 16];
      BYTE8(v22) = byte_444C4A0[16 * v152 - 8];
    }
    else
    {
      *(_QWORD *)&v22 = sub_3007260((__int64)&v152);
      v170[0] = v22;
    }
    v174.m128i_i8[8] = BYTE8(v22);
    v174.m128i_i64[0] = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v23 = sub_CA1930(&v174);
    v24 = v23;
    switch ( v23 )
    {
      case 1u:
        v74 = 2;
        break;
      case 2u:
        v74 = 3;
        break;
      case 4u:
        v74 = 4;
        break;
      case 8u:
        v74 = 5;
        break;
      case 0x10u:
        v74 = 6;
        break;
      case 0x20u:
        v74 = 7;
        break;
      case 0x40u:
        v74 = 8;
        break;
      case 0x80u:
        v74 = 9;
        break;
      default:
        v25 = sub_3007020(*(_QWORD **)(a5 + 64), v23);
        v27 = v26;
        v5 = v25;
        v28 = v152;
        if ( v152 )
          goto LABEL_26;
        goto LABEL_62;
    }
    LOWORD(v5) = v74;
    v28 = v152;
    v27 = 0;
    if ( v152 )
    {
LABEL_26:
      if ( v28 == 1 || (unsigned __int16)(v28 - 504) <= 7u )
        goto LABEL_129;
      v29 = 16LL * (v28 - 1);
      v30 = *(_QWORD *)&byte_444C4A0[v29];
      v31 = byte_444C4A0[v29 + 8];
LABEL_63:
      v174.m128i_i64[0] = v30;
      v174.m128i_i8[8] = v31;
      v76 = sub_CA1930(&v174);
      switch ( v76 )
      {
        case 1u:
          LOWORD(v77) = 2;
          break;
        case 2u:
          LOWORD(v77) = 3;
          break;
        case 4u:
          LOWORD(v77) = 4;
          break;
        case 8u:
          LOWORD(v77) = 5;
          break;
        case 0x10u:
          LOWORD(v77) = 6;
          break;
        case 0x20u:
          LOWORD(v77) = 7;
          break;
        case 0x40u:
          LOWORD(v77) = 8;
          break;
        case 0x80u:
          LOWORD(v77) = 9;
          break;
        default:
          v77 = sub_3007020(*(_QWORD **)(a5 + 64), v76);
          v128 = v77;
LABEL_75:
          v79 = v128;
          v147 = v78;
          LOWORD(v79) = v77;
          if ( !(_WORD)v156 )
          {
            v80 = sub_3007260((__int64)&v156);
            v162 = v80;
            v163 = v81;
            goto LABEL_77;
          }
          if ( (_WORD)v156 != 1 && (unsigned __int16)(v156 - 504) > 7u )
          {
            v81 = 16LL * ((unsigned __int16)v156 - 1);
            v80 = *(_QWORD *)&byte_444C4A0[v81];
            LOBYTE(v81) = byte_444C4A0[v81 + 8];
LABEL_77:
            v174.m128i_i64[0] = v80;
            v174.m128i_i8[8] = v81;
            v82 = sub_CA1930(&v174);
            v174.m128i_i32[2] = v24;
            v83 = v82;
            if ( v24 > 0x40 )
              sub_C43690((__int64)&v174, 0, 0);
            else
              v174.m128i_i64[0] = 0;
            if ( v83 )
            {
              if ( v83 > 0x40 )
              {
                sub_C43C90(&v174, 0, v83);
              }
              else
              {
                v84 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v83);
                if ( v174.m128i_i32[2] > 0x40u )
                  *(_QWORD *)v174.m128i_i64[0] |= v84;
                else
                  v174.m128i_i64[0] |= v84;
              }
            }
            *(_QWORD *)&v137 = sub_34007B0(a5, (__int64)&v174, (__int64)&v150, v5, v27, 0, a2, 0);
            *((_QWORD *)&v137 + 1) = v85;
            if ( v174.m128i_i32[2] > 0x40u && v174.m128i_i64[0] )
              j_j___libc_free_0_0(v174.m128i_u64[0]);
            v86 = *(_QWORD *)(a4 + 112);
            v174 = _mm_loadu_si128((const __m128i *)(v86 + 40));
            v175[0] = _mm_loadu_si128((const __m128i *)(v86 + 56));
            LOBYTE(v87) = *(_BYTE *)(v86 + 34);
            HIBYTE(v87) = 1;
            *(_QWORD *)&v140 = sub_33F1DB0(
                                 (__int64 *)a5,
                                 1,
                                 (__int64)&v150,
                                 v5,
                                 v27,
                                 v87,
                                 v10,
                                 v11.m128i_i64[0],
                                 v11.m128i_i64[1],
                                 *(_OWORD *)v86,
                                 *(_QWORD *)(v86 + 16),
                                 v79,
                                 v147,
                                 *(_WORD *)(v86 + 32),
                                 (__int64)&v174);
            v174.m128i_i64[0] = (__int64)v175;
            *((_QWORD *)&v140 + 1) = v89;
            v174.m128i_i64[1] = 0x800000000LL;
            if ( v142 )
            {
              v90 = 0;
              do
              {
                v111 = (_BYTE *)sub_2E79000(*(__int64 **)(a5 + 40));
                v92 = v90;
                if ( *v111 )
                  v92 = v142 - 1 - v90;
                if ( (_WORD)v156 )
                {
                  if ( (_WORD)v156 == 1 || (unsigned __int16)(v156 - 504) <= 7u )
                    goto LABEL_129;
                  v96 = *(_BYTE **)&byte_444C4A0[16 * (unsigned __int16)v156 - 16];
                  LOBYTE(v95) = byte_444C4A0[16 * (unsigned __int16)v156 - 8];
                }
                else
                {
                  v145 = v92;
                  v91 = sub_3007260((__int64)&v156);
                  v92 = v145;
                  v93 = (_BYTE *)v91;
                  v95 = v94;
                  v171 = v93;
                  v96 = v93;
                  v172 = v95;
                }
                BYTE8(v168) = v95;
                *(_QWORD *)&v168 = (_QWORD)v96 * v92;
                v97 = sub_CA1930(&v168);
                *(_QWORD *)&v98 = sub_3400E40(a5, v97, v5, v27, (__int64)&v150, a2);
                *(_QWORD *)&v100 = sub_3406EB0((_QWORD *)a5, 0xC0u, (__int64)&v150, v5, v27, v99, v140, v98);
                sub_3406EB0((_QWORD *)a5, 0xBAu, (__int64)&v150, v5, v27, v101, v100, v137);
                v105 = sub_33FAF80(a5, 216, (__int64)&v150, (unsigned int)v156, v157, v102, a2);
                v106 = v103;
                if ( v141 )
                {
                  v148 = v103;
                  v112 = sub_33CBCE0(0, v141);
                  v105 = sub_33FAF80(a5, v112, (__int64)&v150, v131, v130, v113, a2);
                  v106 = v114 | v148 & 0xFFFFFFFF00000000LL;
                }
                v107 = v174.m128i_u32[2];
                v108 = v174.m128i_u32[2] + 1LL;
                if ( v108 > v174.m128i_u32[3] )
                {
                  v146 = v105;
                  v149 = v106;
                  sub_C8D5F0((__int64)&v174, v175, v108, 0x10u, v104, v88);
                  v107 = v174.m128i_u32[2];
                  v105 = v146;
                  v106 = v149;
                }
                v109 = (unsigned __int8 **)(v174.m128i_i64[0] + 16 * v107);
                ++v90;
                *v109 = v105;
                v109[1] = (unsigned __int8 *)v106;
                v110 = (unsigned int)++v174.m128i_i32[2];
              }
              while ( v90 != v142 );
              v115 = (_OWORD *)v174.m128i_i64[0];
            }
            else
            {
              v115 = v175;
              v110 = 0;
            }
            *((_QWORD *)&v123 + 1) = v110;
            *(_QWORD *)&v123 = v115;
            v116 = sub_33FC220((_QWORD *)a5, 156, (__int64)&v150, v154, v155, v88, v123);
            v72 = (_OWORD *)v174.m128i_i64[0];
            *(_QWORD *)a1 = v116;
            *(_DWORD *)(a1 + 8) = v117;
            *(_DWORD *)(a1 + 24) = 1;
            *(_QWORD *)(a1 + 16) = v140;
            if ( v72 != v175 )
              goto LABEL_56;
            goto LABEL_57;
          }
LABEL_129:
          BUG();
      }
      v78 = 0;
      goto LABEL_75;
    }
LABEL_62:
    v164 = sub_3007260((__int64)&v152);
    v165 = v75;
    v30 = v164;
    v31 = v165;
    goto LABEL_63;
  }
  v158 = sub_3007260((__int64)&v156);
  v159 = v21;
  if ( !v158 )
    goto LABEL_15;
  v160 = sub_3007260((__int64)&v156);
  v161 = v35;
  if ( (v160 & 7) != 0 )
    goto LABEL_15;
  v34 = sub_3007260((__int64)&v156);
  v166 = v34;
  v167 = v36;
  v33 = v36;
LABEL_38:
  v174.m128i_i64[0] = v34;
  v174.m128i_i8[8] = v33;
  v37 = sub_CA1930(&v174);
  v171 = v173;
  v39 = v37 >> 3;
  v129 = v39;
  v172 = 0x800000000LL;
  v174.m128i_i64[0] = (__int64)v175;
  v174.m128i_i64[1] = 0x800000000LL;
  if ( v142 )
  {
    v144 = 0;
    v127 = (unsigned int)v39;
    v41 = v11.m128i_u64[1];
    v40 = (unsigned __int8 *)v11.m128i_i64[0];
    v139 = 0;
    do
    {
      v56 = *(_QWORD *)(a4 + 112);
      v57 = _mm_loadu_si128((const __m128i *)(v56 + 40));
      v170[0] = v57;
      v170[1] = _mm_loadu_si128((const __m128i *)(v56 + 56));
      LOBYTE(v59) = *(_BYTE *)(v56 + 34);
      v58 = *(_WORD *)(v56 + 32);
      HIBYTE(v59) = 1;
      v60 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v60 )
      {
        v61 = *(_QWORD *)(v56 + 8) + v139;
        v62 = *(_BYTE *)(v56 + 20);
        if ( (*(_QWORD *)v56 & 4) != 0 )
        {
          *((_QWORD *)&v168 + 1) = *(_QWORD *)(v56 + 8) + v139;
          BYTE4(v169) = v62;
          *(_QWORD *)&v168 = v60 | 4;
          LODWORD(v169) = *(_DWORD *)(v60 + 12);
        }
        else
        {
          *(_QWORD *)&v168 = *(_QWORD *)v56 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v168 + 1) = v61;
          BYTE4(v169) = v62;
          v63 = *(_QWORD *)(v60 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v63 + 8) - 17 <= 1 )
            v63 = **(_QWORD **)(v63 + 16);
          LODWORD(v169) = *(_DWORD *)(v63 + 8) >> 8;
        }
      }
      else
      {
        v42 = *(_DWORD *)(v56 + 16);
        v43 = *(_QWORD *)(v56 + 8) + v139;
        BYTE4(v169) = 0;
        *(_QWORD *)&v168 = 0;
        *((_QWORD *)&v168 + 1) = v43;
        LODWORD(v169) = v42;
      }
      v44 = sub_33F1DB0(
              (__int64 *)a5,
              v141,
              (__int64)&v150,
              v131,
              v130,
              v59,
              v10,
              (__int64)v40,
              v41,
              v168,
              v169,
              v156,
              v157,
              v58,
              (__int64)v170);
      BYTE8(v170[0]) = 0;
      *(_QWORD *)&v170[0] = v127;
      v132 = v44;
      v40 = sub_3409320((_QWORD *)a5, (__int64)v40, v41, v127, 0, (__int64)&v150, v57, 1);
      v45 = v135 & 0xFFFFFFFF00000000LL;
      v135 &= 0xFFFFFFFF00000000LL;
      v47 = v132;
      v41 = v46 | v41 & 0xFFFFFFFF00000000LL;
      v48 = (unsigned int)v172;
      v49 = (unsigned int)v172 + 1LL;
      if ( v49 > HIDWORD(v172) )
      {
        v125 = v45;
        sub_C8D5F0((__int64)&v171, v173, v49, 0x10u, v45, v38);
        v48 = (unsigned int)v172;
        v45 = v125;
        v47 = v132;
      }
      v50 = (__m128i **)&v171[16 * v48];
      v50[1] = (__m128i *)v45;
      *v50 = v47;
      v51 = v174.m128i_u32[2];
      LODWORD(v172) = v172 + 1;
      v52 = v136 & 0xFFFFFFFF00000000LL | 1;
      v53 = v174.m128i_u32[2] + 1LL;
      v136 = v52;
      if ( v53 > v174.m128i_u32[3] )
      {
        v124 = v52;
        v133 = v47;
        sub_C8D5F0((__int64)&v174, v175, v53, 0x10u, v52, v38);
        v51 = v174.m128i_u32[2];
        v52 = v124;
        v47 = v133;
      }
      v54 = (__m128i **)(v174.m128i_i64[0] + 16 * v51);
      ++v144;
      *v54 = v47;
      v54[1] = (__m128i *)v52;
      v139 += v129;
      v55 = (unsigned int)++v174.m128i_i32[2];
    }
    while ( v144 != v142 );
    v64 = (_OWORD *)v174.m128i_i64[0];
  }
  else
  {
    v64 = v175;
    v55 = 0;
  }
  *((_QWORD *)&v122 + 1) = v55;
  *(_QWORD *)&v122 = v64;
  v65 = sub_33FC220((_QWORD *)a5, 2, (__int64)&v150, 1, 0, v38, v122);
  v67 = v66;
  *((_QWORD *)&v121 + 1) = (unsigned int)v172;
  *(_QWORD *)&v121 = v171;
  v69 = sub_33FC220((_QWORD *)a5, 156, (__int64)&v150, v154, v155, v68, v121);
  v70 = (_OWORD *)v174.m128i_i64[0];
  *(_QWORD *)a1 = v69;
  *(_DWORD *)(a1 + 8) = v71;
  *(_QWORD *)(a1 + 16) = v65;
  *(_QWORD *)(a1 + 24) = v67;
  if ( v70 != v175 )
    _libc_free((unsigned __int64)v70);
  v72 = v171;
  if ( v171 != v173 )
LABEL_56:
    _libc_free((unsigned __int64)v72);
LABEL_57:
  if ( v150 )
    sub_B91220((__int64)&v150, v150);
  return a1;
}
