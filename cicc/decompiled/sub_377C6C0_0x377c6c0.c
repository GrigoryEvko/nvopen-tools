// Function: sub_377C6C0
// Address: 0x377c6c0
//
void __fastcall sub_377C6C0(__int64 *a1, __int64 a2, unsigned int *a3, unsigned int *a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // rbx
  __int128 v8; // xmm0
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // rcx
  unsigned __int16 *v17; // rax
  int v18; // ebx
  unsigned __int64 v19; // r13
  __int64 v20; // rdx
  unsigned __int16 v21; // r13
  __int64 v22; // rdx
  __m128i v23; // rax
  unsigned __int8 v24; // al
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rdx
  char v28; // al
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 *v32; // rbx
  _QWORD *v33; // rdi
  __m128i *v34; // rax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // r14
  __int128 v37; // rax
  _QWORD *v38; // r15
  unsigned __int64 v39; // rax
  __int64 v40; // rdx
  char v41; // cl
  unsigned __int64 v42; // rax
  unsigned int v43; // edx
  unsigned __int64 v44; // rbx
  __int16 v45; // ax
  __int64 *v46; // rdi
  __m128i v47; // kr00_16
  __int64 v48; // rsi
  __int64 v49; // rdx
  unsigned int v50; // edx
  const __m128i *v51; // roff
  __int16 v52; // ax
  __int64 *v53; // rdi
  unsigned int v54; // edx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // r10
  __int64 v59; // r9
  __int64 v60; // r11
  __int64 v61; // rax
  __int16 v62; // bx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int16 *v66; // rbx
  int v67; // eax
  __int64 v68; // rdx
  unsigned __int16 *v69; // rax
  bool v70; // al
  _QWORD *v71; // r12
  unsigned __int8 *v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdi
  unsigned __int16 *v75; // rdx
  __int64 v76; // r9
  unsigned int v77; // edx
  unsigned int v78; // edx
  unsigned __int8 *v79; // rax
  unsigned int v80; // edx
  __int64 v81; // rax
  unsigned int v82; // edx
  __m128i v83; // rax
  __int64 v84; // rax
  __int64 v85; // rdx
  int v86; // edx
  __int16 v87; // ax
  __int64 v88; // rdx
  __int64 v89; // rax
  _QWORD *v90; // r13
  __m128i v91; // rax
  unsigned int v92; // eax
  int v93; // r9d
  __int16 v94; // r13
  __int64 v95; // rbx
  __int64 v96; // rdi
  unsigned int v97; // edx
  __int64 v98; // rax
  __int16 v99; // dx
  __int64 v100; // rax
  __m128i v101; // rax
  __int128 v102; // rax
  int v103; // r9d
  unsigned int v104; // edx
  __int64 v105; // rdx
  int v106; // eax
  int v107; // ecx
  __int64 v108; // rsi
  unsigned __int16 v109; // ax
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 v114; // rdx
  int v115; // esi
  __int16 v116; // ax
  __int128 v117; // [rsp+0h] [rbp-230h]
  __int128 v118; // [rsp+10h] [rbp-220h]
  __int128 v119; // [rsp+10h] [rbp-220h]
  unsigned __int8 v120; // [rsp+20h] [rbp-210h]
  unsigned int v121; // [rsp+20h] [rbp-210h]
  unsigned int v122; // [rsp+24h] [rbp-20Ch]
  unsigned __int8 *v123; // [rsp+28h] [rbp-208h]
  __int64 v124; // [rsp+28h] [rbp-208h]
  unsigned __int8 *v126; // [rsp+38h] [rbp-1F8h]
  unsigned __int64 v127; // [rsp+38h] [rbp-1F8h]
  __int64 v128; // [rsp+40h] [rbp-1F0h]
  __int64 v129; // [rsp+40h] [rbp-1F0h]
  unsigned int v131; // [rsp+58h] [rbp-1D8h]
  __int64 v132; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 v133; // [rsp+60h] [rbp-1D0h]
  __int64 v134; // [rsp+60h] [rbp-1D0h]
  unsigned int v135; // [rsp+60h] [rbp-1D0h]
  unsigned int v136; // [rsp+60h] [rbp-1D0h]
  __m128i *v138; // [rsp+A0h] [rbp-190h]
  __m128i *v139; // [rsp+B0h] [rbp-180h]
  __int64 v140; // [rsp+F0h] [rbp-140h] BYREF
  int v141; // [rsp+F8h] [rbp-138h]
  __int64 v142; // [rsp+100h] [rbp-130h] BYREF
  unsigned __int64 v143; // [rsp+108h] [rbp-128h]
  __m128i v144; // [rsp+110h] [rbp-120h] BYREF
  unsigned __int64 v145; // [rsp+120h] [rbp-110h] BYREF
  unsigned __int64 v146; // [rsp+128h] [rbp-108h]
  __int64 v147[2]; // [rsp+130h] [rbp-100h] BYREF
  unsigned __int64 v148; // [rsp+140h] [rbp-F0h]
  __int64 v149; // [rsp+148h] [rbp-E8h]
  __int64 v150; // [rsp+150h] [rbp-E0h]
  __int64 v151; // [rsp+158h] [rbp-D8h]
  unsigned __int64 v152; // [rsp+160h] [rbp-D0h]
  __int64 v153; // [rsp+168h] [rbp-C8h]
  __int64 v154; // [rsp+170h] [rbp-C0h]
  __int64 v155; // [rsp+178h] [rbp-B8h]
  __int128 v156; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v157; // [rsp+190h] [rbp-A0h]
  __int128 v158; // [rsp+1A0h] [rbp-90h] BYREF
  __int64 v159; // [rsp+1B0h] [rbp-80h]
  __int128 v160; // [rsp+1C0h] [rbp-70h] BYREF
  __int64 v161; // [rsp+1D0h] [rbp-60h]
  __m128i v162; // [rsp+1E0h] [rbp-50h] BYREF
  __int64 v163; // [rsp+1F0h] [rbp-40h]
  __int64 v164; // [rsp+1F8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(unsigned __int8 **)v5;
  v8 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 40));
  v9 = *(_QWORD *)v5;
  v140 = v6;
  v126 = v7;
  v10 = *(_QWORD *)(v5 + 8);
  v131 = *(_DWORD *)(v5 + 8);
  v118 = (__int128)_mm_loadu_si128((const __m128i *)(v5 + 80));
  v123 = *(unsigned __int8 **)(v5 + 40);
  v122 = *(_DWORD *)(v5 + 48);
  v11 = *(_QWORD *)(v5 + 80);
  if ( v6 )
    sub_B96E90((__int64)&v140, v6, 1);
  v141 = *(_DWORD *)(a2 + 72);
  sub_375E8D0((__int64)a1, v9, v10, (__int64)a3, (__int64)a4);
  v15 = *(_DWORD *)(v11 + 24);
  if ( v15 != 35 && v15 != 11 )
  {
    v16 = (__int64)v126;
    v17 = (unsigned __int16 *)(*((_QWORD *)v126 + 6) + 16LL * v131);
    v18 = *v17;
    v19 = *((_QWORD *)v17 + 1);
    goto LABEL_6;
  }
  v64 = *(_QWORD *)(v11 + 96);
  if ( *(_DWORD *)(v64 + 32) <= 0x40u )
    v65 = *(_QWORD *)(v64 + 24);
  else
    v65 = **(_QWORD **)(v64 + 24);
  v16 = (unsigned int)v65;
  v66 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2]);
  v67 = *v66;
  v68 = *((_QWORD *)v66 + 1);
  v162.m128i_i16[0] = v67;
  v162.m128i_i64[1] = v68;
  if ( (_WORD)v67 )
  {
    v12 = word_4456340[v67 - 1];
    if ( (unsigned int)v16 >= (unsigned int)v12 )
      goto LABEL_28;
LABEL_37:
    *(_QWORD *)a3 = sub_340F900(
                      (_QWORD *)a1[1],
                      0x9Du,
                      (__int64)&v140,
                      *v66,
                      *((_QWORD *)v66 + 1),
                      v14,
                      *(_OWORD *)a3,
                      v8,
                      v118);
    a3[2] = v82;
    goto LABEL_21;
  }
  v135 = v16;
  v81 = sub_3007240((__int64)&v162);
  v16 = v135;
  v12 = v81;
  if ( v135 < (unsigned int)v81 )
    goto LABEL_37;
LABEL_28:
  v69 = (unsigned __int16 *)(*((_QWORD *)v126 + 6) + 16LL * v131);
  v18 = *v69;
  v19 = *((_QWORD *)v69 + 1);
  v162.m128i_i16[0] = v18;
  v162.m128i_i64[1] = v19;
  if ( (_WORD)v18 )
  {
    v70 = (unsigned __int16)(v18 - 176) <= 0x34u;
  }
  else
  {
    v121 = v12;
    v136 = v16;
    v70 = sub_3007100((__int64)&v162);
    v12 = v121;
    v16 = v136;
  }
  if ( !v70 )
  {
    v71 = (_QWORD *)a1[1];
    v72 = sub_3400EE0((__int64)v71, (unsigned int)(v16 - v12), (__int64)&v140, 0, (__m128i)v8);
    v74 = v73;
    v75 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
    *((_QWORD *)&v117 + 1) = v74;
    *(_QWORD *)&v117 = v72;
    *(_QWORD *)a4 = sub_340F900(v71, 0x9Du, (__int64)&v140, *v75, *((_QWORD *)v75 + 1), v76, *(_OWORD *)a4, v8, v117);
    a4[2] = v77;
    goto LABEL_21;
  }
LABEL_6:
  LOWORD(v142) = v18;
  v143 = v19;
  if ( (_WORD)v18 )
  {
    v20 = 0;
    v21 = word_4456580[v18 - 1];
  }
  else
  {
    v21 = sub_3009970((__int64)&v142, v9, v12, v16, v13);
  }
  v144.m128i_i16[0] = v21;
  v144.m128i_i64[1] = v20;
  if ( v21 )
  {
    if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
      goto LABEL_97;
    if ( *(_QWORD *)&byte_444C4A0[16 * v21 - 16] )
    {
      v83.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v144);
      v162 = v83;
      if ( (v83.m128i_i8[0] & 7) == 0 )
        goto LABEL_11;
    }
    if ( (unsigned __int16)(v21 - 17) <= 0xD3u )
    {
      v162.m128i_i16[0] = v21;
      v87 = sub_30369B0((unsigned __int16 *)&v162);
      v88 = 0;
    }
    else
    {
      if ( (unsigned __int16)(v21 - 504) <= 7u )
        goto LABEL_97;
      v84 = 16LL * (v21 - 1);
      v85 = *(_QWORD *)&byte_444C4A0[v84];
      v162.m128i_i8[8] = byte_444C4A0[v84 + 8];
      v162.m128i_i64[0] = v85;
      v86 = sub_CA1930(&v162);
      v87 = 2;
      if ( v86 != 1 )
      {
        v87 = 3;
        if ( v86 != 2 )
        {
          v87 = 4;
          if ( v86 != 4 )
          {
            v87 = 5;
            if ( v86 != 8 )
            {
              v87 = 6;
              if ( v86 != 16 )
              {
                v87 = 7;
                if ( v86 != 32 )
                {
                  v87 = 8;
                  if ( v86 != 64 )
                    v87 = 9 * (v86 == 128);
                }
              }
            }
          }
        }
      }
      v88 = 0;
    }
  }
  else
  {
    v150 = sub_3007260((__int64)&v144);
    v151 = v22;
    if ( v150 )
    {
      v23.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v144);
      v162 = v23;
      if ( (v23.m128i_i8[0] & 7) == 0 )
        goto LABEL_11;
    }
    if ( sub_30070B0((__int64)&v144) )
      v87 = sub_300A990((unsigned __int16 *)&v144, v9);
    else
      v87 = sub_30072B0((__int64)&v144);
  }
  LOWORD(v160) = v87;
  v89 = a1[1];
  *((_QWORD *)&v160 + 1) = v88;
  v90 = *(_QWORD **)(v89 + 64);
  v91.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v160);
  v162 = v91;
  v92 = sub_CA1930(&v162);
  if ( v92 > 8 )
  {
    _BitScanReverse(&v92, v92 - 1);
    v106 = v92 ^ 0x1F;
    v107 = 32 - v106;
    if ( v106 == 28 )
    {
      v94 = 6;
      v144.m128i_i64[1] = 0;
      v144.m128i_i16[0] = 6;
      v162 = _mm_loadu_si128(&v144);
    }
    else if ( v107 == 5 )
    {
      v144.m128i_i16[0] = 7;
      v94 = 7;
      v144.m128i_i64[1] = 0;
      v162 = _mm_loadu_si128(&v144);
    }
    else
    {
      if ( v107 != 6 )
      {
        if ( v107 == 7 )
        {
          v95 = 0;
          v94 = 9;
          v144.m128i_i64[1] = 0;
          v144.m128i_i16[0] = 9;
          v162 = _mm_loadu_si128(&v144);
        }
        else
        {
          v108 = (unsigned int)(1 << (32 - v106));
          v109 = sub_3007020(v90, v108);
          v144.m128i_i16[0] = v109;
          v94 = v109;
          v95 = v110;
          v144.m128i_i64[1] = v110;
          v162 = _mm_loadu_si128(&v144);
          if ( v109 )
          {
            if ( (unsigned __int16)(v109 - 17) <= 0xD3u )
            {
              v95 = 0;
              v94 = word_4456580[v109 - 1];
            }
          }
          else if ( sub_30070B0((__int64)&v162) )
          {
            v94 = sub_3009970((__int64)&v162, v108, v111, v112, v113);
            v95 = v114;
          }
        }
        goto LABEL_67;
      }
      v94 = 8;
      v144.m128i_i64[1] = 0;
      v144.m128i_i16[0] = 8;
      v162 = _mm_loadu_si128(&v144);
    }
  }
  else
  {
    v94 = 5;
    v144.m128i_i64[1] = 0;
    v144.m128i_i16[0] = 5;
    v162 = _mm_loadu_si128(&v144);
  }
  v95 = 0;
LABEL_67:
  v162.m128i_i16[0] = v94;
  v162.m128i_i64[1] = v95;
  if ( (_WORD)v142 )
  {
    if ( (unsigned __int16)(v142 - 17) <= 0xD3u )
    {
      v115 = word_4456340[(unsigned __int16)v142 - 1];
      if ( (unsigned __int16)(v142 - 176) > 0x34u )
        v116 = sub_2D43050(v94, v115);
      else
        v116 = sub_2D43AD0(v94, v115);
      v94 = v116;
      v95 = 0;
    }
  }
  else if ( sub_30070B0((__int64)&v142) )
  {
    v94 = sub_3009490((unsigned __int16 *)&v142, v162.m128i_u32[0], v162.m128i_i64[1]);
    v95 = v105;
  }
  v96 = a1[1];
  LOWORD(v142) = v94;
  v143 = v95;
  v126 = sub_33FAF80(v96, 215, (__int64)&v140, (unsigned int)v142, v95, v93, (__m128i)v8);
  v131 = v97;
  v10 = v97 | v10 & 0xFFFFFFFF00000000LL;
  v98 = *((_QWORD *)v123 + 6) + 16LL * v122;
  v99 = *(_WORD *)v98;
  v100 = *(_QWORD *)(v98 + 8);
  if ( v99 != v144.m128i_i16[0] || !v99 && v100 != v144.m128i_i64[1] )
  {
    LOWORD(v158) = v99;
    *((_QWORD *)&v158 + 1) = v100;
    v101.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v158);
    v162 = v101;
    *(_QWORD *)&v102 = sub_2D5B750((unsigned __int16 *)&v144);
    v160 = v102;
    if ( (BYTE8(v102) || !v162.m128i_i8[8]) && (unsigned __int64)v160 > v162.m128i_i64[0] )
    {
      v123 = sub_33FAF80(a1[1], 215, (__int64)&v140, v144.m128i_u32[0], v144.m128i_i64[1], v103, (__m128i)v8);
      v122 = v104;
    }
  }
LABEL_11:
  v24 = sub_33CD850(a1[1], (unsigned int)v142, v143, 0);
  v25 = a1[1];
  v133 = v24;
  if ( (_WORD)v142 )
  {
    if ( (_WORD)v142 == 1 || (unsigned __int16)(v142 - 504) <= 7u )
      goto LABEL_97;
    v26 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v142 - 16];
    v28 = byte_444C4A0[16 * (unsigned __int16)v142 - 8];
  }
  else
  {
    v154 = sub_3007260((__int64)&v142);
    v26 = v154;
    v155 = v27;
    v28 = v27;
  }
  LOBYTE(v149) = v28;
  v148 = (unsigned __int64)(v26 + 7) >> 3;
  v29 = sub_33EDE90(v25, v148, v149, v133);
  v146 = v30;
  v31 = a1[1];
  v145 = (unsigned __int64)v29;
  v32 = *(__int64 **)(v31 + 40);
  sub_2EAC300((__int64)&v156, (__int64)v32, *((_DWORD *)v29 + 24), 0);
  v33 = (_QWORD *)a1[1];
  v162 = 0u;
  v163 = 0;
  v164 = 0;
  v34 = sub_33F4560(
          v33,
          (unsigned __int64)(v33 + 36),
          0,
          (__int64)&v140,
          (unsigned __int64)v126,
          v131 | v10 & 0xFFFFFFFF00000000LL,
          v145,
          v146,
          v156,
          v157,
          v133,
          0,
          (__int64)&v162);
  v36 = v35;
  v127 = (unsigned __int64)v34;
  *(_QWORD *)&v37 = sub_3466750(*a1, (_QWORD *)a1[1], v145, v146, (unsigned int)v142, v143, (__m128i)v8, v118);
  v162 = 0u;
  v119 = v37;
  v38 = (_QWORD *)a1[1];
  v163 = 0;
  v164 = 0;
  if ( v144.m128i_i16[0] )
  {
    if ( v144.m128i_i16[0] != 1 && (unsigned __int16)(v144.m128i_i16[0] - 504) > 7u )
    {
      v39 = *(_QWORD *)&byte_444C4A0[16 * v144.m128i_u16[0] - 16];
      goto LABEL_15;
    }
LABEL_97:
    BUG();
  }
  v39 = sub_3007260((__int64)&v144);
  v152 = v39;
  v153 = v40;
LABEL_15:
  v41 = -1;
  v42 = -(__int64)((v39 >> 3) | (1LL << v133)) & ((v39 >> 3) | (1LL << v133));
  if ( v42 )
  {
    _BitScanReverse64(&v42, v42);
    v41 = 63 - (v42 ^ 0x3F);
  }
  v120 = v41;
  sub_2EAC3A0((__int64)&v160, v32);
  v139 = sub_33F5040(
           v38,
           v127,
           v36,
           (__int64)&v140,
           (unsigned __int64)v123,
           v122 | *((_QWORD *)&v8 + 1) & 0xFFFFFFFF00000000LL,
           v119,
           *((unsigned __int64 *)&v119 + 1),
           v160,
           v161,
           v144.m128i_i64[0],
           v144.m128i_u64[1],
           v120,
           0,
           (__int64)&v162);
  v44 = v43 | v36 & 0xFFFFFFFF00000000LL;
  sub_33D0340((__int64)&v162, a1[1], &v142);
  HIBYTE(v45) = 1;
  LOBYTE(v45) = v133;
  v46 = (__int64 *)a1[1];
  v47 = v162;
  v48 = v162.m128i_u32[0];
  v49 = v162.m128i_i64[1];
  v124 = v163;
  v128 = v164;
  v162 = 0u;
  v163 = 0;
  v164 = 0;
  v138 = sub_33F1F00(
           v46,
           v48,
           v49,
           (__int64)&v140,
           (__int64)v139,
           v44,
           v145,
           v146,
           v156,
           v157,
           v45,
           0,
           (__int64)&v162,
           0);
  *(_QWORD *)a3 = v138;
  a3[2] = v50;
  v51 = (const __m128i *)v138[7].m128i_i64[0];
  v158 = (__int128)_mm_loadu_si128(v51);
  v159 = v51[1].m128i_i64[0];
  sub_3777490(
    (__int64)a1,
    (__int64)v138,
    v47.m128i_u32[0],
    v47.m128i_i64[1],
    (__int64)&v158,
    (unsigned int *)&v145,
    (__m128i)v8,
    0);
  HIBYTE(v52) = 1;
  LOBYTE(v52) = v133;
  v53 = (__int64 *)a1[1];
  v162 = 0u;
  v163 = 0;
  v164 = 0;
  *(_QWORD *)a4 = sub_33F1F00(
                    v53,
                    v124,
                    v128,
                    (__int64)&v140,
                    (__int64)v139,
                    v44,
                    v145,
                    v146,
                    v158,
                    v159,
                    v52,
                    0,
                    (__int64)&v162,
                    0);
  a4[2] = v54;
  v55 = *(_QWORD *)(a2 + 48);
  v56 = a1[1];
  LOWORD(v54) = *(_WORD *)v55;
  v57 = *(_QWORD *)(v55 + 8);
  LOWORD(v147[0]) = v54;
  v147[1] = v57;
  sub_33D0340((__int64)&v162, v56, v147);
  v58 = v164;
  v59 = v164;
  v60 = v163;
  v61 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v62 = v163;
  if ( *(_WORD *)v61 != v162.m128i_i16[0] || !v162.m128i_i16[0] && *(_QWORD *)(v61 + 8) != v162.m128i_i64[1] )
  {
    v129 = v164;
    v132 = v163;
    v134 = v164;
    v79 = sub_33FAF80(a1[1], 216, (__int64)&v140, v162.m128i_u32[0], v162.m128i_i64[1], v164, (__m128i)v8);
    v58 = v129;
    v60 = v132;
    v59 = v134;
    *(_QWORD *)a3 = v79;
    a3[2] = v80;
  }
  v63 = *(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2];
  if ( *(_WORD *)v63 != v62 || !v62 && *(_QWORD *)(v63 + 8) != v59 )
  {
    *(_QWORD *)a4 = sub_33FAF80(a1[1], 216, (__int64)&v140, v60, v58, v59, (__m128i)v8);
    a4[2] = v78;
  }
LABEL_21:
  if ( v140 )
    sub_B91220((__int64)&v140, v140);
}
