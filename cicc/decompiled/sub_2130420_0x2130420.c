// Function: sub_2130420
// Address: 0x2130420
//
void __fastcall sub_2130420(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rax
  __m128i v13; // xmm2
  unsigned __int8 v14; // dl
  __int64 v15; // rbx
  __m128i v16; // xmm3
  unsigned int v17; // eax
  char v18; // r8
  unsigned __int8 v19; // dl
  unsigned int v20; // eax
  int v21; // edx
  char v22; // di
  __int64 v23; // rax
  int v24; // eax
  char v25; // r14
  int v26; // ebx
  int v27; // eax
  unsigned int v28; // ebx
  __int64 v29; // r10
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rbx
  unsigned int v33; // ebx
  __int64 v34; // r10
  __int64 v35; // rbx
  __int64 v36; // r14
  __int128 v37; // rax
  __int64 *v38; // rax
  _QWORD *v39; // rdi
  unsigned int v40; // edx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r9
  unsigned __int64 v44; // rcx
  int v45; // eax
  int v46; // edx
  unsigned int v47; // edx
  const __m128i *v48; // r9
  __int64 v49; // rcx
  __int64 v50; // r9
  int v51; // edx
  _QWORD *v52; // rdi
  int v53; // edx
  _QWORD *v54; // rbx
  const __m128i *v55; // r9
  unsigned int v56; // eax
  char v57; // di
  __int64 v58; // rax
  int v59; // esi
  unsigned int v60; // eax
  __int64 v61; // r14
  unsigned int v62; // r10d
  unsigned int v63; // ebx
  unsigned int v64; // esi
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rsi
  __int64 v68; // r14
  int v69; // edx
  __int128 v70; // rax
  __int64 *v71; // rax
  __int64 v72; // r10
  __int64 v73; // r14
  unsigned int v74; // edx
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rdi
  __int64 v79; // rax
  unsigned __int64 v80; // rcx
  int v81; // eax
  int v82; // edx
  unsigned int v83; // edx
  unsigned int v84; // eax
  __int64 *v85; // r14
  __int64 v86; // rax
  unsigned int v87; // edx
  unsigned __int8 v88; // al
  __int128 v89; // rax
  __int128 v90; // rax
  int v91; // edx
  __int64 *v92; // r14
  __int64 v93; // rax
  unsigned int v94; // edx
  unsigned __int8 v95; // al
  int v96; // eax
  __int64 v97; // rcx
  int v98; // esi
  __int128 v99; // rax
  int v100; // edx
  int v101; // eax
  __int64 v102; // rdx
  int v103; // edx
  __int64 v104; // r10
  char v105; // r8
  __int64 v106; // rax
  char v107; // si
  __int64 v108; // rbx
  __int64 v109; // rax
  unsigned int v110; // eax
  __int64 v111; // rax
  char v112; // di
  __int64 v113; // rax
  int v114; // ebx
  __int64 *v115; // r14
  __int64 v116; // rax
  unsigned int v117; // edx
  unsigned __int8 v118; // al
  __int128 v119; // rax
  int v120; // edx
  __int128 v121; // [rsp-10h] [rbp-230h]
  __int128 v122; // [rsp-10h] [rbp-230h]
  __int64 v123; // [rsp+8h] [rbp-218h]
  __int64 v124; // [rsp+10h] [rbp-210h]
  __int64 v125; // [rsp+18h] [rbp-208h]
  __int8 v126; // [rsp+20h] [rbp-200h]
  __int64 v127; // [rsp+20h] [rbp-200h]
  unsigned __int8 v128; // [rsp+28h] [rbp-1F8h]
  unsigned __int8 v129; // [rsp+28h] [rbp-1F8h]
  unsigned __int8 v130; // [rsp+28h] [rbp-1F8h]
  unsigned int v131; // [rsp+30h] [rbp-1F0h]
  __int64 v132; // [rsp+30h] [rbp-1F0h]
  __int64 v133; // [rsp+30h] [rbp-1F0h]
  unsigned int v134; // [rsp+30h] [rbp-1F0h]
  __int64 v135; // [rsp+38h] [rbp-1E8h]
  unsigned int v136; // [rsp+40h] [rbp-1E0h]
  __int64 *v137; // [rsp+40h] [rbp-1E0h]
  __int64 *v138; // [rsp+40h] [rbp-1E0h]
  __int64 v139; // [rsp+48h] [rbp-1D8h]
  unsigned int v140; // [rsp+50h] [rbp-1D0h]
  unsigned __int16 v143; // [rsp+68h] [rbp-1B8h]
  unsigned int v144; // [rsp+6Ch] [rbp-1B4h]
  __int64 v145; // [rsp+70h] [rbp-1B0h]
  int v146; // [rsp+70h] [rbp-1B0h]
  __int64 v147; // [rsp+70h] [rbp-1B0h]
  __int64 v148; // [rsp+70h] [rbp-1B0h]
  __int64 v149; // [rsp+78h] [rbp-1A8h]
  unsigned __int64 v150; // [rsp+78h] [rbp-1A8h]
  __int64 *v151; // [rsp+80h] [rbp-1A0h]
  __int64 *v152; // [rsp+80h] [rbp-1A0h]
  __m128i *v153; // [rsp+88h] [rbp-198h]
  __int64 v154; // [rsp+C0h] [rbp-160h]
  __int64 v155; // [rsp+100h] [rbp-120h]
  __m128i v156; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v157; // [rsp+180h] [rbp-A0h] BYREF
  int v158; // [rsp+188h] [rbp-98h]
  __m128i v159; // [rsp+190h] [rbp-90h] BYREF
  __int64 v160; // [rsp+1A0h] [rbp-80h]
  __int128 v161; // [rsp+1B0h] [rbp-70h] BYREF
  __int64 v162; // [rsp+1C0h] [rbp-60h]
  __m128i v163; // [rsp+1D0h] [rbp-50h] BYREF
  __int64 v164; // [rsp+1E0h] [rbp-40h]

  if ( *(_WORD *)(a2 + 24) == 185 && (*(_BYTE *)(a2 + 27) & 0xC) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 )
  {
    sub_2144300();
    return;
  }
  sub_1F40D10(
    (__int64)&v163,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 104);
  v156.m128i_i8[0] = v163.m128i_i8[8];
  v156.m128i_i64[1] = v164;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = _mm_loadu_si128((const __m128i *)v7);
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v139 = *(_QWORD *)(v7 + 40);
  v136 = *(_DWORD *)(v7 + 48);
  v144 = (*(_BYTE *)(a2 + 27) >> 2) & 3;
  v10 = sub_1E34390(v6);
  v11 = *(_QWORD *)(a2 + 72);
  v140 = v10;
  v12 = *(_QWORD *)(a2 + 104);
  v157 = v11;
  v13 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v143 = *(_WORD *)(v12 + 32);
  v160 = *(_QWORD *)(v12 + 56);
  v159 = v13;
  if ( v11 )
    sub_1623A60((__int64)&v157, v11, 2);
  v14 = *(_BYTE *)(a2 + 88);
  v15 = *(_QWORD *)(a2 + 96);
  v16 = _mm_loadu_si128(&v156);
  v158 = *(_DWORD *)(a2 + 64);
  LOBYTE(v161) = v14;
  *((_QWORD *)&v161 + 1) = v15;
  v163 = v16;
  if ( v14 == v156.m128i_i8[0] )
  {
    if ( v14 || v15 == v163.m128i_i64[1] )
      goto LABEL_28;
  }
  else if ( v14 )
  {
    v128 = v14;
    v17 = sub_2127930(v14);
    v19 = v128;
    v131 = v17;
    goto LABEL_10;
  }
  v126 = v156.m128i_i8[0];
  v130 = v14;
  v56 = sub_1F58D40((__int64)&v161);
  v18 = v126;
  v19 = v130;
  v131 = v56;
LABEL_10:
  v129 = v19;
  if ( v18 )
    v20 = sub_2127930(v18);
  else
    v20 = sub_1F58D40((__int64)&v163);
  v14 = v129;
  if ( v20 >= v131 )
  {
LABEL_28:
    v49 = sub_1D2B810(
            *(_QWORD **)(a1 + 8),
            v144,
            (__int64)&v157,
            v156.m128i_u32[0],
            v156.m128i_i64[1],
            v140,
            *(_OWORD *)&v8,
            v9.m128i_i64[0],
            v9.m128i_i64[1],
            *(_OWORD *)*(_QWORD *)(a2 + 104),
            *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
            v14,
            v15,
            v143,
            (__int64)&v159);
    *(_QWORD *)a3 = v49;
    v152 = (__int64 *)v49;
    *(_DWORD *)(a3 + 8) = v51;
    v153 = (__m128i *)(v8.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1);
    if ( v144 == 2 )
    {
      v111 = *(_QWORD *)(v49 + 40) + 16LL * *(unsigned int *)(a3 + 8);
      v112 = *(_BYTE *)v111;
      v113 = *(_QWORD *)(v111 + 8);
      v163.m128i_i8[0] = v112;
      v163.m128i_i64[1] = v113;
      if ( v112 )
        v114 = sub_2127930(v112);
      else
        v114 = sub_1F58D40((__int64)&v163);
      v115 = *(__int64 **)(a1 + 8);
      v116 = sub_1E0A0C0(v115[4]);
      v117 = 8 * sub_15A9520(v116, 0);
      if ( v117 == 32 )
      {
        v118 = 5;
      }
      else if ( v117 > 0x20 )
      {
        v118 = 6;
        if ( v117 != 64 )
        {
          v118 = 0;
          if ( v117 == 128 )
            v118 = 7;
        }
      }
      else
      {
        v118 = 3;
        if ( v117 != 8 )
          v118 = 4 * (v117 == 16);
      }
      *(_QWORD *)&v119 = sub_1D38BB0(
                           (__int64)v115,
                           (unsigned int)(v114 - 1),
                           (__int64)&v157,
                           v118,
                           0,
                           0,
                           v8,
                           *(double *)v9.m128i_i64,
                           v13,
                           0);
      *(_QWORD *)a4 = sub_1D332F0(
                        v115,
                        123,
                        (__int64)&v157,
                        v156.m128i_u32[0],
                        (const void **)v156.m128i_i64[1],
                        0,
                        *(double *)v8.m128i_i64,
                        *(double *)v9.m128i_i64,
                        v13,
                        *(_QWORD *)a3,
                        *(_QWORD *)(a3 + 8),
                        v119);
      *(_DWORD *)(a4 + 8) = v120;
    }
    else
    {
      v52 = *(_QWORD **)(a1 + 8);
      if ( v144 == 3 )
      {
        *(_QWORD *)a4 = sub_1D38BB0(
                          (__int64)v52,
                          0,
                          (__int64)&v157,
                          v156.m128i_u32[0],
                          (const void **)v156.m128i_i64[1],
                          0,
                          v8,
                          *(double *)v9.m128i_i64,
                          v13,
                          0);
        *(_DWORD *)(a4 + 8) = v103;
      }
      else
      {
        v163.m128i_i64[0] = 0;
        v163.m128i_i32[2] = 0;
        v54 = sub_1D2B300(v52, 0x30u, (__int64)&v163, v156.m128i_u32[0], v156.m128i_i64[1], v50);
        if ( v163.m128i_i64[0] )
        {
          v146 = v53;
          sub_161E7C0((__int64)&v163, v163.m128i_i64[0]);
          v53 = v146;
        }
        *(_QWORD *)a4 = v54;
        *(_DWORD *)(a4 + 8) = v53;
      }
    }
    goto LABEL_33;
  }
  if ( !*(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
  {
    *(_QWORD *)a3 = sub_1D2B730(
                      *(_QWORD **)(a1 + 8),
                      v156.m128i_u32[0],
                      v156.m128i_i64[1],
                      (__int64)&v157,
                      v8.m128i_i64[0],
                      v8.m128i_i64[1],
                      v9.m128i_i64[0],
                      v9.m128i_i64[1],
                      *(_OWORD *)*(_QWORD *)(a2 + 104),
                      *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                      v140,
                      v143,
                      (__int64)&v159,
                      0);
    *(_DWORD *)(a3 + 8) = v21;
    v22 = *(_BYTE *)(a2 + 88);
    v23 = *(_QWORD *)(a2 + 96);
    v163.m128i_i8[0] = v22;
    v163.m128i_i64[1] = v23;
    if ( v22 )
    {
      v101 = sub_2127930(v22);
      v25 = v156.m128i_i8[0];
      v26 = v101;
      if ( v156.m128i_i8[0] )
        goto LABEL_16;
    }
    else
    {
      v24 = sub_1F58D40((__int64)&v163);
      v25 = v156.m128i_i8[0];
      v26 = v24;
      if ( v156.m128i_i8[0] )
      {
LABEL_16:
        v27 = sub_2127930(v25);
        goto LABEL_17;
      }
    }
    v27 = sub_1F58D40((__int64)&v156);
LABEL_17:
    v28 = v26 - v27;
    v29 = *(_QWORD *)(a1 + 8);
    if ( v28 == 32 )
    {
      LOBYTE(v30) = 5;
    }
    else if ( v28 > 0x20 )
    {
      if ( v28 == 64 )
      {
        LOBYTE(v30) = 6;
      }
      else
      {
        if ( v28 != 128 )
        {
LABEL_76:
          v30 = sub_1F58CC0(*(_QWORD **)(v29 + 48), v28);
          v29 = *(_QWORD *)(a1 + 8);
          v25 = v156.m128i_i8[0];
          v123 = v30;
          goto LABEL_22;
        }
        LOBYTE(v30) = 7;
      }
    }
    else if ( v28 == 8 )
    {
      LOBYTE(v30) = 3;
    }
    else
    {
      LOBYTE(v30) = 4;
      if ( v28 != 16 )
      {
        LOBYTE(v30) = 2;
        if ( v28 != 1 )
          goto LABEL_76;
      }
    }
    v31 = 0;
LABEL_22:
    v32 = v123;
    v135 = v31;
    LOBYTE(v32) = v30;
    v132 = v32;
    if ( v25 )
    {
      v33 = sub_2127930(v25);
    }
    else
    {
      v127 = v29;
      v110 = sub_1F58D40((__int64)&v156);
      v34 = v127;
      v33 = v110;
    }
    v35 = v33 >> 3;
    v36 = v136;
    v137 = (__int64 *)v34;
    v36 *= 16;
    *(_QWORD *)&v37 = sub_1D38BB0(
                        v34,
                        v35,
                        (__int64)&v157,
                        *(unsigned __int8 *)(v36 + *(_QWORD *)(v139 + 40)),
                        *(const void ***)(v36 + *(_QWORD *)(v139 + 40) + 8),
                        0,
                        v8,
                        *(double *)v9.m128i_i64,
                        v13,
                        0);
    v38 = sub_1D332F0(
            v137,
            52,
            (__int64)&v157,
            *(unsigned __int8 *)(*(_QWORD *)(v139 + 40) + v36),
            *(const void ***)(*(_QWORD *)(v139 + 40) + v36 + 8),
            0,
            *(double *)v8.m128i_i64,
            *(double *)v9.m128i_i64,
            v13,
            v9.m128i_i64[0],
            v9.m128i_u64[1],
            v37);
    v39 = *(_QWORD **)(a1 + 8);
    v145 = (__int64)v38;
    v41 = v40 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v42 = *(_QWORD *)(a2 + 104);
    v149 = v41;
    v43 = -(v35 | v140) & (v35 | v140);
    v44 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v44 )
    {
      v107 = *(_BYTE *)(v42 + 16);
      v108 = *(_QWORD *)(v42 + 8) + v35;
      if ( (*(_QWORD *)v42 & 4) != 0 )
      {
        *((_QWORD *)&v161 + 1) = v108;
        LOBYTE(v162) = v107;
        *(_QWORD *)&v161 = v44 | 4;
        HIDWORD(v162) = *(_DWORD *)(v44 + 12);
      }
      else
      {
        *(_QWORD *)&v161 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v161 + 1) = v108;
        LOBYTE(v162) = v107;
        v109 = *(_QWORD *)v44;
        if ( *(_BYTE *)(*(_QWORD *)v44 + 8LL) == 16 )
          v109 = **(_QWORD **)(v109 + 16);
        HIDWORD(v162) = *(_DWORD *)(v109 + 8) >> 8;
      }
    }
    else
    {
      v45 = *(_DWORD *)(v42 + 20);
      LODWORD(v162) = 0;
      v161 = 0u;
      HIDWORD(v162) = v45;
    }
    v155 = sub_1D2B810(
             v39,
             v144,
             (__int64)&v157,
             v156.m128i_u32[0],
             v156.m128i_i64[1],
             v43,
             *(_OWORD *)&v8,
             v145,
             v149,
             v161,
             v162,
             v132,
             v135,
             v143,
             (__int64)&v159);
    *(_QWORD *)a4 = v155;
    *(_DWORD *)(a4 + 8) = v46;
    *((_QWORD *)&v121 + 1) = 1;
    *(_QWORD *)&v121 = v155;
    v151 = sub_1D332F0(
             *(__int64 **)(a1 + 8),
             2,
             (__int64)&v157,
             1,
             0,
             0,
             *(double *)v8.m128i_i64,
             *(double *)v9.m128i_i64,
             v13,
             *(_QWORD *)a3,
             1u,
             v121);
    sub_2013400(a1, a2, 1, (__int64)v151, (__m128i *)(v47 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL), v48);
    goto LABEL_34;
  }
  v57 = *(_BYTE *)(a2 + 88);
  v58 = *(_QWORD *)(a2 + 96);
  LOBYTE(v161) = v57;
  *((_QWORD *)&v161 + 1) = v58;
  if ( v57 )
    v59 = sub_2127930(v57);
  else
    v59 = sub_1F58D40((__int64)&v161);
  if ( v156.m128i_i8[0] )
    v60 = sub_2127930(v156.m128i_i8[0]);
  else
    v60 = sub_1F58D40((__int64)&v156);
  v61 = *(_QWORD *)(a1 + 8);
  v62 = v60 >> 3;
  v63 = 8 * (((unsigned int)(v59 + 7) >> 3) - (v60 >> 3));
  v64 = v59 - v63;
  if ( v64 == 32 )
  {
    LOBYTE(v65) = 5;
    goto LABEL_48;
  }
  if ( v64 <= 0x20 )
  {
    if ( v64 == 8 )
    {
      LOBYTE(v65) = 3;
    }
    else
    {
      LOBYTE(v65) = 4;
      if ( v64 != 16 )
      {
        LOBYTE(v65) = 2;
        if ( v64 != 1 )
          goto LABEL_74;
      }
    }
LABEL_48:
    v66 = 0;
    goto LABEL_49;
  }
  if ( v64 == 64 )
  {
    LOBYTE(v65) = 6;
    goto LABEL_48;
  }
  if ( v64 == 128 )
  {
    LOBYTE(v65) = 7;
    goto LABEL_48;
  }
LABEL_74:
  v134 = v62;
  v65 = sub_1F58CC0(*(_QWORD **)(v61 + 48), v64);
  v62 = v134;
  v125 = v65;
  v66 = v102;
LABEL_49:
  v67 = v125;
  LOBYTE(v67) = v65;
  LODWORD(v133) = v62;
  *(_QWORD *)a4 = sub_1D2B810(
                    (_QWORD *)v61,
                    v144,
                    (__int64)&v157,
                    v156.m128i_u32[0],
                    v156.m128i_i64[1],
                    v140,
                    *(_OWORD *)&v8,
                    v9.m128i_i64[0],
                    v9.m128i_i64[1],
                    *(_OWORD *)*(_QWORD *)(a2 + 104),
                    *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
                    v67,
                    v66,
                    v143,
                    (__int64)&v159);
  v68 = 16LL * v136;
  *(_DWORD *)(a4 + 8) = v69;
  v133 = (unsigned int)v133;
  v138 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v70 = sub_1D38BB0(
                      (__int64)v138,
                      (unsigned int)v133,
                      (__int64)&v157,
                      *(unsigned __int8 *)(v68 + *(_QWORD *)(v139 + 40)),
                      *(const void ***)(v68 + *(_QWORD *)(v139 + 40) + 8),
                      0,
                      v8,
                      *(double *)v9.m128i_i64,
                      v13,
                      0);
  v71 = sub_1D332F0(
          v138,
          52,
          (__int64)&v157,
          *(unsigned __int8 *)(*(_QWORD *)(v139 + 40) + v68),
          *(const void ***)(*(_QWORD *)(v139 + 40) + v68 + 8),
          0,
          *(double *)v8.m128i_i64,
          *(double *)v9.m128i_i64,
          v13,
          v9.m128i_i64[0],
          v9.m128i_u64[1],
          v70);
  v72 = (unsigned int)v133;
  v73 = *(_QWORD *)(a1 + 8);
  v147 = (__int64)v71;
  v150 = v74 | v9.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v75 = -(v133 | v140) & (v133 | v140);
  if ( v63 == 32 )
  {
    LOBYTE(v76) = 5;
    goto LABEL_53;
  }
  if ( v63 > 0x20 )
  {
    if ( v63 == 64 )
    {
      LOBYTE(v76) = 6;
      goto LABEL_53;
    }
    if ( v63 == 128 )
    {
      LOBYTE(v76) = 7;
      goto LABEL_53;
    }
LABEL_77:
    v76 = sub_1F58CC0(*(_QWORD **)(v73 + 48), v63);
    v72 = (unsigned int)v133;
    LODWORD(v75) = -(v133 | v140) & (v133 | v140);
    v124 = v76;
    goto LABEL_54;
  }
  if ( v63 == 8 )
  {
    LOBYTE(v76) = 3;
    goto LABEL_53;
  }
  LOBYTE(v76) = 4;
  if ( v63 != 16 )
    goto LABEL_77;
LABEL_53:
  v77 = 0;
LABEL_54:
  v78 = v124;
  LOBYTE(v78) = v76;
  v79 = *(_QWORD *)(a2 + 104);
  v80 = *(_QWORD *)v79 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v80 )
  {
    v104 = *(_QWORD *)(v79 + 8) + v72;
    v105 = *(_BYTE *)(v79 + 16);
    if ( (*(_QWORD *)v79 & 4) != 0 )
    {
      v163.m128i_i64[1] = v104;
      LOBYTE(v164) = v105;
      v163.m128i_i64[0] = v80 | 4;
      HIDWORD(v164) = *(_DWORD *)(v80 + 12);
    }
    else
    {
      v163.m128i_i64[0] = *(_QWORD *)v79 & 0xFFFFFFFFFFFFFFF8LL;
      v163.m128i_i64[1] = v104;
      LOBYTE(v164) = v105;
      v106 = *(_QWORD *)v80;
      if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16 )
        v106 = **(_QWORD **)(v106 + 16);
      HIDWORD(v164) = *(_DWORD *)(v106 + 8) >> 8;
    }
  }
  else
  {
    v81 = *(_DWORD *)(v79 + 20);
    LODWORD(v164) = 0;
    v163 = 0u;
    HIDWORD(v164) = v81;
  }
  v154 = sub_1D2B810(
           (_QWORD *)v73,
           3u,
           (__int64)&v157,
           v156.m128i_u32[0],
           v156.m128i_i64[1],
           v75,
           *(_OWORD *)&v8,
           v147,
           v150,
           *(_OWORD *)&v163,
           v164,
           v78,
           v77,
           v143,
           (__int64)&v159);
  *(_QWORD *)a3 = v154;
  *(_DWORD *)(a3 + 8) = v82;
  *((_QWORD *)&v122 + 1) = 1;
  *(_QWORD *)&v122 = *(_QWORD *)a4;
  v152 = sub_1D332F0(
           *(__int64 **)(a1 + 8),
           2,
           (__int64)&v157,
           1,
           0,
           0,
           *(double *)v8.m128i_i64,
           *(double *)v9.m128i_i64,
           v13,
           v154,
           1u,
           v122);
  v153 = (__m128i *)(v83 | v8.m128i_i64[1] & 0xFFFFFFFF00000000LL);
  if ( v156.m128i_i8[0] )
    v84 = sub_2127930(v156.m128i_i8[0]);
  else
    v84 = sub_1F58D40((__int64)&v156);
  if ( v63 < v84 )
  {
    v85 = *(__int64 **)(a1 + 8);
    v86 = sub_1E0A0C0(v85[4]);
    v87 = 8 * sub_15A9520(v86, 0);
    if ( v87 == 32 )
    {
      v88 = 5;
    }
    else if ( v87 > 0x20 )
    {
      v88 = 6;
      if ( v87 != 64 )
      {
        v88 = 0;
        if ( v87 == 128 )
          v88 = 7;
      }
    }
    else
    {
      v88 = 3;
      if ( v87 != 8 )
        v88 = 4 * (v87 == 16);
    }
    *(_QWORD *)&v89 = sub_1D38BB0((__int64)v85, v63, (__int64)&v157, v88, 0, 0, v8, *(double *)v9.m128i_i64, v13, 0);
    *(_QWORD *)&v90 = sub_1D332F0(
                        v85,
                        122,
                        (__int64)&v157,
                        v156.m128i_u32[0],
                        (const void **)v156.m128i_i64[1],
                        0,
                        *(double *)v8.m128i_i64,
                        *(double *)v9.m128i_i64,
                        v13,
                        *(_QWORD *)a4,
                        *(_QWORD *)(a4 + 8),
                        v89);
    *(_QWORD *)a3 = sub_1D332F0(
                      v85,
                      119,
                      (__int64)&v157,
                      v156.m128i_u32[0],
                      (const void **)v156.m128i_i64[1],
                      0,
                      *(double *)v8.m128i_i64,
                      *(double *)v9.m128i_i64,
                      v13,
                      *(_QWORD *)a3,
                      *(_QWORD *)(a3 + 8),
                      v90);
    *(_DWORD *)(a3 + 8) = v91;
    v92 = *(__int64 **)(a1 + 8);
    v93 = sub_1E0A0C0(v92[4]);
    v94 = 8 * sub_15A9520(v93, 0);
    if ( v94 == 32 )
    {
      v95 = 5;
    }
    else if ( v94 > 0x20 )
    {
      v95 = 6;
      if ( v94 != 64 )
      {
        v95 = 0;
        if ( v94 == 128 )
          v95 = 7;
      }
    }
    else
    {
      v95 = 3;
      if ( v94 != 8 )
        v95 = 4 * (v94 == 16);
    }
    if ( v156.m128i_i8[0] )
    {
      v98 = sub_2127930(v156.m128i_i8[0]);
    }
    else
    {
      v148 = v95;
      v96 = sub_1F58D40((__int64)&v156);
      v97 = v148;
      v98 = v96;
    }
    *(_QWORD *)&v99 = sub_1D38BB0(
                        (__int64)v92,
                        v98 - v63,
                        (__int64)&v157,
                        v97,
                        0,
                        0,
                        v8,
                        *(double *)v9.m128i_i64,
                        v13,
                        0);
    *(_QWORD *)a4 = sub_1D332F0(
                      v92,
                      (unsigned int)(v144 != 2) + 123,
                      (__int64)&v157,
                      v156.m128i_u32[0],
                      (const void **)v156.m128i_i64[1],
                      0,
                      *(double *)v8.m128i_i64,
                      *(double *)v9.m128i_i64,
                      v13,
                      *(_QWORD *)a4,
                      *(_QWORD *)(a4 + 8),
                      v99);
    *(_DWORD *)(a4 + 8) = v100;
  }
LABEL_33:
  sub_2013400(a1, a2, 1, (__int64)v152, v153, v55);
LABEL_34:
  if ( v157 )
    sub_161E7C0((__int64)&v157, v157);
}
