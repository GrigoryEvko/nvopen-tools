// Function: sub_1FECF20
// Address: 0x1fecf20
//
__int64 __fastcall sub_1FECF20(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  const __m128i *v8; // rax
  __m128 v9; // xmm0
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r14
  unsigned __int8 *v13; // rax
  const void **v14; // rbx
  unsigned int v15; // r15d
  __int128 v16; // rax
  __int128 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int8 v20; // r14
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int8 v23; // cl
  unsigned int v24; // eax
  unsigned int v25; // r14d
  __int64 v26; // rsi
  unsigned __int64 v27; // rdx
  __int128 v28; // rax
  __int128 v29; // rax
  unsigned int v30; // r11d
  __int64 v31; // rbx
  char v32; // di
  unsigned int v33; // eax
  unsigned int v34; // r11d
  unsigned int v35; // r15d
  __int64 v36; // rax
  char v37; // di
  __int64 v38; // rax
  unsigned int v39; // r11d
  int v40; // eax
  __int64 *v41; // r15
  bool v42; // zf
  __int64 v43; // rax
  char v44; // di
  __int64 v45; // rax
  __int64 v46; // rax
  char v47; // di
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r14
  char v53; // dl
  __int64 v54; // rdx
  __int128 v55; // rax
  __int128 v56; // rax
  __int128 v57; // rax
  __int64 v58; // r15
  __int64 v59; // rax
  unsigned __int8 v60; // al
  __int64 v61; // rdx
  const void **v62; // rbx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rax
  const void **v67; // r8
  unsigned int v68; // ebx
  __int64 v69; // rdx
  __int64 *v70; // r14
  __int64 *v71; // r12
  unsigned int v72; // edx
  __int16 *v73; // rcx
  __int64 v74; // rdx
  const void **v75; // r10
  char v76; // al
  __int64 v77; // rdx
  __int16 *v78; // r15
  bool v79; // al
  unsigned int v80; // esi
  char v81; // si
  unsigned int v82; // eax
  __int64 v83; // rax
  unsigned int v84; // edx
  __int128 v85; // rax
  __int64 *v86; // rax
  unsigned int v87; // edx
  __int64 v88; // rax
  unsigned int v89; // edx
  __int128 v90; // rax
  __int128 v91; // [rsp-10h] [rbp-270h]
  __int128 v92; // [rsp-10h] [rbp-270h]
  unsigned int v93; // [rsp+8h] [rbp-258h]
  int v94; // [rsp+14h] [rbp-24Ch]
  int v95; // [rsp+18h] [rbp-248h]
  unsigned __int8 v96; // [rsp+1Fh] [rbp-241h]
  __int64 v97; // [rsp+28h] [rbp-238h]
  const void **v98; // [rsp+28h] [rbp-238h]
  int v99; // [rsp+30h] [rbp-230h]
  __int64 *v100; // [rsp+30h] [rbp-230h]
  unsigned int v101; // [rsp+30h] [rbp-230h]
  unsigned int v102; // [rsp+30h] [rbp-230h]
  unsigned int v103; // [rsp+38h] [rbp-228h]
  unsigned int v104; // [rsp+38h] [rbp-228h]
  __int64 v105; // [rsp+38h] [rbp-228h]
  unsigned __int8 v106; // [rsp+40h] [rbp-220h]
  const void **v107; // [rsp+40h] [rbp-220h]
  unsigned int v108; // [rsp+40h] [rbp-220h]
  __int64 (__fastcall *v109)(__int64, __int64, __int64, __int64, const void **); // [rsp+40h] [rbp-220h]
  unsigned int v110; // [rsp+40h] [rbp-220h]
  __int64 v111; // [rsp+48h] [rbp-218h]
  __int128 v112; // [rsp+50h] [rbp-210h]
  __int128 v113; // [rsp+50h] [rbp-210h]
  __int128 v114; // [rsp+50h] [rbp-210h]
  __int128 v115; // [rsp+50h] [rbp-210h]
  unsigned __int32 v116; // [rsp+60h] [rbp-200h]
  const void **v117; // [rsp+60h] [rbp-200h]
  __int128 v118; // [rsp+60h] [rbp-200h]
  __int64 v119; // [rsp+70h] [rbp-1F0h]
  const void **v120; // [rsp+70h] [rbp-1F0h]
  __int128 v121; // [rsp+70h] [rbp-1F0h]
  __int128 v122; // [rsp+80h] [rbp-1E0h]
  unsigned __int64 v123; // [rsp+88h] [rbp-1D8h]
  __int64 v124; // [rsp+C0h] [rbp-1A0h] BYREF
  int v125; // [rsp+C8h] [rbp-198h]
  unsigned __int64 v126; // [rsp+D0h] [rbp-190h] BYREF
  unsigned int v127; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v128; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v129; // [rsp+E8h] [rbp-178h]
  _BYTE v130[8]; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v131; // [rsp+F8h] [rbp-168h]
  __int64 v132; // [rsp+100h] [rbp-160h]
  int v133; // [rsp+108h] [rbp-158h]
  __int64 v134; // [rsp+110h] [rbp-150h]
  int v135; // [rsp+118h] [rbp-148h]
  __int64 v136; // [rsp+120h] [rbp-140h]
  int v137; // [rsp+128h] [rbp-138h]
  __int64 v138; // [rsp+130h] [rbp-130h]
  __int64 v139; // [rsp+138h] [rbp-128h]
  __int64 v140; // [rsp+140h] [rbp-120h]
  __int64 v141; // [rsp+148h] [rbp-118h]
  __int64 v142; // [rsp+150h] [rbp-110h]
  __int64 v143; // [rsp+158h] [rbp-108h]
  __int64 v144; // [rsp+160h] [rbp-100h]
  unsigned __int64 v145; // [rsp+168h] [rbp-F8h]
  __int64 v146; // [rsp+170h] [rbp-F0h] BYREF
  unsigned int v147; // [rsp+178h] [rbp-E8h]
  unsigned __int8 v148; // [rsp+180h] [rbp-E0h]
  unsigned int v149; // [rsp+190h] [rbp-D0h] BYREF
  __int64 v150; // [rsp+198h] [rbp-C8h]
  __int64 v151; // [rsp+1A0h] [rbp-C0h]
  int v152; // [rsp+1A8h] [rbp-B8h]
  __int64 v153; // [rsp+1B0h] [rbp-B0h]
  int v154; // [rsp+1B8h] [rbp-A8h]
  __int64 v155; // [rsp+1C0h] [rbp-A0h]
  int v156; // [rsp+1C8h] [rbp-98h]
  __int64 v157; // [rsp+1D0h] [rbp-90h]
  __int64 v158; // [rsp+1D8h] [rbp-88h]
  __int64 v159; // [rsp+1E0h] [rbp-80h]
  __int64 v160; // [rsp+1E8h] [rbp-78h]
  __int64 v161; // [rsp+1F0h] [rbp-70h]
  __int64 v162; // [rsp+1F8h] [rbp-68h]
  __int64 v163; // [rsp+200h] [rbp-60h]
  unsigned __int64 v164; // [rsp+208h] [rbp-58h]
  __int64 v165; // [rsp+210h] [rbp-50h] BYREF
  unsigned int v166; // [rsp+218h] [rbp-48h]
  unsigned __int8 v167; // [rsp+220h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 72);
  v124 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v124, v7, 2);
  v125 = *(_DWORD *)(a2 + 64);
  v8 = *(const __m128i **)(a2 + 32);
  v9 = (__m128)_mm_loadu_si128(v8);
  v10 = v8[2].m128i_i64[1];
  v11 = v8[3].m128i_i64[0];
  v12 = v8->m128i_i64[0];
  v116 = v8->m128i_u32[2];
  v130[0] = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  LODWORD(v145) = 0;
  v147 = 1;
  v146 = 0;
  sub_1FEBFF0(a1, (__int64)v130, (__int64)&v124, v10, v11, *(double *)v9.m128_u64, a4, a5);
  v13 = (unsigned __int8 *)(*(_QWORD *)(v144 + 40) + 16LL * (unsigned int)v145);
  v14 = (const void **)*((_QWORD *)v13 + 1);
  v106 = *v13;
  v15 = *v13;
  *(_QWORD *)&v16 = sub_1D38970(
                      *(_QWORD *)(a1 + 16),
                      (__int64)&v146,
                      (__int64)&v124,
                      v15,
                      v14,
                      0,
                      (__m128i)v9,
                      a4,
                      a5,
                      0);
  *(_QWORD *)&v17 = sub_1D332F0(
                      *(__int64 **)(a1 + 16),
                      118,
                      (__int64)&v124,
                      v15,
                      v14,
                      0,
                      *(double *)v9.m128_u64,
                      a4,
                      a5,
                      v144,
                      v145,
                      v16);
  v18 = *(_QWORD *)(a1 + 8);
  v122 = v17;
  v119 = v17;
  v19 = *(_QWORD *)(v12 + 40) + 16LL * v116;
  v103 = DWORD2(v17);
  v20 = *(_BYTE *)v19;
  if ( *(_BYTE *)v19 == 1 )
  {
    v53 = *(_BYTE *)(v18 + 2844);
    if ( v53 && v53 != 4 )
      goto LABEL_5;
    v54 = 1;
  }
  else
  {
    if ( !v20 )
      goto LABEL_5;
    v54 = v20;
    if ( !*(_QWORD *)(v18 + 8LL * v20 + 120) )
      goto LABEL_5;
    v81 = *(_BYTE *)(v18 + 259LL * v20 + 2585);
    if ( v81 )
    {
      if ( v81 != 4 || !*(_QWORD *)(v18 + 8 * (v20 + 14LL) + 8) )
        goto LABEL_5;
    }
  }
  if ( (*(_BYTE *)(v18 + 259 * v54 + 2584) & 0xFB) == 0 )
  {
    v120 = *(const void ***)(v19 + 8);
    *(_QWORD *)&v55 = sub_1D309E0(
                        *(__int64 **)(a1 + 16),
                        163,
                        (__int64)&v124,
                        v20,
                        v120,
                        0,
                        *(double *)v9.m128_u64,
                        a4,
                        *(double *)a5.m128i_i64,
                        *(_OWORD *)&v9);
    v98 = v120;
    v118 = v55;
    *(_QWORD *)&v56 = sub_1D309E0(
                        *(__int64 **)(a1 + 16),
                        162,
                        (__int64)&v124,
                        v20,
                        v120,
                        0,
                        *(double *)v9.m128_u64,
                        a4,
                        *(double *)a5.m128i_i64,
                        v55);
    v100 = *(__int64 **)(a1 + 16);
    v115 = v56;
    *(_QWORD *)&v57 = sub_1D38BB0((__int64)v100, 0, (__int64)&v124, v15, v14, 0, (__m128i)v9, a4, a5, 0);
    v58 = *(_QWORD *)(a1 + 8);
    v121 = v57;
    *(_QWORD *)&v57 = *(_QWORD *)(a1 + 16);
    v105 = v106;
    v109 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(*(_QWORD *)v58 + 264LL);
    v111 = *(_QWORD *)(v57 + 48);
    v59 = sub_1E0A0C0(*(_QWORD *)(v57 + 32));
    v60 = v109(v58, v59, v111, v105, v14);
    v62 = (const void **)v61;
    LODWORD(v58) = v60;
    v66 = sub_1D28D50(v100, 0x16u, v61, v63, v64, v65);
    v67 = v62;
    v68 = v20;
    v70 = sub_1D3A900(
            v100,
            0x89u,
            (__int64)&v124,
            (unsigned int)v58,
            v67,
            0,
            v9,
            a4,
            a5,
            v122,
            *((__int16 **)&v122 + 1),
            v121,
            v66,
            v69);
    v71 = *(__int64 **)(a1 + 16);
    v73 = (__int16 *)v72;
    v74 = v70[5] + 16LL * v72;
    v75 = v98;
    v76 = *(_BYTE *)v74;
    v77 = *(_QWORD *)(v74 + 8);
    v78 = v73;
    LOBYTE(v149) = v76;
    v150 = v77;
    if ( v76 )
    {
      v80 = ((unsigned __int8)(v76 - 14) < 0x60u) + 134;
    }
    else
    {
      v79 = sub_1F58D20((__int64)&v149);
      v75 = v98;
      v80 = 134 - (!v79 - 1);
    }
    v51 = (__int64)sub_1D3A900(
                     v71,
                     v80,
                     (__int64)&v124,
                     v68,
                     v75,
                     0,
                     v9,
                     a4,
                     a5,
                     (unsigned __int64)v70,
                     v78,
                     v115,
                     v118,
                     *((__int64 *)&v118 + 1));
    goto LABEL_27;
  }
LABEL_5:
  LOBYTE(v149) = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  LODWORD(v164) = 0;
  v166 = 1;
  v165 = 0;
  sub_1FEBFF0(a1, (__int64)&v149, (__int64)&v124, v9.m128_i64[0], v9.m128_i64[1], *(double *)v9.m128_u64, a4, a5);
  v21 = *(_QWORD *)(a1 + 16);
  v22 = *(_QWORD *)(v163 + 40) + 16LL * (unsigned int)v164;
  v23 = *(_BYTE *)v22;
  v117 = *(const void ***)(v22 + 8);
  v24 = v166;
  v25 = v23;
  v96 = v23;
  v127 = v166;
  if ( v166 > 0x40 )
  {
    sub_16A4FD0((__int64)&v126, (const void **)&v165);
    v24 = v127;
    if ( v127 > 0x40 )
    {
      sub_16A8F40((__int64 *)&v126);
      v24 = v127;
      v27 = v126;
      goto LABEL_8;
    }
    v26 = v126;
  }
  else
  {
    v26 = v165;
  }
  v27 = ~v26 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
  v126 = v27;
LABEL_8:
  LODWORD(v129) = v24;
  v128 = v27;
  v127 = 0;
  *(_QWORD *)&v28 = sub_1D38970(v21, (__int64)&v128, (__int64)&v124, v25, v117, 0, (__m128i)v9, a4, a5, 0);
  if ( (unsigned int)v129 > 0x40 && v128 )
  {
    v112 = v28;
    j_j___libc_free_0_0(v128);
    v28 = v112;
  }
  if ( v127 > 0x40 && v126 )
  {
    v113 = v28;
    j_j___libc_free_0_0(v126);
    v28 = v113;
  }
  *(_QWORD *)&v29 = sub_1D332F0(
                      *(__int64 **)(a1 + 16),
                      118,
                      (__int64)&v124,
                      v25,
                      v117,
                      0,
                      *(double *)v9.m128_u64,
                      a4,
                      a5,
                      v163,
                      v164,
                      v28);
  v114 = v29;
  v30 = v106;
  v107 = v14;
  v31 = v103;
  v95 = v148;
  v99 = v148 - v167;
  *(_QWORD *)&v29 = *(_QWORD *)(v119 + 40) + 16LL * v103;
  v94 = v167;
  v32 = *(_BYTE *)v29;
  *(_QWORD *)&v29 = *(_QWORD *)(v29 + 8);
  LOBYTE(v128) = v32;
  v129 = v29;
  if ( v32 )
  {
    v35 = sub_1FEB8F0(v32);
  }
  else
  {
    v104 = v30;
    v33 = sub_1F58D40((__int64)&v128);
    v34 = v104;
    v35 = v33;
  }
  v97 = 16LL * DWORD2(v114);
  v36 = *(_QWORD *)(v114 + 40) + v97;
  v37 = *(_BYTE *)v36;
  v38 = *(_QWORD *)(v36 + 8);
  LOBYTE(v128) = v37;
  v129 = v38;
  if ( v37 )
  {
    if ( v35 >= (unsigned int)sub_1FEB8F0(v37) )
      goto LABEL_18;
LABEL_49:
    v110 = v39;
    v83 = sub_1D309E0(
            *(__int64 **)(a1 + 16),
            143,
            (__int64)&v124,
            v25,
            v117,
            0,
            *(double *)v9.m128_u64,
            a4,
            *(double *)a5.m128i_i64,
            v122);
    v39 = v110;
    LOBYTE(v39) = v96;
    v119 = v83;
    v40 = v99;
    v41 = *(__int64 **)(a1 + 16);
    v107 = v117;
    v31 = v84;
    *((_QWORD *)&v122 + 1) = v84 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
    v42 = v99 == 0;
    if ( v99 <= 0 )
      goto LABEL_19;
LABEL_50:
    v101 = v39;
    *(_QWORD *)&v85 = sub_1D38BB0((__int64)v41, v40, (__int64)&v124, v39, v107, 0, (__m128i)v9, a4, a5, 0);
    v123 = v31 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
    v86 = sub_1D332F0(
            *(__int64 **)(a1 + 16),
            124,
            (__int64)&v124,
            v101,
            v107,
            0,
            *(double *)v9.m128_u64,
            a4,
            a5,
            v119,
            v123,
            v85);
    goto LABEL_51;
  }
  v93 = v34;
  v82 = sub_1F58D40((__int64)&v128);
  v39 = v93;
  if ( v35 < v82 )
    goto LABEL_49;
LABEL_18:
  v40 = v99;
  v41 = *(__int64 **)(a1 + 16);
  v42 = v99 == 0;
  if ( v99 > 0 )
    goto LABEL_50;
LABEL_19:
  if ( v42 )
    goto LABEL_20;
  v102 = v39;
  *(_QWORD *)&v90 = sub_1D38BB0((__int64)v41, v94 - v95, (__int64)&v124, v39, v107, 0, (__m128i)v9, a4, a5, 0);
  v123 = v31 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
  v86 = sub_1D332F0(
          *(__int64 **)(a1 + 16),
          122,
          (__int64)&v124,
          v102,
          v107,
          0,
          *(double *)v9.m128_u64,
          a4,
          a5,
          v119,
          v123,
          v90);
LABEL_51:
  v31 = v87;
  v119 = (__int64)v86;
  *((_QWORD *)&v122 + 1) = v87 | v123 & 0xFFFFFFFF00000000LL;
  v41 = *(__int64 **)(a1 + 16);
LABEL_20:
  v43 = *(_QWORD *)(v119 + 40) + 16 * v31;
  v44 = *(_BYTE *)v43;
  v45 = *(_QWORD *)(v43 + 8);
  LOBYTE(v128) = v44;
  v129 = v45;
  if ( v44 )
    v108 = sub_1FEB8F0(v44);
  else
    v108 = sub_1F58D40((__int64)&v128);
  v46 = *(_QWORD *)(v114 + 40) + v97;
  v47 = *(_BYTE *)v46;
  v48 = *(_QWORD *)(v46 + 8);
  LOBYTE(v128) = v47;
  v129 = v48;
  if ( !v47 )
  {
    if ( v108 <= (unsigned int)sub_1F58D40((__int64)&v128) )
      goto LABEL_24;
LABEL_53:
    *((_QWORD *)&v122 + 1) = v31 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v92 + 1) = *((_QWORD *)&v122 + 1);
    *(_QWORD *)&v92 = v119;
    v88 = sub_1D309E0(v41, 145, (__int64)&v124, v25, v117, 0, *(double *)v9.m128_u64, a4, *(double *)a5.m128i_i64, v92);
    v41 = *(__int64 **)(a1 + 16);
    v119 = v88;
    v31 = v89;
    goto LABEL_24;
  }
  if ( v108 > (unsigned int)sub_1FEB8F0(v47) )
    goto LABEL_53;
LABEL_24:
  *((_QWORD *)&v91 + 1) = v31 | *((_QWORD *)&v122 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v91 = v119;
  v49 = sub_1D332F0(
          v41,
          119,
          (__int64)&v124,
          v25,
          v117,
          0,
          *(double *)v9.m128_u64,
          a4,
          a5,
          v114,
          *((unsigned __int64 *)&v114 + 1),
          v91);
  v51 = sub_1FEB540(a1, &v149, (__int64)&v124, (__int64)v49, v50, *(double *)v9.m128_u64, a4, *(double *)a5.m128i_i64);
  if ( v166 > 0x40 && v165 )
    j_j___libc_free_0_0(v165);
LABEL_27:
  if ( v147 > 0x40 && v146 )
    j_j___libc_free_0_0(v146);
  if ( v124 )
    sub_161E7C0((__int64)&v124, v124);
  return v51;
}
