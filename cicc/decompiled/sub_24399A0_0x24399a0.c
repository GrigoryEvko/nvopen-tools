// Function: sub_24399A0
// Address: 0x24399a0
//
void __fastcall sub_24399A0(__int64 a1, __int64 a2, unsigned __int8 a3, __int16 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int8 *v13; // r14
  unsigned __int8 *v14; // r10
  __int64 (__fastcall *v15)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v19; // r14
  unsigned __int8 *v20; // r15
  __int64 (__fastcall *v21)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v22; // r9
  __int64 v23; // r14
  __int64 v24; // r15
  __int64 v25; // rax
  unsigned __int8 *v26; // r14
  unsigned __int8 *v27; // r10
  __int64 (__fastcall *v28)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 **v31; // r14
  __int64 (__fastcall *v32)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v33; // r10
  __int64 v34; // r14
  __int64 v35; // rax
  char v36; // al
  _QWORD *v37; // rax
  __int64 v38; // r9
  _BYTE *v39; // r15
  __int64 v40; // r14
  unsigned int *v41; // r14
  unsigned int *v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // r14
  __int64 v46; // r15
  __int64 v47; // rax
  unsigned int v48; // eax
  __int8 *v49; // rsi
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rax
  __m128i *v52; // rax
  __int64 *v53; // rdi
  __int64 v54; // r15
  _QWORD **v55; // rax
  __int64 v56; // r15
  unsigned __int64 v57; // rsi
  __int64 v58; // r14
  unsigned int *v59; // r14
  unsigned int *v60; // rbx
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int8 *v63; // rsi
  unsigned __int64 v64; // rcx
  unsigned __int64 v65; // rax
  __m128i *v66; // rax
  __m128i *v67; // rax
  __int64 *v68; // rdi
  __int64 v69; // r15
  _QWORD **v70; // rax
  __int64 v71; // r14
  unsigned int *v72; // r14
  unsigned int *v73; // rbx
  __int64 v74; // rdx
  unsigned int v75; // esi
  unsigned int *v76; // r15
  unsigned int *v77; // r14
  __int64 v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  __int8 *v81; // rsi
  unsigned __int64 v82; // rcx
  unsigned __int64 v83; // rax
  __m128i *v84; // rax
  __int64 *v85; // rdi
  __int64 v86; // r15
  __int64 v87; // r14
  _QWORD **v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // r15
  unsigned int *v93; // r15
  unsigned int *v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // esi
  char v97; // al
  __int64 v98; // r10
  int v99; // r14d
  unsigned int *v100; // rbx
  unsigned int *v101; // r15
  __int64 v102; // r14
  __int64 v103; // rdx
  unsigned int v104; // esi
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // [rsp-10h] [rbp-1F0h]
  unsigned __int16 v108; // [rsp+4h] [rbp-1DCh]
  __int64 v109; // [rsp+20h] [rbp-1C0h]
  __int64 **v110; // [rsp+28h] [rbp-1B8h]
  char v111; // [rsp+28h] [rbp-1B8h]
  __int64 v112; // [rsp+28h] [rbp-1B8h]
  char v113; // [rsp+30h] [rbp-1B0h]
  unsigned __int8 *v114; // [rsp+30h] [rbp-1B0h]
  __int64 v115; // [rsp+30h] [rbp-1B0h]
  __int64 v116; // [rsp+30h] [rbp-1B0h]
  __int64 v117; // [rsp+30h] [rbp-1B0h]
  __int64 v118; // [rsp+30h] [rbp-1B0h]
  __int64 v119; // [rsp+30h] [rbp-1B0h]
  __int64 v120; // [rsp+30h] [rbp-1B0h]
  __int64 v121; // [rsp+30h] [rbp-1B0h]
  __int64 v122; // [rsp+30h] [rbp-1B0h]
  unsigned __int8 *v123; // [rsp+30h] [rbp-1B0h]
  __int64 v125; // [rsp+38h] [rbp-1A8h]
  __int64 v126; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v127; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v128; // [rsp+48h] [rbp-198h]
  __int64 v129; // [rsp+48h] [rbp-198h]
  unsigned __int8 *v130; // [rsp+48h] [rbp-198h]
  __int64 v131; // [rsp+58h] [rbp-188h] BYREF
  __int64 v132; // [rsp+60h] [rbp-180h] BYREF
  __int64 v133; // [rsp+68h] [rbp-178h] BYREF
  __int64 v134; // [rsp+70h] [rbp-170h]
  __int64 v135; // [rsp+78h] [rbp-168h]
  _BYTE *v136; // [rsp+80h] [rbp-160h]
  __int64 v137[2]; // [rsp+90h] [rbp-150h] BYREF
  char v138; // [rsp+A0h] [rbp-140h] BYREF
  __int16 v139; // [rsp+B0h] [rbp-130h]
  __m128i *v140; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v141; // [rsp+C8h] [rbp-118h]
  __m128i v142; // [rsp+D0h] [rbp-110h] BYREF
  __int16 v143; // [rsp+E0h] [rbp-100h]
  __m128i *v144; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v145; // [rsp+F8h] [rbp-E8h]
  __m128i v146; // [rsp+100h] [rbp-E0h] BYREF
  __int16 v147; // [rsp+110h] [rbp-D0h]
  unsigned int *v148; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v149; // [rsp+128h] [rbp-B8h]
  _BYTE v150[32]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v151; // [rsp+150h] [rbp-90h]
  __int64 v152; // [rsp+158h] [rbp-88h]
  __int64 v153; // [rsp+160h] [rbp-80h]
  __int64 v154; // [rsp+168h] [rbp-78h]
  void **v155; // [rsp+170h] [rbp-70h]
  void **v156; // [rsp+178h] [rbp-68h]
  __int64 v157; // [rsp+180h] [rbp-60h]
  int v158; // [rsp+188h] [rbp-58h]
  __int16 v159; // [rsp+18Ch] [rbp-54h]
  char v160; // [rsp+18Eh] [rbp-52h]
  __int64 v161; // [rsp+190h] [rbp-50h]
  __int64 v162; // [rsp+198h] [rbp-48h]
  void *v163; // [rsp+1A0h] [rbp-40h] BYREF
  void *v164; // [rsp+1A8h] [rbp-38h] BYREF

  v7 = a1;
  v113 = a4;
  v108 = a4 | (16 * a3) | (32 * *(unsigned __int8 *)(a1 + 161));
  sub_2438890((unsigned __int64 *)&v132, a1, a2, a5, a6, a7);
  v8 = v132;
  v154 = sub_BD5C60(v132);
  v155 = &v163;
  v156 = &v164;
  v148 = (unsigned int *)v150;
  v163 = &unk_49DA100;
  v159 = 512;
  LOWORD(v153) = 0;
  v149 = 0x200000000LL;
  v164 = &unk_49DA0B0;
  v157 = 0;
  v158 = 0;
  v160 = 7;
  v161 = 0;
  v162 = 0;
  v151 = 0;
  v152 = 0;
  sub_D5F1F0((__int64)&v148, v8);
  v9 = *(_QWORD *)(a1 + 136);
  v147 = 257;
  v10 = (_BYTE *)sub_AD64C0(v9, 15, 0);
  v11 = sub_92B530(&v148, 0x22u, (__int64)v136, v10, (__int64)&v144);
  v144 = *(__m128i **)v7;
  v12 = sub_B8C340(&v144);
  v127 = sub_F38250(v11, (__int64 *)(v132 + 24), 0, *(_BYTE *)(v7 + 161) ^ 1u, v12, a6, a7, 0);
  sub_D5F1F0((__int64)&v148, v132);
  v143 = 257;
  v13 = (unsigned __int8 *)v133;
  v139 = 257;
  v110 = *(__int64 ***)(v7 + 136);
  v14 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v133 + 8), 15, 0);
  v15 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v155 + 2);
  if ( v15 == sub_9202E0 )
  {
    if ( *v13 > 0x15u || *v14 > 0x15u )
      goto LABEL_55;
    v128 = v14;
    if ( (unsigned __int8)sub_AC47B0(28) )
      v16 = sub_AD5570(28, (__int64)v13, v128, 0, 0);
    else
      v16 = sub_AABE40(0x1Cu, v13, v128);
    v14 = v128;
    v17 = v16;
  }
  else
  {
    v130 = v14;
    v105 = v15((__int64)v155, 28u, v13, v14);
    v14 = v130;
    v17 = v105;
  }
  if ( v17 )
    goto LABEL_8;
LABEL_55:
  v147 = 257;
  v17 = sub_B504D0(28, (__int64)v13, (__int64)v14, (__int64)&v144, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v156 + 2))(v156, v17, v137, v152, v153);
  v58 = 4LL * (unsigned int)v149;
  if ( v148 == &v148[v58] )
  {
LABEL_8:
    if ( v110 != *(__int64 ***)(v17 + 8) )
      goto LABEL_9;
LABEL_59:
    v19 = (unsigned __int8 *)v17;
    goto LABEL_14;
  }
  v129 = v7;
  v59 = &v148[v58];
  v60 = v148;
  do
  {
    v61 = *((_QWORD *)v60 + 1);
    v62 = *v60;
    v60 += 4;
    sub_B99FD0(v17, v62, v61);
  }
  while ( v59 != v60 );
  v7 = v129;
  if ( v110 == *(__int64 ***)(v17 + 8) )
    goto LABEL_59;
LABEL_9:
  v18 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v155 + 15);
  if ( v18 != sub_920130 )
  {
    v19 = (unsigned __int8 *)v18((__int64)v155, 38u, (_BYTE *)v17, (__int64)v110);
    goto LABEL_13;
  }
  if ( *(_BYTE *)v17 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v19 = (unsigned __int8 *)sub_ADAB70(38, v17, v110, 0);
    else
      v19 = (unsigned __int8 *)sub_AA93C0(0x26u, v17, (__int64)v110);
LABEL_13:
    if ( v19 )
      goto LABEL_14;
  }
  v147 = 257;
  v19 = (unsigned __int8 *)sub_B51D30(38, v17, (__int64)v110, (__int64)&v144, 0, 0);
  (*((void (__fastcall **)(void **, unsigned __int8 *, __m128i **, __int64, __int64))*v156 + 2))(
    v156,
    v19,
    &v140,
    v152,
    v153);
  v92 = 4LL * (unsigned int)v149;
  if ( v148 != &v148[v92] )
  {
    v112 = v7;
    v93 = &v148[v92];
    v94 = v148;
    do
    {
      v95 = *((_QWORD *)v94 + 1);
      v96 = *v94;
      v94 += 4;
      sub_B99FD0((__int64)v19, v96, v95);
    }
    while ( v93 != v94 );
    v7 = v112;
  }
LABEL_14:
  v143 = 257;
  v20 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v7 + 136), (1 << v113) - 1, 0);
  v21 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v155 + 4);
  if ( v21 != sub_9201A0 )
  {
    v22 = v21((__int64)v155, 13u, v19, v20, 0, 0);
    goto LABEL_19;
  }
  if ( *v19 <= 0x15u && *v20 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
      v22 = sub_AD5570(13, (__int64)v19, v20, 0, 0);
    else
      v22 = sub_AABE40(0xDu, v19, v20);
LABEL_19:
    if ( v22 )
      goto LABEL_20;
  }
  v147 = 257;
  v117 = sub_B504D0(13, (__int64)v19, (__int64)v20, (__int64)&v144, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i **, __int64, __int64))*v156 + 2))(v156, v117, &v140, v152, v153);
  v76 = v148;
  v22 = v117;
  v77 = &v148[4 * (unsigned int)v149];
  if ( v148 != v77 )
  {
    v118 = v7;
    v78 = v22;
    do
    {
      v79 = *((_QWORD *)v76 + 1);
      v80 = *v76;
      v76 += 4;
      sub_B99FD0(v78, v80, v79);
    }
    while ( v77 != v76 );
    v22 = v78;
    v7 = v118;
  }
LABEL_20:
  v147 = 257;
  v23 = sub_92B530(&v148, 0x23u, v22, v136, (__int64)&v144);
  v24 = *(_QWORD *)(v127 + 40);
  v144 = *(__m128i **)v7;
  v25 = sub_B8C340(&v144);
  sub_F38250(v23, (__int64 *)(v132 + 24), 0, 0, v25, a6, a7, v24);
  sub_D5F1F0((__int64)&v148, v132);
  v26 = (unsigned __int8 *)v134;
  v143 = 257;
  v27 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v134 + 8), 15, 0);
  v28 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v155 + 2);
  if ( v28 != sub_9202E0 )
  {
    v123 = v27;
    v106 = v28((__int64)v155, 29u, v26, v27);
    v27 = v123;
    v30 = v106;
    goto LABEL_26;
  }
  if ( *v26 <= 0x15u && *v27 <= 0x15u )
  {
    v114 = v27;
    if ( (unsigned __int8)sub_AC47B0(29) )
      v29 = sub_AD5570(29, (__int64)v26, v114, 0, 0);
    else
      v29 = sub_AABE40(0x1Du, v26, v114);
    v27 = v114;
    v30 = v29;
LABEL_26:
    if ( v30 )
      goto LABEL_27;
  }
  v147 = 257;
  v30 = sub_B504D0(29, (__int64)v26, (__int64)v27, (__int64)&v144, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __m128i **, __int64, __int64))*v156 + 2))(v156, v30, &v140, v152, v153);
  v71 = 4LL * (unsigned int)v149;
  if ( v148 != &v148[v71] )
  {
    v116 = v7;
    v72 = &v148[v71];
    v73 = v148;
    do
    {
      v74 = *((_QWORD *)v73 + 1);
      v75 = *v73;
      v73 += 4;
      sub_B99FD0(v30, v75, v74);
    }
    while ( v72 != v73 );
    v7 = v116;
  }
LABEL_27:
  v31 = *(__int64 ***)(v7 + 128);
  v143 = 257;
  if ( v31 == *(__int64 ***)(v30 + 8) )
  {
    v33 = v30;
    goto LABEL_33;
  }
  v32 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v155 + 15);
  if ( v32 != sub_920130 )
  {
    v33 = v32((__int64)v155, 48u, (_BYTE *)v30, (__int64)v31);
    goto LABEL_32;
  }
  if ( *(_BYTE *)v30 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x30u) )
      v33 = sub_ADAB70(48, v30, v31, 0);
    else
      v33 = sub_AA93C0(0x30u, v30, (__int64)v31);
LABEL_32:
    if ( v33 )
      goto LABEL_33;
  }
  v147 = 257;
  v119 = sub_B51D30(48, v30, (__int64)v31, (__int64)&v144, 0, 0);
  v97 = sub_920620(v119);
  v98 = v119;
  if ( v97 )
  {
    v99 = v158;
    if ( v157 )
    {
      sub_B99FD0(v119, 3u, v157);
      v98 = v119;
    }
    v120 = v98;
    sub_B45150(v98, v99);
    v98 = v120;
  }
  v121 = v98;
  (*((void (__fastcall **)(void **, __int64, __m128i **, __int64, __int64))*v156 + 2))(v156, v98, &v140, v152, v153);
  v33 = v121;
  if ( v148 != &v148[4 * (unsigned int)v149] )
  {
    v122 = v7;
    v100 = v148;
    v101 = &v148[4 * (unsigned int)v149];
    v102 = v33;
    do
    {
      v103 = *((_QWORD *)v100 + 1);
      v104 = *v100;
      v100 += 4;
      sub_B99FD0(v102, v104, v103);
    }
    while ( v101 != v100 );
    v7 = v122;
    v33 = v102;
  }
LABEL_33:
  v34 = *(_QWORD *)(v7 + 136);
  v109 = v33;
  v143 = 257;
  v35 = sub_AA4E30(v151);
  v36 = sub_AE5020(v35, v34);
  v147 = 257;
  v111 = v36;
  v37 = sub_BD2C40(80, unk_3F10A14);
  v39 = v37;
  if ( v37 )
  {
    sub_B4D190((__int64)v37, v34, v109, (__int64)&v144, 0, v111, 0, 0);
    v38 = v107;
  }
  (*((void (__fastcall **)(void **, _BYTE *, __m128i **, __int64, __int64, __int64))*v156 + 2))(
    v156,
    v39,
    &v140,
    v152,
    v153,
    v38);
  v40 = 4LL * (unsigned int)v149;
  if ( v148 != &v148[v40] )
  {
    v115 = v7;
    v41 = &v148[v40];
    v42 = v148;
    do
    {
      v43 = *((_QWORD *)v42 + 1);
      v44 = *v42;
      v42 += 4;
      sub_B99FD0((__int64)v39, v44, v43);
    }
    while ( v41 != v42 );
    v7 = v115;
  }
  v147 = 257;
  v45 = sub_92B530(&v148, 0x21u, v135, v39, (__int64)&v144);
  v46 = *(_QWORD *)(v127 + 40);
  v144 = *(__m128i **)v7;
  v47 = sub_B8C340(&v144);
  sub_F38250(v45, (__int64 *)(v132 + 24), 0, 0, v47, a6, a7, v46);
  sub_D5F1F0((__int64)&v148, v127);
  v48 = *(_DWORD *)(v7 + 56);
  if ( v48 == 29 )
  {
    v81 = &v146.m128i_i8[5];
    v82 = v108 + 64LL;
    do
    {
      *--v81 = v82 % 0xA + 48;
      v83 = v82;
      v82 /= 0xAu;
    }
    while ( v83 > 9 );
    v140 = &v142;
    sub_2434550((__int64 *)&v140, v81, (__int64)v146.m128i_i64 + 5);
    v84 = (__m128i *)sub_2241130((unsigned __int64 *)&v140, 0, 0, "ebreak\naddiw x0, x11, ", 0x16u);
    v144 = &v146;
    if ( (__m128i *)v84->m128i_i64[0] == &v84[1] )
    {
      v146 = _mm_loadu_si128(v84 + 1);
    }
    else
    {
      v144 = (__m128i *)v84->m128i_i64[0];
      v146.m128i_i64[0] = v84[1].m128i_i64[0];
    }
    v145 = v84->m128i_i64[1];
    v84->m128i_i64[0] = (__int64)v84[1].m128i_i64;
    v84->m128i_i64[1] = 0;
    v84[1].m128i_i8[0] = 0;
    v85 = *(__int64 **)(v7 + 112);
    v86 = (__int64)v144;
    v87 = v145;
    v137[0] = *(_QWORD *)(v133 + 8);
    v88 = (_QWORD **)sub_BCF480(v85, v137, 1, 0);
    v56 = sub_B41A60(v88, v86, v87, (__int64)"{x10}", 5, 1, 0, 0, 0);
    sub_2240A30((unsigned __int64 *)&v144);
LABEL_48:
    sub_2240A30((unsigned __int64 *)&v140);
    goto LABEL_49;
  }
  if ( v48 <= 0x1D )
  {
    if ( v48 - 3 <= 1 )
    {
      v49 = &v146.m128i_i8[5];
      v50 = v108 + 2304LL;
      do
      {
        *--v49 = v50 % 0xA + 48;
        v51 = v50;
        v50 /= 0xAu;
      }
      while ( v51 > 9 );
      v140 = &v142;
      sub_2434550((__int64 *)&v140, v49, (__int64)v146.m128i_i64 + 5);
      v52 = (__m128i *)sub_2241130((unsigned __int64 *)&v140, 0, 0, "brk #", 5u);
      v144 = &v146;
      if ( (__m128i *)v52->m128i_i64[0] == &v52[1] )
      {
        v146 = _mm_loadu_si128(v52 + 1);
      }
      else
      {
        v144 = (__m128i *)v52->m128i_i64[0];
        v146.m128i_i64[0] = v52[1].m128i_i64[0];
      }
      v145 = v52->m128i_i64[1];
      v52->m128i_i64[0] = (__int64)v52[1].m128i_i64;
      v52->m128i_i64[1] = 0;
      v52[1].m128i_i8[0] = 0;
      v53 = *(__int64 **)(v7 + 112);
      v54 = v145;
      v125 = (__int64)v144;
      v137[0] = *(_QWORD *)(v133 + 8);
      v55 = (_QWORD **)sub_BCF480(v53, v137, 1, 0);
      v56 = sub_B41A60(v55, v125, v54, (__int64)"{x0}", 4, 1, 0, 0, 0);
      if ( v144 != &v146 )
        j_j___libc_free_0((unsigned __int64)v144);
      goto LABEL_48;
    }
LABEL_116:
    sub_C64ED0("unsupported architecture", 1u);
  }
  if ( v48 != 39 )
    goto LABEL_116;
  v63 = &v146.m128i_i8[5];
  v64 = v108 + 64LL;
  do
  {
    *--v63 = v64 % 0xA + 48;
    v65 = v64;
    v64 /= 0xAu;
  }
  while ( v65 > 9 );
  v137[0] = (__int64)&v138;
  sub_2434550(v137, v63, (__int64)v146.m128i_i64 + 5);
  v66 = (__m128i *)sub_2241130((unsigned __int64 *)v137, 0, 0, "int3\nnopl ", 0xAu);
  v140 = &v142;
  if ( (__m128i *)v66->m128i_i64[0] == &v66[1] )
  {
    v142 = _mm_loadu_si128(v66 + 1);
  }
  else
  {
    v140 = (__m128i *)v66->m128i_i64[0];
    v142.m128i_i64[0] = v66[1].m128i_i64[0];
  }
  v141 = v66->m128i_i64[1];
  v66->m128i_i64[0] = (__int64)v66[1].m128i_i64;
  v66->m128i_i64[1] = 0;
  v66[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v141) <= 5 )
    sub_4262D8((__int64)"basic_string::append");
  v67 = (__m128i *)sub_2241490((unsigned __int64 *)&v140, "(%rax)", 6u);
  v144 = &v146;
  if ( (__m128i *)v67->m128i_i64[0] == &v67[1] )
  {
    v146 = _mm_loadu_si128(v67 + 1);
  }
  else
  {
    v144 = (__m128i *)v67->m128i_i64[0];
    v146.m128i_i64[0] = v67[1].m128i_i64[0];
  }
  v145 = v67->m128i_i64[1];
  v67->m128i_i64[0] = (__int64)v67[1].m128i_i64;
  v67->m128i_i64[1] = 0;
  v67[1].m128i_i8[0] = 0;
  v68 = *(__int64 **)(v7 + 112);
  v69 = v145;
  v126 = (__int64)v144;
  v131 = *(_QWORD *)(v133 + 8);
  v70 = (_QWORD **)sub_BCF480(v68, &v131, 1, 0);
  v56 = sub_B41A60(v70, v126, v69, (__int64)"{rdi}", 5, 1, 0, 0, 0);
  sub_2240A30((unsigned __int64 *)&v144);
  sub_2240A30((unsigned __int64 *)&v140);
  sub_2240A30((unsigned __int64 *)v137);
LABEL_49:
  v57 = 0;
  v147 = 257;
  if ( v56 )
    v57 = sub_B3B7D0(v56);
  sub_921880(&v148, v57, v56, (int)&v133, 1, (__int64)&v144, 0);
  if ( *(_BYTE *)(v7 + 161) )
  {
    v89 = *(_QWORD *)(v132 + 40);
    if ( *(_QWORD *)(v127 - 32) )
    {
      v90 = *(_QWORD *)(v127 - 24);
      **(_QWORD **)(v127 - 16) = v90;
      if ( v90 )
        *(_QWORD *)(v90 + 16) = *(_QWORD *)(v127 - 16);
    }
    *(_QWORD *)(v127 - 32) = v89;
    if ( v89 )
    {
      v91 = *(_QWORD *)(v89 + 16);
      *(_QWORD *)(v127 - 24) = v91;
      if ( v91 )
        *(_QWORD *)(v91 + 16) = v127 - 24;
      *(_QWORD *)(v127 - 16) = v89 + 16;
      *(_QWORD *)(v89 + 16) = v127 - 32;
    }
  }
  nullsub_61();
  v163 = &unk_49DA100;
  nullsub_63();
  if ( v148 != (unsigned int *)v150 )
    _libc_free((unsigned __int64)v148);
}
