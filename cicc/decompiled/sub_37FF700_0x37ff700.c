// Function: sub_37FF700
// Address: 0x37ff700
//
void __fastcall sub_37FF700(__int64 *a1, unsigned __int64 a2, const __m128i *a3, const __m128i *a4)
{
  __int16 *v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  unsigned __int16 v9; // bx
  __int64 v10; // r8
  __int64 v11; // r10
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  int v13; // r9d
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  __m128i v17; // xmm0
  __int64 v18; // rax
  unsigned __int16 v19; // r15
  __int64 v20; // rax
  const __m128i *v21; // roff
  __int32 v22; // eax
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // ecx
  unsigned __int8 *v37; // rax
  __int64 v38; // rdx
  void *v39; // rax
  void *v40; // r13
  const __m128i *v41; // rsi
  __int64 v42; // rdx
  int v43; // r9d
  __int64 *v44; // rdi
  __int64 v45; // r8
  __m128i v46; // xmm3
  unsigned int *v47; // rax
  __int64 v48; // rdx
  unsigned __int8 *v49; // rax
  __int64 v50; // r9
  const __m128i *v51; // rdi
  bool v52; // zf
  __int64 v53; // rdx
  unsigned __int8 *v54; // rax
  __int64 v55; // rdx
  unsigned int v56; // r11d
  __int64 v57; // r10
  _WORD *v58; // rsi
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  const __m128i *v63; // rbx
  unsigned int v64; // eax
  unsigned __int8 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rax
  __int16 v69; // dx
  __int64 v70; // rax
  _QWORD *v71; // r9
  __int64 v72; // r15
  void *v73; // rax
  __m128i v74; // rax
  __int64 v75; // r9
  __int64 *v76; // rdi
  unsigned __int8 *v77; // rax
  const __m128i *v78; // rdi
  __int64 v79; // rdx
  _QWORD *v80; // r14
  __m128i v81; // rax
  __m128i v82; // xmm4
  __m128i v83; // xmm5
  __int64 v84; // r10
  __int64 v85; // r11
  __int64 v86; // r15
  __int64 v87; // rbx
  __int128 v88; // rax
  __int64 v89; // r9
  unsigned __int8 *v90; // rax
  const __m128i *v91; // rcx
  const __m128i *v92; // r8
  __int64 v93; // rdx
  unsigned __int8 *v94; // rax
  const __m128i *v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // r8
  __m128i v99; // xmm7
  __m128i v100; // xmm7
  unsigned int *v101; // rax
  __int64 v102; // rdx
  unsigned __int8 *v103; // rax
  const __m128i *v104; // rdi
  __int64 v105; // rdx
  __int64 v106; // [rsp-10h] [rbp-230h]
  void *v107; // [rsp+8h] [rbp-218h]
  __int64 v108; // [rsp+10h] [rbp-210h]
  int v109; // [rsp+18h] [rbp-208h]
  unsigned int v110; // [rsp+18h] [rbp-208h]
  int v111; // [rsp+20h] [rbp-200h]
  unsigned __int16 v112; // [rsp+26h] [rbp-1FAh]
  __int64 (__fastcall *v113)(__int64, __int64, unsigned int); // [rsp+28h] [rbp-1F8h]
  __m128i v114; // [rsp+30h] [rbp-1F0h] BYREF
  __int128 v115; // [rsp+40h] [rbp-1E0h]
  __int128 v116; // [rsp+50h] [rbp-1D0h]
  const __m128i *v117; // [rsp+60h] [rbp-1C0h]
  const __m128i *v118; // [rsp+68h] [rbp-1B8h]
  __int128 v119; // [rsp+70h] [rbp-1B0h]
  __m128i v120; // [rsp+80h] [rbp-1A0h] BYREF
  unsigned __int8 *v121; // [rsp+90h] [rbp-190h]
  __int64 v122; // [rsp+98h] [rbp-188h]
  unsigned __int8 *v123; // [rsp+A0h] [rbp-180h]
  __int64 v124; // [rsp+A8h] [rbp-178h]
  unsigned __int8 *v125; // [rsp+B0h] [rbp-170h]
  __int64 v126; // [rsp+B8h] [rbp-168h]
  unsigned __int8 *v127; // [rsp+C0h] [rbp-160h]
  __int64 v128; // [rsp+C8h] [rbp-158h]
  unsigned __int8 *v129; // [rsp+D0h] [rbp-150h]
  __int64 v130; // [rsp+D8h] [rbp-148h]
  unsigned __int8 *v131; // [rsp+E0h] [rbp-140h]
  __int64 v132; // [rsp+E8h] [rbp-138h]
  unsigned __int8 *v133; // [rsp+F0h] [rbp-130h]
  __int64 v134; // [rsp+F8h] [rbp-128h]
  unsigned __int8 *v135; // [rsp+100h] [rbp-120h]
  __int64 v136; // [rsp+108h] [rbp-118h]
  __int64 v137; // [rsp+110h] [rbp-110h]
  __int64 v138; // [rsp+118h] [rbp-108h]
  unsigned int v139; // [rsp+120h] [rbp-100h] BYREF
  __int64 v140; // [rsp+128h] [rbp-F8h]
  __m128i v141; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v142; // [rsp+140h] [rbp-E0h] BYREF
  __int64 v143; // [rsp+148h] [rbp-D8h]
  __int64 v144; // [rsp+150h] [rbp-D0h] BYREF
  int v145; // [rsp+158h] [rbp-C8h]
  unsigned __int64 v146; // [rsp+160h] [rbp-C0h] BYREF
  unsigned int v147; // [rsp+168h] [rbp-B8h]
  __int64 v148; // [rsp+170h] [rbp-B0h]
  __int64 v149; // [rsp+178h] [rbp-A8h]
  __int64 v150; // [rsp+180h] [rbp-A0h]
  __int64 v151; // [rsp+188h] [rbp-98h]
  __int64 v152; // [rsp+190h] [rbp-90h]
  __int64 v153; // [rsp+198h] [rbp-88h]
  __int64 v154; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v155; // [rsp+1A8h] [rbp-78h]
  __int64 v156; // [rsp+1B0h] [rbp-70h]
  unsigned int v157; // [rsp+1B8h] [rbp-68h]
  __m128i v158; // [rsp+1C0h] [rbp-60h] BYREF
  __m128i v159; // [rsp+1D0h] [rbp-50h]
  __m128i v160; // [rsp+1E0h] [rbp-40h]

  v6 = *(__int16 **)(a2 + 48);
  v7 = *a1;
  v118 = a3;
  v8 = a1[1];
  v117 = a4;
  v9 = *v6;
  v10 = *((_QWORD *)v6 + 1);
  v11 = *(_QWORD *)(v8 + 64);
  v112 = *v6;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  v113 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))v10;
  if ( v12 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v158, v7, v11, v9, v10);
    LOWORD(v139) = v158.m128i_i16[4];
    v140 = v159.m128i_i64[0];
  }
  else
  {
    v109 = v112;
    v139 = v12(v7, v11, v112, v10);
    v140 = v97;
  }
  v14 = *(_DWORD *)(a2 + 24);
  if ( v14 > 239 )
  {
    v15 = (unsigned int)(v14 - 242) < 2 ? 0x28 : 0;
    LOBYTE(v119) = (unsigned int)(v14 - 242) < 2;
  }
  else if ( v14 > 237 )
  {
    LOBYTE(v119) = 1;
    v15 = 40;
  }
  else
  {
    v15 = (unsigned int)(v14 - 101) < 0x30 ? 0x28 : 0;
    LOBYTE(v119) = (unsigned int)(v14 - 101) < 0x30;
  }
  v16 = *(_QWORD *)(a2 + 80);
  v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v15));
  v141 = v17;
  v18 = *(_QWORD *)(v17.m128i_i64[0] + 48) + 16LL * v17.m128i_u32[2];
  v19 = *(_WORD *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  v144 = v16;
  v143 = v20;
  LOWORD(v142) = v19;
  v114.m128i_i8[0] = v14 == 220 || v14 == 143;
  if ( v16 )
    sub_B96E90((__int64)&v144, v16, 1);
  v145 = *(_DWORD *)(a2 + 72);
  if ( (_BYTE)v119 )
  {
    v21 = *(const __m128i **)(a2 + 40);
    v22 = v21->m128i_i32[2];
    *(_QWORD *)&v115 = v21->m128i_i64[0];
    LODWORD(v116) = v22;
    v120 = _mm_loadu_si128(v21);
  }
  else
  {
    LODWORD(v116) = 0;
    *(_QWORD *)&v115 = a1[1] + 288;
  }
  v23 = 4096;
  if ( (*(_DWORD *)(a2 + 28) & 0x1000) == 0 )
    v23 = 0;
  v111 = v23;
  if ( v19 == 7 )
    goto LABEL_28;
  if ( v19 )
  {
    if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
LABEL_80:
      BUG();
    v27 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
    LOBYTE(v26) = byte_444C4A0[16 * v19 - 8];
  }
  else
  {
    v24 = sub_3007260((__int64)&v142);
    v26 = v25;
    v148 = v24;
    v27 = v24;
    v149 = v26;
  }
  if ( !(_BYTE)v26 && v27 <= 0x20 )
  {
LABEL_28:
    v108 = a1[1];
    v107 = sub_300AC80((unsigned __int16 *)&v139, v16);
    v39 = sub_C33340();
    v40 = v39;
    if ( v107 == v39 )
    {
      sub_C3C500(&v158, (__int64)v39);
      if ( (void *)v158.m128i_i64[0] != v40 )
        goto LABEL_30;
    }
    else
    {
      sub_C373C0(&v158, (__int64)v107);
      if ( (void *)v158.m128i_i64[0] != v40 )
      {
LABEL_30:
        sub_C37310((__int64)&v158, 0);
        goto LABEL_31;
      }
    }
    sub_C3CEB0((void **)&v158, 0);
LABEL_31:
    v41 = v118;
    v137 = sub_33FE6E0(v108, v158.m128i_i64, (__int64)&v144, v139, v140, 0, v17);
    v138 = v42;
    v118->m128i_i64[0] = v137;
    v41->m128i_i32[2] = v138;
    sub_91D830(&v158);
    if ( (_BYTE)v119 )
    {
      v44 = (__int64 *)a1[1];
      v45 = (unsigned int)v116;
      v46 = _mm_loadu_si128(&v141);
      *((_QWORD *)&v115 + 1) = 2;
      v120.m128i_i64[0] = v115;
      *(_QWORD *)&v116 = v44;
      v159 = v46;
      v120.m128i_i64[1] = v45 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v158 = _mm_load_si128(&v120);
      v47 = (unsigned int *)sub_33E5110(v44, v139, v140, 1, 0);
      v49 = sub_3410740(v44, *(unsigned int *)(a2 + 24), (__int64)&v144, v47, v48, v111, v17, &v158, 2);
      v51 = v117;
      v52 = v114.m128i_i8[0] == 0;
      LODWORD(v116) = 1;
      v136 = v53;
      v135 = v49;
      v117->m128i_i64[0] = (__int64)v49;
      *(_QWORD *)&v115 = v49;
      v51->m128i_i32[2] = v136;
      if ( !v52 || v19 == 7 )
        goto LABEL_33;
    }
    else
    {
      v106 = v141.m128i_i64[0];
      v94 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v144, v139, v140, v43, v17);
      v95 = v117;
      v52 = v114.m128i_i8[0] == 0;
      v133 = v94;
      v134 = v96;
      v117->m128i_i64[0] = (__int64)v94;
      v95->m128i_i32[2] = v134;
      v50 = v106;
      if ( !v52 || v19 == 7 )
        goto LABEL_34;
    }
    goto LABEL_41;
  }
  if ( v19 == 8 )
    goto LABEL_38;
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 504) <= 7u )
      goto LABEL_80;
    v31 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
    LOBYTE(v30) = byte_444C4A0[16 * v19 - 8];
  }
  else
  {
    v28 = sub_3007260((__int64)&v142);
    v30 = v29;
    v150 = v28;
    v31 = v28;
    v151 = v30;
  }
  if ( (_BYTE)v30 || v31 > 0x40 )
  {
    if ( v19 == 9 )
      goto LABEL_26;
    if ( v19 )
    {
      if ( (unsigned __int16)(v19 - 504) <= 7u )
        goto LABEL_80;
      v35 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
      LOBYTE(v34) = byte_444C4A0[16 * v19 - 8];
    }
    else
    {
      v32 = sub_3007260((__int64)&v142);
      v34 = v33;
      v152 = v32;
      v35 = v32;
      v153 = v34;
    }
    v36 = 729;
    if ( !(_BYTE)v34 && v35 <= 0x80 )
    {
LABEL_26:
      v37 = sub_33FAF80(a1[1], 213, (__int64)&v144, 9, 0, v13, v17);
      v36 = 418;
      v129 = v37;
      v130 = v38;
      v141.m128i_i64[0] = (__int64)v37;
      v141.m128i_i32[2] = v38;
    }
  }
  else
  {
LABEL_38:
    v54 = sub_33FAF80(a1[1], (unsigned int)(v114.m128i_i8[0] == 0) + 213, (__int64)&v144, 8, 0, v13, v17);
    v36 = 412;
    v131 = v54;
    v132 = v55;
    v141.m128i_i64[0] = (__int64)v54;
    v141.m128i_i32[2] = v55;
  }
  v160.m128i_i8[0] = 5;
  v120.m128i_i64[0] = v115;
  HIWORD(v56) = HIWORD(v109);
  LOWORD(v56) = v112;
  v159 = 0u;
  v57 = a1[1];
  v120.m128i_i64[1] = (unsigned int)v116 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v58 = (_WORD *)*a1;
  v158 = 0u;
  sub_3494590(
    (__int64)&v154,
    v58,
    v57,
    v36,
    v56,
    v113,
    (__int64)&v141,
    1u,
    0,
    0,
    0,
    0,
    5,
    (__int64)&v144,
    v115,
    v120.m128i_i64[1]);
  if ( (_BYTE)v119 )
  {
    v120.m128i_i64[0] = v156;
    *(_QWORD *)&v115 = v156;
    v120.m128i_i64[1] = v157 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    LODWORD(v116) = v157;
    sub_375AEA0(a1, v154, v155, (__int64)v118, (__int64)v117, v17);
    if ( v114.m128i_i8[0] )
      goto LABEL_33;
  }
  else
  {
    sub_375AEA0(a1, v154, v155, (__int64)v118, (__int64)v117, v17);
    if ( v114.m128i_i8[0] )
      goto LABEL_34;
  }
LABEL_41:
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 504) <= 7u )
      goto LABEL_80;
    v62 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
    LOBYTE(v61) = byte_444C4A0[16 * v19 - 8];
  }
  else
  {
    v59 = sub_3007260((__int64)&v142);
    v61 = v60;
    v154 = v59;
    v62 = v59;
    v155 = v61;
  }
  if ( !(_BYTE)v61 && v62 <= 0x20 )
  {
    if ( !(_BYTE)v119 )
      goto LABEL_34;
LABEL_33:
    v120.m128i_i64[0] = v115;
    sub_3760E70((__int64)a1, a2, 1, v115, (unsigned int)v116 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL);
    goto LABEL_34;
  }
  v63 = v117;
  HIWORD(v64) = HIWORD(v109);
  LOWORD(v64) = v112;
  v110 = v64;
  v65 = sub_3406EB0((_QWORD *)a1[1], 0x36u, (__int64)&v144, v64, (__int64)v113, v50, (__int128)*v118, (__int128)*v117);
  v128 = v66;
  v67 = v141.m128i_i64[0];
  v127 = v65;
  v117->m128i_i64[0] = (__int64)v65;
  v63->m128i_i32[2] = v128;
  v68 = *(_QWORD *)(v67 + 48) + 16LL * v141.m128i_u32[2];
  v69 = *(_WORD *)v68;
  v70 = *(_QWORD *)(v68 + 8);
  LOWORD(v142) = v69;
  v143 = v70;
  if ( v69 == 8 )
  {
    v71 = &unk_4529780;
  }
  else
  {
    v71 = &unk_4529770;
    if ( v69 != 9 )
    {
      v71 = &unk_4529790;
      if ( v69 != 7 )
        goto LABEL_80;
    }
  }
  v114.m128i_i64[0] = (__int64)&v146;
  v72 = a1[1];
  sub_C438C0((__int64)&v146, 128, v71, 2u);
  v73 = sub_C33340();
  sub_C3C640(&v158, (__int64)v73, &v146);
  v74.m128i_i64[0] = sub_33FE6E0(v72, v158.m128i_i64, (__int64)&v144, 0x10u, 0, 0, v17);
  v114 = v74;
  sub_91D830(&v158);
  if ( v147 > 0x40 && v146 )
    j_j___libc_free_0_0(v146);
  v76 = (__int64 *)a1[1];
  if ( (_BYTE)v119 )
  {
    v98 = (unsigned int)v116;
    *((_QWORD *)&v116 + 1) = 3;
    v120.m128i_i64[0] = v115;
    *(_QWORD *)&v116 = &v158;
    v120.m128i_i64[1] = v98 | v120.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v99 = _mm_loadu_si128(v117);
    v158 = _mm_load_si128(&v120);
    v119 = (__int128)v99;
    v159 = v99;
    v100 = _mm_load_si128(&v114);
    *(_QWORD *)&v119 = v76;
    v160 = v100;
    v101 = (unsigned int *)sub_33E5110(v76, v110, (__int64)v113, 1, 0);
    v103 = sub_3410740(
             (_QWORD *)v119,
             101,
             (__int64)&v144,
             v101,
             v102,
             v111,
             v17,
             (__m128i *)v116,
             *((__int64 *)&v116 + 1));
    v104 = v118;
    v125 = v103;
    v126 = v105;
    v118->m128i_i64[0] = (__int64)v103;
    v120.m128i_i64[0] = (__int64)v103;
    v104->m128i_i32[2] = v126;
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v103, v120.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1);
  }
  else
  {
    v77 = sub_3406EB0(v76, 0x60u, (__int64)&v144, v110, (__int64)v113, v75, (__int128)*v117, *(_OWORD *)&v114);
    v78 = v118;
    v123 = v77;
    v124 = v79;
    v118->m128i_i64[0] = (__int64)v77;
    v78->m128i_i32[2] = v124;
  }
  v80 = (_QWORD *)a1[1];
  v81.m128i_i64[0] = (__int64)sub_3400BD0((__int64)v80, 0, (__int64)&v144, v142, v143, 0, v17, 0);
  v82 = _mm_loadu_si128(&v141);
  v120 = v81;
  v83 = _mm_loadu_si128(v118);
  v84 = v117->m128i_i64[0];
  v85 = v117->m128i_i64[1];
  v86 = v118->m128i_i64[0];
  v116 = (__int128)v82;
  v87 = v118->m128i_u32[2];
  *(_QWORD *)&v115 = v84;
  *((_QWORD *)&v115 + 1) = v85;
  v119 = (__int128)v83;
  *(_QWORD *)&v88 = sub_33ED040(v80, 0x14u);
  v90 = sub_33FC1D0(
          v80,
          207,
          (__int64)&v144,
          *(unsigned __int16 *)(*(_QWORD *)(v86 + 48) + 16 * v87),
          *(_QWORD *)(*(_QWORD *)(v86 + 48) + 16 * v87 + 8),
          v89,
          v116,
          *(_OWORD *)&v120,
          v119,
          v115,
          v88);
  v91 = v118;
  v92 = v117;
  v121 = v90;
  v122 = v93;
  v118->m128i_i64[0] = (__int64)v90;
  v91->m128i_i32[2] = v122;
  sub_375AEA0(a1, (__int64)v90, v91->m128i_i64[1], (__int64)v91, (__int64)v92, v17);
LABEL_34:
  if ( v144 )
    sub_B91220((__int64)&v144, v144);
}
