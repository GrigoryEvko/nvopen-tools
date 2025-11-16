// Function: sub_17E06D0
// Address: 0x17e06d0
//
__int64 __fastcall sub_17E06D0(__int64 a1, __int64 a2, __m128 a3, __m128 a4, double a5)
{
  __int64 v6; // rax
  _QWORD *v7; // r13
  _UNKNOWN **v8; // rax
  char v9; // al
  const char *v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // r13
  _QWORD *v18; // rdi
  __int64 v19; // r14
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  _QWORD *v23; // r14
  __int64 v24; // r15
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  _QWORD *v28; // r14
  __int64 v29; // r15
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r14
  _QWORD *v33; // r14
  __int64 v34; // r15
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r14
  const char *v38; // r15
  __int64 *v39; // rax
  __int64 **v40; // rax
  __int64 v41; // rax
  bool v42; // cf
  _QWORD *v43; // rdi
  __int64 v44; // r14
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 *v48; // r15
  _QWORD *v49; // r14
  _QWORD *v50; // rax
  _QWORD *v51; // r14
  __int64 *v52; // rax
  __int64 *v53; // r15
  _QWORD *v54; // rax
  _QWORD *v55; // r14
  __int64 *v56; // r15
  _QWORD *v57; // rax
  _QWORD *v58; // r14
  __int64 *v59; // rax
  __int64 *v60; // r15
  _QWORD *v61; // rax
  _QWORD *v62; // r14
  __int64 v63; // r15
  _QWORD *v64; // rax
  _QWORD *v65; // r14
  __int64 v66; // r15
  _QWORD *v67; // rax
  _QWORD *v68; // r14
  __int64 i; // rbx
  __m128i *v70; // rax
  __int64 v71; // rcx
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  __m128i *v75; // rax
  __m128 *v76; // rdi
  unsigned __int64 v77; // rsi
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 v80; // r14
  __int64 v81; // r14
  __int64 v82; // r15
  __int64 *v83; // rax
  __int64 v84; // rax
  __int64 v85; // r14
  unsigned __int64 v86; // r15
  __int64 v87; // r14
  __int64 *v88; // rax
  __int64 v89; // rax
  __int64 v90; // r13
  __int64 v91; // rax
  _QWORD *v92; // rax
  unsigned int v93; // r12d
  _QWORD *v94; // rbx
  _QWORD *v95; // r14
  __int64 v96; // rax
  _QWORD *v97; // rbx
  _QWORD *v98; // r14
  __int64 v99; // rax
  _QWORD *v101; // rbx
  _QWORD *v102; // r13
  _QWORD *v103; // rbx
  _QWORD *v104; // r13
  size_t v105; // rdx
  __int64 v106; // [rsp+20h] [rbp-5E0h]
  __int64 v107; // [rsp+28h] [rbp-5D8h]
  __int64 v108; // [rsp+28h] [rbp-5D8h]
  __int64 v109; // [rsp+30h] [rbp-5D0h]
  __int64 v110; // [rsp+48h] [rbp-5B8h]
  __int64 v111; // [rsp+50h] [rbp-5B0h]
  void *src; // [rsp+80h] [rbp-580h]
  void *srca; // [rsp+80h] [rbp-580h]
  __int64 *v115; // [rsp+98h] [rbp-568h]
  __int64 *v116; // [rsp+98h] [rbp-568h]
  __int64 *v117; // [rsp+98h] [rbp-568h]
  unsigned __int64 v118; // [rsp+98h] [rbp-568h]
  __m128 *dest; // [rsp+A0h] [rbp-560h]
  __int64 v120; // [rsp+A8h] [rbp-558h]
  size_t v121; // [rsp+A8h] [rbp-558h]
  __m128 v122; // [rsp+B0h] [rbp-550h] BYREF
  _QWORD v123[2]; // [rsp+C0h] [rbp-540h] BYREF
  _QWORD v124[2]; // [rsp+D0h] [rbp-530h] BYREF
  __m128 *v125; // [rsp+E0h] [rbp-520h] BYREF
  size_t n; // [rsp+E8h] [rbp-518h]
  __m128 v127; // [rsp+F0h] [rbp-510h] BYREF
  __int64 v128; // [rsp+100h] [rbp-500h]
  __int64 v129; // [rsp+108h] [rbp-4F8h]
  void *v130; // [rsp+110h] [rbp-4F0h] BYREF
  __int64 v131; // [rsp+118h] [rbp-4E8h] BYREF
  __int64 v132; // [rsp+120h] [rbp-4E0h]
  __int64 v133; // [rsp+128h] [rbp-4D8h]
  __int64 v134; // [rsp+130h] [rbp-4D0h]
  int v135; // [rsp+138h] [rbp-4C8h]
  __int64 v136; // [rsp+140h] [rbp-4C0h]
  __int64 v137; // [rsp+148h] [rbp-4B8h]
  __int64 *v138; // [rsp+160h] [rbp-4A0h] BYREF
  __int64 v139; // [rsp+168h] [rbp-498h] BYREF
  __int64 v140; // [rsp+170h] [rbp-490h] BYREF
  __int64 v141; // [rsp+178h] [rbp-488h]
  __int64 *v142; // [rsp+180h] [rbp-480h]
  __int64 *v143; // [rsp+188h] [rbp-478h]
  __int64 v144; // [rsp+190h] [rbp-470h]
  __int64 v145; // [rsp+198h] [rbp-468h]
  __int64 v146; // [rsp+1A0h] [rbp-460h]
  __int64 v147; // [rsp+1A8h] [rbp-458h]
  __int64 v148; // [rsp+1B0h] [rbp-450h]
  __int64 v149; // [rsp+1B8h] [rbp-448h]
  _QWORD v150[2]; // [rsp+1C0h] [rbp-440h] BYREF
  _BYTE *v151; // [rsp+1D0h] [rbp-430h]
  __int64 v152; // [rsp+1D8h] [rbp-428h]
  _BYTE v153[128]; // [rsp+1E0h] [rbp-420h] BYREF
  _BYTE *v154; // [rsp+260h] [rbp-3A0h]
  __int64 v155; // [rsp+268h] [rbp-398h]
  _BYTE v156[128]; // [rsp+270h] [rbp-390h] BYREF
  __int64 v157; // [rsp+2F0h] [rbp-310h] BYREF
  __int64 v158; // [rsp+2F8h] [rbp-308h]
  unsigned int v159; // [rsp+308h] [rbp-2F8h]
  _QWORD *v160; // [rsp+318h] [rbp-2E8h]
  unsigned int v161; // [rsp+328h] [rbp-2D8h]
  char v162; // [rsp+330h] [rbp-2D0h]
  char v163; // [rsp+339h] [rbp-2C7h]
  __int64 v164; // [rsp+340h] [rbp-2C0h] BYREF
  __int64 v165; // [rsp+348h] [rbp-2B8h]
  unsigned int v166; // [rsp+358h] [rbp-2A8h]
  _QWORD *v167; // [rsp+368h] [rbp-298h]
  unsigned int v168; // [rsp+378h] [rbp-288h]
  char v169; // [rsp+380h] [rbp-280h]
  char v170; // [rsp+389h] [rbp-277h]
  _QWORD *v171; // [rsp+390h] [rbp-270h]
  __int64 v172; // [rsp+398h] [rbp-268h]
  __int64 v173; // [rsp+3A0h] [rbp-260h]
  char v174; // [rsp+3A8h] [rbp-258h]
  char v175; // [rsp+3A9h] [rbp-257h]
  __int16 v176; // [rsp+3AAh] [rbp-256h]
  char v177; // [rsp+3ACh] [rbp-254h]
  _BYTE *v178; // [rsp+3B0h] [rbp-250h]
  __int64 v179; // [rsp+3B8h] [rbp-248h]
  _BYTE v180[384]; // [rsp+3C0h] [rbp-240h] BYREF
  _BYTE *v181; // [rsp+540h] [rbp-C0h]
  __int64 v182; // [rsp+548h] [rbp-B8h]
  _BYTE v183[176]; // [rsp+550h] [rbp-B0h] BYREF

  v150[1] = a1;
  v151 = v153;
  v150[0] = a2;
  v152 = 0x1000000000LL;
  v154 = v156;
  v155 = 0x1000000000LL;
  v157 = 0;
  v159 = 128;
  v158 = sub_22077B0(6144);
  sub_17D3930((__int64)&v157);
  v162 = 0;
  v163 = 1;
  v164 = 0;
  v166 = 128;
  v165 = sub_22077B0(6144);
  sub_17D3930((__int64)&v164);
  v6 = *(_QWORD *)(a2 + 40);
  LOWORD(v132) = 260;
  v130 = (void *)(v6 + 240);
  v169 = 0;
  v170 = 1;
  sub_16E1010((__int64)&v138, (__int64)&v130);
  if ( (_DWORD)v142 == 32 )
  {
    v7 = (_QWORD *)sub_22077B0(192);
    if ( !v7 )
      goto LABEL_6;
    v8 = &off_4985108;
    goto LABEL_5;
  }
  if ( (unsigned int)((_DWORD)v142 - 12) > 1 )
  {
    if ( (_DWORD)v142 == 3 )
    {
      v7 = (_QWORD *)sub_22077B0(192);
      if ( !v7 )
        goto LABEL_6;
      v8 = &off_4985188;
      goto LABEL_5;
    }
    if ( (unsigned int)((_DWORD)v142 - 17) > 1 )
    {
      v7 = (_QWORD *)sub_22077B0(8);
      if ( v7 )
        *v7 = off_4985208;
      goto LABEL_6;
    }
    v7 = (_QWORD *)sub_22077B0(192);
    if ( v7 )
    {
      v8 = &off_49851C8;
      goto LABEL_5;
    }
  }
  else
  {
    v7 = (_QWORD *)sub_22077B0(192);
    if ( v7 )
    {
      v8 = &off_4985148;
LABEL_5:
      *v7 = v8;
      v7[2] = a1;
      v7[1] = a2;
      v7[4] = 0;
      v7[3] = v150;
      v7[5] = 0;
      v7[6] = v7 + 8;
      v7[7] = 0x1000000000LL;
    }
  }
LABEL_6:
  if ( v138 != &v140 )
    j_j___libc_free_0(v138, v140 + 1);
  v178 = v180;
  v171 = v7;
  v179 = 0x1000000000LL;
  v181 = v183;
  v182 = 0x1000000000LL;
  v9 = sub_1560180(a2 + 112, 44);
  v174 = v9;
  v175 = v9;
  if ( v9 )
  {
    LOBYTE(v176) = byte_4FA4C20;
    HIBYTE(v176) = byte_4FA4980;
    v10 = sub_1649960(a2);
    v9 = 0;
    if ( v11 == 4 )
      v9 = *(_DWORD *)v10 == 1852399981;
  }
  else
  {
    v176 = 0;
  }
  v12 = *(__int64 **)(a1 + 8);
  v177 = v9;
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_142:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F9B6E8 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_142;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, const char *))(**(_QWORD **)(v13 + 8) + 104LL))(
          *(_QWORD *)(v13 + 8),
          &unk_4F9B6E8,
          v14,
          v10);
  v16 = *(_BYTE *)(a1 + 248) == 0;
  v172 = v15 + 360;
  v17 = *(_QWORD *)(a2 + 40);
  if ( !v16 )
    goto LABEL_67;
  v18 = *(_QWORD **)(a1 + 168);
  v130 = 0;
  v132 = 0;
  v133 = (__int64)v18;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v131 = 0;
  v19 = sub_1643350(v18);
  v20 = (__int64 *)sub_1643350((_QWORD *)v133);
  v140 = v19;
  v138 = &v140;
  v139 = 0x100000001LL;
  v21 = sub_1644EA0(v20, &v140, 1, 0);
  v22 = sub_1632080(v17, (__int64)"__msan_chain_origin", 19, v21, 0);
  if ( v138 != &v140 )
    _libc_free((unsigned __int64)v138);
  *(_QWORD *)(a1 + 344) = v22;
  v115 = *(__int64 **)(a1 + 176);
  v23 = (_QWORD *)sub_16471D0((_QWORD *)v133, 0);
  v24 = sub_16471D0((_QWORD *)v133, 0);
  v25 = (__int64 *)sub_16471D0((_QWORD *)v133, 0);
  v141 = (__int64)v23;
  v142 = v115;
  v138 = &v140;
  v140 = v24;
  v139 = 0x300000003LL;
  v26 = sub_1644EA0(v25, &v140, 3, 0);
  v27 = sub_1632080(v17, (__int64)"__msan_memmove", 14, v26, 0);
  if ( v138 != &v140 )
    _libc_free((unsigned __int64)v138);
  *(_QWORD *)(a1 + 352) = v27;
  v116 = *(__int64 **)(a1 + 176);
  v28 = (_QWORD *)sub_16471D0((_QWORD *)v133, 0);
  v29 = sub_16471D0((_QWORD *)v133, 0);
  v30 = (__int64 *)sub_16471D0((_QWORD *)v133, 0);
  v141 = (__int64)v28;
  v142 = v116;
  v138 = &v140;
  v140 = v29;
  v139 = 0x300000003LL;
  v31 = sub_1644EA0(v30, &v140, 3, 0);
  v32 = sub_1632080(v17, (__int64)"__msan_memcpy", 13, v31, 0);
  if ( v138 != &v140 )
    _libc_free((unsigned __int64)v138);
  *(_QWORD *)(a1 + 360) = v32;
  v117 = *(__int64 **)(a1 + 176);
  v33 = (_QWORD *)sub_1643350((_QWORD *)v133);
  v34 = sub_16471D0((_QWORD *)v133, 0);
  v35 = (__int64 *)sub_16471D0((_QWORD *)v133, 0);
  v141 = (__int64)v33;
  v142 = v117;
  v138 = &v140;
  v140 = v34;
  v139 = 0x300000003LL;
  v36 = sub_1644EA0(v35, &v140, 3, 0);
  v37 = sub_1632080(v17, (__int64)"__msan_memset", 13, v36, 0);
  if ( v138 != &v140 )
    _libc_free((unsigned __int64)v138);
  *(_QWORD *)(a1 + 368) = v37;
  v38 = "__msan_warning";
  v39 = (__int64 *)sub_1643270((_QWORD *)v133);
  v40 = (__int64 **)sub_16453E0(v39, 0);
  v41 = sub_15EE570(v40, byte_3F871B3, 0, byte_3F871B3, 0, 1, 0, 0);
  v42 = *(_BYTE *)(a1 + 160) == 0;
  v43 = *(_QWORD **)(a1 + 168);
  v138 = 0;
  *(_QWORD *)(a1 + 432) = v41;
  v141 = (__int64)v43;
  v140 = 0;
  v44 = v42 ? 23LL : 14LL;
  v16 = *(_BYTE *)(a1 + 160) == 0;
  v142 = 0;
  if ( v16 )
    v38 = "__msan_warning_noreturn";
  LODWORD(v143) = 0;
  v144 = 0;
  v145 = 0;
  v139 = 0;
  v45 = (__int64 *)sub_1643270(v43);
  n = 0;
  v46 = sub_1644EA0(v45, &v127, 0, 0);
  *(_QWORD *)(a1 + 256) = sub_1632080(v17, (__int64)v38, v44, v46, 0);
  v47 = (__int64 *)sub_1643360((_QWORD *)v141);
  v48 = sub_1645D80(v47, 100);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_retval_tls";
  v49 = sub_1648A60(88, 1u);
  if ( v49 )
    sub_15E51E0((__int64)v49, v17, (__int64)v48, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 208) = v49;
  v125 = (__m128 *)"__msan_retval_origin_tls";
  v127.m128_i16[0] = 259;
  v50 = sub_1648A60(88, 1u);
  v51 = v50;
  if ( v50 )
    sub_15E51E0((__int64)v50, v17, *(_QWORD *)(a1 + 184), 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 216) = v51;
  v52 = (__int64 *)sub_1643360((_QWORD *)v141);
  v53 = sub_1645D80(v52, 100);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_param_tls";
  v54 = sub_1648A60(88, 1u);
  v55 = v54;
  if ( v54 )
    sub_15E51E0((__int64)v54, v17, (__int64)v53, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 192) = v55;
  v56 = sub_1645D80(*(__int64 **)(a1 + 184), 200);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_param_origin_tls";
  v57 = sub_1648A60(88, 1u);
  v58 = v57;
  if ( v57 )
    sub_15E51E0((__int64)v57, v17, (__int64)v56, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 200) = v58;
  v59 = (__int64 *)sub_1643360((_QWORD *)v141);
  v60 = sub_1645D80(v59, 100);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_va_arg_tls";
  v61 = sub_1648A60(88, 1u);
  v62 = v61;
  if ( v61 )
    sub_15E51E0((__int64)v61, v17, (__int64)v60, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 224) = v62;
  v63 = sub_1643360((_QWORD *)v141);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_va_arg_overflow_size_tls";
  v64 = sub_1648A60(88, 1u);
  v65 = v64;
  if ( v64 )
    sub_15E51E0((__int64)v64, v17, v63, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 232) = v65;
  v66 = sub_1643350((_QWORD *)v141);
  v127.m128_i16[0] = 259;
  v125 = (__m128 *)"__msan_origin_tls";
  v67 = sub_1648A60(88, 1u);
  v68 = v67;
  if ( v67 )
    sub_15E51E0((__int64)v67, v17, v66, 0, 0, 0, (__int64)&v125, 0, 3, 0, 0);
  *(_QWORD *)(a1 + 240) = v68;
  v110 = a1;
  for ( i = 0; i != 4; ++i )
  {
    v125 = &v127;
    v127.m128_i16[0] = (unsigned __int8)((1 << i) + 48);
    n = 1;
    v70 = (__m128i *)sub_2241130(&v125, 0, 0, "__msan_maybe_warning_", 21);
    dest = &v122;
    if ( (__m128i *)v70->m128i_i64[0] == &v70[1] )
    {
      a3 = (__m128)_mm_loadu_si128(v70 + 1);
      v122 = a3;
    }
    else
    {
      dest = (__m128 *)v70->m128i_i64[0];
      v122.m128_u64[0] = v70[1].m128i_u64[0];
    }
    v71 = v70->m128i_i64[1];
    v70[1].m128i_i8[0] = 0;
    v120 = v71;
    v70->m128i_i64[0] = (__int64)v70[1].m128i_i64;
    v70->m128i_i64[1] = 0;
    if ( v125 != &v127 )
      j_j___libc_free_0(v125, v127.m128_u64[0] + 1);
    v106 = sub_1643350((_QWORD *)v141);
    v107 = sub_1644C60((_QWORD *)v141, 8 << i);
    v72 = (__int64 *)sub_1643270((_QWORD *)v141);
    v127.m128_u64[0] = v107;
    v127.m128_u64[1] = v106;
    v125 = &v127;
    n = 0x200000002LL;
    v73 = sub_1644EA0(v72, &v127, 2, 0);
    v74 = sub_1632080(v17, (__int64)dest, v120, v73, 0);
    if ( v125 != &v127 )
    {
      src = (void *)v74;
      _libc_free((unsigned __int64)v125);
      v74 = (__int64)src;
    }
    LOBYTE(v124[0]) = (1 << i) + 48;
    *(_QWORD *)(v110 + 8 * i + 264) = v74;
    v123[1] = 1;
    v123[0] = v124;
    BYTE1(v124[0]) = 0;
    v75 = (__m128i *)sub_2241130(v123, 0, 0, "__msan_maybe_store_origin_", 26);
    v125 = &v127;
    if ( (__m128i *)v75->m128i_i64[0] == &v75[1] )
    {
      a4 = (__m128)_mm_loadu_si128(v75 + 1);
      v127 = a4;
    }
    else
    {
      v125 = (__m128 *)v75->m128i_i64[0];
      v127.m128_u64[0] = v75[1].m128i_u64[0];
    }
    n = v75->m128i_u64[1];
    v75->m128i_i64[0] = (__int64)v75[1].m128i_i64;
    v76 = dest;
    v75->m128i_i64[1] = 0;
    v75[1].m128i_i8[0] = 0;
    if ( v125 == &v127 )
    {
      v105 = n;
      if ( n )
      {
        if ( n == 1 )
          dest->m128_i8[0] = v127.m128_i8[0];
        else
          memcpy(dest, &v127, n);
        v105 = n;
        v76 = dest;
      }
      v121 = v105;
      v76->m128_i8[v105] = 0;
      v76 = v125;
    }
    else
    {
      if ( dest == &v122 )
      {
        dest = v125;
        v121 = n;
        v122.m128_u64[0] = v127.m128_u64[0];
      }
      else
      {
        v77 = v122.m128_u64[0];
        dest = v125;
        v121 = n;
        v122.m128_u64[0] = v127.m128_u64[0];
        if ( v76 )
        {
          v125 = v76;
          v127.m128_u64[0] = v77;
          goto LABEL_53;
        }
      }
      v76 = &v127;
      v125 = &v127;
    }
LABEL_53:
    n = 0;
    v76->m128_i8[0] = 0;
    if ( v125 != &v127 )
      j_j___libc_free_0(v125, v127.m128_u64[0] + 1);
    if ( (_QWORD *)v123[0] != v124 )
      j_j___libc_free_0(v123[0], v124[0] + 1LL);
    v108 = sub_1643350((_QWORD *)v141);
    v109 = sub_16471D0((_QWORD *)v141, 0);
    srca = (void *)sub_1644C60((_QWORD *)v141, 8 << i);
    v78 = (__int64 *)sub_1643270((_QWORD *)v141);
    v125 = &v127;
    v127.m128_u64[0] = (unsigned __int64)srca;
    v127.m128_u64[1] = v109;
    v128 = v108;
    n = 0x300000003LL;
    v79 = sub_1644EA0(v78, &v127, 3, 0);
    v80 = sub_1632080(v17, (__int64)dest, v121, v79, 0);
    if ( v125 != &v127 )
      _libc_free((unsigned __int64)v125);
    *(_QWORD *)(v110 + 8 * i + 296) = v80;
    if ( dest != &v122 )
      j_j___libc_free_0(dest, v122.m128_u64[0] + 1);
  }
  v111 = *(_QWORD *)(v110 + 176);
  v81 = sub_16471D0((_QWORD *)v141, 0);
  v118 = *(_QWORD *)(v110 + 176);
  v82 = sub_16471D0((_QWORD *)v141, 0);
  v83 = (__int64 *)sub_1643270((_QWORD *)v141);
  v127.m128_u64[0] = v82;
  v128 = v81;
  v127.m128_u64[1] = v118;
  v129 = v111;
  n = 0x400000004LL;
  v125 = &v127;
  v84 = sub_1644EA0(v83, &v127, 4, 0);
  v85 = sub_1632080(v17, (__int64)"__msan_set_alloca_origin4", 25, v84, 0);
  if ( v125 != &v127 )
    _libc_free((unsigned __int64)v125);
  *(_QWORD *)(v110 + 328) = v85;
  v86 = *(_QWORD *)(v110 + 176);
  v87 = sub_16471D0((_QWORD *)v141, 0);
  v88 = (__int64 *)sub_1643270((_QWORD *)v141);
  v127.m128_u64[1] = v86;
  n = 0x200000002LL;
  v127.m128_u64[0] = v87;
  v125 = &v127;
  v89 = sub_1644EA0(v88, &v127, 2, 0);
  v90 = sub_1632080(v17, (__int64)"__msan_poison_stack", 19, v89, 0);
  if ( v125 != &v127 )
    _libc_free((unsigned __int64)v125);
  *(_QWORD *)(v110 + 336) = v90;
  sub_17CD270((__int64 *)&v138);
  *(_BYTE *)(v110 + 248) = 1;
  sub_17CD270((__int64 *)&v130);
LABEL_67:
  v142 = &v140;
  v143 = &v140;
  v91 = *(_QWORD *)(a2 + 80);
  v138 = 0;
  LODWORD(v140) = 0;
  v141 = 0;
  v144 = 0;
  if ( v91 )
    v91 -= 24;
  v145 = 0;
  v173 = v91;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v92 = sub_15606E0(&v138, 37);
  sub_15606E0(v92, 36);
  sub_15E0EF0(a2, -1, &v138);
  v93 = sub_17DDCE0(v150, a3, a4, a5);
  sub_17CCFA0((_QWORD *)v141);
  if ( v181 != v183 )
    _libc_free((unsigned __int64)v181);
  if ( v178 != v180 )
    _libc_free((unsigned __int64)v178);
  if ( v171 )
    (*(void (__fastcall **)(_QWORD *))(*v171 + 8LL))(v171);
  if ( v169 )
  {
    if ( v168 )
    {
      v103 = v167;
      v104 = &v167[2 * v168];
      do
      {
        if ( *v103 != -4 && *v103 != -8 )
          sub_17CD270(v103 + 1);
        v103 += 2;
      }
      while ( v104 != v103 );
    }
    j___libc_free_0(v167);
  }
  if ( v166 )
  {
    v94 = (_QWORD *)v165;
    v131 = 2;
    v132 = 0;
    v95 = (_QWORD *)(v165 + 48LL * v166);
    v133 = -8;
    v130 = &unk_49F04B0;
    v134 = 0;
    v139 = 2;
    v140 = 0;
    v141 = -16;
    v138 = (__int64 *)&unk_49F04B0;
    v142 = 0;
    do
    {
      v96 = v94[3];
      *v94 = &unk_49EE2B0;
      if ( v96 != 0 && v96 != -8 && v96 != -16 )
        sub_1649B30(v94 + 1);
      v94 += 6;
    }
    while ( v95 != v94 );
    v138 = (__int64 *)&unk_49EE2B0;
    if ( v141 != 0 && v141 != -8 && v141 != -16 )
      sub_1649B30(&v139);
    v130 = &unk_49EE2B0;
    if ( v133 != 0 && v133 != -8 && v133 != -16 )
      sub_1649B30(&v131);
  }
  j___libc_free_0(v165);
  if ( v162 )
  {
    if ( v161 )
    {
      v101 = v160;
      v102 = &v160[2 * v161];
      do
      {
        if ( *v101 != -8 && *v101 != -4 )
          sub_17CD270(v101 + 1);
        v101 += 2;
      }
      while ( v102 != v101 );
    }
    j___libc_free_0(v160);
  }
  if ( v159 )
  {
    v97 = (_QWORD *)v158;
    v131 = 2;
    v132 = 0;
    v98 = (_QWORD *)(v158 + 48LL * v159);
    v133 = -8;
    v130 = &unk_49F04B0;
    v134 = 0;
    v139 = 2;
    v140 = 0;
    v141 = -16;
    v138 = (__int64 *)&unk_49F04B0;
    v142 = 0;
    do
    {
      v99 = v97[3];
      *v97 = &unk_49EE2B0;
      if ( v99 != -8 && v99 != 0 && v99 != -16 )
        sub_1649B30(v97 + 1);
      v97 += 6;
    }
    while ( v98 != v97 );
    v138 = (__int64 *)&unk_49EE2B0;
    if ( v141 != 0 && v141 != -8 && v141 != -16 )
      sub_1649B30(&v139);
    v130 = &unk_49EE2B0;
    if ( v133 != 0 && v133 != -8 && v133 != -16 )
      sub_1649B30(&v131);
  }
  j___libc_free_0(v158);
  if ( v154 != v156 )
    _libc_free((unsigned __int64)v154);
  if ( v151 != v153 )
    _libc_free((unsigned __int64)v151);
  return v93;
}
