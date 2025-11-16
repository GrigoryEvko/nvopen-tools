// Function: sub_2491B40
// Address: 0x2491b40
//
void __fastcall sub_2491B40(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  bool v7; // zf
  __int64 i; // r12
  char v10; // al
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int32 v13; // ebx
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // rax
  __int64 *v22; // rsi
  char *v23; // r13
  __int64 v24; // rax
  _OWORD *v25; // rax
  __m128i si128; // xmm0
  __m128i *v27; // rax
  __int64 v28; // r14
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // r14
  _QWORD *v34; // rdi
  __int64 v35; // r14
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // rdx
  __int64 v40; // r14
  _QWORD *v41; // rdi
  __int64 (__fastcall ***v42)(_QWORD, _QWORD); // r14
  char v43; // bl
  __int64 v44; // rax
  __m128i v45; // xmm0
  __m128i *v46; // rax
  __m128i *v47; // rax
  __m128i *v48; // rax
  unsigned __int64 v49; // rbx
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rbx
  __int64 *v54; // rsi
  __m128i *v55; // rdi
  char v56; // bl
  __int64 v57; // rax
  __m128i v58; // xmm0
  __m128i *v59; // rax
  __m128i *v60; // rax
  __m128i *v61; // rax
  unsigned __int64 v62; // r14
  __int64 v63; // rbx
  __int64 v64; // r15
  unsigned __int64 v65; // rax
  __int64 v66; // r14
  __int64 v67; // rdx
  __int64 v68; // rbx
  __m128i *v69; // rdi
  __int64 v70; // rbx
  unsigned __int64 v71; // rax
  __int64 v72; // r13
  __int64 v73; // rdx
  __int64 v74; // rbx
  __int64 v75; // rbx
  unsigned __int64 v76; // rax
  __int64 v77; // r13
  __int64 v78; // rdx
  __int64 v79; // rbx
  _BYTE *v80; // rax
  _QWORD *v81; // rdi
  __int64 *v82; // rax
  _QWORD *v83; // rax
  _BYTE *v84; // rax
  __int64 v85; // rdx
  _BYTE *v86; // rax
  _QWORD *v87; // rdi
  __int64 *v88; // rax
  _QWORD *v89; // rax
  _BYTE *v90; // rax
  __int64 v91; // rdx
  _QWORD *v92; // rax
  __int64 v93; // rax
  __m128i *v94; // r13
  __m128i *v95; // r12
  const __m128i *v96; // rdx
  const __m128i *v97; // rsi
  __int8 v98; // al
  const char *v99; // rax
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // r9
  __int64 v103; // rcx
  __int64 v104; // r8
  __int64 v105; // r9
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  __int64 *v112; // [rsp+10h] [rbp-380h]
  __int64 v113; // [rsp+18h] [rbp-378h]
  __int64 v114; // [rsp+20h] [rbp-370h]
  __int64 v115; // [rsp+28h] [rbp-368h]
  __int64 v116; // [rsp+38h] [rbp-358h]
  __int64 v117; // [rsp+38h] [rbp-358h]
  __int64 *v118; // [rsp+40h] [rbp-350h]
  _QWORD *v119; // [rsp+48h] [rbp-348h]
  __int64 v120; // [rsp+50h] [rbp-340h]
  const char *v121; // [rsp+58h] [rbp-338h]
  __int64 *v122; // [rsp+60h] [rbp-330h]
  __int64 v123; // [rsp+80h] [rbp-310h]
  __int64 v124; // [rsp+80h] [rbp-310h]
  __int64 v125; // [rsp+88h] [rbp-308h]
  __int64 v126; // [rsp+88h] [rbp-308h]
  unsigned __int64 v127; // [rsp+90h] [rbp-300h]
  __int64 v128; // [rsp+A0h] [rbp-2F0h]
  __int64 *v129; // [rsp+A8h] [rbp-2E8h]
  int j; // [rsp+B4h] [rbp-2DCh]
  unsigned __int32 v132; // [rsp+C4h] [rbp-2CCh]
  unsigned __int32 v133; // [rsp+C8h] [rbp-2C8h]
  unsigned __int32 v134; // [rsp+CCh] [rbp-2C4h]
  char v135; // [rsp+D0h] [rbp-2C0h] BYREF
  char *v136; // [rsp+100h] [rbp-290h] BYREF
  char v137; // [rsp+120h] [rbp-270h]
  char v138; // [rsp+121h] [rbp-26Fh]
  __m128i v139[3]; // [rsp+130h] [rbp-260h] BYREF
  __m128i v140[2]; // [rsp+160h] [rbp-230h] BYREF
  __int16 v141; // [rsp+180h] [rbp-210h]
  __m128i v142[3]; // [rsp+190h] [rbp-200h] BYREF
  __m128i v143; // [rsp+1C0h] [rbp-1D0h] BYREF
  __m128i v144; // [rsp+1D0h] [rbp-1C0h] BYREF
  char v145; // [rsp+1E0h] [rbp-1B0h]
  char v146; // [rsp+1E1h] [rbp-1AFh]
  __m128i v147; // [rsp+1F0h] [rbp-1A0h] BYREF
  _QWORD v148[4]; // [rsp+200h] [rbp-190h] BYREF
  __m128i v149; // [rsp+220h] [rbp-170h] BYREF
  _QWORD v150[2]; // [rsp+230h] [rbp-160h] BYREF
  __int16 v151; // [rsp+240h] [rbp-150h]
  __m128i v152; // [rsp+250h] [rbp-140h] BYREF
  __m128i v153; // [rsp+260h] [rbp-130h] BYREF
  char v154; // [rsp+270h] [rbp-120h]
  char v155; // [rsp+271h] [rbp-11Fh]
  __m128i v156; // [rsp+280h] [rbp-110h] BYREF
  __m128i v157; // [rsp+290h] [rbp-100h] BYREF
  __int16 v158; // [rsp+2A0h] [rbp-F0h]
  __m128i v159; // [rsp+2B0h] [rbp-E0h] BYREF
  __m128i v160; // [rsp+2C0h] [rbp-D0h] BYREF
  char v161; // [rsp+2D0h] [rbp-C0h]
  char v162; // [rsp+2D1h] [rbp-BFh]
  __m128i v163; // [rsp+2E0h] [rbp-B0h] BYREF
  __m128i v164; // [rsp+2F0h] [rbp-A0h] BYREF
  __int16 v165; // [rsp+300h] [rbp-90h]
  __m128i v166; // [rsp+310h] [rbp-80h] BYREF
  const char *v167; // [rsp+320h] [rbp-70h] BYREF
  __int64 v168; // [rsp+328h] [rbp-68h]
  const char *v169; // [rsp+330h] [rbp-60h]
  __int64 v170; // [rsp+338h] [rbp-58h]
  __int64 *v171; // [rsp+340h] [rbp-50h]
  __int64 v172; // [rsp+348h] [rbp-48h]
  __int64 v173; // [rsp+350h] [rbp-40h]

  *(_QWORD *)a1 = a2 + 39;
  v6 = *a2;
  v7 = qword_4FEA770 == 3;
  *(_QWORD *)(a1 + 8) = *a2;
  *(_QWORD *)(a1 + 16) = v6;
  *(_QWORD *)(a1 + 40) = 0;
  *(_OWORD *)(a1 + 24) = 0;
  if ( !v7 )
  {
    v162 = 1;
    v95 = &v166;
    v163.m128i_i64[0] = (__int64)&qword_4FEA768;
    v96 = &v163;
    v165 = 260;
    v97 = &v159;
    v159.m128i_i64[0] = (__int64)"Invalid nsan mapping: ";
    v161 = 3;
LABEL_103:
    sub_9C6370(v95, v97, v96, a4, a5, a6);
    sub_C64D30((__int64)v95, 1u);
  }
  for ( i = 0; i != 3; ++i )
  {
    v10 = *(_BYTE *)(qword_4FEA768 + i);
    if ( v10 == 108 )
    {
      v11 = (_QWORD *)sub_22077B0(8u);
      if ( !v11 )
        goto LABEL_104;
      *v11 = off_49D2D58;
    }
    else if ( v10 > 108 )
    {
      if ( v10 != 113 )
      {
LABEL_101:
        v94 = &v159;
        v156.m128i_i8[0] = *(_BYTE *)(qword_4FEA768 + i);
        v163.m128i_i64[0] = (__int64)"'";
        v95 = &v166;
        v165 = 259;
        v158 = 264;
        v155 = 1;
        v152.m128i_i64[0] = (__int64)"nsan: invalid shadow type id '";
        v154 = 3;
        sub_9C6370(&v159, &v152, &v156, a4, a5, a6);
        v96 = &v163;
        goto LABEL_102;
      }
      v92 = (_QWORD *)sub_22077B0(8u);
      v11 = v92;
      if ( !v92 )
        goto LABEL_104;
      *v92 = off_49D2D88;
    }
    else if ( v10 == 100 )
    {
      v11 = (_QWORD *)sub_22077B0(8u);
      if ( !v11 )
        goto LABEL_104;
      *v11 = off_49D2D28;
    }
    else
    {
      if ( v10 != 101 )
        goto LABEL_101;
      v11 = (_QWORD *)sub_22077B0(8u);
      if ( !v11 )
      {
LABEL_104:
        v96 = &v163;
        v97 = &v166;
        v98 = *(_BYTE *)(qword_4FEA768 + i);
        v95 = &v159;
        v165 = 264;
        v163.m128i_i8[0] = v98;
        v166.m128i_i64[0] = (__int64)"Failed to get ShadowTypeConfig for ";
        LOWORD(v169) = 259;
        goto LABEL_103;
      }
      *v11 = off_49D2DB8;
    }
    if ( !v11 )
      goto LABEL_104;
    v12 = sub_2491250(i, *(_QWORD **)(a1 + 16));
    v13 = sub_BCB060(v12);
    v14 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD))*v11)(v11, *(_QWORD *)(a1 + 16));
    v15 = sub_BCB060(v14);
    if ( v15 > 2 * v13 )
    {
      a5 = 266;
      a6 = 265;
      v136 = " times the application type size";
      v138 = 1;
      v137 = 3;
      v140[0].m128i_i64[0] = 2;
      v141 = 266;
      v146 = 1;
      v143.m128i_i64[0] = (__int64)": The shadow type size should be at most ";
      v145 = 3;
      v151 = 265;
      v158 = 259;
      v165 = 265;
      v163.m128i_i32[0] = v13;
      BYTE1(v169) = 1;
      v149.m128i_i32[0] = v15;
      v156.m128i_i64[0] = (__int64)"->f";
      v99 = "Invalid nsan mapping f";
      goto LABEL_106;
    }
    v16 = *(_QWORD *)(a1 + 8 * i + 24);
    *(&v132 + i) = v15;
    *(_QWORD *)(a1 + 8 * i + 24) = v11;
    if ( v16 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 24LL))(v16);
  }
  a4 = v134;
  if ( v132 > v133 || v133 > v134 )
  {
    v149.m128i_i32[0] = v133;
    v136 = " }";
    v156.m128i_i64[0] = (__int64)"; double->f";
    v99 = "Invalid nsan mapping: { float->f";
    v138 = 1;
    v137 = 3;
    v141 = 265;
    v140[0].m128i_i32[0] = v134;
    v146 = 1;
    v143.m128i_i64[0] = (__int64)"; long double->f";
    v145 = 3;
    v151 = 265;
    v158 = 259;
    v165 = 265;
    v163.m128i_i32[0] = v132;
    BYTE1(v169) = 1;
LABEL_106:
    v166.m128i_i64[0] = (__int64)v99;
    LOBYTE(v169) = 3;
    sub_9C6370(&v159, &v166, &v163, a4, a5, a6);
    sub_9C6370(&v152, &v159, &v156, v100, v101, v102);
    sub_9C6370(&v147, &v152, &v149, v103, v104, v105);
    sub_9C6370(v142, &v147, &v143, v106, v107, v108);
    v94 = v139;
    v95 = (__m128i *)&v135;
    sub_9C6370(v139, v142, v140, v109, v110, v111);
    v96 = (const __m128i *)&v136;
LABEL_102:
    v97 = v94;
    goto LABEL_103;
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_OWORD *)(a1 + 56) = 0;
  *(_OWORD *)(a1 + 72) = 0;
  *(_OWORD *)(a1 + 88) = 0;
  *(_OWORD *)(a1 + 104) = 0;
  *(_OWORD *)(a1 + 120) = 0;
  *(_OWORD *)(a1 + 136) = 0;
  *(_OWORD *)(a1 + 152) = 0;
  *(_OWORD *)(a1 + 168) = 0;
  *(_OWORD *)(a1 + 184) = 0;
  *(_OWORD *)(a1 + 200) = 0;
  *(_OWORD *)(a1 + 216) = 0;
  *(_OWORD *)(a1 + 232) = 0;
  v166.m128i_i64[0] = (__int64)"__nsan_copy_4";
  v167 = "__nsan_copy_8";
  v169 = "__nsan_copy_16";
  v166.m128i_i64[1] = 13;
  v168 = 13;
  v170 = 14;
  sub_2491800(a1 + 248, a2, (__int64)&v166, 3, (__int64)"__nsan_copy_values", 0x12u, 3);
  v166.m128i_i64[0] = (__int64)"__nsan_set_value_unknown_4";
  v167 = "__nsan_set_value_unknown_8";
  v169 = "__nsan_set_value_unknown_16";
  v166.m128i_i64[1] = 26;
  v168 = 26;
  v170 = 27;
  sub_2491800(a1 + 320, a2, (__int64)&v166, 3, (__int64)"__nsan_set_value_unknown", 0x18u, 2);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_BYTE *)(a1 + 488) = 0;
  v19 = sub_AE4420(v18, v17, 0);
  v20 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 48) = v19;
  v122 = (__int64 *)sub_BCE3C0(v20, 0);
  v118 = (__int64 *)sub_BCB2D0(*(_QWORD **)(a1 + 8));
  v113 = sub_BCB2A0(*(_QWORD **)(a1 + 8));
  v21 = sub_BCB120(*(_QWORD **)(a1 + 8));
  v22 = *(__int64 **)(a1 + 8);
  v112 = (__int64 *)v21;
  v142[0].m128i_i64[0] = 0;
  v128 = a1;
  v142[0].m128i_i64[0] = sub_A7A090(v142[0].m128i_i64, v22, -1, 41);
  v129 = (__int64 *)(a1 + 56);
  v119 = (_QWORD *)(a1 + 24);
  for ( j = 0; j != 3; ++j )
  {
    v23 = "double";
    if ( j != 1 )
    {
      v23 = "float";
      if ( j == 2 )
        v23 = "longdouble";
    }
    v24 = sub_2491250(j, *(_QWORD **)(v128 + 8));
    v166.m128i_i64[0] = (__int64)&v167;
    v121 = (const char *)v24;
    v163.m128i_i64[0] = 26;
    v25 = (_OWORD *)sub_22409D0((__int64)&v166, (unsigned __int64 *)&v163, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_4386080);
    v166.m128i_i64[0] = (__int64)v25;
    v167 = (const char *)v163.m128i_i64[0];
    qmemcpy(v25 + 1, "w_ptr_for_", 10);
    *v25 = si128;
    v166.m128i_i64[1] = v163.m128i_i64[0];
    *(_BYTE *)(v166.m128i_i64[0] + v163.m128i_i64[0]) = 0;
    v127 = strlen(v23);
    if ( v127 > 0x3FFFFFFFFFFFFFFFLL - v166.m128i_i64[1] )
      goto LABEL_100;
    v27 = (__m128i *)sub_2241490((unsigned __int64 *)&v166, v23, v127);
    v143.m128i_i64[0] = (__int64)&v144;
    if ( (__m128i *)v27->m128i_i64[0] == &v27[1] )
    {
      v144 = _mm_loadu_si128(v27 + 1);
    }
    else
    {
      v143.m128i_i64[0] = v27->m128i_i64[0];
      v144.m128i_i64[0] = v27[1].m128i_i64[0];
    }
    v143.m128i_i64[1] = v27->m128i_i64[1];
    v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
    v27->m128i_i64[1] = 0;
    v27[1].m128i_i8[0] = 0;
    if ( (const char **)v166.m128i_i64[0] != &v167 )
      j_j___libc_free_0(v166.m128i_u64[0]);
    v147.m128i_i64[0] = (__int64)v148;
    v28 = *(_QWORD *)(v128 + 48);
    sub_2491480(v147.m128i_i64, v143.m128i_i64[0], v143.m128i_i64[0] + v143.m128i_i64[1]);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v147.m128i_i64[1]) <= 5 )
      goto LABEL_100;
    sub_2241490((unsigned __int64 *)&v147, "_store", 6u);
    v168 = v28;
    v166.m128i_i64[0] = (__int64)&v167;
    v29 = v147.m128i_u64[1];
    v123 = v147.m128i_i64[0];
    v125 = v142[0].m128i_i64[0];
    v167 = (const char *)v122;
    v166.m128i_i64[1] = 0x200000002LL;
    v30 = sub_BCF480(v122, &v167, 2, 0);
    v31 = sub_BA8C10((__int64)a2, v123, v29, v30, v125);
    v33 = v32;
    if ( (const char **)v166.m128i_i64[0] != &v167 )
      _libc_free(v166.m128i_u64[0]);
    v34 = (_QWORD *)v147.m128i_i64[0];
    *v129 = v31;
    v129[1] = v33;
    if ( v34 != v148 )
      j_j___libc_free_0((unsigned __int64)v34);
    v149.m128i_i64[0] = (__int64)v150;
    v35 = *(_QWORD *)(v128 + 48);
    sub_2491480(v149.m128i_i64, v143.m128i_i64[0], v143.m128i_i64[0] + v143.m128i_i64[1]);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v149.m128i_i64[1]) <= 4 )
      goto LABEL_100;
    sub_2241490((unsigned __int64 *)&v149, "_load", 5u);
    v168 = v35;
    v166.m128i_i64[0] = (__int64)&v167;
    v36 = v149.m128i_u64[1];
    v124 = v149.m128i_i64[0];
    v126 = v142[0].m128i_i64[0];
    v167 = (const char *)v122;
    v166.m128i_i64[1] = 0x200000002LL;
    v37 = sub_BCF480(v122, &v167, 2, 0);
    v38 = sub_BA8C10((__int64)a2, v124, v36, v37, v126);
    v40 = v39;
    if ( (const char **)v166.m128i_i64[0] != &v167 )
      _libc_free(v166.m128i_u64[0]);
    v41 = (_QWORD *)v149.m128i_i64[0];
    v129[6] = v38;
    v129[7] = v40;
    if ( v41 != v150 )
      j_j___libc_free_0((unsigned __int64)v41);
    v42 = (__int64 (__fastcall ***)(_QWORD, _QWORD))*v119;
    v120 = (**(__int64 (__fastcall ***)(_QWORD, _QWORD))*v119)(*v119, *(_QWORD *)(v128 + 8));
    v116 = *(_QWORD *)(v128 + 48);
    v43 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD, _QWORD)))(*v42)[1])(v42);
    v166.m128i_i64[0] = 22;
    v156.m128i_i64[0] = (__int64)&v157;
    v44 = sub_22409D0((__int64)&v156, (unsigned __int64 *)&v166, 0);
    v45 = _mm_load_si128((const __m128i *)&xmmword_4386090);
    v156.m128i_i64[0] = v44;
    v157.m128i_i64[0] = v166.m128i_i64[0];
    *(_DWORD *)(v44 + 16) = 1667590243;
    *(_WORD *)(v44 + 20) = 24427;
    *(__m128i *)v44 = v45;
    v156.m128i_i64[1] = v166.m128i_i64[0];
    *(_BYTE *)(v156.m128i_i64[0] + v166.m128i_i64[0]) = 0;
    if ( v127 > 0x3FFFFFFFFFFFFFFFLL - v156.m128i_i64[1] )
      goto LABEL_100;
    v46 = (__m128i *)sub_2241490((unsigned __int64 *)&v156, v23, v127);
    v152.m128i_i64[0] = (__int64)&v153;
    if ( (__m128i *)v46->m128i_i64[0] == &v46[1] )
    {
      v153 = _mm_loadu_si128(v46 + 1);
    }
    else
    {
      v152.m128i_i64[0] = v46->m128i_i64[0];
      v153.m128i_i64[0] = v46[1].m128i_i64[0];
    }
    v152.m128i_i64[1] = v46->m128i_i64[1];
    v46->m128i_i64[0] = (__int64)v46[1].m128i_i64;
    v46->m128i_i64[1] = 0;
    v46[1].m128i_i8[0] = 0;
    if ( v152.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_100;
    v47 = (__m128i *)sub_2241490((unsigned __int64 *)&v152, "_", 1u);
    v163.m128i_i64[0] = (__int64)&v164;
    if ( (__m128i *)v47->m128i_i64[0] == &v47[1] )
    {
      v164 = _mm_loadu_si128(v47 + 1);
    }
    else
    {
      v163.m128i_i64[0] = v47->m128i_i64[0];
      v164.m128i_i64[0] = v47[1].m128i_i64[0];
    }
    v163.m128i_i64[1] = v47->m128i_i64[1];
    v47->m128i_i64[0] = (__int64)v47[1].m128i_i64;
    v47->m128i_i64[1] = 0;
    v47[1].m128i_i8[0] = 0;
    v48 = (__m128i *)sub_2240FD0((unsigned __int64 *)&v163, v163.m128i_u64[1], 0, 1u, v43);
    v159.m128i_i64[0] = (__int64)&v160;
    if ( (__m128i *)v48->m128i_i64[0] == &v48[1] )
    {
      v160 = _mm_loadu_si128(v48 + 1);
    }
    else
    {
      v159.m128i_i64[0] = v48->m128i_i64[0];
      v160.m128i_i64[0] = v48[1].m128i_i64[0];
    }
    v159.m128i_i64[1] = v48->m128i_i64[1];
    v48->m128i_i64[0] = (__int64)v48[1].m128i_i64;
    v48->m128i_i64[1] = 0;
    v48[1].m128i_i8[0] = 0;
    v169 = (const char *)v118;
    v167 = v121;
    v49 = v159.m128i_u64[1];
    v114 = v159.m128i_i64[0];
    v168 = v120;
    v115 = v142[0].m128i_i64[0];
    v170 = v116;
    v166.m128i_i64[0] = (__int64)&v167;
    v166.m128i_i64[1] = 0x400000004LL;
    v50 = sub_BCF480(v118, &v167, 4, 0);
    v51 = sub_BA8C10((__int64)a2, v114, v49, v50, v115);
    v53 = v52;
    if ( (const char **)v166.m128i_i64[0] != &v167 )
    {
      v117 = v51;
      _libc_free(v166.m128i_u64[0]);
      v51 = v117;
    }
    v54 = v129;
    v55 = (__m128i *)v159.m128i_i64[0];
    v129[12] = v51;
    v129[13] = v53;
    if ( v55 != &v160 )
    {
      v54 = (__int64 *)(v160.m128i_i64[0] + 1);
      j_j___libc_free_0((unsigned __int64)v55);
    }
    if ( (__m128i *)v163.m128i_i64[0] != &v164 )
    {
      v54 = (__int64 *)(v164.m128i_i64[0] + 1);
      j_j___libc_free_0(v163.m128i_u64[0]);
    }
    if ( (__m128i *)v152.m128i_i64[0] != &v153 )
    {
      v54 = (__int64 *)(v153.m128i_i64[0] + 1);
      j_j___libc_free_0(v152.m128i_u64[0]);
    }
    if ( (__m128i *)v156.m128i_i64[0] != &v157 )
    {
      v54 = (__int64 *)(v157.m128i_i64[0] + 1);
      j_j___libc_free_0(v156.m128i_u64[0]);
    }
    v56 = ((__int64 (__fastcall **)(_QWORD, __int64 *))*v42)[1](v42, v54);
    v166.m128i_i64[0] = 17;
    v163.m128i_i64[0] = (__int64)&v164;
    v57 = sub_22409D0((__int64)&v163, (unsigned __int64 *)&v166, 0);
    v58 = _mm_load_si128((const __m128i *)&xmmword_43860A0);
    v163.m128i_i64[0] = v57;
    v164.m128i_i64[0] = v166.m128i_i64[0];
    *(_BYTE *)(v57 + 16) = 95;
    *(__m128i *)v57 = v58;
    v163.m128i_i64[1] = v166.m128i_i64[0];
    *(_BYTE *)(v163.m128i_i64[0] + v166.m128i_i64[0]) = 0;
    if ( v127 > 0x3FFFFFFFFFFFFFFFLL - v163.m128i_i64[1] )
      goto LABEL_100;
    v59 = (__m128i *)sub_2241490((unsigned __int64 *)&v163, v23, v127);
    v159.m128i_i64[0] = (__int64)&v160;
    if ( (__m128i *)v59->m128i_i64[0] == &v59[1] )
    {
      v160 = _mm_loadu_si128(v59 + 1);
    }
    else
    {
      v159.m128i_i64[0] = v59->m128i_i64[0];
      v160.m128i_i64[0] = v59[1].m128i_i64[0];
    }
    v159.m128i_i64[1] = v59->m128i_i64[1];
    v59->m128i_i64[0] = (__int64)v59[1].m128i_i64;
    v59->m128i_i64[1] = 0;
    v59[1].m128i_i8[0] = 0;
    if ( v159.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
LABEL_100:
      sub_4262D8((__int64)"basic_string::append");
    v60 = (__m128i *)sub_2241490((unsigned __int64 *)&v159, "_", 1u);
    v156.m128i_i64[0] = (__int64)&v157;
    if ( (__m128i *)v60->m128i_i64[0] == &v60[1] )
    {
      v157 = _mm_loadu_si128(v60 + 1);
    }
    else
    {
      v156.m128i_i64[0] = v60->m128i_i64[0];
      v157.m128i_i64[0] = v60[1].m128i_i64[0];
    }
    v156.m128i_i64[1] = v60->m128i_i64[1];
    v60->m128i_i64[0] = (__int64)v60[1].m128i_i64;
    v60->m128i_i64[1] = 0;
    v60[1].m128i_i8[0] = 0;
    v61 = (__m128i *)sub_2240FD0((unsigned __int64 *)&v156, v156.m128i_u64[1], 0, 1u, v56);
    v152.m128i_i64[0] = (__int64)&v153;
    if ( (__m128i *)v61->m128i_i64[0] == &v61[1] )
    {
      v153 = _mm_loadu_si128(v61 + 1);
    }
    else
    {
      v152.m128i_i64[0] = v61->m128i_i64[0];
      v153.m128i_i64[0] = v61[1].m128i_i64[0];
    }
    v152.m128i_i64[1] = v61->m128i_i64[1];
    v61->m128i_i64[0] = (__int64)v61[1].m128i_i64;
    v61->m128i_i64[1] = 0;
    v61[1].m128i_i8[0] = 0;
    v62 = v152.m128i_u64[1];
    v63 = v142[0].m128i_i64[0];
    v166.m128i_i64[0] = (__int64)&v167;
    v167 = v121;
    v64 = v152.m128i_i64[0];
    v168 = (__int64)v121;
    v169 = (const char *)v120;
    v170 = v120;
    v171 = v118;
    v172 = v113;
    v173 = v113;
    v166.m128i_i64[1] = 0x700000007LL;
    v65 = sub_BCF480(v112, &v167, 7, 0);
    v66 = sub_BA8C10((__int64)a2, v64, v62, v65, v63);
    v68 = v67;
    if ( (const char **)v166.m128i_i64[0] != &v167 )
      _libc_free(v166.m128i_u64[0]);
    v69 = (__m128i *)v152.m128i_i64[0];
    v129[18] = v66;
    v129[19] = v68;
    if ( v69 != &v153 )
      j_j___libc_free_0((unsigned __int64)v69);
    if ( (__m128i *)v156.m128i_i64[0] != &v157 )
      j_j___libc_free_0(v156.m128i_u64[0]);
    if ( (__m128i *)v159.m128i_i64[0] != &v160 )
      j_j___libc_free_0(v159.m128i_u64[0]);
    if ( (__m128i *)v163.m128i_i64[0] != &v164 )
      j_j___libc_free_0(v163.m128i_u64[0]);
    if ( (__m128i *)v143.m128i_i64[0] != &v144 )
      j_j___libc_free_0(v143.m128i_u64[0]);
    v129 += 2;
    ++v119;
  }
  v166.m128i_i64[0] = (__int64)&v167;
  v70 = v142[0].m128i_i64[0];
  v167 = (const char *)v122;
  v166.m128i_i64[1] = 0x100000001LL;
  v71 = sub_BCF480(v122, &v167, 1, 0);
  v72 = sub_BA8C10((__int64)a2, (__int64)"__nsan_internal_get_raw_shadow_type_ptr", 0x27u, v71, v70);
  v74 = v73;
  if ( (const char **)v166.m128i_i64[0] != &v167 )
    _libc_free(v166.m128i_u64[0]);
  v166.m128i_i64[0] = (__int64)&v167;
  *(_QWORD *)(v128 + 392) = v72;
  *(_QWORD *)(v128 + 400) = v74;
  v75 = v142[0].m128i_i64[0];
  v167 = (const char *)v122;
  v166.m128i_i64[1] = 0x100000001LL;
  v76 = sub_BCF480(v122, &v167, 1, 0);
  v77 = sub_BA8C10((__int64)a2, (__int64)"__nsan_internal_get_raw_shadow_ptr", 0x22u, v76, v75);
  v79 = v78;
  if ( (const char **)v166.m128i_i64[0] != &v167 )
    _libc_free(v166.m128i_u64[0]);
  *(_QWORD *)(v128 + 416) = v79;
  *(_QWORD *)(v128 + 408) = v77;
  v80 = sub_24913C0("__nsan_shadow_ret_tag", (__int64)a2, *(_QWORD *)(v128 + 48));
  v81 = *(_QWORD **)(v128 + 8);
  *(_QWORD *)(v128 + 424) = v80;
  v82 = (__int64 *)sub_BCB2B0(v81);
  v83 = sub_BCD420(v82, 128);
  *(_QWORD *)(v128 + 432) = v83;
  v84 = sub_24913C0("__nsan_shadow_ret_ptr", (__int64)a2, (__int64)v83);
  v85 = *(_QWORD *)(v128 + 48);
  *(_QWORD *)(v128 + 440) = v84;
  v86 = sub_24913C0("__nsan_shadow_args_tag", (__int64)a2, v85);
  v87 = *(_QWORD **)(v128 + 8);
  *(_QWORD *)(v128 + 448) = v86;
  v88 = (__int64 *)sub_BCB2B0(v87);
  v89 = sub_BCD420(v88, 0x4000);
  *(_QWORD *)(v128 + 456) = v89;
  v90 = sub_24913C0("__nsan_shadow_args_ptr", (__int64)a2, (__int64)v89);
  v91 = qword_4FEA590;
  *(_QWORD *)(v128 + 464) = v90;
  if ( v91 )
  {
    sub_C88F40((__int64)&v159, qword_4FEA588, v91, 0);
    v7 = *(_BYTE *)(v128 + 488) == 0;
    v166.m128i_i64[0] = (__int64)&v167;
    v166.m128i_i64[1] = 0;
    LOBYTE(v167) = 0;
    if ( v7 )
    {
      sub_C88FD0(v128 + 472, v159.m128i_i64);
      *(_BYTE *)(v128 + 488) = 1;
    }
    else
    {
      sub_C88FD0((__int64)&v163, v159.m128i_i64);
      v93 = *(_QWORD *)(v128 + 472);
      *(_QWORD *)(v128 + 472) = v163.m128i_i64[0];
      v163.m128i_i64[0] = v93;
      LODWORD(v93) = *(_DWORD *)(v128 + 480);
      *(_DWORD *)(v128 + 480) = v163.m128i_i32[2];
      v163.m128i_i32[2] = v93;
      sub_C88FF0(&v163);
    }
    if ( (const char **)v166.m128i_i64[0] != &v167 )
      j_j___libc_free_0(v166.m128i_u64[0]);
    sub_C88FF0(&v159);
  }
}
