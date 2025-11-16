// Function: sub_23635D0
// Address: 0x23635d0
//
__int64 *__fastcall sub_23635D0(__m128i *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 *v5; // r14
  __int64 *v6; // r14
  __int64 *v7; // r14
  __int64 *v8; // r14
  __int64 *v9; // r14
  __int64 *v10; // r14
  __int64 *v11; // r14
  __int64 *v12; // r14
  __int64 *v13; // r14
  __int64 *v14; // r14
  __int64 *v15; // r14
  __int64 *v16; // r14
  __int64 *v17; // r14
  __int64 *v18; // r14
  __int64 *v19; // r14
  __int64 *v20; // r14
  __int64 *v21; // r14
  __int64 *v22; // r14
  __int64 *v23; // r14
  __int64 *v24; // r14
  __int64 *v25; // r14
  __int64 *v26; // r14
  __int64 *v27; // r14
  __int64 *v28; // r14
  __int64 *v29; // r14
  __int64 *v30; // r14
  __int64 *v31; // r14
  __int64 *v32; // r14
  __int64 *v33; // r14
  __int64 *v34; // r14
  __int64 *v35; // r14
  __int64 *v36; // r14
  __int64 *v37; // r14
  __int64 *v38; // r14
  __int64 *v39; // r14
  __int64 *v40; // r14
  __int64 *v41; // r14
  __int64 *v42; // r14
  __int64 *v43; // r14
  __int64 *v44; // r14
  __int64 *v45; // r14
  __int64 *v46; // r14
  __m128i *v47; // rsi
  __int64 v48; // rdi
  __int64 *result; // rax
  void *v50; // rdx
  __int64 *v51; // r12
  __int64 v52; // r12
  __int64 i; // r13
  __m128i *v54; // rsi
  __int64 *v55; // r14
  _QWORD *v56; // rax
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 v60; // rdi
  _QWORD *v61; // rax
  __int64 v62; // rdi
  _QWORD *v63; // rax
  __int64 v64; // rdi
  _QWORD *v65; // rax
  __int64 v66; // rdi
  _QWORD *v67; // rax
  __int64 v68; // rdi
  _QWORD *v69; // rax
  __int64 v70; // rdi
  _QWORD *v71; // rax
  __int64 v72; // rdi
  _QWORD *v73; // rax
  __int64 v74; // rdi
  _QWORD *v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rax
  __int64 v78; // r15
  bool v79; // zf
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // r14
  __int64 v83; // r15
  __int64 v84; // rax
  __int64 v85; // rcx
  __m128i v86; // xmm0
  __m128i v87; // xmm1
  __int64 v88; // rdx
  __int64 v89; // rdx
  __int64 v90; // rdi
  _QWORD *v91; // rax
  __int64 v92; // rdi
  _QWORD *v93; // rax
  __int64 v94; // rdi
  _QWORD *v95; // rax
  __int64 v96; // rdi
  _QWORD *v97; // rax
  __int64 v98; // rdi
  _QWORD *v99; // rax
  __int64 v100; // rdi
  _QWORD *v101; // rax
  __int64 v102; // rdi
  _QWORD *v103; // rax
  __int64 v104; // rdi
  _QWORD *v105; // rax
  __int64 v106; // rdi
  __int64 v107; // r15
  _QWORD *v108; // rax
  __int64 v109; // rdi
  _QWORD *v110; // rax
  __int64 v111; // rdi
  _QWORD *v112; // rax
  __int64 v113; // rdi
  _QWORD *v114; // rax
  __int64 v115; // rdi
  __int32 v116; // r15d
  __int64 v117; // rax
  __int64 v118; // rdi
  _QWORD *v119; // rax
  __int64 v120; // rdi
  _QWORD *v121; // rax
  __int64 v122; // rdi
  _QWORD *v123; // rax
  __int64 v124; // rdi
  _QWORD *v125; // rax
  _QWORD *v126; // r15
  __int64 v127; // rdi
  _QWORD *v128; // rax
  __int64 v129; // rdi
  __int64 v130; // r15
  _QWORD *v131; // rax
  __int64 v132; // rdi
  _QWORD *v133; // rax
  __int64 v134; // rdi
  _QWORD *v135; // rax
  __int64 v136; // rdi
  _QWORD *v137; // rax
  __int64 v138; // rdi
  _QWORD *v139; // rax
  __int64 v140; // rdi
  _QWORD *v141; // rax
  __int64 v142; // rdi
  _QWORD *v143; // rax
  __int64 v144; // rdi
  _QWORD *v145; // rax
  __int64 v146; // rdi
  _QWORD *v147; // rax
  __int64 v148; // rdi
  _QWORD *v149; // rax
  __int64 v150; // rdi
  __int64 v151; // r15
  _QWORD *v152; // rax
  __int64 v153; // rdi
  _QWORD *v154; // rax
  __int64 v155; // rdi
  _QWORD *v156; // rax
  __int64 v157; // rdi
  _QWORD *v158; // rax
  __int64 v159; // r8
  __int64 v160; // r9
  __int64 v161; // rdi
  _QWORD *v162; // [rsp+8h] [rbp-128h]
  _QWORD *v163; // [rsp+8h] [rbp-128h]
  __m128i v164[18]; // [rsp+10h] [rbp-120h] BYREF

  v164[0].m128i_i64[0] = (__int64)&unk_4F86540;
  v4 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v4 )
  {
    v54 = a1;
    v55 = v4;
    sub_23A1380(v164);
    v56 = (_QWORD *)sub_22077B0(0x38u);
    if ( v56 )
    {
      v56[2] = 0x400000000LL;
      v59 = v164[0].m128i_u32[2];
      *v56 = &unk_4A089C8;
      v56[1] = v56 + 3;
      if ( (_DWORD)v59 )
      {
        v54 = v164;
        v163 = v56;
        sub_23038C0((__int64)(v56 + 1), (char **)v164, (__int64)(v56 + 3), v59, v57, v58);
        v56 = v163;
      }
    }
    v60 = *v55;
    *v55 = (__int64)v56;
    if ( v60 )
      (*(void (__fastcall **)(__int64, __m128i *))(*(_QWORD *)v60 + 8LL))(v60, v54);
    if ( (__m128i *)v164[0].m128i_i64[0] != &v164[1] )
      _libc_free(v164[0].m128i_u64[0]);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F86540;
  v5 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v5 )
  {
    v164[2].m128i_i64[1] = 0;
    *(__m128i *)((char *)v164 + 8) = 0;
    v164[0].m128i_i64[0] = (__int64)v164[1].m128i_i64;
    v164[0].m128i_i32[3] = 4;
    *(__m128i *)((char *)&v164[1] + 8) = 0;
    v158 = (_QWORD *)sub_22077B0(0x38u);
    if ( v158 )
    {
      v158[2] = 0x400000000LL;
      *v158 = &unk_4A089C8;
      v158[1] = v158 + 3;
      if ( v164[0].m128i_i32[2] )
      {
        v162 = v158;
        sub_23038C0((__int64)(v158 + 1), (char **)v164, v164[0].m128i_u32[2], 0x400000000LL, v159, v160);
        v158 = v162;
      }
    }
    v161 = *v5;
    *v5 = (__int64)v158;
    if ( v161 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v161 + 8LL))(v161);
    if ( (__m128i *)v164[0].m128i_i64[0] != &v164[1] )
      _libc_free(v164[0].m128i_u64[0]);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F86D28;
  v6 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v6 )
  {
    v156 = (_QWORD *)sub_22077B0(0x10u);
    if ( v156 )
      *v156 = &unk_4A0BAD8;
    v157 = *v6;
    *v6 = (__int64)v156;
    if ( v157 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v157 + 8LL))(v157);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F86630;
  v7 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v7 )
  {
    v154 = (_QWORD *)sub_22077B0(0x10u);
    if ( v154 )
      *v154 = &unk_4A0BB08;
    v155 = *v7;
    *v7 = (__int64)v154;
    if ( v155 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v155 + 8LL))(v155);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5016950;
  v8 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v8 )
  {
    v151 = a1->m128i_i64[0];
    v152 = (_QWORD *)sub_22077B0(0x10u);
    if ( v152 )
    {
      v152[1] = v151;
      *v152 = &unk_4A0BB38;
    }
    v153 = *v8;
    *v8 = (__int64)v152;
    if ( v153 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v153 + 8LL))(v153);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8D9A8;
  v9 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v9 )
  {
    v149 = (_QWORD *)sub_22077B0(0x10u);
    if ( v149 )
      *v149 = &unk_4A0BB68;
    v150 = *v9;
    *v9 = (__int64)v149;
    if ( v150 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v150 + 8LL))(v150);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8E5A8;
  v10 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v10 )
  {
    v147 = (_QWORD *)sub_22077B0(0x10u);
    if ( v147 )
      *v147 = &unk_4A0BB98;
    v148 = *v10;
    *v10 = (__int64)v147;
    if ( v148 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v148 + 8LL))(v148);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F92388;
  v11 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v11 )
  {
    v145 = (_QWORD *)sub_22077B0(0x10u);
    if ( v145 )
      *v145 = &unk_4A0BBC8;
    v146 = *v11;
    *v11 = (__int64)v145;
    if ( v146 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v146 + 8LL))(v146);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDB350;
  v12 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v12 )
  {
    v143 = (_QWORD *)sub_22077B0(0x10u);
    if ( v143 )
      *v143 = &unk_4A0BBF8;
    v144 = *v12;
    *v12 = (__int64)v143;
    if ( v144 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v144 + 8LL))(v144);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_50165D8;
  v13 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v13 )
  {
    v141 = (_QWORD *)sub_22077B0(0x10u);
    if ( v141 )
      *v141 = &unk_4A0BC28;
    v142 = *v13;
    *v13 = (__int64)v141;
    if ( v142 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v142 + 8LL))(v142);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F86B68;
  v14 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v14 )
  {
    v139 = (_QWORD *)sub_22077B0(0x10u);
    if ( v139 )
      *v139 = &unk_4A0BC58;
    v140 = *v14;
    *v14 = (__int64)v139;
    if ( v140 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v140 + 8LL))(v140);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDB678;
  v15 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v15 )
  {
    v137 = (_QWORD *)sub_22077B0(0x10u);
    if ( v137 )
      *v137 = &unk_4A0BC88;
    v138 = *v15;
    *v15 = (__int64)v137;
    if ( v138 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v138 + 8LL))(v138);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F81450;
  v16 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v16 )
  {
    v135 = (_QWORD *)sub_22077B0(0x10u);
    if ( v135 )
      *v135 = &unk_4A0BCB8;
    v136 = *v16;
    *v16 = (__int64)v135;
    if ( v136 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v136 + 8LL))(v136);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_502ED90;
  v17 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v17 )
  {
    v133 = (_QWORD *)sub_22077B0(0x10u);
    if ( v133 )
      *v133 = &unk_4A0BCE8;
    v134 = *v17;
    *v17 = (__int64)v133;
    if ( v134 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v134 + 8LL))(v134);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5020008;
  v18 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v18 )
  {
    v130 = a1->m128i_i64[0];
    v131 = (_QWORD *)sub_22077B0(0x10u);
    if ( v131 )
    {
      v131[1] = v130;
      *v131 = &unk_4A0BD18;
    }
    v132 = *v18;
    *v18 = (__int64)v131;
    if ( v132 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v132 + 8LL))(v132);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_501DA10;
  v19 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v19 )
  {
    v128 = (_QWORD *)sub_22077B0(0x10u);
    if ( v128 )
      *v128 = &unk_4A0BD48;
    v129 = *v19;
    *v19 = (__int64)v128;
    if ( v129 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v129 + 8LL))(v129);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5031220;
  v20 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v20 )
  {
    sub_30EBEE0(v164);
    v125 = (_QWORD *)sub_22077B0(0x10u);
    v126 = v125;
    if ( v125 )
    {
      *v125 = &unk_4A0BD78;
      sub_30EBEF0(v125 + 1, v164);
    }
    v127 = *v20;
    *v20 = (__int64)v126;
    if ( v127 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v127 + 8LL))(v127);
    sub_30EBF00(v164);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8ED68;
  v21 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v21 )
  {
    v123 = (_QWORD *)sub_22077B0(0x10u);
    if ( v123 )
      *v123 = &unk_4A0BDA8;
    v124 = *v21;
    *v21 = (__int64)v123;
    if ( v124 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v124 + 8LL))(v124);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDBCC8;
  v22 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v22 )
  {
    v121 = (_QWORD *)sub_22077B0(0x10u);
    if ( v121 )
      *v121 = &unk_4A0BDD8;
    v122 = *v22;
    *v22 = (__int64)v121;
    if ( v122 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v122 + 8LL))(v122);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F875F0;
  v23 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v23 )
  {
    v119 = (_QWORD *)sub_22077B0(0x10u);
    if ( v119 )
      *v119 = &unk_4A0BE08;
    v120 = *v23;
    *v23 = (__int64)v119;
    if ( v120 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v120 + 8LL))(v120);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8EE60;
  v24 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v24 )
  {
    sub_102BD30(v164);
    v116 = v164[0].m128i_i32[0];
    v117 = sub_22077B0(0x10u);
    if ( v117 )
    {
      *(_DWORD *)(v117 + 8) = v116;
      *(_QWORD *)v117 = &unk_4A0BE38;
    }
    v118 = *v24;
    *v24 = v117;
    if ( v118 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v118 + 8LL))(v118);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8F810;
  v25 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v25 )
  {
    v114 = (_QWORD *)sub_22077B0(0x10u);
    if ( v114 )
      *v114 = &unk_4A0BE68;
    v115 = *v25;
    *v25 = (__int64)v114;
    if ( v115 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v115 + 8LL))(v115);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDC278;
  v26 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v26 )
  {
    v112 = (_QWORD *)sub_22077B0(0x10u);
    if ( v112 )
      *v112 = &unk_4A0BE98;
    v113 = *v26;
    *v26 = (__int64)v112;
    if ( v113 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v113 + 8LL))(v113);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8FAE8;
  v27 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v27 )
  {
    v110 = (_QWORD *)sub_22077B0(0x10u);
    if ( v110 )
      *v110 = &unk_4A0BEC8;
    v111 = *v27;
    *v27 = (__int64)v110;
    if ( v111 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v111 + 8LL))(v111);
  }
  v164[0].m128i_i64[0] = (__int64)&qword_4F8A320;
  v28 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v28 )
  {
    v107 = a1[12].m128i_i64[1];
    v108 = (_QWORD *)sub_22077B0(0x10u);
    if ( v108 )
    {
      v108[1] = v107;
      *v108 = &unk_4A0BEF8;
    }
    v109 = *v28;
    *v28 = (__int64)v108;
    if ( v109 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v109 + 8LL))(v109);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDBCF8;
  v29 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v29 )
  {
    v105 = (_QWORD *)sub_22077B0(0x10u);
    if ( v105 )
      *v105 = &unk_4A0BF28;
    v106 = *v29;
    *v29 = (__int64)v105;
    if ( v106 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v106 + 8LL))(v106);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8FBC8;
  v30 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v30 )
  {
    v103 = (_QWORD *)sub_22077B0(0x10u);
    if ( v103 )
      *v103 = &unk_4A0BF58;
    v104 = *v30;
    *v30 = (__int64)v103;
    if ( v104 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v104 + 8LL))(v104);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDBD00;
  v31 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v31 )
  {
    v101 = (_QWORD *)sub_22077B0(0x10u);
    if ( v101 )
      *v101 = &unk_4A0BF88;
    v102 = *v31;
    *v31 = (__int64)v101;
    if ( v102 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v102 + 8LL))(v102);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F881D0;
  v32 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v32 )
  {
    v99 = (_QWORD *)sub_22077B0(0x10u);
    if ( v99 )
      *v99 = &unk_4A0BFB8;
    v100 = *v32;
    *v32 = (__int64)v99;
    if ( v100 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v100 + 8LL))(v100);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4FDADD0;
  v33 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v33 )
  {
    v97 = (_QWORD *)sub_22077B0(0x10u);
    if ( v97 )
      *v97 = &unk_4A0BFE8;
    v98 = *v33;
    *v33 = (__int64)v97;
    if ( v98 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v98 + 8LL))(v98);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_500CD08;
  v34 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v34 )
  {
    v95 = (_QWORD *)sub_22077B0(0x10u);
    if ( v95 )
      *v95 = &unk_4A0C018;
    v96 = *v34;
    *v34 = (__int64)v95;
    if ( v96 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v96 + 8LL))(v96);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5026090;
  v35 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v35 )
  {
    v93 = (_QWORD *)sub_22077B0(0x10u);
    if ( v93 )
      *v93 = &unk_4A0C048;
    v94 = *v35;
    *v35 = (__int64)v93;
    if ( v94 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v94 + 8LL))(v94);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F87F28;
  v36 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v36 )
  {
    v91 = (_QWORD *)sub_22077B0(0x10u);
    if ( v91 )
      *v91 = &unk_4A0C078;
    v92 = *v36;
    *v36 = (__int64)v91;
    if ( v92 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v92 + 8LL))(v92);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F89C30;
  v37 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v37 )
  {
    if ( a1->m128i_i64[0] )
      sub_23CF4B0(v164);
    else
      sub_DFE980(v164);
    v84 = sub_22077B0(0x28u);
    if ( v84 )
    {
      v85 = *(_QWORD *)(v84 + 32);
      v86 = _mm_loadu_si128(v164);
      v87 = _mm_loadu_si128((const __m128i *)(v84 + 8));
      *(_QWORD *)v84 = &unk_4A0C0A8;
      v88 = v164[1].m128i_i64[0];
      v164[1].m128i_i64[0] = 0;
      *(_QWORD *)(v84 + 24) = v88;
      v89 = v164[1].m128i_i64[1];
      v164[1].m128i_i64[1] = v85;
      *(_QWORD *)(v84 + 32) = v89;
      v164[0] = v87;
      *(__m128i *)(v84 + 8) = v86;
    }
    v90 = *v37;
    *v37 = v84;
    if ( v90 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v90 + 8LL))(v90);
    sub_A17130((__int64)v164);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F6D3F8;
  v38 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v38 )
  {
    memset(v164, 0, 0xE8u);
    v77 = sub_22077B0(0xF0u);
    v78 = v77;
    if ( v77 )
    {
      *(_BYTE *)(v77 + 232) = 0;
      v79 = v164[14].m128i_i8[0] == 0;
      *(_QWORD *)v77 = &unk_4A089F8;
      if ( !v79 )
      {
        sub_97F4E0(v77 + 8, (__int64)v164);
        *(_BYTE *)(v78 + 232) = 1;
      }
    }
    v80 = *v38;
    *v38 = v78;
    if ( v80 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v80 + 8LL))(v80);
    if ( v164[14].m128i_i8[0] )
    {
      v164[14].m128i_i8[0] = 0;
      if ( v164[12].m128i_i64[1] )
        j_j___libc_free_0(v164[12].m128i_u64[1]);
      if ( v164[11].m128i_i64[0] )
        j_j___libc_free_0(v164[11].m128i_u64[0]);
      v81 = v164[10].m128i_u32[0];
      if ( v164[10].m128i_i32[0] )
      {
        v82 = v164[9].m128i_i64[0];
        v83 = v164[9].m128i_i64[0] + 40LL * v164[10].m128i_u32[0];
        do
        {
          if ( *(_DWORD *)v82 <= 0xFFFFFFFD )
            sub_2240A30((unsigned __int64 *)(v82 + 8));
          v82 += 40;
        }
        while ( v83 != v82 );
        v81 = v164[10].m128i_u32[0];
      }
      sub_C7D6A0(v164[9].m128i_i64[0], 40 * v81, 8);
    }
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8FC88;
  v39 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v39 )
  {
    v75 = (_QWORD *)sub_22077B0(0x10u);
    if ( v75 )
      *v75 = &unk_4A0C0D8;
    v76 = *v39;
    *v39 = (__int64)v75;
    if ( v76 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v76 + 8LL))(v76);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F836C8;
  v40 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v40 )
  {
    v73 = (_QWORD *)sub_22077B0(0x10u);
    if ( v73 )
      *v73 = &unk_4A0C108;
    v74 = *v40;
    *v40 = (__int64)v73;
    if ( v74 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v74 + 8LL))(v74);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F8D468;
  v41 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v41 )
  {
    v71 = (_QWORD *)sub_22077B0(0x10u);
    if ( v71 )
      *v71 = &unk_4A0C138;
    v72 = *v41;
    *v41 = (__int64)v71;
    if ( v72 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v72 + 8LL))(v72);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5010CC8;
  v42 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v42 )
  {
    v69 = (_QWORD *)sub_22077B0(0x10u);
    if ( v69 )
      *v69 = &unk_4A0C168;
    v70 = *v42;
    *v42 = (__int64)v69;
    if ( v70 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v70 + 8LL))(v70);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F86710;
  v43 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v43 )
  {
    v67 = (_QWORD *)sub_22077B0(0x10u);
    if ( v67 )
      *v67 = &unk_4A0C198;
    v68 = *v43;
    *v43 = (__int64)v67;
    if ( v68 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v68 + 8LL))(v68);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_5031CE0;
  v44 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v44 )
  {
    v65 = (_QWORD *)sub_22077B0(0x10u);
    if ( v65 )
      *v65 = &unk_4A0C1C8;
    v66 = *v44;
    *v44 = (__int64)v65;
    if ( v66 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v66 + 8LL))(v66);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F89B38;
  v45 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v45 )
  {
    v63 = (_QWORD *)sub_22077B0(0x10u);
    if ( v63 )
      *v63 = &unk_4A0C1F8;
    v64 = *v45;
    *v45 = (__int64)v63;
    if ( v64 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v64 + 8LL))(v64);
  }
  v164[0].m128i_i64[0] = (__int64)&unk_4F89B48;
  v46 = sub_2363490(a2, v164[0].m128i_i64);
  if ( !*v46 )
  {
    v61 = (_QWORD *)sub_22077B0(0x10u);
    if ( v61 )
      *v61 = &unk_4A0C228;
    v62 = *v46;
    *v46 = (__int64)v61;
    if ( v62 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v62 + 8LL))(v62);
  }
  v47 = v164;
  v48 = a2;
  v164[0].m128i_i64[0] = (__int64)&unk_4F89FB0;
  result = sub_2363490(a2, v164[0].m128i_i64);
  v51 = result;
  if ( !*result )
  {
    result = (__int64 *)sub_22077B0(0x10u);
    if ( result )
    {
      v50 = &unk_4A0C258;
      *result = (__int64)&unk_4A0C258;
    }
    v48 = *v51;
    *v51 = (__int64)result;
    if ( v48 )
      result = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v48 + 8LL))(v48);
  }
  v52 = a1[103].m128i_i64[0];
  for ( i = v52 + 32LL * a1[103].m128i_u32[2]; v52 != i; v52 += 32 )
  {
    if ( !*(_QWORD *)(v52 + 16) )
      sub_4263D6(v48, v47, v50);
    v48 = v52;
    v47 = (__m128i *)a2;
    result = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(v52 + 24))(v52, a2);
  }
  return result;
}
