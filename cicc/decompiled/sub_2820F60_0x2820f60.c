// Function: sub_2820F60
// Address: 0x2820f60
//
__int64 __fastcall sub_2820F60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int16 a4,
        unsigned __int8 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        char a10,
        char a11)
{
  __int64 **v14; // rbx
  __int64 v15; // rax
  int v16; // r14d
  unsigned __int64 v17; // rsi
  int v18; // eax
  unsigned __int64 v19; // rsi
  bool v20; // cf
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 *v27; // rax
  _QWORD *v28; // r14
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 *v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned int v34; // r12d
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rcx
  char v38; // al
  __int64 *v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 *v44; // r14
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 *v47; // rax
  __int64 v48; // rdx
  __int64 *v49; // r14
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 *v53; // r14
  __int64 *v54; // rdi
  __int64 v55; // r12
  __int64 v56; // rax
  unsigned __int8 *v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // r14
  __int64 v63; // rdx
  unsigned __int64 *v64; // r13
  unsigned __int64 *v65; // r12
  unsigned __int64 v66; // rdi
  __int64 *v67; // rax
  __int64 *v68; // r14
  __int64 v69; // r13
  __int64 *v70; // r12
  __int64 *v71; // r13
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 *v75; // rax
  __m128i v76; // xmm1
  __int64 *v77; // rdi
  int v78; // ecx
  __int64 v79; // rsi
  __int64 v80; // rcx
  unsigned int v81; // edx
  __int64 *v82; // rax
  __int64 v83; // r8
  __int64 v84; // rsi
  __int64 *v85; // rax
  int v86; // eax
  __int64 *v87; // rax
  __int64 *v88; // r12
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  int v91; // edx
  unsigned __int64 v92; // r14
  _QWORD *v93; // rdx
  _QWORD *v94; // rax
  __int64 v95; // r12
  __int64 v96; // rax
  _QWORD *v97; // rbx
  char *v98; // rax
  __int64 v99; // rdx
  char *v100; // rax
  __int64 v101; // rdx
  _QWORD *v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // [rsp-8h] [rbp-768h]
  _QWORD *v106; // [rsp+8h] [rbp-758h]
  unsigned __int8 v107; // [rsp+25h] [rbp-73Bh]
  __int64 *v109; // [rsp+30h] [rbp-730h]
  __int64 v110; // [rsp+38h] [rbp-728h]
  __int64 **v111; // [rsp+38h] [rbp-728h]
  __int64 v112; // [rsp+40h] [rbp-720h]
  __int64 *v113; // [rsp+40h] [rbp-720h]
  __int64 v115; // [rsp+50h] [rbp-710h]
  __int64 v116; // [rsp+58h] [rbp-708h]
  _QWORD *v117; // [rsp+58h] [rbp-708h]
  _QWORD *v118; // [rsp+78h] [rbp-6E8h]
  _QWORD *v119; // [rsp+78h] [rbp-6E8h]
  __int64 v120; // [rsp+80h] [rbp-6E0h]
  __int64 *v121; // [rsp+80h] [rbp-6E0h]
  __int64 v122; // [rsp+80h] [rbp-6E0h]
  int v123; // [rsp+80h] [rbp-6E0h]
  __int64 v124; // [rsp+80h] [rbp-6E0h]
  __int64 v125; // [rsp+88h] [rbp-6D8h]
  __m128i v126; // [rsp+90h] [rbp-6D0h] BYREF
  __m128i v127; // [rsp+A0h] [rbp-6C0h] BYREF
  _BYTE *v128; // [rsp+B0h] [rbp-6B0h] BYREF
  char v129; // [rsp+B8h] [rbp-6A8h]
  __m128i v130; // [rsp+C0h] [rbp-6A0h] BYREF
  __m128i v131; // [rsp+D0h] [rbp-690h]
  unsigned __int64 v132[2]; // [rsp+E0h] [rbp-680h] BYREF
  _QWORD v133[2]; // [rsp+F0h] [rbp-670h] BYREF
  _QWORD *v134; // [rsp+100h] [rbp-660h]
  _QWORD v135[4]; // [rsp+110h] [rbp-650h] BYREF
  __m128i v136; // [rsp+130h] [rbp-630h] BYREF
  _QWORD v137[2]; // [rsp+140h] [rbp-620h] BYREF
  _QWORD *v138; // [rsp+150h] [rbp-610h]
  _QWORD v139[4]; // [rsp+160h] [rbp-600h] BYREF
  unsigned int *v140[2]; // [rsp+180h] [rbp-5E0h] BYREF
  _BYTE v141[32]; // [rsp+190h] [rbp-5D0h] BYREF
  __int64 v142; // [rsp+1B0h] [rbp-5B0h]
  __int64 v143; // [rsp+1B8h] [rbp-5A8h]
  __int16 v144; // [rsp+1C0h] [rbp-5A0h]
  __int64 *v145; // [rsp+1C8h] [rbp-598h]
  void **v146; // [rsp+1D0h] [rbp-590h]
  void **v147; // [rsp+1D8h] [rbp-588h]
  __int64 v148; // [rsp+1E0h] [rbp-580h]
  int v149; // [rsp+1E8h] [rbp-578h]
  __int16 v150; // [rsp+1ECh] [rbp-574h]
  char v151; // [rsp+1EEh] [rbp-572h]
  __int64 v152; // [rsp+1F0h] [rbp-570h]
  __int64 v153; // [rsp+1F8h] [rbp-568h]
  void *v154; // [rsp+200h] [rbp-560h] BYREF
  void *v155; // [rsp+208h] [rbp-558h] BYREF
  const char *v156; // [rsp+210h] [rbp-550h] BYREF
  __int64 v157; // [rsp+218h] [rbp-548h]
  _QWORD *v158; // [rsp+220h] [rbp-540h] BYREF
  _QWORD *v159; // [rsp+228h] [rbp-538h]
  __int64 v160; // [rsp+230h] [rbp-530h]
  unsigned __int64 *v161; // [rsp+260h] [rbp-500h]
  unsigned int v162; // [rsp+268h] [rbp-4F8h]
  char v163; // [rsp+270h] [rbp-4F0h] BYREF
  _BYTE v164[928]; // [rsp+3C0h] [rbp-3A0h] BYREF

  v14 = (__int64 **)a1;
  v109 = (__int64 *)sub_B43CA0(a6);
  v110 = 0;
  v112 = sub_98A180(a5, *(_QWORD *)(a1 + 56));
  if ( !v112 )
    v110 = sub_281EA60((__int64)a5, *(_BYTE **)(a1 + 56));
  v15 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  v16 = *(_DWORD *)(v15 + 8) >> 8;
  v125 = sub_D4B130(*(_QWORD *)a1);
  v116 = v125 + 48;
  v17 = *(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v125 + 48 == v17 )
  {
    v22 = 0;
  }
  else
  {
    if ( !v17 )
      BUG();
    v18 = *(unsigned __int8 *)(v17 - 24);
    v19 = v17 - 24;
    v20 = (unsigned int)(v18 - 30) < 0xB;
    v21 = 0;
    if ( v20 )
      v21 = v19;
    v22 = v21;
  }
  v145 = (__int64 *)sub_BD5C60(v22);
  v146 = &v154;
  v147 = &v155;
  v150 = 512;
  v144 = 0;
  v154 = &unk_49DA100;
  v155 = &unk_49DA0B0;
  v140[0] = (unsigned int *)v141;
  v140[1] = (unsigned int *)0x200000000LL;
  v148 = 0;
  v149 = 0;
  v151 = 7;
  v152 = 0;
  v153 = 0;
  v142 = 0;
  v143 = 0;
  sub_D5F1F0((__int64)v140, v22);
  sub_27C1C30((__int64)v164, *(__int64 **)(a1 + 32), *(_QWORD *)(a1 + 56), (__int64)"loop-idiom", 1);
  v128 = v164;
  v129 = 0;
  v115 = sub_BCE3C0(v145, v16);
  v120 = sub_AE4570(*(_QWORD *)(a1 + 56), *(_QWORD *)(a2 + 8));
  v27 = *(__int64 **)(a8 + 32);
  v28 = (_QWORD *)*v27;
  if ( a10 )
    v28 = sub_281DF50(*v27, a9, v120, a3, *(__int64 **)(a1 + 32));
  v29 = (__int64)v28;
  v34 = sub_F80610((__int64)v164, (__int64)v28, v23, v24, v25, v26);
  if ( !(_BYTE)v34 )
    goto LABEL_13;
  v36 = *(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v116 == v36 )
  {
    v37 = 0;
  }
  else
  {
    if ( !v36 )
      BUG();
    v37 = v36 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 >= 0xB )
      v37 = 0;
  }
  v29 = 3;
  v106 = sub_F8DB90((__int64)v164, (__int64)v28, v115, v37 + 24, 0);
  v38 = sub_281E0D0((__int64)v106, 3u, *(_QWORD *)a1, a9, a3, *(_QWORD **)(a1 + 8), a7);
  v32 = v105;
  if ( v38 )
    goto LABEL_13;
  v31 = *(__int64 **)a1;
  if ( *(_BYTE *)(a1 + 72) )
  {
    if ( (unsigned int)((v31[5] - v31[4]) >> 3) > 1 && !*v31 && !a11 )
      goto LABEL_13;
  }
  v39 = *(__int64 **)(a1 + 32);
  v118 = sub_DE5A20(v39, a9, v120, (__int64)v31);
  v159 = sub_DC5760((__int64)v39, a3, v120, 0);
  v156 = (const char *)&v158;
  v158 = v118;
  v157 = 0x200000002LL;
  v44 = sub_DC8BD0(v39, (__int64)&v156, 2u, 0);
  if ( v156 != (const char *)&v158 )
    _libc_free((unsigned __int64)v156);
  v29 = (__int64)v44;
  v107 = sub_F80610((__int64)v164, (__int64)v44, v40, v41, v42, v43);
  if ( !v107 )
    goto LABEL_13;
  v45 = *(_QWORD *)(v125 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v116 == v45 )
  {
    v46 = 0;
  }
  else
  {
    if ( !v45 )
      BUG();
    v46 = v45 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v45 - 24) - 30 >= 0xB )
      v46 = 0;
  }
  v119 = sub_F8DB90((__int64)v164, (__int64)v44, v120, v46 + 24, 0);
  if ( !v112 )
  {
    v29 = *(_QWORD *)(a1 + 40);
    if ( !sub_11C99B0(v109, (__int64 *)v29, 0x16Bu) )
      goto LABEL_13;
  }
  sub_B91FC0(v130.m128i_i64, a6);
  v47 = *(__int64 **)(a7 + 8);
  if ( *(_BYTE *)(a7 + 28) )
    v48 = *(unsigned int *)(a7 + 20);
  else
    v48 = *(unsigned int *)(a7 + 16);
  v49 = &v47[v48];
  if ( v47 != v49 )
  {
    while ( 1 )
    {
      v50 = *v47;
      if ( (unsigned __int64)*v47 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v49 == ++v47 )
        goto LABEL_38;
    }
    if ( v49 != v47 )
    {
      v71 = v47;
      do
      {
        sub_B91FC0((__int64 *)&v156, v50);
        sub_E01E30(v126.m128i_i64, v130.m128i_i64, (__int64 *)&v156, v72, v73, v74);
        v75 = v71 + 1;
        v76 = _mm_loadu_si128(&v127);
        v130 = _mm_loadu_si128(&v126);
        v131 = v76;
        if ( v71 + 1 == v49 )
          break;
        while ( 1 )
        {
          v50 = *v75;
          v71 = v75;
          if ( (unsigned __int64)*v75 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v49 == ++v75 )
            goto LABEL_94;
        }
      }
      while ( v49 != v75 );
LABEL_94:
      v14 = (__int64 **)a1;
    }
  }
LABEL_38:
  v51 = v130.m128i_i64[0];
  if ( *(_BYTE *)v119 == 17 )
  {
    v52 = v119[3];
    if ( *((_DWORD *)v119 + 8) > 0x40u )
      v52 = *(_QWORD *)v52;
    if ( v130.m128i_i64[0] )
      goto LABEL_42;
  }
  else if ( v130.m128i_i64[0] )
  {
    v52 = -1;
LABEL_42:
    v51 = sub_E00A50(v130.m128i_i64[0], v52);
  }
  v130.m128i_i64[0] = v51;
  if ( v112 )
  {
    v53 = (__int64 *)sub_B34240(
                       (__int64)v140,
                       (__int64)v106,
                       v112,
                       (__int64)v119,
                       a4,
                       0,
                       v130.m128i_i64[0],
                       v131.m128i_i64[0],
                       v131.m128i_i64[1]);
  }
  else
  {
    v87 = (__int64 *)sub_BCB120(v145);
    v88 = v14[5];
    v158 = (_QWORD *)v115;
    v159 = (_QWORD *)v115;
    v156 = (const char *)&v158;
    v160 = v120;
    v157 = 0x300000003LL;
    v89 = sub_BCF480(v87, &v158, 3, 0);
    v90 = sub_11C96C0((__int64)v109, v88, 0x16Bu, v89, 0);
    v123 = v91;
    v92 = v90;
    if ( v156 != (const char *)&v158 )
      _libc_free((unsigned __int64)v156);
    sub_11C9500((__int64)v109, (__int64)"memset_pattern16", 0x10u, v14[5]);
    v93 = *(_QWORD **)(v110 + 8);
    v156 = ".memset_pattern";
    v117 = v93;
    LOWORD(v160) = 259;
    v136.m128i_i8[4] = 0;
    v94 = sub_BD2C40(88, unk_3F0FAE8);
    v95 = (__int64)v94;
    if ( v94 )
      sub_B30000((__int64)v94, (__int64)v109, v117, 1, 8, v110, (__int64)&v156, 0, 0, v136.m128i_i64[0], 0);
    *(_BYTE *)(v95 + 32) = *(_BYTE *)(v95 + 32) & 0x3F | 0x80;
    sub_B2F770(v95, 4u);
    LOWORD(v160) = 257;
    v136.m128i_i64[0] = (__int64)v106;
    v136.m128i_i64[1] = v95;
    v137[0] = v119;
    v96 = sub_921880(v140, v92, v123, (int)&v136, 3, (__int64)&v156, 0);
    v53 = (__int64 *)v96;
    if ( v130.m128i_i64[0] )
      sub_B99FD0(v96, 1u, v130.m128i_i64[0]);
    if ( v131.m128i_i64[0] )
      sub_B99FD0((__int64)v53, 7u, v131.m128i_i64[0]);
    if ( v131.m128i_i64[1] )
      sub_B99FD0((__int64)v53, 8u, v131.m128i_i64[1]);
  }
  v121 = v53 + 6;
  v29 = *(_QWORD *)(a6 + 48);
  v156 = (const char *)v29;
  if ( v29 )
  {
    sub_B96E90((__int64)&v156, v29, 1);
    if ( v53 + 6 == (__int64 *)&v156 )
    {
      v29 = (__int64)v156;
      if ( v156 )
        sub_B91220((__int64)&v156, (__int64)v156);
      goto LABEL_49;
    }
    v29 = v53[6];
    if ( !v29 )
    {
LABEL_102:
      v29 = (__int64)v156;
      v53[6] = (__int64)v156;
      if ( v29 )
        sub_B976B0((__int64)&v156, (unsigned __int8 *)v29, (__int64)v121);
      goto LABEL_49;
    }
LABEL_101:
    sub_B91220((__int64)v121, v29);
    goto LABEL_102;
  }
  if ( v121 != (__int64 *)&v156 )
  {
    v29 = v53[6];
    if ( v29 )
      goto LABEL_101;
  }
LABEL_49:
  v54 = v14[10];
  if ( v54 )
  {
    v29 = sub_D694D0(v54, (__int64)v53, 0, v53[5], 2u, 1u);
    sub_D75120(v14[10], (__int64 *)v29, 1);
  }
  v55 = *v14[8];
  v113 = v14[8];
  v56 = sub_B2BE50(v55);
  if ( sub_B6EA50(v56)
    || (v103 = sub_B2BE50(v55),
        v104 = sub_B6F970(v103),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v104 + 48LL))(v104)) )
  {
    sub_B157E0((__int64)&v136, v121);
    sub_B17430((__int64)&v156, (__int64)"loop-idiom", (__int64)"ProcessLoopStridedStore", 23, &v136, v125);
    sub_B18290((__int64)&v156, "Transformed loop-strided store in ", 0x22u);
    v57 = (unsigned __int8 *)sub_B43CB0(a6);
    sub_B16080((__int64)&v136, "Function", 8, v57);
    v122 = sub_23FD640((__int64)&v156, (__int64)&v136);
    sub_B18290(v122, " function into a call to ", 0x19u);
    v58 = *(v53 - 4);
    if ( v58 )
    {
      if ( *(_BYTE *)v58 )
      {
        v58 = 0;
      }
      else if ( *(_QWORD *)(v58 + 24) != v53[10] )
      {
        v58 = 0;
      }
    }
    sub_B16080((__int64)v132, "NewFunction", 11, (unsigned __int8 *)v58);
    v59 = sub_23FD640(v122, (__int64)v132);
    sub_B18290(v59, "() intrinsic", 0xCu);
    if ( v134 != v135 )
      j_j___libc_free_0((unsigned __int64)v134);
    if ( (_QWORD *)v132[0] != v133 )
      j_j___libc_free_0(v132[0]);
    if ( v138 != v139 )
      j_j___libc_free_0((unsigned __int64)v138);
    if ( (_QWORD *)v136.m128i_i64[0] != v137 )
      j_j___libc_free_0(v136.m128i_u64[0]);
    if ( *(_DWORD *)(a7 + 20) != *(_DWORD *)(a7 + 24) )
      sub_B17B50((__int64)&v156);
    v60 = *(_QWORD **)(a7 + 8);
    if ( *(_BYTE *)(a7 + 28) )
      v61 = *(unsigned int *)(a7 + 20);
    else
      v61 = *(unsigned int *)(a7 + 16);
    v62 = &v60[v61];
    if ( v60 != v62 )
    {
      while ( 1 )
      {
        v63 = *v60;
        if ( *v60 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v62 == ++v60 )
          goto LABEL_71;
      }
      if ( v62 != v60 )
      {
        v111 = v14;
        v97 = v60;
        do
        {
          v98 = (char *)sub_BD5D20(*(_QWORD *)(v63 + 40));
          sub_B16430((__int64)&v136, "FromBlock", 9u, v98, v99);
          v124 = sub_23FD640((__int64)&v156, (__int64)&v136);
          v100 = (char *)sub_BD5D20(v125);
          sub_B16430((__int64)v132, "ToBlock", 7u, v100, v101);
          sub_23FD640(v124, (__int64)v132);
          if ( v134 != v135 )
            j_j___libc_free_0((unsigned __int64)v134);
          if ( (_QWORD *)v132[0] != v133 )
            j_j___libc_free_0(v132[0]);
          if ( v138 != v139 )
            j_j___libc_free_0((unsigned __int64)v138);
          if ( (_QWORD *)v136.m128i_i64[0] != v137 )
            j_j___libc_free_0(v136.m128i_u64[0]);
          v102 = v97 + 1;
          if ( v97 + 1 == v62 )
            break;
          while ( 1 )
          {
            v63 = *v102;
            v97 = v102;
            if ( *v102 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v62 == ++v102 )
              goto LABEL_150;
          }
        }
        while ( v62 != v102 );
LABEL_150:
        v14 = v111;
      }
    }
LABEL_71:
    v29 = (__int64)&v156;
    sub_1049740(v113, (__int64)&v156);
    v64 = v161;
    v156 = (const char *)&unk_49D9D40;
    v65 = &v161[10 * v162];
    if ( v161 != v65 )
    {
      do
      {
        v65 -= 10;
        v66 = v65[4];
        if ( (unsigned __int64 *)v66 != v65 + 6 )
        {
          v29 = v65[6] + 1;
          j_j___libc_free_0(v66);
        }
        if ( (unsigned __int64 *)*v65 != v65 + 2 )
        {
          v29 = v65[2] + 1;
          j_j___libc_free_0(*v65);
        }
      }
      while ( v64 != v65 );
      v65 = v161;
    }
    if ( v65 != (unsigned __int64 *)&v163 )
      _libc_free((unsigned __int64)v65);
  }
  v31 = (__int64 *)a7;
  v67 = *(__int64 **)(a7 + 8);
  if ( *(_BYTE *)(a7 + 28) )
  {
    v30 = *(unsigned int *)(a7 + 20);
  }
  else
  {
    v31 = (__int64 *)a7;
    v30 = *(unsigned int *)(a7 + 16);
  }
  v68 = &v67[v30];
  if ( v67 != v68 )
  {
    while ( 1 )
    {
      v69 = *v67;
      v70 = v67;
      if ( (unsigned __int64)*v67 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v68 == ++v67 )
        goto LABEL_85;
    }
    while ( v68 != v70 )
    {
      v77 = v14[10];
      if ( v77 )
      {
        v78 = *(_DWORD *)(*v77 + 56);
        v79 = *(_QWORD *)(*v77 + 40);
        if ( v78 )
        {
          v80 = (unsigned int)(v78 - 1);
          v81 = v80 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
          v82 = (__int64 *)(v79 + 16LL * v81);
          v83 = *v82;
          if ( v69 == *v82 )
          {
LABEL_112:
            v84 = v82[1];
            if ( v84 )
              sub_D6E4B0(v77, v84, 1, v80, v83, v33);
          }
          else
          {
            v86 = 1;
            while ( v83 != -4096 )
            {
              v33 = (unsigned int)(v86 + 1);
              v81 = v80 & (v86 + v81);
              v82 = (__int64 *)(v79 + 16LL * v81);
              v83 = *v82;
              if ( v69 == *v82 )
                goto LABEL_112;
              v86 = v33;
            }
          }
        }
      }
      v29 = sub_ACADE0(*(__int64 ***)(v69 + 8));
      sub_BD84D0(v69, v29);
      sub_B43D60((_QWORD *)v69);
      v85 = v70 + 1;
      if ( v70 + 1 == v68 )
        break;
      while ( 1 )
      {
        v69 = *v85;
        v70 = v85;
        if ( (unsigned __int64)*v85 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v68 == ++v85 )
          goto LABEL_85;
      }
    }
  }
LABEL_85:
  if ( v14[10] )
  {
    v30 = (__int64)byte_4F8F8E8;
    if ( byte_4F8F8E8[0] )
    {
      v29 = 0;
      nullsub_390();
    }
  }
  v129 = 1;
  v34 = v107;
LABEL_13:
  sub_F82D10((__int64)&v128, v29, v30, v31, v32, v33);
  sub_27C20B0((__int64)v164);
  nullsub_61();
  v154 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v140[0] != v141 )
    _libc_free((unsigned __int64)v140[0]);
  return v34;
}
