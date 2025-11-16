// Function: sub_1BDB410
// Address: 0x1bdb410
//
__int64 __fastcall sub_1BDB410(
        __int64 a1,
        __int64 ***a2,
        unsigned __int64 a3,
        __int64 a4,
        int a5,
        char a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 ***v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  int v19; // r9d
  __int64 **v20; // rbx
  __int64 **v21; // r14
  __int64 v22; // r15
  __int64 *v23; // r14
  __int64 v24; // rax
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  _QWORD *v27; // rbx
  _QWORD *v28; // r13
  _QWORD *v29; // rdi
  _QWORD *v30; // rbx
  _QWORD *v31; // r12
  _QWORD *v32; // rdi
  unsigned int v33; // r15d
  int v35; // esi
  __int64 *v36; // rax
  __int64 ***v37; // r14
  __int64 *v38; // r15
  __int64 **v39; // rax
  unsigned int v40; // r14d
  unsigned int v41; // edx
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rbx
  unsigned int v44; // r13d
  void **v46; // rsi
  __int64 ***v47; // r9
  int v48; // r10d
  __int64 v49; // r11
  unsigned int v50; // r12d
  unsigned int v51; // ebx
  unsigned int v52; // edx
  __int64 v53; // rdi
  __int64 *v54; // r15
  __int64 *v55; // rcx
  __int64 *v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rbx
  __int64 *v59; // r12
  __int64 v60; // rax
  int v61; // r10d
  __int64 v62; // r11
  __int64 *v63; // r9
  char v64; // al
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // rdx
  __int64 v69; // rcx
  _QWORD *v70; // r8
  int v71; // r9d
  int v72; // eax
  int v73; // ecx
  int v74; // eax
  __int64 *v75; // r14
  __int64 v76; // rax
  __int64 v77; // r12
  __int64 v78; // rbx
  _QWORD *v79; // rbx
  _QWORD *v80; // r15
  _QWORD *v81; // rdi
  _QWORD *v82; // rbx
  _QWORD *v83; // r12
  _QWORD *v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 *v87; // r13
  __int64 v88; // rax
  __m128i v89; // xmm4
  __m128i v90; // xmm5
  double v91; // xmm4_8
  double v92; // xmm5_8
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 *v97; // [rsp+8h] [rbp-5F8h]
  __int64 *v98; // [rsp+8h] [rbp-5F8h]
  __int64 v99; // [rsp+18h] [rbp-5E8h]
  __int64 v100; // [rsp+18h] [rbp-5E8h]
  __int64 v101; // [rsp+18h] [rbp-5E8h]
  int v102; // [rsp+18h] [rbp-5E8h]
  unsigned int v103; // [rsp+20h] [rbp-5E0h]
  char v104; // [rsp+20h] [rbp-5E0h]
  __int64 v105; // [rsp+20h] [rbp-5E0h]
  unsigned int v106; // [rsp+28h] [rbp-5D8h]
  unsigned __int8 v107; // [rsp+28h] [rbp-5D8h]
  unsigned int v108; // [rsp+28h] [rbp-5D8h]
  __int64 v109; // [rsp+28h] [rbp-5D8h]
  __int64 v110; // [rsp+30h] [rbp-5D0h]
  __int64 v111; // [rsp+38h] [rbp-5C8h]
  unsigned int v112; // [rsp+38h] [rbp-5C8h]
  int v113; // [rsp+40h] [rbp-5C0h]
  int v116; // [rsp+50h] [rbp-5B0h]
  int v117; // [rsp+50h] [rbp-5B0h]
  __int64 v118; // [rsp+50h] [rbp-5B0h]
  _QWORD *v119; // [rsp+50h] [rbp-5B0h]
  __int64 v121; // [rsp+60h] [rbp-5A0h] BYREF
  __int64 v122; // [rsp+68h] [rbp-598h]
  _QWORD v123[2]; // [rsp+80h] [rbp-580h] BYREF
  _QWORD v124[2]; // [rsp+90h] [rbp-570h] BYREF
  __int64 *v125; // [rsp+A0h] [rbp-560h]
  __int64 v126; // [rsp+B0h] [rbp-550h] BYREF
  _BYTE *v127; // [rsp+E0h] [rbp-520h] BYREF
  size_t v128; // [rsp+E8h] [rbp-518h]
  _QWORD v129[2]; // [rsp+F0h] [rbp-510h] BYREF
  __int64 *v130; // [rsp+100h] [rbp-500h] BYREF
  __int64 v131; // [rsp+110h] [rbp-4F0h] BYREF
  __int64 *v132; // [rsp+140h] [rbp-4C0h] BYREF
  __int64 v133; // [rsp+148h] [rbp-4B8h]
  __int64 v134; // [rsp+150h] [rbp-4B0h] BYREF
  __int64 v135; // [rsp+158h] [rbp-4A8h]
  int v136; // [rsp+160h] [rbp-4A0h]
  _QWORD *v137; // [rsp+168h] [rbp-498h]
  void *v138; // [rsp+210h] [rbp-3F0h] BYREF
  int v139; // [rsp+218h] [rbp-3E8h]
  char v140; // [rsp+21Ch] [rbp-3E4h]
  __int64 v141; // [rsp+220h] [rbp-3E0h]
  __m128i v142; // [rsp+228h] [rbp-3D8h] BYREF
  __int64 v143; // [rsp+238h] [rbp-3C8h]
  __int64 v144; // [rsp+240h] [rbp-3C0h]
  __m128i v145; // [rsp+248h] [rbp-3B8h]
  __int64 v146; // [rsp+258h] [rbp-3A8h]
  char v147; // [rsp+260h] [rbp-3A0h]
  _BYTE *v148; // [rsp+268h] [rbp-398h] BYREF
  __int64 v149; // [rsp+270h] [rbp-390h]
  _BYTE v150[352]; // [rsp+278h] [rbp-388h] BYREF
  char v151; // [rsp+3D8h] [rbp-228h]
  int v152; // [rsp+3DCh] [rbp-224h]
  __int64 v153; // [rsp+3E0h] [rbp-220h]
  void *v154; // [rsp+3F0h] [rbp-210h] BYREF
  __int64 v155; // [rsp+3F8h] [rbp-208h]
  __int64 v156; // [rsp+400h] [rbp-200h]
  __m128i v157; // [rsp+408h] [rbp-1F8h] BYREF
  __int64 v158; // [rsp+418h] [rbp-1E8h]
  __int64 v159; // [rsp+420h] [rbp-1E0h]
  __m128i v160; // [rsp+428h] [rbp-1D8h] BYREF
  __int64 v161; // [rsp+438h] [rbp-1C8h]
  char v162; // [rsp+440h] [rbp-1C0h]
  _QWORD *v163; // [rsp+448h] [rbp-1B8h] BYREF
  unsigned int v164; // [rsp+450h] [rbp-1B0h]
  _BYTE v165[352]; // [rsp+458h] [rbp-1A8h] BYREF
  char v166; // [rsp+5B8h] [rbp-48h]
  int v167; // [rsp+5BCh] [rbp-44h]
  __int64 v168; // [rsp+5C0h] [rbp-40h]

  if ( a3 <= 1 )
    return 0;
  v15 = a2;
  sub_1BBCA40(&v121, a2, a3, 0);
  if ( !v122 || *(_BYTE *)(v122 + 16) == 24 )
    return 0;
  v110 = v121;
  v106 = sub_1BBE2D0(a4, v121, v16, v17, v18, v19);
  v103 = *(_DWORD *)(a4 + 1396);
  v20 = (__int64 **)&a2[a3];
  v111 = 8 * a3;
  if ( a2 != (__int64 ***)v20 )
  {
    v21 = (__int64 **)a2;
    while ( 1 )
    {
      v22 = **v21;
      if ( !(unsigned __int8)sub_1643F10(v22) || (*(_BYTE *)(v22 + 8) & 0xFD) == 4 )
        break;
      if ( v20 == ++v21 )
        goto LABEL_41;
    }
    v23 = *(__int64 **)(a4 + 1384);
    v24 = sub_15E0530(*v23);
    if ( sub_1602790(v24)
      || (v85 = sub_15E0530(*v23),
          v86 = sub_16033E0(v85),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v86 + 48LL))(v86)) )
    {
      v137 = v123;
      v123[0] = v124;
      v132 = (__int64 *)&unk_49EFBE0;
      v123[1] = 0;
      LOBYTE(v124[0]) = 0;
      v136 = 1;
      v135 = 0;
      v134 = 0;
      v133 = 0;
      sub_154E060(v22, (__int64)&v132, 0, 0);
      sub_15CA5C0((__int64)&v154, (__int64)"slp-vectorizer", (__int64)"UnsupportedType", 15, v110);
      sub_15CAB20((__int64)&v154, "Cannot SLP vectorize list: type ", 0x20u);
      if ( v135 != v133 )
        sub_16E7BA0((__int64 *)&v132);
      v127 = v129;
      sub_1BBA100((__int64 *)&v127, (_BYTE *)*v137, *v137 + v137[1]);
      if ( 0x3FFFFFFFFFFFFFFFLL - v128 <= 0x1C )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v127, " is unsupported by vectorizer", 29);
      sub_15CAB20((__int64)&v154, v127, v128);
      v25 = _mm_loadu_si128(&v157);
      v26 = _mm_loadu_si128(&v160);
      v139 = v155;
      v142 = v25;
      v140 = BYTE4(v155);
      v145 = v26;
      v141 = v156;
      v143 = v158;
      v138 = &unk_49ECF68;
      v144 = v159;
      v147 = v162;
      if ( v162 )
        v146 = v161;
      v148 = v150;
      v149 = 0x400000000LL;
      if ( v164 )
        sub_1BC2970((__int64)&v148, (__int64)&v163);
      v151 = v166;
      v152 = v167;
      v153 = v168;
      v138 = &unk_49ECFC8;
      if ( v127 != (_BYTE *)v129 )
        j_j___libc_free_0(v127, v129[0] + 1LL);
      v27 = v163;
      v154 = &unk_49ECF68;
      v28 = &v163[11 * v164];
      if ( v163 != v28 )
      {
        do
        {
          v28 -= 11;
          v29 = (_QWORD *)v28[4];
          if ( v29 != v28 + 6 )
            j_j___libc_free_0(v29, v28[6] + 1LL);
          if ( (_QWORD *)*v28 != v28 + 2 )
            j_j___libc_free_0(*v28, v28[2] + 1LL);
        }
        while ( v27 != v28 );
        v28 = v163;
      }
      if ( v28 != (_QWORD *)v165 )
        _libc_free((unsigned __int64)v28);
      sub_16E7BC0((__int64 *)&v132);
      if ( (_QWORD *)v123[0] != v124 )
        j_j___libc_free_0(v123[0], v124[0] + 1LL);
      sub_143AA50(v23, (__int64)&v138);
      v30 = v148;
      v138 = &unk_49ECF68;
      v31 = &v148[88 * (unsigned int)v149];
      if ( v148 != (_BYTE *)v31 )
      {
        do
        {
          v31 -= 11;
          v32 = (_QWORD *)v31[4];
          if ( v32 != v31 + 6 )
            j_j___libc_free_0(v32, v31[6] + 1LL);
          if ( (_QWORD *)*v31 != v31 + 2 )
            j_j___libc_free_0(*v31, v31[2] + 1LL);
        }
        while ( v30 != v31 );
        v31 = v148;
      }
      if ( v31 != (_QWORD *)v150 )
        _libc_free((unsigned __int64)v31);
    }
    return 0;
  }
LABEL_41:
  v35 = 0;
  v133 = 0x800000000LL;
  v113 = dword_4FB9620;
  v36 = &v134;
  v132 = &v134;
  v99 = v111 >> 3;
  if ( (unsigned __int64)v111 > 0x40 )
  {
    sub_170B450((__int64)&v132, v111 >> 3);
    v35 = v133;
    v36 = &v132[3 * (unsigned int)v133];
  }
  if ( v15 != (__int64 ***)v20 )
  {
    v37 = v15;
    v38 = v36;
    do
    {
      if ( v38 )
      {
        v39 = *v37;
        *v38 = 6;
        v38[1] = 0;
        v38[2] = (__int64)v39;
        if ( v39 + 1 != 0 && v39 != 0 && v39 != (__int64 **)-16LL )
          sub_164C220((__int64)v38);
      }
      ++v37;
      v38 += 3;
    }
    while ( v20 != (__int64 **)v37 );
    v35 = v133;
  }
  v40 = a3;
  v41 = 2;
  if ( v103 / v106 >= 2 )
    v41 = v103 / v106;
  _BitScanReverse64(&v42, a3);
  v43 = 0x8000000000000000LL >> ((unsigned __int8)v42 ^ 0x3Fu);
  v112 = v41;
  if ( v41 > (unsigned int)v43 )
    LODWORD(v43) = v41;
  LODWORD(v133) = v35 + v99;
  if ( (unsigned int)a3 <= 1 || v41 > (unsigned int)v43 )
    goto LABEL_126;
  v104 = 0;
  v44 = 0;
  v107 = 0;
  do
  {
    v46 = (void **)sub_16463B0(**v15, v43);
    if ( (unsigned int)sub_14A35F0(*(_QWORD *)(a1 + 8)) != (_DWORD)v43 && v40 > v44 )
    {
      v47 = v15;
      v48 = v44;
      v49 = a1;
      v50 = v43;
      do
      {
        v51 = v50 + v44;
        v52 = v40 - v44;
        if ( v50 + v44 <= v40 )
          v52 = v50;
        if ( (v52 & (v52 - 1)) != 0 || v52 == 1 )
          break;
        v53 = v52;
        v54 = (__int64 *)&v47[v44];
        v55 = &v54[v52];
        v56 = &v132[3 * v44];
        v57 = v54;
        do
        {
          v46 = (void **)v56[2];
          if ( (void **)*v57 != v46 )
            goto LABEL_67;
          ++v57;
          v56 += 3;
        }
        while ( v55 != v57 );
        v97 = (__int64 *)v47;
        v100 = v49;
        v116 = v48;
        sub_1BD8550(a4, (__int64)&v47[v44], v53, 0, 0, a7);
        v46 = (void **)a4;
        sub_1BC2C70((__int64)v123, a4);
        v61 = v116;
        v62 = v100;
        v63 = v97;
        if ( a6 && LOBYTE(v124[0]) )
        {
          v46 = &v154;
          v154 = (void *)v54[1];
          v155 = *v54;
          sub_1BD8550(a4, (__int64)&v154, 2, 0, 0, a7);
          v63 = v97;
          v62 = v100;
          v61 = v116;
        }
        v98 = v63;
        v101 = v62;
        v117 = v61;
        v64 = sub_1BBD300((__int64 *)a4);
        v48 = v117;
        v49 = v101;
        v47 = (__int64 ***)v98;
        if ( v64 )
        {
LABEL_67:
          ++v44;
          continue;
        }
        v118 = v101;
        v102 = v48;
        sub_1BC4C80((__int64 *)a4, (__int64)v46, v65, v66, v67, v98);
        v72 = sub_1BD8A90(a4, (__int64)v46, v68, v69, v70, v71);
        v73 = v113;
        v74 = v72 - a5;
        v49 = v118;
        v47 = (__int64 ***)v98;
        if ( v113 > v74 )
          v73 = v74;
        v113 = v73;
        if ( v74 < -dword_4FB9620 )
        {
          v108 = v74;
          v105 = v118;
          v119 = *(_QWORD **)(a4 + 1384);
          sub_15CA3B0((__int64)&v154, (__int64)"slp-vectorizer", (__int64)"VectorizedList", 14, *v54);
          sub_15CAB20((__int64)&v154, "SLP vectorized with cost ", 0x19u);
          sub_15C9890((__int64)&v127, "Cost", 4, v108);
          v109 = sub_17C2270((__int64)&v154, (__int64)&v127);
          sub_15CAB20(v109, " and with tree size ", 0x14u);
          sub_15C9C50(
            (__int64)&v138,
            "TreeSize",
            8,
            -1171354717 * ((__int64)(*(_QWORD *)(a4 + 8) - *(_QWORD *)a4) >> 4));
          v46 = (void **)sub_17C2270(v109, (__int64)&v138);
          sub_143AA50(v119, (__int64)v46);
          sub_2240A30(&v142.m128i_u64[1]);
          v44 += v50;
          sub_2240A30(&v138);
          sub_2240A30(&v130);
          sub_2240A30(&v127);
          v154 = &unk_49ECF68;
          sub_1897B80((__int64)&v163);
          sub_1BD47F0((__int64 ***)a4, (__m128)a7, a8, a9, a10, v91, v92, a13, a14);
          v107 = 1;
          v47 = (__int64 ***)v98;
          v48 = v51;
          v49 = v105;
        }
        else
        {
          v48 = v102;
          ++v44;
        }
        v104 = 1;
      }
      while ( v40 > v44 );
      LODWORD(v43) = v50;
      a1 = v49;
      v15 = v47;
      v44 = v48;
    }
    LODWORD(v43) = (unsigned int)v43 >> 1;
  }
  while ( v44 + 1 < v40 && (unsigned int)v43 >= v112 );
  v33 = v107;
  if ( v107 )
    goto LABEL_75;
  if ( v104 )
  {
    v75 = *(__int64 **)(a4 + 1384);
    v76 = sub_15E0530(*v75);
    if ( sub_1602790(v76)
      || (v93 = sub_15E0530(*v75),
          v94 = sub_16033E0(v93),
          (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v94 + 48LL))(v94, v46)) )
    {
      sub_15CA5C0((__int64)&v154, (__int64)"slp-vectorizer", (__int64)"NotBeneficial", 13, v110);
      sub_15CAB20((__int64)&v154, "List vectorization was possible but not beneficial with cost ", 0x3Du);
      sub_15C9890((__int64)&v127, "Cost", 4, (unsigned int)v113);
      v77 = sub_17C21B0((__int64)&v154, (__int64)&v127);
      sub_15CAB20(v77, " >= ", 4u);
      sub_15C9890((__int64)v123, "Treshold", 8, (unsigned int)-dword_4FB9620);
      v78 = sub_17C21B0(v77, (__int64)v123);
      v139 = *(_DWORD *)(v78 + 8);
      v140 = *(_BYTE *)(v78 + 12);
      v141 = *(_QWORD *)(v78 + 16);
      v142 = _mm_loadu_si128((const __m128i *)(v78 + 24));
      v143 = *(_QWORD *)(v78 + 40);
      v138 = &unk_49ECF68;
      v144 = *(_QWORD *)(v78 + 48);
      v145 = _mm_loadu_si128((const __m128i *)(v78 + 56));
      v147 = *(_BYTE *)(v78 + 80);
      if ( v147 )
        v146 = *(_QWORD *)(v78 + 72);
      v148 = v150;
      v149 = 0x400000000LL;
      if ( *(_DWORD *)(v78 + 96) )
        sub_1BC2970((__int64)&v148, v78 + 88);
      v151 = *(_BYTE *)(v78 + 456);
      v152 = *(_DWORD *)(v78 + 460);
      v153 = *(_QWORD *)(v78 + 464);
      v138 = &unk_49ECFC8;
      if ( v125 != &v126 )
        j_j___libc_free_0(v125, v126 + 1);
      if ( (_QWORD *)v123[0] != v124 )
        j_j___libc_free_0(v123[0], v124[0] + 1LL);
      if ( v130 != &v131 )
        j_j___libc_free_0(v130, v131 + 1);
      if ( v127 != (_BYTE *)v129 )
        j_j___libc_free_0(v127, v129[0] + 1LL);
      v79 = v163;
      v154 = &unk_49ECF68;
      v80 = &v163[11 * v164];
      if ( v163 != v80 )
      {
        do
        {
          v80 -= 11;
          v81 = (_QWORD *)v80[4];
          if ( v81 != v80 + 6 )
            j_j___libc_free_0(v81, v80[6] + 1LL);
          if ( (_QWORD *)*v80 != v80 + 2 )
            j_j___libc_free_0(*v80, v80[2] + 1LL);
        }
        while ( v79 != v80 );
        v80 = v163;
      }
      if ( v80 != (_QWORD *)v165 )
        _libc_free((unsigned __int64)v80);
      sub_143AA50(v75, (__int64)&v138);
      v82 = v148;
      v138 = &unk_49ECF68;
      v83 = &v148[88 * (unsigned int)v149];
      if ( v148 != (_BYTE *)v83 )
      {
        do
        {
          v83 -= 11;
          v84 = (_QWORD *)v83[4];
          if ( v84 != v83 + 6 )
            j_j___libc_free_0(v84, v83[6] + 1LL);
          if ( (_QWORD *)*v83 != v83 + 2 )
            j_j___libc_free_0(*v83, v83[2] + 1LL);
        }
        while ( v82 != v83 );
        v83 = v148;
      }
      if ( v83 != (_QWORD *)v150 )
        _libc_free((unsigned __int64)v83);
    }
  }
  else
  {
LABEL_126:
    v87 = *(__int64 **)(a4 + 1384);
    v88 = sub_15E0530(*v87);
    if ( sub_1602790(v88)
      || (v95 = sub_15E0530(*v87),
          v96 = sub_16033E0(v95),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v96 + 48LL))(v96)) )
    {
      sub_15CA5C0((__int64)&v154, (__int64)"slp-vectorizer", (__int64)"NotPossible", 11, v110);
      sub_15CAB20((__int64)&v154, "Cannot SLP vectorize list: vectorization was impossible", 0x37u);
      sub_15CAB20((__int64)&v154, " with available vectorization factors", 0x25u);
      v89 = _mm_loadu_si128(&v157);
      v90 = _mm_loadu_si128(&v160);
      v139 = v155;
      v142 = v89;
      v140 = BYTE4(v155);
      v145 = v90;
      v141 = v156;
      v143 = v158;
      v138 = &unk_49ECF68;
      v144 = v159;
      v147 = v162;
      if ( v162 )
        v146 = v161;
      v148 = v150;
      v149 = 0x400000000LL;
      if ( v164 )
        sub_1BC2970((__int64)&v148, (__int64)&v163);
      v154 = &unk_49ECF68;
      v151 = v166;
      v152 = v167;
      v153 = v168;
      v138 = &unk_49ECFC8;
      sub_1897B80((__int64)&v163);
      sub_143AA50(v87, (__int64)&v138);
      v138 = &unk_49ECF68;
      sub_1897B80((__int64)&v148);
    }
  }
  v33 = 0;
LABEL_75:
  v58 = (__int64)v132;
  v59 = &v132[3 * (unsigned int)v133];
  if ( v132 != v59 )
  {
    do
    {
      v60 = *(v59 - 1);
      v59 -= 3;
      if ( v60 != 0 && v60 != -8 && v60 != -16 )
        sub_1649B30(v59);
    }
    while ( (__int64 *)v58 != v59 );
    v59 = v132;
  }
  if ( v59 != &v134 )
    _libc_free((unsigned __int64)v59);
  return v33;
}
