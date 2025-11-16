// Function: sub_340CA20
// Address: 0x340ca20
//
unsigned __int8 *__fastcall sub_340CA20(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        __int128 a9,
        unsigned __int8 a10,
        char a11,
        bool a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        const __m128i *a17)
{
  unsigned __int64 v19; // r13
  __int64 v21; // r11
  bool v22; // bl
  int v23; // eax
  __int64 v24; // r11
  __int64 v25; // rdi
  __int64 (*v26)(); // rax
  __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v30; // rdx
  int v31; // eax
  char v32; // cl
  unsigned __int64 v33; // rax
  unsigned __int8 *v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rsi
  int v38; // eax
  __int64 v39; // r15
  __int64 v40; // rax
  const char *v41; // r13
  __int64 v42; // rax
  __m128i v43; // xmm0
  unsigned __int16 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  __m128i v51; // xmm1
  __int64 v52; // rax
  __m128i v53; // xmm2
  __int64 v54; // rdi
  __int64 (__fastcall *v55)(__int64, __int64, unsigned int); // rax
  int v56; // edx
  unsigned __int16 v57; // ax
  __int64 v58; // r13
  __int64 v59; // rdx
  __int64 v60; // r15
  unsigned __int16 *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // rax
  unsigned __int64 v73; // rdi
  void (***v74)(); // rdi
  void (*v75)(); // rax
  __int64 v76; // r13
  __int64 v77; // r15
  int v78; // eax
  unsigned __int8 v79; // dl
  __int64 v80; // rax
  __m128i v81; // xmm3
  __int64 v82; // rax
  __m128i v83; // xmm4
  __int64 v84; // rdi
  __int64 (__fastcall *v85)(__int64, __int64, unsigned int); // rax
  int v86; // edx
  unsigned __int16 v87; // ax
  __int64 v88; // r15
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // rcx
  unsigned __int64 v92; // rdi
  unsigned int v93; // r13d
  __int64 v94; // rax
  void (***v95)(); // rdi
  void (*v96)(); // rax
  int v97; // [rsp+18h] [rbp-1278h]
  __int64 v98; // [rsp+20h] [rbp-1270h]
  __int64 v99; // [rsp+28h] [rbp-1268h]
  __int64 v100; // [rsp+28h] [rbp-1268h]
  __int64 v101; // [rsp+28h] [rbp-1268h]
  __int64 v102; // [rsp+28h] [rbp-1268h]
  bool v103; // [rsp+36h] [rbp-125Ah]
  char v104; // [rsp+37h] [rbp-1259h]
  unsigned __int8 v105; // [rsp+38h] [rbp-1258h]
  __int64 *v106; // [rsp+38h] [rbp-1258h]
  __m128i v107; // [rsp+40h] [rbp-1250h] BYREF
  __int64 v108; // [rsp+50h] [rbp-1240h]
  __int64 v109; // [rsp+58h] [rbp-1238h]
  __m128i v110; // [rsp+60h] [rbp-1230h]
  __m128i v111; // [rsp+70h] [rbp-1220h]
  __int64 v112; // [rsp+80h] [rbp-1210h]
  __int64 v113; // [rsp+88h] [rbp-1208h]
  __m128i v114; // [rsp+90h] [rbp-1200h]
  __m128i v115; // [rsp+A0h] [rbp-11F0h]
  __m128i v116; // [rsp+B0h] [rbp-11E0h]
  unsigned __int64 v117; // [rsp+C0h] [rbp-11D0h]
  unsigned __int64 v118; // [rsp+C8h] [rbp-11C8h]
  __int16 v119; // [rsp+D0h] [rbp-11C0h] BYREF
  __int64 v120; // [rsp+D8h] [rbp-11B8h]
  unsigned __int64 v121; // [rsp+E0h] [rbp-11B0h] BYREF
  __int64 v122; // [rsp+E8h] [rbp-11A8h]
  __int64 v123; // [rsp+F0h] [rbp-11A0h]
  __m128i v124; // [rsp+100h] [rbp-1190h] BYREF
  unsigned __int64 v125; // [rsp+110h] [rbp-1180h]
  __int64 v126; // [rsp+118h] [rbp-1178h]
  __int64 v127; // [rsp+120h] [rbp-1170h]
  __int64 v128; // [rsp+128h] [rbp-1168h]
  unsigned __int64 v129; // [rsp+130h] [rbp-1160h] BYREF
  __int64 v130; // [rsp+138h] [rbp-1158h]
  __int64 v131; // [rsp+140h] [rbp-1150h]
  unsigned __int64 v132; // [rsp+148h] [rbp-1148h]
  __int64 v133; // [rsp+150h] [rbp-1140h]
  __int64 v134; // [rsp+158h] [rbp-1138h]
  __int64 v135; // [rsp+160h] [rbp-1130h]
  unsigned __int64 v136; // [rsp+168h] [rbp-1128h] BYREF
  __int64 v137; // [rsp+170h] [rbp-1120h]
  __int64 v138; // [rsp+178h] [rbp-1118h]
  __int64 v139; // [rsp+180h] [rbp-1110h]
  __int64 v140; // [rsp+188h] [rbp-1108h] BYREF
  int v141; // [rsp+190h] [rbp-1100h]
  __int64 v142; // [rsp+198h] [rbp-10F8h]
  _BYTE *v143; // [rsp+1A0h] [rbp-10F0h]
  __int64 v144; // [rsp+1A8h] [rbp-10E8h]
  _BYTE v145[1792]; // [rsp+1B0h] [rbp-10E0h] BYREF
  _BYTE *v146; // [rsp+8B0h] [rbp-9E0h]
  __int64 v147; // [rsp+8B8h] [rbp-9D8h]
  _BYTE v148[512]; // [rsp+8C0h] [rbp-9D0h] BYREF
  _BYTE *v149; // [rsp+AC0h] [rbp-7D0h]
  __int64 v150; // [rsp+AC8h] [rbp-7C8h]
  _BYTE v151[1792]; // [rsp+AD0h] [rbp-7C0h] BYREF
  _BYTE *v152; // [rsp+11D0h] [rbp-C0h]
  __int64 v153; // [rsp+11D8h] [rbp-B8h]
  _BYTE v154[64]; // [rsp+11E0h] [rbp-B0h] BYREF
  __int64 v155; // [rsp+1220h] [rbp-70h]
  __int64 v156; // [rsp+1228h] [rbp-68h]
  int v157; // [rsp+1230h] [rbp-60h]
  char v158; // [rsp+1250h] [rbp-40h]

  v19 = a2;
  v21 = a9;
  v107.m128i_i64[0] = a5;
  v22 = a12;
  v107.m128i_i64[1] = a6;
  v105 = a11;
  v23 = *(_DWORD *)(a9 + 24);
  v104 = a12;
  if ( v23 == 35 || v23 == 11 )
  {
    v30 = *(_QWORD *)(a9 + 96);
    if ( *(_DWORD *)(v30 + 32) <= 0x40u )
    {
      if ( !*(_QWORD *)(v30 + 24) )
        return (unsigned __int8 *)v19;
      v32 = a11;
      v33 = *(_QWORD *)(v30 + 24);
    }
    else
    {
      v97 = *(_DWORD *)(v30 + 32);
      v98 = a9;
      v99 = *(_QWORD *)(a9 + 96);
      v31 = sub_C444A0(v30 + 24);
      v21 = v98;
      if ( v97 == v31 )
        return (unsigned __int8 *)v19;
      v32 = v105;
      v33 = **(_QWORD **)(v99 + 24);
    }
    v101 = v21;
    v34 = sub_340B880(
            a1,
            a4,
            a2,
            a3,
            v107.m128i_i64[0],
            v107.m128i_i64[1],
            a7,
            a8,
            *((__int64 *)&a8 + 1),
            v33,
            a10,
            v32,
            0,
            a14,
            a15,
            a16,
            a17);
    v24 = v101;
    if ( v34 )
      return v34;
  }
  else
  {
    v24 = 0;
  }
  v25 = *(_QWORD *)(a1 + 8);
  if ( v25 )
  {
    v26 = *(__int64 (**)())(*(_QWORD *)v25 + 56LL);
    if ( v26 != sub_33C7D00 )
    {
      v100 = v24;
      v34 = (unsigned __int8 *)((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, unsigned __int64, _QWORD, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, bool, __int64, __int64, __int64))v26)(
                                 v25,
                                 a1,
                                 a4,
                                 a2,
                                 a3,
                                 a10,
                                 v107.m128i_i64[0],
                                 v107.m128i_i64[1],
                                 a8,
                                 *((_QWORD *)&a8 + 1),
                                 a9,
                                 *((_QWORD *)&a9 + 1),
                                 v105,
                                 v22,
                                 a14,
                                 a15,
                                 a16);
      v24 = v100;
      if ( v34 )
        return v34;
    }
  }
  if ( v22 )
  {
    v27 = *(_QWORD *)(v24 + 96);
    v28 = *(_QWORD **)(v27 + 24);
    if ( *(_DWORD *)(v27 + 32) > 0x40u )
      v28 = (_QWORD *)*v28;
    return sub_340B880(
             a1,
             a4,
             a2,
             a3,
             v107.m128i_i64[0],
             v107.m128i_i64[1],
             a7,
             a8,
             *((__int64 *)&a8 + 1),
             (unsigned __int64)v28,
             a10,
             v105,
             1,
             a14,
             a15,
             a16,
             a17);
  }
  else
  {
    v35 = sub_2EAC1E0((__int64)&a14);
    sub_33C8580(*(_QWORD *)(a1 + 16), v35);
    v106 = *(__int64 **)(a1 + 64);
    v36 = sub_2E79000(*(__int64 **)(a1 + 40));
    v129 = 0;
    v102 = v36;
    v132 = 0xFFFFFFFF00000020LL;
    v143 = v145;
    v144 = 0x2000000000LL;
    v146 = v148;
    v37 = *(_QWORD *)a4;
    v147 = 0x2000000000LL;
    v150 = 0x2000000000LL;
    v152 = v154;
    v130 = 0;
    v131 = 0;
    v133 = 0;
    v134 = 0;
    v135 = 0;
    v136 = 0;
    v137 = 0;
    v138 = 0;
    v139 = a1;
    v141 = 0;
    v142 = 0;
    v149 = v151;
    v153 = 0x400000000LL;
    v155 = 0;
    v156 = 0;
    v157 = 0;
    v158 = 0;
    v140 = v37;
    if ( v37 )
      sub_B96E90((__int64)&v140, v37, 1);
    v38 = *(_DWORD *)(a4 + 8);
    v39 = a8;
    v117 = v19;
    v118 = a3;
    v141 = v38;
    v40 = *(_QWORD *)(a1 + 16);
    v129 = v19;
    v41 = *(const char **)(v40 + 529064);
    LODWORD(v130) = a3;
    v121 = 0;
    v122 = 0;
    v123 = 0;
    v103 = sub_33CF170(a8) && v41 != 0;
    if ( v103 )
    {
      v80 = sub_BCE3C0(v106, 0);
      v81 = _mm_load_si128(&v107);
      v126 = v80;
      v111 = v81;
      v124.m128i_i64[1] = v107.m128i_i64[0];
      v124.m128i_i64[0] = 0;
      LODWORD(v125) = v81.m128i_i32[2];
      v127 = 0;
      v128 = 0;
      sub_33ECB40(&v121, &v124);
      v82 = sub_AE4420(v102, (__int64)v106, 0);
      v83 = _mm_loadu_si128((const __m128i *)&a9);
      v126 = v82;
      v124.m128i_i64[1] = a9;
      v110 = v83;
      v124.m128i_i64[0] = 0;
      v127 = 0;
      v128 = 0;
      LODWORD(v125) = v83.m128i_i32[2];
      sub_33ECB40(&v121, &v124);
      v84 = *(_QWORD *)(a1 + 16);
      v85 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v84 + 32LL);
      if ( v85 == sub_2D42F30 )
      {
        v86 = sub_AE2980(v102, 0)[1];
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
      }
      else
      {
        v87 = v85(v84, v102, 0);
      }
      v88 = sub_33EED90(a1, v41, v87, 0);
      v107.m128i_i64[0] = v89;
      v90 = sub_BCB120(v106);
      v91 = *(_QWORD *)(a1 + 16);
      v131 = v90;
      v92 = v136;
      v93 = *(_DWORD *)(v91 + 533016);
      v109 = v107.m128i_i64[0];
      v108 = v88;
      v134 = v88;
      LODWORD(v135) = v107.m128i_i32[0];
      LODWORD(v133) = v93;
      v136 = v121;
      LODWORD(v90) = -1431655765 * ((__int64)(v122 - v121) >> 4);
      v137 = v122;
      v121 = 0;
      v122 = 0;
      HIDWORD(v132) = v90;
      v94 = v123;
      v123 = 0;
      v138 = v94;
      if ( v92 )
        j_j___libc_free_0(v92);
      v95 = *(void (****)())(v139 + 16);
      v96 = **v95;
      if ( v96 != nullsub_1688 )
        ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v96)(
          v95,
          *(_QWORD *)(v139 + 40),
          v93,
          &v136);
    }
    else
    {
      v42 = sub_BCE3C0(v106, 0);
      v43 = _mm_load_si128(&v107);
      v126 = v42;
      v116 = v43;
      v124.m128i_i64[1] = v107.m128i_i64[0];
      v124.m128i_i64[0] = 0;
      LODWORD(v125) = v43.m128i_i32[2];
      v127 = 0;
      v128 = 0;
      sub_33ECB40(&v121, &v124);
      v44 = (unsigned __int16 *)(*(_QWORD *)(v39 + 48) + 16LL * DWORD2(a8));
      v45 = *v44;
      v46 = *((_QWORD *)v44 + 1);
      v119 = v45;
      v120 = v46;
      v50 = sub_3007410((__int64)&v119, v106, v45, v47, v48, v49);
      v51 = _mm_loadu_si128((const __m128i *)&a8);
      v126 = v50;
      v115 = v51;
      v124.m128i_i64[1] = a8;
      v124.m128i_i64[0] = 0;
      LODWORD(v125) = v51.m128i_i32[2];
      v127 = 0;
      v128 = 0;
      sub_33ECB40(&v121, &v124);
      v52 = sub_AE4420(v102, (__int64)v106, 0);
      v53 = _mm_loadu_si128((const __m128i *)&a9);
      v126 = v52;
      v124.m128i_i64[1] = a9;
      v114 = v53;
      v124.m128i_i64[0] = 0;
      v127 = 0;
      v128 = 0;
      LODWORD(v125) = v53.m128i_i32[2];
      sub_33ECB40(&v121, &v124);
      v54 = *(_QWORD *)(a1 + 16);
      v55 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v54 + 32LL);
      if ( v55 == sub_2D42F30 )
      {
        v56 = sub_AE2980(v102, 0)[1];
        v57 = 2;
        if ( v56 != 1 )
        {
          v57 = 3;
          if ( v56 != 2 )
          {
            v57 = 4;
            if ( v56 != 4 )
            {
              v57 = 5;
              if ( v56 != 8 )
              {
                v57 = 6;
                if ( v56 != 16 )
                {
                  v57 = 7;
                  if ( v56 != 32 )
                  {
                    v57 = 8;
                    if ( v56 != 64 )
                      v57 = 9 * (v56 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v57 = v55(v54, v102, 0);
      }
      v58 = sub_33EED90(a1, *(const char **)(*(_QWORD *)(a1 + 16) + 529048LL), v57, 0);
      v60 = v59;
      v61 = (unsigned __int16 *)(*(_QWORD *)(v107.m128i_i64[0] + 48) + 16LL * v107.m128i_u32[2]);
      v62 = *v61;
      v63 = *((_QWORD *)v61 + 1);
      v124.m128i_i16[0] = v62;
      v124.m128i_i64[1] = v63;
      v67 = sub_3007410((__int64)&v124, v106, v62, v64, v65, v66);
      v68 = *(_QWORD *)(a1 + 16);
      v131 = v67;
      v69 = v121;
      v70 = *(unsigned int *)(v68 + 533008);
      v113 = v60;
      v137 = v122;
      v112 = v58;
      LODWORD(v135) = v60;
      v71 = v123;
      v134 = v58;
      v72 = (__int64)(v122 - v121) >> 4;
      LODWORD(v133) = v70;
      v121 = 0;
      v73 = v136;
      v122 = 0;
      v136 = v69;
      HIDWORD(v132) = -1431655765 * v72;
      v123 = 0;
      v138 = v71;
      if ( v73 )
      {
        v107.m128i_i32[0] = v70;
        j_j___libc_free_0(v73);
        v70 = v107.m128i_u32[0];
      }
      v74 = *(void (****)())(v139 + 16);
      v75 = **v74;
      if ( v75 != nullsub_1688 )
        ((void (__fastcall *)(void (***)(), _QWORD, __int64, unsigned __int64 *))v75)(
          v74,
          *(_QWORD *)(v139 + 40),
          v70,
          &v136);
    }
    if ( v121 )
      j_j___libc_free_0(v121);
    v76 = *(_QWORD *)(a1 + 16);
    v77 = *(_QWORD *)(v76 + 529048);
    if ( v77 && strlen(*(const char **)(v76 + 529048)) == 6 )
    {
      if ( *(_DWORD *)v77 != 1936549229 || (v78 = 0, *(_WORD *)(v77 + 4) != 29797) )
        v78 = 1;
      v22 = v78 == 0;
    }
    if ( a13 )
    {
      v79 = sub_34B9CE0(a13);
      if ( (*(_WORD *)(a13 + 2) & 3u) - 1 <= 1 )
        v104 = sub_34B9AF0(a13, *(_QWORD *)a1, v79 & !v103 & (unsigned __int8)v22);
      v76 = *(_QWORD *)(a1 + 16);
    }
    LOBYTE(v132) = v132 & 0xDF;
    BYTE2(v132) = v104;
    sub_3377410((__int64)&v124, (_WORD *)v76, (__int64)&v129);
    v19 = v125;
    if ( v152 != v154 )
      _libc_free((unsigned __int64)v152);
    if ( v149 != v151 )
      _libc_free((unsigned __int64)v149);
    if ( v146 != v148 )
      _libc_free((unsigned __int64)v146);
    if ( v143 != v145 )
      _libc_free((unsigned __int64)v143);
    if ( v140 )
      sub_B91220((__int64)&v140, v140);
    if ( v136 )
      j_j___libc_free_0(v136);
  }
  return (unsigned __int8 *)v19;
}
