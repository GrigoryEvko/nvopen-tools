// Function: sub_19DBD20
// Address: 0x19dbd20
//
__int64 __fastcall sub_19DBD20(
        __int64 *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // r12
  _QWORD *v13; // r13
  __m128i *v15; // rdi
  __int64 v16; // rsi
  void (__fastcall *v17)(__m128i *, __m128i *, __int64); // rax
  int v18; // r12d
  __int64 v19; // rdx
  __int64 v20; // r13
  _QWORD *v21; // r12
  _QWORD *v22; // r15
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __m128 v25; // xmm0
  __m128i v26; // xmm1
  unsigned __int64 v27; // rbx
  __int64 v28; // r15
  __int64 v29; // rdi
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  unsigned __int64 v34; // rbx
  __int64 v35; // r15
  __int64 v36; // rdi
  _QWORD *v37; // rax
  unsigned __int8 *v38; // r12
  _QWORD *v39; // r13
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r10
  __int64 v46; // r8
  __int64 v47; // r11
  _QWORD *v48; // r10
  __int64 v49; // r9
  __m128i v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // r11
  __int64 v56; // rbx
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // rcx
  __int64 v61; // rsi
  __int64 **v62; // rdx
  __int64 *v63; // rsi
  unsigned __int64 v64; // rdi
  __int64 v65; // rdi
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // rcx
  __int64 v71; // rcx
  __int64 v72; // rsi
  _QWORD *v73; // rdx
  __int64 v74; // rsi
  unsigned __int64 v75; // rdi
  __int64 v76; // rsi
  unsigned int v77; // edx
  __int64 v78; // rsi
  __int64 v79; // rcx
  unsigned __int64 v80; // rdi
  __int64 v81; // rcx
  double v82; // xmm4_8
  double v83; // xmm5_8
  double v84; // xmm4_8
  double v85; // xmm5_8
  __int64 v86; // rax
  __int64 v87; // [rsp+0h] [rbp-1E0h]
  __int64 v88; // [rsp+8h] [rbp-1D8h]
  _QWORD *v89; // [rsp+10h] [rbp-1D0h]
  _QWORD *v90; // [rsp+10h] [rbp-1D0h]
  __int64 v91; // [rsp+10h] [rbp-1D0h]
  __int64 v93; // [rsp+20h] [rbp-1C0h]
  __int64 v94; // [rsp+28h] [rbp-1B8h]
  _QWORD *v95; // [rsp+28h] [rbp-1B8h]
  __int64 v96; // [rsp+28h] [rbp-1B8h]
  _QWORD *v97; // [rsp+28h] [rbp-1B8h]
  __int64 v98; // [rsp+28h] [rbp-1B8h]
  __int64 v99; // [rsp+30h] [rbp-1B0h]
  __int64 v100; // [rsp+38h] [rbp-1A8h]
  __int64 *v101; // [rsp+38h] [rbp-1A8h]
  __int64 v102; // [rsp+38h] [rbp-1A8h]
  __int64 v103; // [rsp+40h] [rbp-1A0h]
  __int64 v104; // [rsp+40h] [rbp-1A0h]
  __int64 v105; // [rsp+40h] [rbp-1A0h]
  __int64 v106; // [rsp+40h] [rbp-1A0h]
  __int64 v107; // [rsp+48h] [rbp-198h]
  __int64 v108; // [rsp+48h] [rbp-198h]
  __int64 v109; // [rsp+48h] [rbp-198h]
  char v110; // [rsp+48h] [rbp-198h]
  __int64 v111; // [rsp+50h] [rbp-190h]
  unsigned __int8 v112; // [rsp+58h] [rbp-188h]
  __int64 v113; // [rsp+58h] [rbp-188h]
  __int64 v114; // [rsp+58h] [rbp-188h]
  int v115; // [rsp+60h] [rbp-180h]
  int v116; // [rsp+64h] [rbp-17Ch]
  _QWORD *v117; // [rsp+68h] [rbp-178h]
  unsigned __int64 v118; // [rsp+70h] [rbp-170h]
  __int64 v119; // [rsp+70h] [rbp-170h]
  _QWORD *v120; // [rsp+78h] [rbp-168h]
  _QWORD *v121; // [rsp+78h] [rbp-168h]
  _QWORD *v122; // [rsp+78h] [rbp-168h]
  _QWORD *v123; // [rsp+78h] [rbp-168h]
  _QWORD *v125; // [rsp+88h] [rbp-158h]
  __m128i v126; // [rsp+90h] [rbp-150h]
  __m128i v127; // [rsp+A0h] [rbp-140h] BYREF
  void (__fastcall *v128)(__m128i *, __m128i *, __int64); // [rsp+B0h] [rbp-130h]
  unsigned __int8 (__fastcall *v129)(__m128i *, __int64); // [rsp+B8h] [rbp-128h]
  __m128i v130; // [rsp+C0h] [rbp-120h] BYREF
  __m128i v131; // [rsp+D0h] [rbp-110h] BYREF
  void (__fastcall *v132)(__m128i *, __m128i *, __int64); // [rsp+E0h] [rbp-100h]
  __int64 v133; // [rsp+E8h] [rbp-F8h]
  __m128i v134; // [rsp+F0h] [rbp-F0h] BYREF
  __m128i v135; // [rsp+100h] [rbp-E0h] BYREF
  void (__fastcall *v136)(__m128i *, __m128i *, __int64); // [rsp+110h] [rbp-D0h]
  unsigned __int8 (__fastcall *v137)(__m128i *, __int64); // [rsp+118h] [rbp-C8h]
  __m128i v138; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v139; // [rsp+130h] [rbp-B0h] BYREF
  void (__fastcall *v140)(__m128i *, __m128i *, __int64); // [rsp+140h] [rbp-A0h]
  __int64 v141; // [rsp+148h] [rbp-98h]
  __m128i v142; // [rsp+150h] [rbp-90h] BYREF
  __m128i v143; // [rsp+160h] [rbp-80h] BYREF
  void (__fastcall *v144)(__m128i *, __m128i *, __int64); // [rsp+170h] [rbp-70h]
  unsigned __int8 (__fastcall *v145)(__m128i *, __int64); // [rsp+178h] [rbp-68h]
  __m128i v146; // [rsp+180h] [rbp-60h]
  __m128i v147; // [rsp+190h] [rbp-50h] BYREF
  void (__fastcall *v148)(__m128i *, __m128i *, __int64); // [rsp+1A0h] [rbp-40h]
  __int64 v149; // [rsp+1A8h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 8);
  do
  {
    if ( !v10 )
      goto LABEL_147;
    v11 = sub_1648700(v10);
    v10 = *(_QWORD *)(v10 + 8);
    v12 = v11;
  }
  while ( (unsigned __int8)(*((_BYTE *)v11 + 16) - 25) > 9u );
  if ( !v10 )
LABEL_147:
    BUG();
  while ( 1 )
  {
    v13 = sub_1648700(v10);
    if ( (unsigned __int8)(*((_BYTE *)v13 + 16) - 25) <= 9u )
      break;
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      goto LABEL_147;
  }
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      break;
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) <= 9u )
      return 0;
  }
  v93 = v12[5];
  v111 = v13[5];
  if ( v111 == v93 )
    return 0;
  v15 = &v142;
  sub_1580910(&v142);
  v132 = 0;
  v130 = v146;
  if ( v148 )
  {
    v15 = &v131;
    v148(&v131, &v147, 2);
    v133 = v149;
    v132 = v148;
  }
  v128 = 0;
  v126 = v142;
  if ( v144 )
  {
    v15 = &v127;
    v144(&v127, &v143, 2);
    v129 = v145;
    v128 = v144;
  }
  v140 = 0;
  v138 = v130;
  if ( v132 )
  {
    v15 = &v139;
    v132(&v139, &v131, 2);
    v141 = v133;
    v140 = v132;
  }
  v16 = v126.m128i_i64[0];
  v136 = 0;
  v17 = v128;
  v134 = v126;
  if ( v128 )
  {
    v15 = &v135;
    v128(&v135, &v127, 2);
    v16 = v134.m128i_i64[0];
    v137 = v129;
    v17 = v128;
    v136 = v128;
    if ( v138.m128i_i64[0] == v134.m128i_i64[0] )
    {
      v115 = 0;
LABEL_32:
      if ( v17 )
        v17(&v135, &v135, 3);
      goto LABEL_34;
    }
LABEL_21:
    v18 = 0;
    do
    {
      v16 = *(_QWORD *)(v16 + 8);
      v134.m128i_i64[0] = v16;
      if ( v16 != v134.m128i_i64[1] )
      {
        while ( 1 )
        {
          v19 = v16 - 24;
          if ( v16 )
            v16 -= 24;
          if ( !v17 )
            sub_4263D6(v15, v16, v19);
          v15 = &v135;
          if ( v137(&v135, v16) )
            break;
          v16 = *(_QWORD *)(v134.m128i_i64[0] + 8);
          v17 = v136;
          v134.m128i_i64[0] = v16;
          if ( v134.m128i_i64[1] == v16 )
            goto LABEL_30;
        }
        v16 = v134.m128i_i64[0];
        v17 = v136;
      }
LABEL_30:
      ++v18;
    }
    while ( v138.m128i_i64[0] != v16 );
    v115 = v18;
    goto LABEL_32;
  }
  if ( v126.m128i_i64[0] != v138.m128i_i64[0] )
    goto LABEL_21;
  v115 = 0;
LABEL_34:
  if ( v140 )
    v140(&v139, &v139, 3);
  if ( v128 )
    v128(&v127, &v127, 3);
  if ( v132 )
    v132(&v131, &v131, 3);
  v112 = 0;
  v116 = 0;
  v117 = (_QWORD *)(v93 + 40);
  v120 = (_QWORD *)(*(_QWORD *)(v93 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  v125 = (_QWORD *)(v111 + 40);
  if ( (_QWORD *)(v93 + 40) != v120 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v120 )
LABEL_148:
          BUG();
        v118 = *v120 & 0xFFFFFFFFFFFFFFF8LL;
        if ( *((_BYTE *)v120 - 8) == 55 && !sub_15F32D0((__int64)(v120 - 3)) && (*((_BYTE *)v120 - 6) & 1) == 0 )
        {
          ++v116;
          if ( v115 * v116 >= *((_DWORD *)a1 + 2) )
            goto LABEL_128;
          v100 = v120[2];
          v107 = *(_QWORD *)(v111 + 40);
          if ( v125 != (_QWORD *)(v107 & 0xFFFFFFFFFFFFFFF8LL) )
            break;
        }
LABEL_132:
        v120 = (_QWORD *)v118;
        if ( v117 == (_QWORD *)v118 )
          goto LABEL_128;
      }
      v20 = (__int64)(v120 - 3);
      v21 = (_QWORD *)(v107 & 0xFFFFFFFFFFFFFFF8LL);
      while ( 1 )
      {
        if ( !v21 )
          goto LABEL_148;
        if ( *((_BYTE *)v21 - 8) == 55 )
        {
          v22 = v21 - 3;
          sub_141EDF0(&v130, v20);
          sub_141EDF0(&v134, (__int64)(v21 - 3));
          if ( (unsigned __int8)sub_134CB50(*a1, (__int64)&v130, (__int64)&v134) == 3 )
          {
            if ( sub_15F4220(v20, (__int64)(v21 - 3), 0) )
            {
              v23 = 0;
              if ( (*(_QWORD *)(v111 + 40) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                v23 = (*(_QWORD *)(v111 + 40) & 0xFFFFFFFFFFFFFFF8LL) - 24;
              v24 = v21[1];
              v108 = v23;
              if ( v24 == v21[2] + 40LL || !v24 )
                v103 = 0;
              else
                v103 = v24 - 24;
              v25 = (__m128)_mm_loadu_si128(&v134);
              v26 = _mm_loadu_si128(&v135);
              v140 = v136;
              v138 = (__m128i)v25;
              v139 = v26;
              if ( v23 + 24 != v103 + 24 )
              {
                v27 = v23 + 24;
                v28 = v103 + 24;
                do
                {
                  v29 = v28 - 24;
                  if ( !v28 )
                    v29 = 0;
                  if ( sub_15F3330(v29) )
                    goto LABEL_48;
                  v28 = *(_QWORD *)(v28 + 8);
                }
                while ( v27 != v28 );
                v22 = v21 - 3;
              }
              if ( !(unsigned __int8)sub_134F310((_QWORD *)*a1, v103, v108, &v138, 7u) )
              {
                v30 = 0;
                if ( (*(_QWORD *)(v100 + 40) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                  v30 = (*(_QWORD *)(v100 + 40) & 0xFFFFFFFFFFFFFFF8LL) - 24;
                v109 = v30;
                v31 = v120[1];
                if ( v31 == v120[2] + 40LL || !v31 )
                  v104 = 0;
                else
                  v104 = v31 - 24;
                v32 = _mm_loadu_si128(&v130);
                v33 = _mm_loadu_si128(&v131);
                v140 = v132;
                v138 = v32;
                v139 = v33;
                if ( v30 + 24 != v104 + 24 )
                {
                  v34 = v30 + 24;
                  v89 = v22;
                  v35 = v104 + 24;
                  do
                  {
                    v36 = v35 - 24;
                    if ( !v35 )
                      v36 = 0;
                    if ( sub_15F3330(v36) )
                      goto LABEL_48;
                    v35 = *(_QWORD *)(v35 + 8);
                  }
                  while ( v34 != v35 );
                  v22 = v89;
                }
                if ( !(unsigned __int8)sub_134F310((_QWORD *)*a1, v104, v109, &v138, 7u) )
                  break;
              }
            }
          }
        }
LABEL_48:
        v21 = (_QWORD *)(*v21 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v125 == v21 )
          goto LABEL_132;
      }
      v37 = v21;
      v38 = (unsigned __int8 *)(v120 - 3);
      v39 = v37;
      v119 = *(v120 - 6);
      if ( *(_BYTE *)(v119 + 16) <= 0x17u )
        break;
      v40 = *(v37 - 6);
      if ( !v40 )
        goto LABEL_139;
      if ( *(_BYTE *)(v40 + 16) > 0x17u )
      {
        v105 = *(v37 - 6);
        v110 = sub_15F41F0(v119, v105);
        if ( v110 )
        {
          v41 = *(_QWORD *)(v119 + 8);
          if ( v41 )
          {
            if ( !*(_QWORD *)(v41 + 8) && *(_QWORD *)(v119 + 40) == v120[2] )
            {
              v42 = *(_QWORD *)(v105 + 8);
              if ( v42 )
              {
                if ( !*(_QWORD *)(v42 + 8) && *(_QWORD *)(v105 + 40) == v39[2] && *(_BYTE *)(v119 + 16) == 56 )
                {
                  v113 = v105;
                  v43 = sub_157EE30(a2);
                  if ( v43 )
                  {
                    v94 = v43;
                    sub_15F2780(v38, (__int64)v22);
                    sub_1624960((__int64)v38, 0, 0);
                    v106 = sub_15F4880((__int64)v38);
                    v44 = sub_15F4880(v119);
                    v45 = v113;
                    v114 = v44;
                    v46 = v94 - 24;
                  }
                  else
                  {
                    sub_15F2780(v38, (__int64)v22);
                    sub_1624960((__int64)v38, 0, 0);
                    v106 = sub_15F4880((__int64)v38);
                    v86 = sub_15F4880(v119);
                    v45 = v113;
                    v46 = 0;
                    v114 = v86;
                  }
                  v95 = (_QWORD *)v45;
                  sub_15F2120(v106, v46);
                  sub_15F2120(v114, v106);
                  v47 = *(v39 - 9);
                  v48 = v95;
                  v101 = (__int64 *)*(v120 - 9);
                  if ( v101 != (__int64 *)v47 )
                  {
                    v49 = *(_QWORD *)(a2 + 48);
                    if ( v49 )
                      v49 -= 24;
                    v90 = v95;
                    v96 = *(v39 - 9);
                    v87 = v49;
                    v50.m128i_i64[0] = (__int64)sub_1649960(v47);
                    v138.m128i_i64[0] = (__int64)&v134;
                    v134 = v50;
                    v138.m128i_i64[1] = (__int64)".sink";
                    v139.m128i_i16[0] = 773;
                    v51 = *v101;
                    v88 = *v101;
                    v52 = sub_1648B60(64);
                    v55 = v96;
                    v48 = v90;
                    v56 = v52;
                    if ( v52 )
                    {
                      v91 = v96;
                      v97 = v48;
                      sub_15F1EA0(v52, v88, 53, 0, 0, v87);
                      *(_DWORD *)(v56 + 56) = 2;
                      sub_164B780(v56, v138.m128i_i64);
                      v51 = *(unsigned int *)(v56 + 56);
                      sub_1648880(v56, v51, 1);
                      v55 = v91;
                      v48 = v97;
                    }
                    v57 = *(unsigned int *)(v56 + 20);
                    v58 = v120[2];
                    v59 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
                    if ( (_DWORD)v59 == *(_DWORD *)(v56 + 56) )
                    {
                      v98 = v120[2];
                      v99 = v55;
                      v122 = v48;
                      sub_15F55D0(v56, v51, v57, v59, v53, v54);
                      LODWORD(v57) = *(_DWORD *)(v56 + 20);
                      v58 = v98;
                      v55 = v99;
                      v48 = v122;
                    }
                    v60 = ((_DWORD)v57 + 1) & 0xFFFFFFF;
                    *(_DWORD *)(v56 + 20) = v60 | v57 & 0xF0000000;
                    if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
                      v61 = *(_QWORD *)(v56 - 8);
                    else
                      v61 = v56 - 24 * v60;
                    v62 = (__int64 **)(v61 + 24LL * (unsigned int)(v60 - 1));
                    if ( *v62 )
                    {
                      v63 = v62[1];
                      v64 = (unsigned __int64)v62[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v64 = v63;
                      if ( v63 )
                        v63[2] = v64 | v63[2] & 3;
                    }
                    *v62 = v101;
                    v65 = v101[1];
                    v62[1] = (__int64 *)v65;
                    if ( v65 )
                    {
                      v53 = (__int64)(v62 + 1);
                      *(_QWORD *)(v65 + 16) = (unsigned __int64)(v62 + 1) | *(_QWORD *)(v65 + 16) & 3LL;
                    }
                    v62[2] = (__int64 *)((unsigned __int64)(v101 + 1) | (unsigned __int64)v62[2] & 3);
                    v101[1] = (__int64)v62;
                    v66 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
                    if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
                      v67 = *(_QWORD *)(v56 - 8);
                    else
                      v67 = v56 - 24 * v66;
                    *(_QWORD *)(v67 + 24LL * *(unsigned int *)(v56 + 56) + 8LL * (unsigned int)(v66 - 1) + 8) = v58;
                    v68 = *(unsigned int *)(v56 + 20);
                    v69 = v39[2];
                    v70 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
                    if ( (_DWORD)v70 == *(_DWORD *)(v56 + 56) )
                    {
                      v102 = v55;
                      v123 = v48;
                      sub_15F55D0(v56, v67, v68, v70, v53, v54);
                      LODWORD(v68) = *(_DWORD *)(v56 + 20);
                      v55 = v102;
                      v48 = v123;
                    }
                    v71 = ((_DWORD)v68 + 1) & 0xFFFFFFF;
                    *(_DWORD *)(v56 + 20) = v71 | v68 & 0xF0000000;
                    if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
                      v72 = *(_QWORD *)(v56 - 8);
                    else
                      v72 = v56 - 24 * v71;
                    v73 = (_QWORD *)(v72 + 24LL * (unsigned int)(v71 - 1));
                    if ( *v73 )
                    {
                      v74 = v73[1];
                      v75 = v73[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v75 = v74;
                      if ( v74 )
                        *(_QWORD *)(v74 + 16) = v75 | *(_QWORD *)(v74 + 16) & 3LL;
                    }
                    *v73 = v55;
                    if ( v55 )
                    {
                      v76 = *(_QWORD *)(v55 + 8);
                      v73[1] = v76;
                      if ( v76 )
                        *(_QWORD *)(v76 + 16) = (unsigned __int64)(v73 + 1) | *(_QWORD *)(v76 + 16) & 3LL;
                      v73[2] = (v55 + 8) | v73[2] & 3LL;
                      *(_QWORD *)(v55 + 8) = v73;
                    }
                    v77 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
                    if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
                      v78 = *(_QWORD *)(v56 - 8);
                    else
                      v78 = v56 - 24LL * v77;
                    *(_QWORD *)(v78 + 24LL * *(unsigned int *)(v56 + 56) + 8LL * (v77 - 1) + 8) = v69;
                    if ( *(_QWORD *)(v106 - 48) )
                    {
                      v79 = *(_QWORD *)(v106 - 40);
                      v80 = *(_QWORD *)(v106 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v80 = v79;
                      if ( v79 )
                        *(_QWORD *)(v79 + 16) = v80 | *(_QWORD *)(v79 + 16) & 3LL;
                    }
                    *(_QWORD *)(v106 - 48) = v56;
                    v81 = *(_QWORD *)(v56 + 8);
                    *(_QWORD *)(v106 - 40) = v81;
                    if ( v81 )
                      *(_QWORD *)(v81 + 16) = (v106 - 40) | *(_QWORD *)(v81 + 16) & 3LL;
                    *(_QWORD *)(v106 - 32) = (v56 + 8) | *(_QWORD *)(v106 - 32) & 3LL;
                    *(_QWORD *)(v56 + 8) = v106 - 48;
                  }
                  v121 = v48;
                  sub_15F20C0(v38);
                  sub_15F20C0(v22);
                  sub_164D160(
                    v119,
                    v114,
                    v25,
                    *(double *)v26.m128i_i64,
                    *(double *)v32.m128i_i64,
                    *(double *)v33.m128i_i64,
                    v82,
                    v83,
                    a9,
                    a10);
                  sub_15F20C0((_QWORD *)v119);
                  sub_164D160(
                    (__int64)v121,
                    v114,
                    v25,
                    *(double *)v26.m128i_i64,
                    *(double *)v32.m128i_i64,
                    *(double *)v33.m128i_i64,
                    v84,
                    v85,
                    a9,
                    a10);
                  sub_15F20C0(v121);
                  v120 = (_QWORD *)(*(_QWORD *)(v93 + 40) & 0xFFFFFFFFFFFFFFF8LL);
                  v112 = v110;
                  if ( v117 != v120 )
                    continue;
                }
              }
            }
          }
        }
      }
      goto LABEL_128;
    }
    if ( !*(v37 - 6) )
LABEL_139:
      BUG();
  }
LABEL_128:
  if ( v148 )
    v148(&v147, &v147, 3);
  if ( v144 )
    v144(&v143, &v143, 3);
  return v112;
}
