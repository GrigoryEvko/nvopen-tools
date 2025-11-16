// Function: sub_32BC110
// Address: 0x32bc110
//
__int64 __fastcall sub_32BC110(__int64 **a1, __int64 a2)
{
  __int64 **v3; // r12
  __int64 *v4; // rax
  __int64 v5; // rcx
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  __int64 v8; // rbx
  int v9; // eax
  __int64 v10; // rbx
  int v11; // eax
  unsigned __int16 *v12; // rax
  __int64 v13; // rsi
  unsigned __int16 v14; // cx
  __int64 v15; // rax
  int v16; // edx
  unsigned int v17; // r15d
  __m128i si128; // xmm4
  __int64 *v19; // rax
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __int64 v22; // rdi
  __int64 v23; // rdx
  int v24; // esi
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rsi
  __int64 *v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // rdi
  __int128 v38; // rax
  void *v39; // r9
  __int64 v40; // rax
  void *v41; // r9
  void *v42; // rax
  __int64 v43; // r10
  __int64 v44; // rax
  void *v45; // r9
  __int64 v46; // rax
  __int64 v47; // rax
  void *v48; // rax
  __int64 v49; // rcx
  void *v50; // rdx
  int v51; // r9d
  __int64 *v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // edx
  __int64 v55; // r8
  __int64 *v56; // rdi
  __int64 (*v57)(); // rax
  __int64 *v58; // rbx
  __int64 v59; // rax
  unsigned int v60; // edx
  int v61; // r9d
  __int64 v62; // rax
  __int64 v63; // rdi
  bool v64; // al
  __int64 v65; // rdi
  bool v66; // al
  __int64 v67; // rax
  int v68; // r9d
  __int64 v69; // rax
  unsigned int v70; // edx
  int v71; // r9d
  __int64 v72; // rdx
  _QWORD *v73; // rdi
  _QWORD *v74; // r12
  __int64 v75; // rsi
  _QWORD *v76; // rdx
  _QWORD *v77; // r12
  __int64 v78; // rdi
  bool v79; // al
  __int64 v80; // rax
  int v81; // ebx
  __int64 v82; // rdx
  double v83; // rax
  __int64 v84; // rcx
  __int128 v85; // rax
  int v86; // r9d
  __int64 v87; // rax
  unsigned int v88; // edx
  __int64 v89; // rax
  _QWORD *v90; // rcx
  _QWORD *v91; // r12
  int v92; // r9d
  __int64 v93; // rax
  unsigned int v94; // edx
  bool v95; // al
  bool v96; // al
  int v97; // r9d
  __int64 (*v98)(); // rax
  __int64 v99; // rax
  unsigned int v100; // edx
  __int128 v101; // [rsp-30h] [rbp-210h]
  __int128 v102; // [rsp-20h] [rbp-200h]
  __int128 v103; // [rsp-10h] [rbp-1F0h]
  __int128 v104; // [rsp-10h] [rbp-1F0h]
  __int128 v105; // [rsp-10h] [rbp-1F0h]
  __int64 v106; // [rsp-8h] [rbp-1E8h]
  __int128 v107; // [rsp+10h] [rbp-1D0h]
  __int64 v108; // [rsp+10h] [rbp-1D0h]
  __int64 **v109; // [rsp+10h] [rbp-1D0h]
  __int64 **v110; // [rsp+10h] [rbp-1D0h]
  void *v111; // [rsp+10h] [rbp-1D0h]
  __int64 **v112; // [rsp+10h] [rbp-1D0h]
  __int64 v113; // [rsp+20h] [rbp-1C0h]
  __int64 v114; // [rsp+20h] [rbp-1C0h]
  char v115; // [rsp+20h] [rbp-1C0h]
  void *v116; // [rsp+30h] [rbp-1B0h]
  void *v117; // [rsp+30h] [rbp-1B0h]
  __int64 v118; // [rsp+30h] [rbp-1B0h]
  int v119; // [rsp+38h] [rbp-1A8h]
  int v120; // [rsp+3Ch] [rbp-1A4h]
  __int64 v121; // [rsp+40h] [rbp-1A0h]
  __int64 *v122; // [rsp+48h] [rbp-198h]
  __int64 v123; // [rsp+50h] [rbp-190h]
  _DWORD *v124; // [rsp+50h] [rbp-190h]
  bool v125; // [rsp+50h] [rbp-190h]
  __int64 v126; // [rsp+58h] [rbp-188h]
  _DWORD *v127; // [rsp+58h] [rbp-188h]
  bool v128; // [rsp+58h] [rbp-188h]
  __int64 v129; // [rsp+58h] [rbp-188h]
  _DWORD *v130; // [rsp+58h] [rbp-188h]
  bool v131; // [rsp+58h] [rbp-188h]
  __int64 v132; // [rsp+58h] [rbp-188h]
  __int64 v133; // [rsp+60h] [rbp-180h]
  unsigned __int16 v134; // [rsp+68h] [rbp-178h]
  __int64 v135; // [rsp+68h] [rbp-178h]
  __int64 v136; // [rsp+78h] [rbp-168h]
  __m128i v137; // [rsp+80h] [rbp-160h] BYREF
  __m128i v138[2]; // [rsp+90h] [rbp-150h] BYREF
  int v139; // [rsp+B8h] [rbp-128h] BYREF
  int v140; // [rsp+BCh] [rbp-124h] BYREF
  __m128i v141; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v142; // [rsp+D0h] [rbp-110h] BYREF
  int v143; // [rsp+D8h] [rbp-108h]
  __int64 *v144; // [rsp+E0h] [rbp-100h] BYREF
  int v145; // [rsp+E8h] [rbp-F8h]
  __int64 v146; // [rsp+F0h] [rbp-F0h]
  void *v147; // [rsp+100h] [rbp-E0h] BYREF
  _QWORD *v148; // [rsp+108h] [rbp-D8h]
  __m128i v149; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v150; // [rsp+130h] [rbp-B0h]
  __m128i v151; // [rsp+140h] [rbp-A0h]
  __int64 v152; // [rsp+150h] [rbp-90h]
  __int64 v153; // [rsp+158h] [rbp-88h]
  __int64 v154; // [rsp+160h] [rbp-80h]
  int v155; // [rsp+168h] [rbp-78h]
  __int64 v156; // [rsp+170h] [rbp-70h]
  __int64 v157; // [rsp+178h] [rbp-68h]
  __int64 v158; // [rsp+180h] [rbp-60h] BYREF
  int v159; // [rsp+188h] [rbp-58h]
  __m128i *v160; // [rsp+190h] [rbp-50h]
  __int64 v161; // [rsp+198h] [rbp-48h]
  __int64 v162; // [rsp+1A0h] [rbp-40h] BYREF

  v3 = a1;
  v4 = *(__int64 **)(a2 + 40);
  v5 = *v4;
  v6 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v7 = _mm_loadu_si128((const __m128i *)v4 + 5);
  v120 = *((_DWORD *)v4 + 2);
  v8 = v4[10];
  LODWORD(v4) = *((_DWORD *)v4 + 22);
  v133 = v5;
  v138[0] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v119 = (int)v4;
  v9 = *(_DWORD *)(v5 + 24);
  v121 = v8;
  v141 = v6;
  v137 = v7;
  if ( v9 == 12 || (v126 = 0, v9 == 36) )
    v126 = v5;
  v10 = v141.m128i_i64[0];
  v11 = *(_DWORD *)(v141.m128i_i64[0] + 24);
  if ( v11 != 36 && v11 != 12 )
    v10 = 0;
  v12 = *(unsigned __int16 **)(a2 + 48);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *v12;
  v15 = *((_QWORD *)v12 + 1);
  v142 = v13;
  v134 = v14;
  v136 = v15;
  if ( v13 )
    sub_B96E90((__int64)&v142, v13, 1);
  v16 = *(_DWORD *)(a2 + 28);
  v17 = v134;
  si128 = _mm_load_si128(v138);
  v143 = *(_DWORD *)(a2 + 72);
  v19 = *a1;
  v20 = _mm_loadu_si128(&v141);
  v21 = _mm_load_si128(&v137);
  v22 = **a1;
  v145 = v16;
  v23 = v19[128];
  v144 = v19;
  v123 = v22;
  v146 = v23;
  v19[128] = (__int64)&v144;
  v24 = *(_DWORD *)(a2 + 24);
  v122 = *v3;
  v149 = si128;
  v150 = v20;
  v151 = v21;
  v25 = sub_3402EA0((_DWORD)v122, v24, (unsigned int)&v142, v134, v136, 0, (__int64)&v149, 3);
  if ( v25 )
    goto LABEL_9;
  v29 = v3[1];
  v139 = 2;
  v140 = 2;
  v30 = v138[0].m128i_i64[0];
  *(_QWORD *)&v107 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, int *, _QWORD))(*v29 + 2264))(
                       v29,
                       v138[0].m128i_i64[0],
                       v138[0].m128i_i64[1],
                       *v3,
                       *((unsigned __int8 *)v3 + 33),
                       *((unsigned __int8 *)v3 + 35),
                       &v139,
                       0);
  *((_QWORD *)&v107 + 1) = v34;
  if ( (_QWORD)v107 )
  {
    v35 = sub_33ECD10(1, v30, v106, v31, v32, v33);
    v162 = 0;
    v152 = v35;
    v154 = 0x100000000LL;
    v157 = 0xFFFFFFFFLL;
    v160 = &v149;
    v149 = 0u;
    v150.m128i_i64[0] = 0;
    v150.m128i_i64[1] = 328;
    v151.m128i_i64[0] = -65536;
    v153 = 0;
    v155 = 0;
    v156 = 0;
    v161 = 0;
    v138[1] = (__m128i)v107;
    v158 = v107;
    v159 = DWORD2(v107);
    v36 = *(_QWORD *)(v107 + 56);
    v162 = v36;
    if ( v36 )
      *(_QWORD *)(v36 + 24) = &v162;
    v161 = v107 + 56;
    *(_QWORD *)(v107 + 56) = &v158;
    v37 = v3[1];
    LODWORD(v154) = 1;
    v151.m128i_i64[1] = (__int64)&v158;
    *(_QWORD *)&v38 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64 *, _QWORD, _QWORD, int *, _QWORD))(*v37 + 2264))(
                        v37,
                        v141.m128i_i64[0],
                        v141.m128i_i64[1],
                        *v3,
                        *((unsigned __int8 *)v3 + 33),
                        *((unsigned __int8 *)v3 + 35),
                        &v140,
                        0);
    if ( (_QWORD)v38 && (!v139 || !v140) )
    {
      v26 = sub_340F900((_DWORD)v122, 150, (unsigned int)&v142, v134, v136, v139, v107, v38, *(_OWORD *)&v137);
      sub_33CF710(&v149);
      goto LABEL_10;
    }
    sub_33CF710(&v149);
  }
  if ( (*(_BYTE *)(v123 + 864) & 1) != 0 )
  {
    if ( v126 )
    {
      v113 = *(_QWORD *)(v126 + 96);
      v39 = sub_C33340();
      if ( *(void **)(v113 + 24) == v39 )
        v46 = *(_QWORD *)(v113 + 32);
      else
        v46 = v113 + 24;
      if ( (*(_BYTE *)(v46 + 20) & 7) == 3 )
        goto LABEL_86;
      if ( !v10 )
      {
LABEL_25:
        v116 = v39;
        v127 = sub_C33320();
        sub_C3B1B0((__int64)&v149, 1.0);
        sub_C407B0(&v147, v149.m128i_i64, v127);
        sub_C338F0((__int64)&v149);
        sub_C41640((__int64 *)&v147, *(_DWORD **)(v113 + 24), 1, (bool *)v149.m128i_i8);
        v40 = (__int64)v147;
        v128 = 0;
        v41 = v116;
        if ( *(void **)(v113 + 24) == v147 )
        {
          v65 = v113 + 24;
          if ( v147 == v116 )
            v66 = sub_C3E590(v65, (__int64)&v147);
          else
            v66 = sub_C33D00(v65, (__int64)&v147);
          v41 = v116;
          v128 = v66;
          v40 = (__int64)v147;
        }
        if ( v41 == (void *)v40 )
        {
          if ( v148 )
          {
            v75 = 3LL * *(v148 - 1);
            v76 = &v148[v75];
            if ( v148 != &v148[v75] )
            {
              v110 = v3;
              v77 = &v148[v75];
              do
              {
                v77 -= 3;
                sub_91D830(v77);
              }
              while ( v148 != v77 );
              v76 = v77;
              v3 = v110;
            }
            j_j_j___libc_free_0_0((unsigned __int64)(v76 - 1));
          }
        }
        else
        {
          sub_C338F0((__int64)&v147);
        }
        if ( v128 )
        {
          v62 = sub_3406EB0(
                  (_DWORD)v122,
                  96,
                  (unsigned int)&v142,
                  v134,
                  v136,
                  (_DWORD)v41,
                  *(_OWORD *)&v141,
                  *(_OWORD *)&v137);
          goto LABEL_71;
        }
        goto LABEL_29;
      }
    }
    else
    {
      if ( !v10 )
        goto LABEL_35;
      v39 = sub_C33340();
    }
    v43 = *(_QWORD *)(v10 + 96);
    v47 = v43 + 24;
    if ( v39 == *(void **)(v43 + 24) )
      v47 = *(_QWORD *)(v43 + 32);
    if ( (*(_BYTE *)(v47 + 20) & 7) != 3 )
    {
      if ( !v126 )
        goto LABEL_31;
      v113 = *(_QWORD *)(v126 + 96);
      goto LABEL_25;
    }
LABEL_86:
    v25 = v137.m128i_i64[0];
LABEL_9:
    v26 = v25;
    goto LABEL_10;
  }
  if ( v126 )
  {
    v113 = *(_QWORD *)(v126 + 96);
    v39 = sub_C33340();
    goto LABEL_25;
  }
LABEL_29:
  if ( !v10 )
    goto LABEL_35;
  v129 = *(_QWORD *)(v10 + 96);
  v42 = sub_C33340();
  v43 = v129;
  v39 = v42;
LABEL_31:
  v117 = v39;
  v114 = v43;
  v130 = sub_C33320();
  sub_C3B1B0((__int64)&v149, 1.0);
  sub_C407B0(&v147, v149.m128i_i64, v130);
  sub_C338F0((__int64)&v149);
  sub_C41640((__int64 *)&v147, *(_DWORD **)(v114 + 24), 1, (bool *)v149.m128i_i8);
  v44 = (__int64)v147;
  v131 = 0;
  v45 = v117;
  if ( *(void **)(v114 + 24) == v147 )
  {
    v63 = v114 + 24;
    if ( v147 == v117 )
      v64 = sub_C3E590(v63, (__int64)&v147);
    else
      v64 = sub_C33D00(v63, (__int64)&v147);
    v45 = v117;
    v131 = v64;
    v44 = (__int64)v147;
  }
  if ( v45 == (void *)v44 )
  {
    if ( v148 )
    {
      v72 = *(v148 - 1);
      v73 = &v148[3 * v72];
      if ( v148 != v73 )
      {
        v109 = v3;
        v74 = &v148[3 * v72];
        do
        {
          v74 -= 3;
          sub_91D830(v74);
        }
        while ( v148 != v74 );
        v73 = v74;
        v3 = v109;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v73 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v147);
  }
  if ( v131 )
  {
LABEL_70:
    v62 = sub_3406EB0((_DWORD)v122, 96, (unsigned int)&v142, v134, v136, (_DWORD)v45, *(_OWORD *)v138, *(_OWORD *)&v137);
LABEL_71:
    v26 = v62;
    goto LABEL_10;
  }
LABEL_35:
  if ( (unsigned __int8)sub_33E2470(*v3, v138[0].m128i_i64[0], v138[0].m128i_i64[1])
    && !(unsigned __int8)sub_33E2470(*v3, v141.m128i_i64[0], v141.m128i_i64[1]) )
  {
    v26 = sub_340F900(
            (_DWORD)v122,
            150,
            (unsigned int)&v142,
            v134,
            v136,
            v71,
            *(_OWORD *)&v141,
            *(_OWORD *)v138,
            *(_OWORD *)&v137);
    goto LABEL_10;
  }
  if ( (*(_BYTE *)(v123 + 864) & 1) != 0 || (*(_BYTE *)(a2 + 29) & 8) != 0 )
  {
    if ( *(_DWORD *)(v121 + 24) == 98 )
    {
      v67 = *(_QWORD *)(v121 + 40);
      if ( *(_QWORD *)v67 == v133
        && *(_DWORD *)(v67 + 8) == v120
        && (unsigned __int8)sub_33E2470(*v3, v141.m128i_i64[0], v141.m128i_i64[1])
        && (unsigned __int8)sub_33E2470(
                              *v3,
                              *(_QWORD *)(*(_QWORD *)(v121 + 40) + 40LL),
                              *(_QWORD *)(*(_QWORD *)(v121 + 40) + 48LL)) )
      {
        v69 = sub_3406EB0(
                (_DWORD)v122,
                96,
                (unsigned int)&v142,
                v134,
                v136,
                v68,
                *(_OWORD *)&v141,
                *(_OWORD *)(*(_QWORD *)(v121 + 40) + 40LL));
        *((_QWORD *)&v103 + 1) = v70;
        *(_QWORD *)&v103 = v69;
        v26 = sub_3406EB0((_DWORD)v122, 98, (unsigned int)&v142, v134, v136, v69, *(_OWORD *)v138, v103);
        goto LABEL_10;
      }
    }
    if ( *(_DWORD *)(v133 + 24) == 98
      && (unsigned __int8)sub_33E2470(*v3, v141.m128i_i64[0], v141.m128i_i64[1])
      && (unsigned __int8)sub_33E2470(
                            *v3,
                            *(_QWORD *)(*(_QWORD *)(v133 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(v133 + 40) + 48LL)) )
    {
      v93 = sub_3406EB0(
              (_DWORD)v122,
              98,
              (unsigned int)&v142,
              v134,
              v136,
              v92,
              *(_OWORD *)&v141,
              *(_OWORD *)(*(_QWORD *)(v133 + 40) + 40LL));
      *((_QWORD *)&v102 + 1) = v94;
      *(_QWORD *)&v102 = v93;
      v26 = sub_340F900(
              (_DWORD)v122,
              150,
              (unsigned int)&v142,
              v134,
              v136,
              v93,
              *(_OWORD *)*(_QWORD *)(v133 + 40),
              v102,
              *(_OWORD *)&v137);
      goto LABEL_10;
    }
    v115 = 1;
  }
  else
  {
    v115 = 0;
  }
  if ( !v10 )
    goto LABEL_64;
  v132 = *(_QWORD *)(v10 + 96);
  v124 = sub_C33320();
  sub_C3B1B0((__int64)&v149, 1.0);
  sub_C407B0(&v147, v149.m128i_i64, v124);
  sub_C338F0((__int64)&v149);
  sub_C41640((__int64 *)&v147, *(_DWORD **)(v132 + 24), 1, (bool *)v149.m128i_i8);
  v118 = (__int64)v147;
  v108 = *(_QWORD *)(v132 + 24);
  v48 = sub_C33340();
  v49 = v118;
  v125 = 0;
  v50 = v48;
  if ( v108 == v118 )
  {
    v111 = v48;
    v78 = v132 + 24;
    if ( v48 == (void *)v118 )
    {
      v95 = sub_C3E590(v78, (__int64)&v147);
      v49 = (__int64)v147;
      v50 = v111;
      v125 = v95;
    }
    else
    {
      v79 = sub_C33D00(v78, (__int64)&v147);
      v49 = (__int64)v147;
      v125 = v79;
      v50 = v111;
    }
  }
  if ( v50 == (void *)v49 )
  {
    if ( v148 )
    {
      v89 = 3LL * *(v148 - 1);
      v90 = &v148[v89];
      if ( v148 != &v148[v89] )
      {
        v112 = v3;
        v91 = &v148[v89];
        do
        {
          v91 -= 3;
          sub_91D830(v91);
        }
        while ( v148 != v91 );
        v90 = v91;
        v3 = v112;
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v90 - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v147);
  }
  if ( v125 )
    goto LABEL_70;
  if ( (unsigned __int8)sub_11DB340((_DWORD **)(*(_QWORD *)(v10 + 96) + 24LL), -1.0) )
  {
    if ( !*((_BYTE *)v3 + 33)
      || ((v52 = v3[1], v53 = 1, v134 == 1) || v134 && (v53 = v134, v52[v134 + 14]))
      && !*((_BYTE *)v52 + 500 * v53 + 6658) )
    {
      v138[0].m128i_i64[0] = sub_33FAF80((_DWORD)v122, 244, (unsigned int)&v142, v134, v136, v51, *(_OWORD *)v138);
      v138[0].m128i_i64[1] = v54;
      sub_32B3E80((__int64)v3, v138[0].m128i_i64[0], 1, 0, v55, 0);
      v26 = sub_3406EB0(
              (_DWORD)v122,
              96,
              (unsigned int)&v142,
              v134,
              v136,
              v138[0].m128i_i32[2],
              *(_OWORD *)&v137,
              *(_OWORD *)v138);
      goto LABEL_10;
    }
  }
  if ( *(_DWORD *)(v133 + 24) != 244
    || (v135 = (__int64)v3[1], v96 = sub_328D6E0(v135, 0xCu, v17), v97 = v135, !v96)
    && (!(unsigned __int8)sub_3286E00(&v141)
     || (v97 = v135, v98 = *(__int64 (**)())(*(_QWORD *)v135 + 616LL), v98 != sub_2FE3170)
     && ((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD, __int64, _QWORD))v98)(
          v135,
          *(_QWORD *)(v10 + 96) + 24LL,
          v17,
          v136,
          *((unsigned __int8 *)v3 + 35))) )
  {
    if ( v115 )
    {
      if ( v121 == v133 && v120 == v119 )
      {
        v81 = v136;
        v82 = v17;
        v83 = 1.0;
        v84 = v136;
        goto LABEL_114;
      }
      if ( *(_DWORD *)(v121 + 24) == 244 )
      {
        v80 = *(_QWORD *)(v121 + 40);
        if ( *(_QWORD *)v80 == v133 && *(_DWORD *)(v80 + 8) == v120 )
        {
          v81 = v136;
          v82 = v17;
          v83 = -1.0;
          v84 = v136;
LABEL_114:
          *(_QWORD *)&v85 = sub_33FE730(*v3, &v142, v82, v84, 0, v83);
          v87 = sub_3406EB0((_DWORD)v122, 96, (unsigned int)&v142, v17, v81, v86, *(_OWORD *)&v141, v85);
          *((_QWORD *)&v104 + 1) = v88;
          *(_QWORD *)&v104 = v87;
          v26 = sub_3406EB0((_DWORD)v122, 98, (unsigned int)&v142, v17, v81, v87, *(_OWORD *)v138, v104);
          goto LABEL_10;
        }
      }
    }
LABEL_64:
    v56 = v3[1];
    v57 = *(__int64 (**)())(*v56 + 1592);
    if ( v57 != sub_2FE3530 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64 *, _QWORD, __int64))v57)(v56, v17, v136) )
        goto LABEL_69;
      v56 = v3[1];
    }
    v58 = *v3;
    v149.m128i_i32[0] = 2;
    v59 = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, __int64 *, _QWORD, _QWORD, __m128i *, _QWORD))(*v56 + 2264))(
            v56,
            a2,
            0,
            v58,
            *((unsigned __int8 *)v3 + 33),
            *((unsigned __int8 *)v3 + 35),
            &v149,
            0);
    if ( v59 )
    {
      if ( v149.m128i_i32[0] <= 0 )
      {
        *((_QWORD *)&v105 + 1) = v60;
        *(_QWORD *)&v105 = v59;
        v26 = sub_33FAF80((_DWORD)v122, 244, (unsigned int)&v142, v17, v136, v61, v105);
        goto LABEL_10;
      }
      if ( !*(_QWORD *)(v59 + 56) )
        sub_33ECEA0(v58, v59);
    }
LABEL_69:
    v26 = 0;
    goto LABEL_10;
  }
  v99 = sub_33FAF80((_DWORD)v122, 244, (unsigned int)&v142, v17, v136, v97, *(_OWORD *)&v141);
  *((_QWORD *)&v101 + 1) = v100;
  *(_QWORD *)&v101 = v99;
  v26 = sub_340F900(
          (_DWORD)v122,
          150,
          (unsigned int)&v142,
          v17,
          v136,
          v99,
          *(_OWORD *)*(_QWORD *)(v133 + 40),
          v101,
          *(_OWORD *)&v137);
LABEL_10:
  v27 = v142;
  v144[128] = v146;
  if ( v27 )
    sub_B91220((__int64)&v142, v27);
  return v26;
}
