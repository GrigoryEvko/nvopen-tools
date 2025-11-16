// Function: sub_2663580
// Address: 0x2663580
//
__int64 __fastcall sub_2663580(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v6; // rdx
  __int64 *v7; // r15
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // eax
  unsigned int v15; // eax
  _QWORD *v16; // rax
  unsigned int v17; // eax
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // r14
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rdi
  _QWORD *v29; // rdi
  __m128i *v30; // rsi
  _QWORD **v31; // rbx
  _QWORD **v32; // r13
  _QWORD *v33; // r15
  unsigned __int64 v34; // rdi
  __int64 v35; // r14
  unsigned __int64 v36; // r12
  volatile signed __int32 *v37; // rdi
  __int64 v38; // r14
  unsigned __int64 v39; // r12
  volatile signed __int32 *v40; // rdi
  unsigned __int64 v41; // rdi
  __int64 v42; // rbx
  __int64 v43; // r12
  unsigned __int64 v44; // rdi
  _QWORD **v45; // r12
  _QWORD **i; // rbx
  __int64 v47; // rdi
  __int64 v48; // rax
  void *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // r14
  __int64 v53; // rbx
  _DWORD *v54; // rax
  __int64 v55; // rdx
  _DWORD *v56; // r12
  __int64 v57; // r9
  unsigned int *v58; // r12
  unsigned int *v59; // rbx
  unsigned __int64 v60; // rax
  unsigned int v61; // r12d
  unsigned int v62; // edx
  int v63; // ecx
  unsigned int v64; // ecx
  int *v65; // rax
  int v66; // edi
  unsigned __int64 *v67; // rbx
  __m128i si128; // xmm0
  __int64 v69; // r12
  __int64 v70; // rax
  __m128i *v71; // rdx
  __int64 v72; // rdi
  __m128i v73; // xmm0
  __int64 v74; // rax
  __m128i *v75; // rdx
  __int64 v76; // rdi
  __m128i v77; // xmm0
  __int64 v78; // rax
  _DWORD *v79; // rdx
  __int64 v80; // r12
  __int64 v81; // rax
  void *v82; // rdx
  char v83; // al
  _QWORD *v84; // rdx
  __int64 v85; // r8
  __int64 v86; // rax
  __m128i *v87; // rdx
  __m128i v88; // xmm0
  void *v89; // rdx
  __int64 v90; // rdi
  __int64 v91; // rdi
  _BYTE *v92; // rax
  _BYTE *v93; // rax
  __m128i *v94; // rdx
  int v95; // r8d
  __int64 v96; // rdi
  __int64 v97; // rax
  __int64 v98; // rdi
  _QWORD **v99; // r12
  _QWORD **j; // rbx
  int v101; // eax
  int v102; // r9d
  unsigned int v103; // eax
  _QWORD *v104; // rax
  __int64 v105; // rdx
  __int64 *v106; // [rsp-2B8h] [rbp-2B8h]
  char *v107; // [rsp-2B0h] [rbp-2B0h]
  __int64 *v108; // [rsp-2A8h] [rbp-2A8h]
  __int64 v109; // [rsp-2A0h] [rbp-2A0h]
  char *v110; // [rsp-298h] [rbp-298h]
  unsigned __int64 v111; // [rsp-280h] [rbp-280h]
  unsigned __int64 *v112; // [rsp-278h] [rbp-278h]
  int *v113; // [rsp-270h] [rbp-270h]
  unsigned __int8 v114; // [rsp-262h] [rbp-262h]
  char v115; // [rsp-261h] [rbp-261h]
  int v116; // [rsp-260h] [rbp-260h]
  __int64 v117; // [rsp-260h] [rbp-260h]
  char *v118; // [rsp-258h] [rbp-258h] BYREF
  char *v119; // [rsp-250h] [rbp-250h]
  __int64 v120; // [rsp-248h] [rbp-248h]
  __int64 v121[3]; // [rsp-238h] [rbp-238h] BYREF
  unsigned int v122; // [rsp-220h] [rbp-220h]
  __m128i v123; // [rsp-218h] [rbp-218h] BYREF
  _QWORD v124[2]; // [rsp-208h] [rbp-208h] BYREF
  __m128i v125; // [rsp-1F8h] [rbp-1F8h] BYREF
  _DWORD *v126; // [rsp-1E8h] [rbp-1E8h] BYREF
  _DWORD *v127; // [rsp-1E0h] [rbp-1E0h]
  __m128i v128; // [rsp-1D8h] [rbp-1D8h] BYREF
  _DWORD *v129; // [rsp-1C8h] [rbp-1C8h]
  _DWORD *v130; // [rsp-1C0h] [rbp-1C0h]
  __m128i v131; // [rsp-1B8h] [rbp-1B8h] BYREF
  unsigned int v132; // [rsp-1A0h] [rbp-1A0h]
  unsigned __int64 *v133; // [rsp-198h] [rbp-198h]
  unsigned int v134; // [rsp-190h] [rbp-190h]
  unsigned __int64 v135[6]; // [rsp-188h] [rbp-188h] BYREF
  __int64 v136; // [rsp-158h] [rbp-158h] BYREF
  __int64 v137; // [rsp-130h] [rbp-130h]
  unsigned int v138; // [rsp-120h] [rbp-120h]
  __int64 v139; // [rsp-110h] [rbp-110h]
  unsigned int v140; // [rsp-100h] [rbp-100h]
  __int64 v141; // [rsp-F0h] [rbp-F0h]
  unsigned int v142; // [rsp-E0h] [rbp-E0h]
  __int64 v143; // [rsp-D0h] [rbp-D0h]
  unsigned int v144; // [rsp-C0h] [rbp-C0h]
  __int64 *v145; // [rsp-B8h] [rbp-B8h]
  unsigned int v146; // [rsp-B0h] [rbp-B0h]
  __int64 v147[3]; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v148; // [rsp-90h] [rbp-90h]
  _QWORD *v149; // [rsp-88h] [rbp-88h]
  _QWORD **v150; // [rsp-78h] [rbp-78h] BYREF
  _QWORD **v151; // [rsp-70h] [rbp-70h]

  if ( *a1 )
    return sub_265CC40(a1, a2, a5);
  v114 = unk_4FF3388;
  if ( unk_4FF3388 )
  {
    sub_2660640(&v131, (signed __int64)a2, a3, a4);
    if ( (_BYTE)qword_4FF3B88 )
    {
      v47 = sub_C5F790((__int64)&v131, (__int64)a2);
      sub_904010(v47, "CCG before cloning:\n");
      v48 = sub_C5F790(v47, (__int64)"CCG before cloning:\n");
      sub_264D7D0((__int64)&v131, v48);
    }
    if ( byte_4FF4088 )
    {
      sub_263F570(v128.m128i_i64, "postbuild");
      sub_265EF70((__int64)&v131, (__int64)&v128);
      sub_2240A30((unsigned __int64 *)&v128);
    }
    if ( (_BYTE)qword_4FF3AA8 )
    {
      v45 = v151;
      for ( i = v150; v45 != i; ++i )
      {
        if ( *((_BYTE *)*i + 2) )
          sub_264C780(*i);
      }
    }
    v6 = v146;
    v7 = v145;
    v125.m128i_i64[1] = 0;
    v126 = 0;
    v8 = v145;
    v127 = 0;
    v9 = &v145[3 * v146];
    if ( v145 == v9 )
    {
      v125.m128i_i64[0] = 1;
      goto LABEL_47;
    }
    LODWORD(v10) = 0;
    v125.m128i_i64[0] = 1;
LABEL_11:
    if ( !HIDWORD(v126) )
      goto LABEL_16;
    v11 = (unsigned int)v127;
    if ( (unsigned int)v127 <= 0x40 )
    {
LABEL_13:
      v12 = (_QWORD *)v125.m128i_i64[1];
      v6 = v125.m128i_i64[1] + 8 * v11;
      if ( v125.m128i_i64[1] != v6 )
      {
        do
          *v12++ = -4096;
        while ( (_QWORD *)v6 != v12 );
      }
      v126 = 0;
      goto LABEL_16;
    }
LABEL_106:
    v116 = v10;
    sub_C7D6A0(v125.m128i_i64[1], 8 * v11, 8);
    sub_2646720((__int64)&v125, v116);
LABEL_16:
    while ( 1 )
    {
      v13 = v7[2];
      v7 += 3;
      sub_264D230((__int64)&v128, v13, v6);
      sub_2659FF0(&v131, *(v7 - 1), (__int64)&v125, (__int64)&v128);
      sub_C7D6A0(v128.m128i_i64[1], 4LL * (unsigned int)v130, 4);
      if ( v9 == v7 )
        break;
      LODWORD(v10) = (_DWORD)v126;
      ++v125.m128i_i64[0];
      if ( !(_DWORD)v126 )
        goto LABEL_11;
      v14 = 4 * (_DWORD)v126;
      v11 = (unsigned int)v127;
      if ( (unsigned int)(4 * (_DWORD)v126) < 0x40 )
        v14 = 64;
      if ( v14 >= (unsigned int)v127 )
        goto LABEL_13;
      if ( (_DWORD)v126 == 1 )
      {
        LODWORD(v10) = 64;
        goto LABEL_106;
      }
      _BitScanReverse(&v15, (_DWORD)v126 - 1);
      v10 = (unsigned int)(1 << (33 - (v15 ^ 0x1F)));
      if ( (int)v10 < 64 )
        v10 = 64;
      if ( (_DWORD)v10 != (_DWORD)v127 )
        goto LABEL_106;
      v16 = (_QWORD *)v125.m128i_i64[1];
      v126 = 0;
      v6 = v125.m128i_i64[1] + 8 * v10;
      do
      {
        if ( v16 )
          *v16 = -4096;
        ++v16;
      }
      while ( (_QWORD *)v6 != v16 );
    }
    LODWORD(v21) = (_DWORD)v126;
    ++v125.m128i_i64[0];
    if ( (_DWORD)v126 )
    {
      v22 = 4 * (_DWORD)v126;
      v23 = (unsigned int)v127;
      if ( (unsigned int)(4 * (_DWORD)v126) < 0x40 )
        v22 = 64;
      if ( (unsigned int)v127 <= v22 )
        goto LABEL_44;
      if ( (_DWORD)v126 == 1 )
      {
        LODWORD(v21) = 64;
      }
      else
      {
        _BitScanReverse(&v103, (_DWORD)v126 - 1);
        v21 = (unsigned int)(1 << (33 - (v103 ^ 0x1F)));
        if ( (int)v21 < 64 )
          v21 = 64;
        if ( (_DWORD)v21 == (_DWORD)v127 )
        {
          v104 = (_QWORD *)v125.m128i_i64[1];
          v126 = 0;
          v105 = v125.m128i_i64[1] + 8 * v21;
          do
          {
            if ( v104 )
              *v104 = -4096;
            ++v104;
          }
          while ( (_QWORD *)v105 != v104 );
          goto LABEL_99;
        }
      }
    }
    else
    {
      if ( !HIDWORD(v126) )
      {
LABEL_99:
        v8 = v145;
        v6 = v146;
LABEL_47:
        v26 = v8;
        v27 = &v8[3 * v6];
        if ( v8 != v27 )
        {
          do
          {
            v28 = v26[2];
            v26 += 3;
            sub_264FAB0(v28, (__int64)&v125);
          }
          while ( v27 != v26 );
        }
        if ( (_BYTE)qword_4FF3AA8 )
          sub_264C8B0((__int64)&v131);
        v29 = (_QWORD *)v125.m128i_i64[1];
        v30 = (__m128i *)(8LL * (unsigned int)v127);
        sub_C7D6A0(v125.m128i_i64[1], (__int64)v30, 8);
        if ( (_BYTE)qword_4FF3AA8 )
        {
          v99 = v151;
          for ( j = v150; v99 != j; ++j )
          {
            v29 = *j;
            if ( *((_BYTE *)*j + 2) )
              sub_264C780(v29);
          }
        }
        if ( (_BYTE)qword_4FF3B88 )
        {
          v98 = sub_C5F790((__int64)v29, (__int64)v30);
          sub_904010(v98, "CCG after cloning:\n");
          v30 = (__m128i *)sub_C5F790(v98, (__int64)"CCG after cloning:\n");
          sub_264D7D0((__int64)&v131, (__int64)v30);
        }
        if ( byte_4FF4088 )
        {
          sub_263F570(v128.m128i_i64, "cloned");
          v30 = &v128;
          sub_265EF70((__int64)&v131, (__int64)&v128);
          sub_2240A30((unsigned __int64 *)&v128);
        }
        v114 = sub_2655460((__int64)&v131, a5);
        if ( (_BYTE)qword_4FF3B88 )
        {
          v96 = sub_C5F790((__int64)&v131, (__int64)v30);
          sub_904010(v96, "CCG after assigning function clones:\n");
          v97 = sub_C5F790(v96, (__int64)"CCG after assigning function clones:\n");
          sub_264D7D0((__int64)&v131, v97);
        }
        if ( byte_4FF4088 )
        {
          sub_263F570(v128.m128i_i64, "clonefuncassign");
          sub_265EF70((__int64)&v131, (__int64)&v128);
          sub_2240A30((unsigned __int64 *)&v128);
        }
        if ( !LOBYTE(qword_4F8F408[8]) )
        {
LABEL_61:
          v31 = v151;
          v32 = v150;
          if ( v151 != v150 )
          {
            do
            {
              v33 = *v32;
              if ( *v32 )
              {
                v34 = v33[12];
                if ( v34 )
                  j_j___libc_free_0(v34);
                v35 = v33[10];
                v36 = v33[9];
                if ( v35 != v36 )
                {
                  do
                  {
                    v37 = *(volatile signed __int32 **)(v36 + 8);
                    if ( v37 )
                      sub_A191D0(v37);
                    v36 += 16LL;
                  }
                  while ( v35 != v36 );
                  v36 = v33[9];
                }
                if ( v36 )
                  j_j___libc_free_0(v36);
                v38 = v33[7];
                v39 = v33[6];
                if ( v38 != v39 )
                {
                  do
                  {
                    v40 = *(volatile signed __int32 **)(v39 + 8);
                    if ( v40 )
                      sub_A191D0(v40);
                    v39 += 16LL;
                  }
                  while ( v38 != v39 );
                  v39 = v33[6];
                }
                if ( v39 )
                  j_j___libc_free_0(v39);
                v41 = v33[3];
                if ( (_QWORD *)v41 != v33 + 5 )
                  _libc_free(v41);
                j_j___libc_free_0((unsigned __int64)v33);
              }
              ++v32;
            }
            while ( v31 != v32 );
            v32 = v150;
          }
          goto LABEL_84;
        }
        v49 = sub_CB72A0();
        v32 = v150;
        v51 = (__int64)v49;
        v106 = (__int64 *)v151;
        if ( v151 == v150 )
        {
LABEL_84:
          if ( v32 )
            j_j___libc_free_0((unsigned __int64)v32);
          if ( v149 != &v150 )
            _libc_free((unsigned __int64)v149);
          sub_C7D6A0(v147[1], 24LL * v148, 8);
          if ( v145 != v147 )
            _libc_free((unsigned __int64)v145);
          sub_C7D6A0(v143, 24LL * v144, 8);
          sub_C7D6A0(v141, 16LL * v142, 8);
          v17 = v140;
          if ( v140 )
          {
            v42 = v139;
            v43 = v139 + 32LL * v140;
            do
            {
              if ( *(_DWORD *)v42 <= 0xFFFFFFFD )
              {
                v44 = *(_QWORD *)(v42 + 8);
                if ( v44 )
                  j_j___libc_free_0(v44);
              }
              v42 += 32;
            }
            while ( v43 != v42 );
            v17 = v140;
          }
          sub_C7D6A0(v139, 32LL * v17, 8);
          sub_C7D6A0(v137, 8LL * v138, 4);
          sub_2342640((__int64)&v136);
          sub_2641300(v135[2]);
          v18 = v133;
          v19 = &v133[4 * v134];
          if ( v133 != v19 )
          {
            do
            {
              v20 = *(v19 - 3);
              v19 -= 4;
              if ( v20 )
                j_j___libc_free_0(v20);
            }
            while ( v18 != v19 );
            v19 = v133;
          }
          if ( v19 != v135 )
            _libc_free((unsigned __int64)v19);
          sub_C7D6A0(v131.m128i_i64[1], 16LL * v132, 8);
          return v114;
        }
        v108 = (__int64 *)v150;
        while ( 1 )
        {
          v52 = *v108;
          if ( *(_BYTE *)(*v108 + 2) && *(_BYTE *)v52 )
          {
            sub_264D230((__int64)v121, v52, v50);
            v53 = *(_QWORD *)(v52 + 8);
            v125.m128i_i64[0] = *(_QWORD *)(v53 + 72);
            v115 = 0;
            if ( (unsigned __int8)sub_A747A0(&v125, "memprof", 7u) )
            {
              v123.m128i_i64[0] = *(_QWORD *)(v53 + 72);
              v125.m128i_i64[0] = sub_A747B0(&v123, -1, "memprof", 7u);
              v54 = (_DWORD *)sub_A72240(v125.m128i_i64);
              if ( v55 == 4 )
                v115 = (*v54 == 1684828003) + 1;
              else
                v115 = 1;
            }
            v56 = (_DWORD *)(v121[1] + 4LL * v122);
            v117 = v121[0];
            sub_22B0690(&v128, v121);
            v124[0] = v56;
            v124[1] = v56;
            v123.m128i_i64[0] = (__int64)v121;
            v123.m128i_i64[1] = v117;
            v126 = v129;
            v127 = v130;
            v125 = v128;
            v118 = 0;
            v119 = 0;
            v120 = 0;
            sub_2640D80(
              (__int64 *)&v118,
              (__int64)v121,
              v128.m128i_i64[1],
              (__int64)v126,
              v117,
              v57,
              v128.m128i_i32[0],
              v128.m128i_i32[2],
              v126,
              v130,
              (int)v121,
              v117,
              v56);
            v58 = (unsigned int *)v119;
            v107 = v118;
            v59 = (unsigned int *)v118;
            if ( v119 != v118 )
            {
              _BitScanReverse64(&v60, (v119 - v118) >> 2);
              sub_263F8F0(v118, v119, 2LL * (int)(63 - (v60 ^ 0x3F)));
              sub_263F470(v59, v58);
              v107 = v119;
              if ( v119 != v118 )
              {
                v110 = v118;
                while ( 1 )
                {
                  v61 = *(_DWORD *)v110;
                  if ( !v138 )
                    goto LABEL_171;
                  v62 = (v138 - 1) & (37 * v61);
                  v113 = (int *)(v137 + 8LL * v62);
                  v63 = *v113;
                  if ( v61 != *v113 )
                    break;
LABEL_120:
                  if ( v140 )
                  {
                    v64 = (v140 - 1) & (37 * v61);
                    v65 = (int *)(v139 + 32LL * v64);
                    v66 = *v65;
                    if ( v61 == *v65 )
                    {
LABEL_122:
                      if ( v65 != (int *)(v139 + 32LL * v140) )
                      {
                        v67 = (unsigned __int64 *)*((_QWORD *)v65 + 1);
                        v112 = (unsigned __int64 *)*((_QWORD *)v65 + 2);
                        if ( v67 != v112 )
                        {
                          v111 = v61;
                          do
                          {
                            while ( 1 )
                            {
                              v94 = *(__m128i **)(v51 + 32);
                              if ( *(_QWORD *)(v51 + 24) - (_QWORD)v94 > 0x10u )
                              {
                                si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E310);
                                v94[1].m128i_i8[0] = 32;
                                v69 = v51;
                                *v94 = si128;
                                *(_QWORD *)(v51 + 32) += 17LL;
                              }
                              else
                              {
                                v69 = sub_CB6200(v51, "MemProf hinting: ", 0x11u);
                              }
                              sub_2643CE0(&v125, *((_BYTE *)v113 + 4));
                              v70 = sub_CB6200(v69, (unsigned __int8 *)v125.m128i_i64[0], v125.m128i_u64[1]);
                              v71 = *(__m128i **)(v70 + 32);
                              v72 = v70;
                              if ( *(_QWORD *)(v70 + 24) - (_QWORD)v71 <= 0x18u )
                              {
                                v72 = sub_CB6200(v70, " full allocation context ", 0x19u);
                              }
                              else
                              {
                                v73 = _mm_load_si128((const __m128i *)&xmmword_438D560);
                                v71[1].m128i_i8[8] = 32;
                                v71[1].m128i_i64[0] = 0x747865746E6F6320LL;
                                *v71 = v73;
                                *(_QWORD *)(v70 + 32) += 25LL;
                              }
                              v74 = sub_CB59D0(v72, *v67);
                              v75 = *(__m128i **)(v74 + 32);
                              v76 = v74;
                              if ( *(_QWORD *)(v74 + 24) - (_QWORD)v75 <= 0x10u )
                              {
                                v76 = sub_CB6200(v74, " with total size ", 0x11u);
                              }
                              else
                              {
                                v77 = _mm_load_si128((const __m128i *)&xmmword_438D570);
                                v75[1].m128i_i8[0] = 32;
                                *v75 = v77;
                                *(_QWORD *)(v74 + 32) += 17LL;
                              }
                              v78 = sub_CB59D0(v76, v67[1]);
                              v79 = *(_DWORD **)(v78 + 32);
                              v80 = v78;
                              if ( *(_QWORD *)(v78 + 24) - (_QWORD)v79 <= 3u )
                              {
                                v80 = sub_CB6200(v78, (unsigned __int8 *)" is ", 4u);
                              }
                              else
                              {
                                *v79 = 544434464;
                                *(_QWORD *)(v78 + 32) += 4LL;
                              }
                              sub_2643CE0(&v123, *(_BYTE *)(v52 + 2));
                              v81 = sub_CB6200(v80, (unsigned __int8 *)v123.m128i_i64[0], v123.m128i_u64[1]);
                              v82 = *(void **)(v81 + 32);
                              if ( *(_QWORD *)(v81 + 24) - (_QWORD)v82 <= 0xDu )
                              {
                                sub_CB6200(v81, " after cloning", 0xEu);
                              }
                              else
                              {
                                qmemcpy(v82, " after cloning", 14);
                                *(_QWORD *)(v81 + 32) += 14LL;
                              }
                              if ( (_QWORD *)v123.m128i_i64[0] != v124 )
                                j_j___libc_free_0(v123.m128i_u64[0]);
                              if ( (_DWORD **)v125.m128i_i64[0] != &v126 )
                                j_j___libc_free_0(v125.m128i_u64[0]);
                              v83 = *(_BYTE *)(v52 + 2);
                              if ( v83 == 3 )
                                v83 = 1;
                              if ( v83 != v115 )
                              {
                                v84 = *(_QWORD **)(v51 + 32);
                                if ( *(_QWORD *)(v51 + 24) - (_QWORD)v84 <= 7u )
                                {
                                  v85 = sub_CB6200(v51, " marked ", 8u);
                                }
                                else
                                {
                                  v85 = v51;
                                  *v84 = 0x2064656B72616D20LL;
                                  *(_QWORD *)(v51 + 32) += 8LL;
                                }
                                v109 = v85;
                                sub_2643CE0(&v125, v115);
                                v86 = sub_CB6200(v109, (unsigned __int8 *)v125.m128i_i64[0], v125.m128i_u64[1]);
                                v87 = *(__m128i **)(v86 + 32);
                                if ( *(_QWORD *)(v86 + 24) - (_QWORD)v87 <= 0x18u )
                                {
                                  sub_CB6200(v86, " due to cold byte percent", 0x19u);
                                }
                                else
                                {
                                  v88 = _mm_load_si128((const __m128i *)&xmmword_438D580);
                                  v87[1].m128i_i8[8] = 116;
                                  v87[1].m128i_i64[0] = 0x6E65637265702065LL;
                                  *v87 = v88;
                                  *(_QWORD *)(v86 + 32) += 25LL;
                                }
                                if ( (_DWORD **)v125.m128i_i64[0] != &v126 )
                                  j_j___libc_free_0(v125.m128i_u64[0]);
                              }
                              v89 = *(void **)(v51 + 32);
                              if ( *(_QWORD *)(v51 + 24) - (_QWORD)v89 <= 0xCu )
                              {
                                v90 = sub_CB6200(v51, " (context id ", 0xDu);
                              }
                              else
                              {
                                v90 = v51;
                                qmemcpy(v89, " (context id ", 13);
                                *(_QWORD *)(v51 + 32) += 13LL;
                              }
                              v91 = sub_CB59D0(v90, v111);
                              v92 = *(_BYTE **)(v91 + 32);
                              if ( *(_BYTE **)(v91 + 24) == v92 )
                              {
                                sub_CB6200(v91, (unsigned __int8 *)")", 1u);
                              }
                              else
                              {
                                *v92 = 41;
                                ++*(_QWORD *)(v91 + 32);
                              }
                              v93 = *(_BYTE **)(v51 + 32);
                              if ( *(_BYTE **)(v51 + 24) == v93 )
                                break;
                              *v93 = 10;
                              v67 += 2;
                              ++*(_QWORD *)(v51 + 32);
                              if ( v112 == v67 )
                                goto LABEL_156;
                            }
                            v67 += 2;
                            sub_CB6200(v51, (unsigned __int8 *)"\n", 1u);
                          }
                          while ( v112 != v67 );
                        }
                      }
                    }
                    else
                    {
                      v101 = 1;
                      while ( v66 != -1 )
                      {
                        v102 = v101 + 1;
                        v64 = (v140 - 1) & (v101 + v64);
                        v65 = (int *)(v139 + 32LL * v64);
                        v66 = *v65;
                        if ( v61 == *v65 )
                          goto LABEL_122;
                        v101 = v102;
                      }
                    }
                  }
LABEL_156:
                  v110 += 4;
                  if ( v107 == v110 )
                  {
                    v107 = v118;
                    goto LABEL_158;
                  }
                }
                v95 = 1;
                while ( v63 != -1 )
                {
                  v62 = (v138 - 1) & (v95 + v62);
                  v63 = *(_DWORD *)(v137 + 8LL * v62);
                  if ( v61 == v63 )
                  {
                    v113 = (int *)(v137 + 8LL * v62);
                    goto LABEL_120;
                  }
                  ++v95;
                }
LABEL_171:
                v113 = (int *)(v137 + 8LL * v138);
                goto LABEL_120;
              }
            }
LABEL_158:
            if ( v107 )
              j_j___libc_free_0((unsigned __int64)v107);
            sub_2342640((__int64)v121);
          }
          if ( v106 == ++v108 )
            goto LABEL_61;
        }
      }
      v23 = (unsigned int)v127;
      if ( (unsigned int)v127 <= 0x40 )
      {
LABEL_44:
        v24 = (_QWORD *)v125.m128i_i64[1];
        v25 = v125.m128i_i64[1] + 8 * v23;
        if ( v125.m128i_i64[1] != v25 )
        {
          do
            *v24++ = -4096;
          while ( (_QWORD *)v25 != v24 );
        }
        v8 = v145;
        v6 = v146;
        v126 = 0;
        goto LABEL_47;
      }
    }
    sub_C7D6A0(v125.m128i_i64[1], 8 * v23, 8);
    sub_2646720((__int64)&v125, v21);
    goto LABEL_99;
  }
  return v114;
}
