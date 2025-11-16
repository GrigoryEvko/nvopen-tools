// Function: sub_F73BC0
// Address: 0xf73bc0
//
__int64 __fastcall sub_F73BC0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, char a5)
{
  __int64 *v5; // rbx
  __int64 v9; // rsi
  unsigned __int64 *v10; // r8
  unsigned __int64 *v11; // rcx
  unsigned __int64 v12; // r9
  int v13; // edx
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rax
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __int64 v26; // rsi
  unsigned __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // r14
  _QWORD *v31; // r13
  __int64 v32; // r12
  __int64 v33; // r15
  _QWORD *v34; // r14
  __int64 v35; // r12
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // r13
  _QWORD *v40; // r14
  __int64 v41; // r15
  unsigned __int64 v42; // rax
  __int64 v43; // r13
  _QWORD *v44; // r15
  __int64 v45; // rax
  unsigned __int64 *v46; // rbx
  unsigned __int64 *v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  unsigned int *v53; // r14
  unsigned int *v54; // r13
  __int64 v55; // rdx
  _QWORD **v56; // rdx
  int v57; // ecx
  __int64 *v58; // rax
  __int64 v59; // rsi
  unsigned __int64 v60; // r15
  __int64 v61; // r12
  __int64 v62; // rdx
  unsigned int v63; // esi
  _QWORD **v64; // rdx
  int v65; // ecx
  __int64 *v66; // rax
  __int64 v67; // rsi
  unsigned __int64 v68; // r14
  __int64 v69; // r12
  __int64 v70; // rdx
  unsigned int v71; // esi
  unsigned int *v72; // r13
  unsigned int *v73; // r12
  __int64 v74; // rdx
  unsigned int *v75; // r13
  unsigned int *v76; // r12
  __int64 v77; // rdx
  _QWORD **v78; // rdx
  int v79; // ecx
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // r14
  unsigned __int64 i; // r13
  __int64 v84; // rdx
  unsigned int v85; // esi
  unsigned int *v86; // r13
  unsigned int *v87; // r12
  __int64 v88; // rdx
  _QWORD **v89; // rdx
  int v90; // ecx
  __int64 *v91; // rax
  __int64 v92; // rsi
  unsigned __int64 v93; // r15
  __int64 v94; // r13
  __int64 v95; // rdx
  unsigned int v96; // esi
  unsigned __int64 *v97; // rsi
  unsigned __int64 *v98; // rsi
  unsigned __int64 *v99; // [rsp+10h] [rbp-3F0h]
  unsigned __int64 *v100; // [rsp+10h] [rbp-3F0h]
  unsigned __int64 *v101; // [rsp+10h] [rbp-3F0h]
  unsigned __int64 *v102; // [rsp+10h] [rbp-3F0h]
  int v103; // [rsp+10h] [rbp-3F0h]
  unsigned __int64 *v104; // [rsp+18h] [rbp-3E8h]
  unsigned __int64 *v105; // [rsp+18h] [rbp-3E8h]
  unsigned __int64 *v106; // [rsp+18h] [rbp-3E8h]
  unsigned __int64 *v107; // [rsp+18h] [rbp-3E8h]
  char *v108; // [rsp+18h] [rbp-3E8h]
  unsigned __int64 *v109; // [rsp+18h] [rbp-3E8h]
  int v110; // [rsp+18h] [rbp-3E8h]
  __int64 v111; // [rsp+28h] [rbp-3D8h]
  __int64 v112; // [rsp+28h] [rbp-3D8h]
  __int64 *v113; // [rsp+30h] [rbp-3D0h]
  unsigned __int64 *v114; // [rsp+30h] [rbp-3D0h]
  __int64 v115; // [rsp+40h] [rbp-3C0h]
  __int64 v116; // [rsp+40h] [rbp-3C0h]
  __int64 v117; // [rsp+50h] [rbp-3B0h]
  __int64 v118; // [rsp+58h] [rbp-3A8h]
  __int64 v119; // [rsp+60h] [rbp-3A0h]
  __int64 v120; // [rsp+68h] [rbp-398h]
  _QWORD v121[2]; // [rsp+70h] [rbp-390h] BYREF
  __int64 v122; // [rsp+80h] [rbp-380h]
  __int64 v123; // [rsp+88h] [rbp-378h] BYREF
  __int16 v124; // [rsp+90h] [rbp-370h]
  __int64 v125; // [rsp+98h] [rbp-368h]
  __int64 v126; // [rsp+A0h] [rbp-360h]
  unsigned __int64 v127[2]; // [rsp+B0h] [rbp-350h] BYREF
  __int64 v128; // [rsp+C0h] [rbp-340h]
  __m128i v129; // [rsp+C8h] [rbp-338h] BYREF
  __m128i v130; // [rsp+D8h] [rbp-328h] BYREF
  __m128i v131; // [rsp+E8h] [rbp-318h] BYREF
  __m128i v132; // [rsp+F8h] [rbp-308h] BYREF
  __int64 v133; // [rsp+108h] [rbp-2F8h]
  unsigned __int64 v134; // [rsp+110h] [rbp-2F0h] BYREF
  __int64 v135; // [rsp+118h] [rbp-2E8h]
  __int64 v136; // [rsp+120h] [rbp-2E0h] BYREF
  unsigned __int64 v137[2]; // [rsp+128h] [rbp-2D8h] BYREF
  __int64 v138; // [rsp+138h] [rbp-2C8h]
  __int64 v139; // [rsp+140h] [rbp-2C0h]
  unsigned __int64 v140; // [rsp+148h] [rbp-2B8h] BYREF
  __int64 v141; // [rsp+150h] [rbp-2B0h]
  __int64 v142; // [rsp+158h] [rbp-2A8h]
  unsigned __int64 v143; // [rsp+160h] [rbp-2A0h] BYREF
  void **v144; // [rsp+168h] [rbp-298h]
  __int64 v145; // [rsp+170h] [rbp-290h]
  __int64 v146; // [rsp+178h] [rbp-288h]
  __int64 v147; // [rsp+180h] [rbp-280h]
  __int64 v148; // [rsp+188h] [rbp-278h]
  void *v149; // [rsp+190h] [rbp-270h] BYREF
  void *v150; // [rsp+198h] [rbp-268h]
  __int64 v151; // [rsp+1A0h] [rbp-260h]
  __m128i v152; // [rsp+1A8h] [rbp-258h]
  __m128i v153; // [rsp+1B8h] [rbp-248h]
  __m128i v154; // [rsp+1C8h] [rbp-238h]
  __m128i v155; // [rsp+1D8h] [rbp-228h]
  __int64 v156; // [rsp+1E8h] [rbp-218h]
  void *v157; // [rsp+1F0h] [rbp-210h] BYREF
  unsigned __int64 *v158; // [rsp+200h] [rbp-200h] BYREF
  __int64 v159; // [rsp+208h] [rbp-1F8h]
  _BYTE v160[496]; // [rsp+210h] [rbp-1F0h] BYREF

  v5 = *(__int64 **)a3;
  v158 = (unsigned __int64 *)v160;
  v159 = 0x400000000LL;
  v113 = &v5[2 * *(unsigned int *)(a3 + 8)];
  if ( v5 != v113 )
  {
    sub_F71620(v121, *v5, a2, a1, a4, a5);
    while ( 1 )
    {
      sub_F71620(v127, v5[1], a2, a1, a4, a5);
      v134 = 6;
      v135 = 0;
      v136 = v122;
      if ( v122 != -4096 && v122 != 0 && v122 != -8192 )
        sub_BD6050(&v134, v121[0] & 0xFFFFFFFFFFFFFFF8LL);
      v137[0] = 6;
      v137[1] = 0;
      v138 = v125;
      if ( v125 != 0 && v125 != -4096 && v125 != -8192 )
        sub_BD6050(v137, v123 & 0xFFFFFFFFFFFFFFF8LL);
      v140 = 6;
      v141 = 0;
      v139 = v126;
      v142 = v128;
      if ( v128 != 0 && v128 != -4096 && v128 != -8192 )
        sub_BD6050(&v140, v127[0] & 0xFFFFFFFFFFFFFFF8LL);
      v143 = 6;
      v144 = 0;
      v145 = v130.m128i_i64[0];
      if ( v130.m128i_i64[0] == 0 || v130.m128i_i64[0] == -4096 || v130.m128i_i64[0] == -8192 )
      {
        v146 = v130.m128i_i64[1];
      }
      else
      {
        sub_BD6050(&v143, v129.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
        v146 = v130.m128i_i64[1];
        if ( v130.m128i_i64[0] != -8192 && v130.m128i_i64[0] != -4096 && v130.m128i_i64[0] )
          sub_BD60C0(&v129);
      }
      if ( v128 != -4096 && v128 != 0 && v128 != -8192 )
        sub_BD60C0(v127);
      if ( v125 != -4096 && v125 != 0 && v125 != -8192 )
        sub_BD60C0(&v123);
      if ( v122 != 0 && v122 != -4096 && v122 != -8192 )
        sub_BD60C0(v121);
      v9 = (unsigned int)v159;
      v10 = v158;
      v11 = &v134;
      v12 = (unsigned int)v159 + 1LL;
      v13 = v159;
      if ( v12 > HIDWORD(v159) )
      {
        if ( v158 > &v134 || &v134 >= &v158[14 * (unsigned int)v159] )
        {
          v98 = (unsigned __int64 *)sub_C8D7D0(
                                      (__int64)&v158,
                                      (__int64)v160,
                                      (unsigned int)v159 + 1LL,
                                      0x70u,
                                      v127,
                                      v12);
          v109 = v98;
          sub_F73960((__int64)&v158, v98);
          v10 = v98;
          if ( v158 == (unsigned __int64 *)v160 )
          {
            v9 = (unsigned int)v159;
            v158 = v109;
            HIDWORD(v159) = v127[0];
            v11 = &v134;
          }
          else
          {
            v110 = v127[0];
            _libc_free(v158, v98);
            v10 = v98;
            v9 = (unsigned int)v159;
            v11 = &v134;
            v158 = v10;
            HIDWORD(v159) = v110;
          }
          v13 = v159;
        }
        else
        {
          v108 = (char *)((char *)&v134 - (char *)v158);
          v97 = (unsigned __int64 *)sub_C8D7D0(
                                      (__int64)&v158,
                                      (__int64)v160,
                                      (unsigned int)v159 + 1LL,
                                      0x70u,
                                      v127,
                                      v12);
          sub_F73960((__int64)&v158, v97);
          v10 = v97;
          if ( v158 == (unsigned __int64 *)v160 )
          {
            v158 = v97;
            HIDWORD(v159) = v127[0];
          }
          else
          {
            v103 = v127[0];
            _libc_free(v158, v97);
            v10 = v97;
            v158 = v97;
            HIDWORD(v159) = v103;
          }
          v9 = (unsigned int)v159;
          v11 = (unsigned __int64 *)&v108[(_QWORD)v10];
          v13 = v159;
        }
      }
      v14 = &v10[14 * v9];
      if ( v14 )
      {
        *v14 = 6;
        v15 = v11[2];
        v14[1] = 0;
        v14[2] = v15;
        if ( v15 != -4096 && v15 != 0 && v15 != -8192 )
        {
          v99 = v11;
          v104 = &v10[14 * v9];
          sub_BD6050(v14, *v11 & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v99;
          v14 = v104;
        }
        v14[3] = 6;
        v16 = v11[5];
        v14[4] = 0;
        v14[5] = v16;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        {
          v100 = v11;
          v105 = v14;
          sub_BD6050(v14 + 3, v11[3] & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v100;
          v14 = v105;
        }
        v17 = v11[6];
        v14[7] = 6;
        v14[8] = 0;
        v14[6] = v17;
        v18 = v11[9];
        v14[9] = v18;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        {
          v101 = v11;
          v106 = v14;
          sub_BD6050(v14 + 7, v11[7] & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v101;
          v14 = v106;
        }
        v14[10] = 6;
        v19 = v11[12];
        v14[11] = 0;
        v14[12] = v19;
        if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        {
          v102 = v11;
          v107 = v14;
          sub_BD6050(v14 + 10, v11[10] & 0xFFFFFFFFFFFFFFF8LL);
          v11 = v102;
          v14 = v107;
        }
        v14[13] = v11[13];
        v13 = v159;
      }
      LODWORD(v159) = v13 + 1;
      if ( v145 != 0 && v145 != -4096 && v145 != -8192 )
        sub_BD60C0(&v143);
      if ( v142 != 0 && v142 != -4096 && v142 != -8192 )
        sub_BD60C0(&v140);
      if ( v138 != -4096 && v138 != 0 && v138 != -8192 )
        sub_BD60C0(v137);
      if ( v136 != -4096 && v136 != 0 && v136 != -8192 )
        sub_BD60C0(&v134);
      v5 += 2;
      if ( v113 == v5 )
        break;
      sub_F71620(v121, *v5, a2, a1, a4, a5);
    }
  }
  v20 = sub_BD5C60(a1);
  v21 = sub_B43CC0(a1);
  v142 = v20;
  v128 = v21;
  v134 = (unsigned __int64)&v136;
  v135 = 0x200000000LL;
  v143 = (unsigned __int64)&v149;
  v127[1] = (unsigned __int64)&unk_49D94D0;
  v127[0] = (unsigned __int64)&unk_49E5698;
  LOWORD(v141) = 0;
  v149 = &unk_49E5698;
  v129 = (__m128i)(unsigned __int64)v21;
  v144 = &v157;
  v130 = 0u;
  v131 = 0u;
  v132 = 0u;
  LOWORD(v133) = 257;
  v145 = 0;
  LODWORD(v146) = 0;
  WORD2(v146) = 512;
  BYTE6(v146) = 7;
  v147 = 0;
  v148 = 0;
  v139 = 0;
  v140 = 0;
  v150 = &unk_49D94D0;
  v22 = _mm_loadu_si128(&v129);
  v151 = v21;
  v23 = _mm_loadu_si128(&v130);
  v24 = _mm_loadu_si128(&v131);
  v152 = v22;
  v156 = v133;
  v25 = _mm_loadu_si128(&v132);
  v153 = v23;
  v154 = v24;
  v157 = &unk_49DA0B0;
  v155 = v25;
  nullsub_63();
  nullsub_63();
  v26 = a1;
  sub_D5F1F0((__int64)&v134, a1);
  v27 = v158;
  v115 = 0;
  v114 = &v158[14 * (unsigned int)v159];
  if ( v158 != v114 )
  {
    do
    {
      while ( 1 )
      {
        v121[0] = "bound0";
        v124 = 259;
        v29 = v27[12];
        v30 = v27[2];
        v31 = (_QWORD *)(*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)v143 + 56LL))(
                          v143,
                          36,
                          v30,
                          v29);
        if ( !v31 )
        {
          v129.m128i_i16[4] = 257;
          v31 = sub_BD2C40(72, unk_3F10FD0);
          if ( v31 )
          {
            v64 = *(_QWORD ***)(v30 + 8);
            v65 = *((unsigned __int8 *)v64 + 8);
            if ( (unsigned int)(v65 - 17) > 1 )
            {
              v67 = sub_BCB2A0(*v64);
            }
            else
            {
              BYTE4(v117) = (_BYTE)v65 == 18;
              LODWORD(v117) = *((_DWORD *)v64 + 8);
              v66 = (__int64 *)sub_BCB2A0(*v64);
              v67 = sub_BCE1B0(v66, v117);
            }
            sub_B523C0((__int64)v31, v67, 53, 36, v30, v29, (__int64)v127, 0, 0, 0);
          }
          (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
            v144,
            v31,
            v121,
            v140,
            v141);
          v68 = v134;
          v69 = v134 + 16LL * (unsigned int)v135;
          if ( v134 != v69 )
          {
            do
            {
              v70 = *(_QWORD *)(v68 + 8);
              v71 = *(_DWORD *)v68;
              v68 += 16LL;
              sub_B99FD0((__int64)v31, v71, v70);
            }
            while ( v69 != v68 );
          }
        }
        v121[0] = "bound1";
        v124 = 259;
        v32 = v27[5];
        v33 = v27[9];
        v34 = (_QWORD *)(*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)v143 + 56LL))(
                          v143,
                          36,
                          v33,
                          v32);
        if ( !v34 )
        {
          v129.m128i_i16[4] = 257;
          v34 = sub_BD2C40(72, unk_3F10FD0);
          if ( v34 )
          {
            v56 = *(_QWORD ***)(v33 + 8);
            v57 = *((unsigned __int8 *)v56 + 8);
            if ( (unsigned int)(v57 - 17) > 1 )
            {
              v59 = sub_BCB2A0(*v56);
            }
            else
            {
              BYTE4(v118) = (_BYTE)v57 == 18;
              LODWORD(v118) = *((_DWORD *)v56 + 8);
              v58 = (__int64 *)sub_BCB2A0(*v56);
              v59 = sub_BCE1B0(v58, v118);
            }
            sub_B523C0((__int64)v34, v59, 53, 36, v33, v32, (__int64)v127, 0, 0, 0);
          }
          (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
            v144,
            v34,
            v121,
            v140,
            v141);
          v60 = v134;
          v61 = v134 + 16LL * (unsigned int)v135;
          if ( v134 != v61 )
          {
            do
            {
              v62 = *(_QWORD *)(v60 + 8);
              v63 = *(_DWORD *)v60;
              v60 += 16LL;
              sub_B99FD0((__int64)v34, v63, v62);
            }
            while ( v61 != v60 );
          }
        }
        v26 = 28;
        v121[0] = "found.conflict";
        v124 = 259;
        v35 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, _QWORD *, _QWORD *))(*(_QWORD *)v143 + 16LL))(
                v143,
                28,
                v31,
                v34);
        if ( !v35 )
        {
          v129.m128i_i16[4] = 257;
          v35 = sub_B504D0(28, (__int64)v31, (__int64)v34, (__int64)v127, 0, 0);
          v26 = v35;
          (*((void (__fastcall **)(void **, __int64, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
            v144,
            v35,
            v121,
            v140,
            v141);
          v53 = (unsigned int *)v134;
          v54 = (unsigned int *)(v134 + 16LL * (unsigned int)v135);
          if ( (unsigned int *)v134 != v54 )
          {
            do
            {
              v55 = *((_QWORD *)v53 + 1);
              v26 = *v53;
              v53 += 4;
              sub_B99FD0(v35, v26, v55);
            }
            while ( v54 != v53 );
          }
        }
        v36 = v27[6];
        if ( v36 )
        {
          v121[0] = "stride.check";
          v124 = 259;
          v37 = sub_AD64C0(*(_QWORD *)(v36 + 8), 0, 0);
          v38 = v27[6];
          v39 = v37;
          v40 = (_QWORD *)(*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)v143 + 56LL))(
                            v143,
                            40,
                            v38,
                            v37);
          if ( !v40 )
          {
            v129.m128i_i16[4] = 257;
            v40 = sub_BD2C40(72, unk_3F10FD0);
            if ( v40 )
            {
              v89 = *(_QWORD ***)(v38 + 8);
              v90 = *((unsigned __int8 *)v89 + 8);
              if ( (unsigned int)(v90 - 17) > 1 )
              {
                v92 = sub_BCB2A0(*v89);
              }
              else
              {
                BYTE4(v119) = (_BYTE)v90 == 18;
                LODWORD(v119) = *((_DWORD *)v89 + 8);
                v91 = (__int64 *)sub_BCB2A0(*v89);
                v92 = sub_BCE1B0(v91, v119);
              }
              sub_B523C0((__int64)v40, v92, 53, 40, v38, v39, (__int64)v127, 0, 0, 0);
            }
            (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
              v144,
              v40,
              v121,
              v140,
              v141);
            v93 = v134;
            v94 = v134 + 16LL * (unsigned int)v135;
            if ( v134 != v94 )
            {
              do
              {
                v95 = *(_QWORD *)(v93 + 8);
                v96 = *(_DWORD *)v93;
                v93 += 16LL;
                sub_B99FD0((__int64)v40, v96, v95);
              }
              while ( v94 != v93 );
            }
          }
          v124 = 257;
          v26 = 29;
          v41 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, _QWORD *))(*(_QWORD *)v143 + 16LL))(
                  v143,
                  29,
                  v35,
                  v40);
          if ( !v41 )
          {
            v129.m128i_i16[4] = 257;
            v41 = sub_B504D0(29, v35, (__int64)v40, (__int64)v127, 0, 0);
            v26 = v41;
            (*((void (__fastcall **)(void **, __int64, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
              v144,
              v41,
              v121,
              v140,
              v141);
            v86 = (unsigned int *)v134;
            v87 = (unsigned int *)(v134 + 16LL * (unsigned int)v135);
            if ( (unsigned int *)v134 != v87 )
            {
              do
              {
                v88 = *((_QWORD *)v86 + 1);
                v26 = *v86;
                v86 += 4;
                sub_B99FD0(v41, v26, v88);
              }
              while ( v87 != v86 );
            }
          }
          v35 = v41;
        }
        v42 = v27[13];
        if ( v42 )
        {
          v121[0] = "stride.check";
          v124 = 259;
          v43 = sub_AD64C0(*(_QWORD *)(v42 + 8), 0, 0);
          v111 = v27[13];
          v44 = (_QWORD *)(*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)v143 + 56LL))(
                            v143,
                            40,
                            v111,
                            v43);
          if ( !v44 )
          {
            v129.m128i_i16[4] = 257;
            v44 = sub_BD2C40(72, unk_3F10FD0);
            if ( v44 )
            {
              v78 = *(_QWORD ***)(v111 + 8);
              v79 = *((unsigned __int8 *)v78 + 8);
              if ( (unsigned int)(v79 - 17) > 1 )
              {
                v81 = sub_BCB2A0(*v78);
              }
              else
              {
                BYTE4(v120) = (_BYTE)v79 == 18;
                LODWORD(v120) = *((_DWORD *)v78 + 8);
                v80 = (__int64 *)sub_BCB2A0(*v78);
                v81 = sub_BCE1B0(v80, v120);
              }
              sub_B523C0((__int64)v44, v81, 53, 40, v111, v43, (__int64)v127, 0, 0, 0);
            }
            (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
              v144,
              v44,
              v121,
              v140,
              v141);
            v82 = v134 + 16LL * (unsigned int)v135;
            for ( i = v134; v82 != i; i += 16LL )
            {
              v84 = *(_QWORD *)(i + 8);
              v85 = *(_DWORD *)i;
              sub_B99FD0((__int64)v44, v85, v84);
            }
          }
          v26 = 29;
          v124 = 257;
          v45 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, _QWORD *))(*(_QWORD *)v143 + 16LL))(
                  v143,
                  29,
                  v35,
                  v44);
          if ( !v45 )
          {
            v129.m128i_i16[4] = 257;
            v112 = sub_B504D0(29, v35, (__int64)v44, (__int64)v127, 0, 0);
            v26 = v112;
            (*((void (__fastcall **)(void **, __int64, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
              v144,
              v112,
              v121,
              v140,
              v141);
            v75 = (unsigned int *)v134;
            v45 = v112;
            v76 = (unsigned int *)(v134 + 16LL * (unsigned int)v135);
            if ( (unsigned int *)v134 != v76 )
            {
              do
              {
                v77 = *((_QWORD *)v75 + 1);
                v26 = *v75;
                v75 += 4;
                sub_B99FD0(v112, v26, v77);
              }
              while ( v76 != v75 );
              v45 = v112;
            }
          }
          v35 = v45;
        }
        if ( v115 )
          break;
        v115 = v35;
        v27 += 14;
        if ( v114 == v27 )
          goto LABEL_73;
      }
      v26 = 29;
      v121[0] = "conflict.rdx";
      v124 = 259;
      v28 = (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, __int64))(*(_QWORD *)v143 + 16LL))(
              v143,
              29,
              v115,
              v35);
      if ( !v28 )
      {
        v129.m128i_i16[4] = 257;
        v116 = sub_B504D0(29, v115, v35, (__int64)v127, 0, 0);
        v26 = v116;
        (*((void (__fastcall **)(void **, __int64, _QWORD *, unsigned __int64, __int64))*v144 + 2))(
          v144,
          v116,
          v121,
          v140,
          v141);
        v72 = (unsigned int *)v134;
        v28 = v116;
        v73 = (unsigned int *)(v134 + 16LL * (unsigned int)v135);
        if ( (unsigned int *)v134 != v73 )
        {
          do
          {
            v74 = *((_QWORD *)v72 + 1);
            v26 = *v72;
            v72 += 4;
            sub_B99FD0(v116, v26, v74);
          }
          while ( v73 != v72 );
          v28 = v116;
        }
      }
      v115 = v28;
      v27 += 14;
    }
    while ( v114 != v27 );
  }
LABEL_73:
  nullsub_61();
  v149 = &unk_49E5698;
  v150 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( (__int64 *)v134 != &v136 )
    _libc_free(v134, v26);
  v46 = v158;
  v47 = &v158[14 * (unsigned int)v159];
  if ( v158 != v47 )
  {
    do
    {
      v48 = *(v47 - 2);
      v47 -= 14;
      if ( v48 != -4096 && v48 != 0 && v48 != -8192 )
        sub_BD60C0(v47 + 10);
      v49 = v47[9];
      if ( v49 != 0 && v49 != -4096 && v49 != -8192 )
        sub_BD60C0(v47 + 7);
      v50 = v47[5];
      if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
        sub_BD60C0(v47 + 3);
      v51 = v47[2];
      if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
        sub_BD60C0(v47);
    }
    while ( v46 != v47 );
    v47 = v158;
  }
  if ( v47 != (unsigned __int64 *)v160 )
    _libc_free(v47, v26);
  return v115;
}
