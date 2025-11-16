// Function: sub_3380DB0
// Address: 0x3380db0
//
__int64 __fastcall sub_3380DB0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _DWORD **a6,
        int a7,
        unsigned __int8 a8)
{
  _DWORD *v12; // rsi
  unsigned __int64 v13; // r9
  __int64 *v14; // r13
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rdi
  int v18; // edx
  unsigned __int64 v19; // rsi
  unsigned __int8 *v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r15
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rsi
  int v27; // eax
  unsigned __int64 v28; // rcx
  unsigned int v29; // r10d
  int v30; // r8d
  _BYTE *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  int v38; // esi
  _DWORD *v39; // rdx
  unsigned __int64 v40; // rdx
  const __m128i *v41; // r15
  __m128i *v42; // rax
  unsigned __int8 *v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // rax
  unsigned int v48; // edi
  __int64 v49; // rsi
  unsigned __int8 *v50; // r9
  __int64 v51; // rax
  int v52; // r15d
  __int64 v53; // rax
  _BYTE *v54; // rdi
  _DWORD *v55; // rcx
  _DWORD *v56; // rax
  int v57; // edx
  __int64 v58; // rax
  char v59; // al
  __int64 v60; // r8
  unsigned __int8 *v61; // rdi
  __int64 v62; // rdx
  unsigned __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rdi
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // rcx
  unsigned __int8 *v69; // r8
  int v70; // edi
  _DWORD *v71; // rcx
  unsigned int v72; // ebx
  __int64 v73; // rax
  __int64 v74; // rdx
  char *v75; // r15
  __int64 v76; // r13
  unsigned int v77; // r15d
  __int64 v78; // r14
  int v79; // eax
  unsigned int v80; // r12d
  unsigned int v81; // edx
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  _QWORD *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r8
  int v88; // eax
  _DWORD *v89; // rcx
  unsigned __int64 v90; // r8
  const __m128i *v91; // r15
  __int64 v92; // rax
  __m128i *v93; // rax
  char *v94; // r15
  unsigned __int64 v95; // r8
  const __m128i *v96; // r15
  __m128i *v97; // rax
  int v98; // esi
  int v99; // r10d
  int v100; // ecx
  int v101; // r10d
  const __m128i *v102; // rax
  __m128i *v103; // rdx
  const __m128i *v104; // rax
  __m128i *v105; // rdx
  __int64 v106; // rax
  char *v107; // r15
  char *v108; // r15
  unsigned __int64 v109; // [rsp-10h] [rbp-290h]
  __int64 v110; // [rsp+8h] [rbp-278h]
  __int64 v111; // [rsp+10h] [rbp-270h]
  int v112; // [rsp+18h] [rbp-268h]
  __int64 v113; // [rsp+18h] [rbp-268h]
  __int64 v114; // [rsp+18h] [rbp-268h]
  __int64 v115; // [rsp+18h] [rbp-268h]
  __int64 v116; // [rsp+30h] [rbp-250h]
  __int64 v117; // [rsp+40h] [rbp-240h]
  unsigned __int8 v120; // [rsp+68h] [rbp-218h]
  char *v121; // [rsp+68h] [rbp-218h]
  unsigned __int8 *v122; // [rsp+78h] [rbp-208h] BYREF
  __m128i v123; // [rsp+80h] [rbp-200h] BYREF
  __int64 v124; // [rsp+90h] [rbp-1F0h]
  __int64 v125; // [rsp+98h] [rbp-1E8h]
  unsigned __int64 v126; // [rsp+A0h] [rbp-1E0h] BYREF
  __int64 v127; // [rsp+A8h] [rbp-1D8h]
  _BYTE v128[48]; // [rsp+B0h] [rbp-1D0h] BYREF
  _BYTE *v129; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 v130; // [rsp+E8h] [rbp-198h]
  _BYTE v131[48]; // [rsp+F0h] [rbp-190h] BYREF
  char *v132; // [rsp+120h] [rbp-160h] BYREF
  __int64 v133; // [rsp+128h] [rbp-158h]
  char v134; // [rsp+130h] [rbp-150h] BYREF
  _DWORD *v135; // [rsp+190h] [rbp-F0h] BYREF
  unsigned __int8 *v136; // [rsp+198h] [rbp-E8h]
  _DWORD v137[16]; // [rsp+1A0h] [rbp-E0h] BYREF
  char *v138; // [rsp+1E0h] [rbp-A0h]
  char v139; // [rsp+1F8h] [rbp-88h] BYREF
  _BYTE *v140; // [rsp+200h] [rbp-80h]
  _BYTE v141[16]; // [rsp+210h] [rbp-70h] BYREF
  _BYTE *v142; // [rsp+220h] [rbp-60h]
  int v143; // [rsp+228h] [rbp-58h]
  _BYTE v144[80]; // [rsp+230h] [rbp-50h] BYREF

  if ( !a3 )
    return 1;
  v12 = *a6;
  v135 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v135, (__int64)v12, 1);
  v120 = sub_3374760(a1, a2, a3, a4, a5, (int)&v135);
  if ( v135 )
    sub_B91220((__int64)&v135, (__int64)v135);
  if ( v120 )
  {
    return 1;
  }
  else
  {
    v126 = (unsigned __int64)v128;
    v127 = 0x200000000LL;
    v129 = v131;
    v130 = 0x600000000LL;
    if ( &a2[a3] != a2 )
    {
      v116 = a5;
      v14 = &a2[a3];
      v117 = a1 + 8;
      while ( 1 )
      {
        while ( 1 )
        {
          v20 = (unsigned __int8 *)*a2;
          v122 = v20;
          v21 = *v20;
          if ( (unsigned __int8)v21 > 0x14u )
            break;
          v15 = 1454080;
          if ( _bittest64(&v15, v21) )
          {
            v16 = (unsigned int)v127;
            v17 = v126;
            v18 = v127;
            v19 = v126 + 24LL * (unsigned int)v127;
            if ( (unsigned int)v127 >= (unsigned __int64)HIDWORD(v127) )
            {
              v40 = (unsigned int)v127 + 1LL;
              v136 = v20;
              v41 = (const __m128i *)&v135;
              LODWORD(v135) = 1;
              if ( HIDWORD(v127) < v40 )
              {
                if ( v126 > (unsigned __int64)&v135 || v19 <= (unsigned __int64)&v135 )
                {
                  sub_C8D5F0((__int64)&v126, v128, v40, 0x18u, HIDWORD(v127), v13);
                  v17 = v126;
                  v16 = (unsigned int)v127;
                }
                else
                {
                  v94 = (char *)&v135 - v126;
                  sub_C8D5F0((__int64)&v126, v128, v40, 0x18u, HIDWORD(v127), v13);
                  v17 = v126;
                  v16 = (unsigned int)v127;
                  v41 = (const __m128i *)&v94[v126];
                }
              }
              v42 = (__m128i *)(v17 + 24 * v16);
              *v42 = _mm_loadu_si128(v41);
              v42[1].m128i_i64[0] = v41[1].m128i_i64[0];
              LODWORD(v127) = v127 + 1;
            }
            else
            {
              if ( v19 )
              {
                *(_DWORD *)v19 = 1;
                *(_QWORD *)(v19 + 8) = v20;
                v18 = v127;
              }
              LODWORD(v127) = v18 + 1;
            }
            goto LABEL_14;
          }
          if ( (_BYTE)v21 == 5 && *((_WORD *)v20 + 1) == 48 )
          {
            v60 = v126;
            v61 = *(unsigned __int8 **)&v20[-32 * (*((_DWORD *)v20 + 1) & 0x7FFFFFF)];
            v62 = (unsigned int)v127;
            v27 = v127;
            v63 = v126 + 24LL * (unsigned int)v127;
            if ( (unsigned int)v127 < (unsigned __int64)HIDWORD(v127) )
            {
              if ( v63 )
              {
                *(_DWORD *)v63 = 1;
                *(_QWORD *)(v63 + 8) = v61;
                v27 = v127;
              }
              goto LABEL_24;
            }
            v13 = (unsigned int)v127 + 1LL;
            v136 = v61;
            v102 = (const __m128i *)&v135;
            LODWORD(v135) = 1;
            if ( HIDWORD(v127) >= v13 )
              goto LABEL_122;
            v114 = v126;
            if ( v126 <= (unsigned __int64)&v135 && v63 > (unsigned __int64)&v135 )
            {
LABEL_138:
              sub_C8D5F0((__int64)&v126, v128, v62 + 1, 0x18u, v60, v13);
              v62 = (unsigned int)v127;
              v60 = v126;
              v102 = (const __m128i *)((char *)&v135 + v126 - v114);
LABEL_122:
              v103 = (__m128i *)(v60 + 24 * v62);
              *v103 = _mm_loadu_si128(v102);
              v103[1].m128i_i64[0] = v102[1].m128i_i64[0];
              LODWORD(v127) = v127 + 1;
              goto LABEL_14;
            }
LABEL_139:
            sub_C8D5F0((__int64)&v126, v128, v62 + 1, 0x18u, v60, v13);
            v62 = (unsigned int)v127;
            v60 = v126;
            v102 = (const __m128i *)&v135;
            goto LABEL_122;
          }
LABEL_17:
          v123 = _mm_loadu_si128((const __m128i *)sub_337DC20(v117, (__int64 *)&v122));
          if ( !v123.m128i_i64[0] )
          {
            v43 = v122;
            if ( *v122 != 22 )
            {
              v44 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
LABEL_42:
              v45 = *(_QWORD *)(a1 + 960);
              v46 = *(_QWORD *)(v45 + 128);
              v47 = *(unsigned int *)(v45 + 144);
              if ( !(_DWORD)v47 )
                goto LABEL_27;
              v48 = (v47 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v49 = v46 + 16LL * v48;
              v50 = *(unsigned __int8 **)v49;
              if ( *(unsigned __int8 **)v49 != v43 )
              {
                v98 = 1;
                while ( v50 != (unsigned __int8 *)-4096LL )
                {
                  v99 = v98 + 1;
                  v48 = (v47 - 1) & (v98 + v48);
                  v49 = v46 + 16LL * v48;
                  v50 = *(unsigned __int8 **)v49;
                  if ( v43 == *(unsigned __int8 **)v49 )
                    goto LABEL_44;
                  v98 = v99;
                }
                goto LABEL_27;
              }
LABEL_44:
              if ( v49 == v46 + 16 * v47 )
                goto LABEL_27;
              v51 = *(_QWORD *)(a1 + 864);
              v52 = *(_DWORD *)(v49 + 8);
              BYTE4(v132) = 0;
              v110 = v44;
              v111 = *((_QWORD *)v43 + 1);
              v112 = sub_2E79000(*(__int64 **)(v51 + 40));
              v53 = sub_BD5C60((__int64)v122);
              sub_336FEE0((__int64)&v135, v53, v110, v112, v52, v111, (__int64)v132);
              v54 = v142;
              v55 = &v142[4 * v143];
              if ( v142 != (_BYTE *)v55 )
              {
                v56 = v142;
                v57 = 0;
                do
                  v57 += *v56++;
                while ( v55 != v56 );
                if ( v57 > 1 )
                {
                  if ( !a8 )
                  {
                    v72 = 0;
                    v73 = sub_AF3FE0(a4);
                    v133 = v74;
                    v132 = (char *)v73;
                    if ( (_BYTE)v74 )
                      v72 = (unsigned int)v132;
                    sub_AF47B0((__int64)&v132, *(unsigned __int64 **)(v116 + 16), *(unsigned __int64 **)(v116 + 24));
                    if ( v134 )
                      v72 = (unsigned int)v132;
                    sub_3372B70((__int64)&v132, (__int64)&v135);
                    v75 = v132;
                    v121 = &v132[24 * (unsigned int)v133];
                    if ( v121 != v132 && v72 )
                    {
                      v76 = a1;
                      v77 = 0;
                      v78 = (__int64)v132;
                      do
                      {
                        v79 = sub_CA1930((_BYTE *)(v78 + 8));
                        v80 = v77 + v79;
                        v81 = v72 - v77;
                        if ( v77 + v79 <= v72 )
                          v81 = v79;
                        v82 = sub_B0E470(v116, v77, v81);
                        v125 = v83;
                        v124 = v82;
                        if ( (_BYTE)v83 )
                        {
                          v77 = v80;
                          v84 = sub_33E5F30(*(_QWORD *)(v76 + 864), a4, v124, *(_DWORD *)v78, 0, (_DWORD)a6, a7);
                          sub_33F99B0(*(_QWORD *)(v76 + 864), v84, 0);
                        }
                        v78 += 24;
                      }
                      while ( (char *)v78 != v121 && v77 < v72 );
                      v75 = v132;
                    }
                    if ( v75 != &v134 )
                      _libc_free((unsigned __int64)v75);
                    v120 = 1;
                    v54 = v142;
                  }
                  if ( v54 != v144 )
                    _libc_free((unsigned __int64)v54);
                  if ( v140 != v141 )
                    _libc_free((unsigned __int64)v140);
                  if ( v138 != &v139 )
                    _libc_free((unsigned __int64)v138);
                  if ( v135 != v137 )
                    _libc_free((unsigned __int64)v135);
                  goto LABEL_27;
                }
              }
              v86 = (unsigned int)v127;
              v87 = v126;
              v88 = v127;
              v89 = (_DWORD *)(v126 + 24LL * (unsigned int)v127);
              if ( (unsigned int)v127 >= (unsigned __int64)HIDWORD(v127) )
              {
                v13 = (unsigned int)v127 + 1LL;
                LODWORD(v133) = v52;
                v104 = (const __m128i *)&v132;
                LODWORD(v132) = 3;
                if ( HIDWORD(v127) < v13 )
                {
                  v115 = v126;
                  if ( v126 > (unsigned __int64)&v132 || v89 <= (_DWORD *)&v132 )
                  {
                    sub_C8D5F0((__int64)&v126, v128, (unsigned int)v127 + 1LL, 0x18u, v126, v13);
                    v86 = (unsigned int)v127;
                    v104 = (const __m128i *)&v132;
                    v87 = v126;
                  }
                  else
                  {
                    sub_C8D5F0((__int64)&v126, v128, (unsigned int)v127 + 1LL, 0x18u, v126, v13);
                    v86 = (unsigned int)v127;
                    v87 = v126;
                    v104 = (const __m128i *)((char *)&v132 + v126 - v115);
                  }
                }
                v105 = (__m128i *)(v87 + 24 * v86);
                *v105 = _mm_loadu_si128(v104);
                v106 = v104[1].m128i_i64[0];
                LODWORD(v127) = v127 + 1;
                v105[1].m128i_i64[0] = v106;
                v54 = v142;
              }
              else
              {
                if ( v89 )
                {
                  *v89 = 3;
                  v89[2] = v52;
                  v88 = v127;
                  v54 = v142;
                }
                LODWORD(v127) = v88 + 1;
              }
              if ( v54 != v144 )
                _libc_free((unsigned __int64)v54);
              if ( v140 != v141 )
                _libc_free((unsigned __int64)v140);
              if ( v138 != &v139 )
                _libc_free((unsigned __int64)v138);
              if ( v135 != v137 )
                _libc_free((unsigned __int64)v135);
              goto LABEL_14;
            }
            v85 = sub_337DC20(a1 + 40, (__int64 *)&v122);
            v123.m128i_i64[0] = *v85;
            v123.m128i_i32[2] = *((_DWORD *)v85 + 2);
            if ( !v123.m128i_i64[0] )
            {
              v43 = v122;
              v44 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
              if ( *v122 == 22 && *(_WORD *)(a4 + 20) )
              {
                v113 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
                if ( !sub_B10D40((__int64)a6) )
                  goto LABEL_27;
                v43 = v122;
                v44 = v113;
              }
              goto LABEL_42;
            }
          }
          if ( !a8 )
          {
            v58 = sub_B10CD0((__int64)a6);
            v59 = sub_337F9F0(a1, (__int64)v122, a4, (_QWORD *)v116, v58, 0, v123.m128i_i64);
            v13 = v109;
            if ( v59 )
            {
              v120 = v59;
              goto LABEL_27;
            }
          }
          v23 = v123.m128i_i64[0];
          v24 = *(_DWORD *)(v123.m128i_i64[0] + 24);
          if ( v24 == 39 || v24 == 15 )
          {
            v35 = (unsigned int)v130;
            v36 = (unsigned int)v130 + 1LL;
            if ( v36 > HIDWORD(v130) )
            {
              sub_C8D5F0((__int64)&v129, v131, v36, 8u, v22, v13);
              v35 = (unsigned int)v130;
            }
            *(_QWORD *)&v129[8 * v35] = v23;
            v37 = (unsigned int)v127;
            LODWORD(v130) = v130 + 1;
            v38 = *(_DWORD *)(v23 + 96);
            v27 = v127;
            if ( (unsigned int)v127 < (unsigned __int64)HIDWORD(v127) )
            {
              v39 = (_DWORD *)(v126 + 24LL * (unsigned int)v127);
              if ( v39 )
              {
                *v39 = 2;
                v39[2] = v38;
                v27 = v127;
              }
              goto LABEL_24;
            }
            v90 = (unsigned int)v127 + 1LL;
            LODWORD(v136) = *(_DWORD *)(v23 + 96);
            v91 = (const __m128i *)&v135;
            v92 = v126;
            LODWORD(v135) = 2;
            if ( HIDWORD(v127) < v90 )
            {
              if ( v126 > (unsigned __int64)&v135 || (unsigned __int64)&v135 >= v126 + 24LL * (unsigned int)v127 )
              {
                v91 = (const __m128i *)&v135;
                sub_C8D5F0((__int64)&v126, v128, v90, 0x18u, v90, v13);
                v92 = v126;
                v37 = (unsigned int)v127;
              }
              else
              {
                v107 = (char *)&v135 - v126;
                sub_C8D5F0((__int64)&v126, v128, v90, 0x18u, v90, v13);
                v92 = v126;
                v37 = (unsigned int)v127;
                v91 = (const __m128i *)&v107[v126];
              }
            }
            v93 = (__m128i *)(v92 + 24 * v37);
            *v93 = _mm_loadu_si128(v91);
            v93[1].m128i_i64[0] = v91[1].m128i_i64[0];
            LODWORD(v127) = v127 + 1;
          }
          else
          {
            v25 = (unsigned int)v127;
            v26 = v126;
            v27 = v127;
            v28 = v126 + 24LL * (unsigned int)v127;
            if ( (unsigned int)v127 < (unsigned __int64)HIDWORD(v127) )
            {
              if ( v28 )
              {
                *(_DWORD *)(v28 + 16) = v123.m128i_i32[2];
                *(_DWORD *)v28 = 0;
                *(_QWORD *)(v28 + 8) = v23;
                v27 = v127;
              }
              goto LABEL_24;
            }
            v137[0] = v123.m128i_i32[2];
            v95 = (unsigned int)v127 + 1LL;
            v136 = (unsigned __int8 *)v123.m128i_i64[0];
            v96 = (const __m128i *)&v135;
            LODWORD(v135) = 0;
            if ( HIDWORD(v127) < v95 )
            {
              if ( v126 > (unsigned __int64)&v135 || v28 <= (unsigned __int64)&v135 )
              {
                v96 = (const __m128i *)&v135;
                sub_C8D5F0((__int64)&v126, v128, (unsigned int)v127 + 1LL, 0x18u, v95, v13);
                v26 = v126;
                v25 = (unsigned int)v127;
              }
              else
              {
                v108 = (char *)&v135 - v126;
                sub_C8D5F0((__int64)&v126, v128, (unsigned int)v127 + 1LL, 0x18u, v95, v13);
                v26 = v126;
                v25 = (unsigned int)v127;
                v96 = (const __m128i *)&v108[v126];
              }
            }
            v97 = (__m128i *)(v26 + 24 * v25);
            *v97 = _mm_loadu_si128(v96);
            v97[1].m128i_i64[0] = v96[1].m128i_i64[0];
            LODWORD(v127) = v127 + 1;
          }
LABEL_14:
          if ( v14 == ++a2 )
            goto LABEL_25;
        }
        if ( (_BYTE)v21 != 60 )
          goto LABEL_17;
        v64 = *(_QWORD *)(a1 + 960);
        v65 = *(_QWORD *)(v64 + 256);
        v66 = *(unsigned int *)(v64 + 272);
        if ( !(_DWORD)v66 )
          goto LABEL_17;
        v13 = (unsigned int)(v66 - 1);
        v67 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v68 = v65 + 16LL * v67;
        v69 = *(unsigned __int8 **)v68;
        if ( v20 != *(unsigned __int8 **)v68 )
        {
          v100 = 1;
          while ( v69 != (unsigned __int8 *)-4096LL )
          {
            v101 = v100 + 1;
            v67 = v13 & (v100 + v67);
            v68 = v65 + 16LL * v67;
            v69 = *(unsigned __int8 **)v68;
            if ( v20 == *(unsigned __int8 **)v68 )
              goto LABEL_67;
            v100 = v101;
          }
          goto LABEL_17;
        }
LABEL_67:
        if ( v68 == v65 + 16 * v66 )
          goto LABEL_17;
        v62 = (unsigned int)v127;
        v60 = v126;
        v70 = *(_DWORD *)(v68 + 8);
        v27 = v127;
        v71 = (_DWORD *)(v126 + 24LL * (unsigned int)v127);
        if ( (unsigned int)v127 >= (unsigned __int64)HIDWORD(v127) )
        {
          v13 = (unsigned int)v127 + 1LL;
          LODWORD(v136) = v70;
          v102 = (const __m128i *)&v135;
          LODWORD(v135) = 2;
          if ( HIDWORD(v127) >= v13 )
            goto LABEL_122;
          v114 = v126;
          if ( v126 <= (unsigned __int64)&v135 && v71 > (_DWORD *)&v135 )
            goto LABEL_138;
          goto LABEL_139;
        }
        if ( v71 )
        {
          *v71 = 2;
          v71[2] = v70;
          v27 = v127;
        }
LABEL_24:
        ++a2;
        LODWORD(v127) = v27 + 1;
        if ( v14 == a2 )
        {
LABEL_25:
          LODWORD(a5) = v116;
          v29 = v126;
          v30 = v127;
          v31 = v129;
          v32 = (unsigned int)v130;
          goto LABEL_26;
        }
      }
    }
    v31 = v131;
    v29 = (unsigned int)v128;
    v32 = 0;
    v30 = 0;
LABEL_26:
    v33 = sub_33E4BC0(*(_QWORD *)(a1 + 864), a4, a5, v29, v30, 0, (__int64)v31, v32, (__int64)a6, a7, a8);
    sub_33F99B0(*(_QWORD *)(a1 + 864), v33, 0);
    v120 = 1;
LABEL_27:
    if ( v129 != v131 )
      _libc_free((unsigned __int64)v129);
    if ( (_BYTE *)v126 != v128 )
      _libc_free(v126);
  }
  return v120;
}
