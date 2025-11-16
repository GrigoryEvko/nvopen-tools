// Function: sub_10468A0
// Address: 0x10468a0
//
__int64 __fastcall sub_10468A0(__int64 *a1)
{
  __int64 *v1; // r13
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  const __m128i *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  const __m128i *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  const __m128i *v21; // rsi
  const __m128i *v22; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rcx
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  const __m128i *v30; // rcx
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __m128i *v33; // rdi
  __m128i *v34; // rdx
  const __m128i *v35; // rax
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // rsi
  unsigned int v42; // ecx
  __int64 *v43; // rax
  __int64 v44; // r12
  __int64 i; // rbx
  __int64 v46; // rdx
  unsigned int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // rax
  int v51; // eax
  _BYTE *v52; // rsi
  int v53; // edx
  __int8 *v54; // r13
  __int64 v55; // rax
  __int64 v56; // rsi
  _QWORD *v57; // rax
  unsigned __int64 v58; // rax
  __int64 v59; // r10
  unsigned __int64 v60; // r15
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  __int64 v63; // r15
  __int64 *v64; // rax
  __int64 v65; // rcx
  __int64 *v66; // rdx
  __int64 v67; // rbx
  __int64 *v68; // rax
  __int64 v69; // rdi
  int v70; // esi
  __int64 v71; // rdi
  int v72; // eax
  __m128i *v73; // r11
  __int64 v74; // rax
  char v75; // dl
  __m128i *v76; // rdx
  char v77; // cl
  __int64 v78; // rsi
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // r12
  __int64 v82; // rdi
  __int64 v83; // rsi
  __int64 result; // rax
  unsigned __int64 v85; // r12
  _BYTE *v86; // rdi
  _QWORD **v87; // rcx
  unsigned __int8 *v88; // rdx
  __m128i v89; // xmm1
  __m128i v90; // xmm2
  __int64 v91; // rdx
  __int64 j; // rax
  int v93; // ecx
  __int64 v94; // rdx
  __int64 v95; // rdx
  unsigned __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rdx
  int v99; // eax
  unsigned __int8 *v100; // rdx
  int v101; // eax
  __int64 v102; // r12
  int v103; // eax
  int v104; // eax
  __int64 v105; // [rsp+28h] [rbp-428h]
  __int64 v106; // [rsp+38h] [rbp-418h]
  __int64 v107; // [rsp+48h] [rbp-408h]
  __int64 v108; // [rsp+50h] [rbp-400h]
  __int64 *v109; // [rsp+58h] [rbp-3F8h]
  __int64 v110; // [rsp+60h] [rbp-3F0h]
  __int64 *v111; // [rsp+68h] [rbp-3E8h]
  __m128i *v112; // [rsp+70h] [rbp-3E0h] BYREF
  __m128i *v113; // [rsp+78h] [rbp-3D8h] BYREF
  __int64 v114; // [rsp+80h] [rbp-3D0h] BYREF
  __int64 v115; // [rsp+88h] [rbp-3C8h]
  __int64 v116; // [rsp+90h] [rbp-3C0h]
  unsigned int v117; // [rsp+98h] [rbp-3B8h]
  __m128i v118[8]; // [rsp+A0h] [rbp-3B0h] BYREF
  __m128i v119; // [rsp+120h] [rbp-330h] BYREF
  __m128i v120; // [rsp+130h] [rbp-320h]
  __m128i v121; // [rsp+140h] [rbp-310h] BYREF
  __int64 v122; // [rsp+150h] [rbp-300h]
  __int64 v123; // [rsp+180h] [rbp-2D0h] BYREF
  __int64 v124; // [rsp+188h] [rbp-2C8h]
  unsigned __int64 v125; // [rsp+190h] [rbp-2C0h]
  __int64 v126; // [rsp+1A0h] [rbp-2B0h] BYREF
  __int64 *v127; // [rsp+1A8h] [rbp-2A8h]
  unsigned int v128; // [rsp+1B0h] [rbp-2A0h]
  unsigned int v129; // [rsp+1B4h] [rbp-29Ch]
  char v130; // [rsp+1BCh] [rbp-294h]
  _BYTE v131[64]; // [rsp+1C0h] [rbp-290h] BYREF
  __int64 v132; // [rsp+200h] [rbp-250h] BYREF
  __int64 v133; // [rsp+208h] [rbp-248h]
  unsigned __int64 v134; // [rsp+210h] [rbp-240h]
  __int64 v135; // [rsp+220h] [rbp-230h] BYREF
  __int64 v136; // [rsp+228h] [rbp-228h]
  __int64 v137; // [rsp+230h] [rbp-220h]
  __int64 v138; // [rsp+238h] [rbp-218h]
  _QWORD v139[8]; // [rsp+240h] [rbp-210h] BYREF
  unsigned __int128 v140; // [rsp+280h] [rbp-1D0h]
  __int64 v141; // [rsp+290h] [rbp-1C0h]
  _QWORD *v142; // [rsp+2A0h] [rbp-1B0h] BYREF
  __int64 v143; // [rsp+2A8h] [rbp-1A8h]
  _QWORD v144[16]; // [rsp+2B0h] [rbp-1A0h] BYREF
  __m128i v145; // [rsp+330h] [rbp-120h] BYREF
  __int64 v146; // [rsp+340h] [rbp-110h]
  __int64 v147; // [rsp+348h] [rbp-108h]
  _QWORD v148[8]; // [rsp+350h] [rbp-100h] BYREF
  const __m128i *v149; // [rsp+390h] [rbp-C0h]
  const __m128i *v150; // [rsp+398h] [rbp-B8h]
  unsigned __int64 v151; // [rsp+3A0h] [rbp-B0h]
  char v152[8]; // [rsp+3A8h] [rbp-A8h] BYREF
  __int64 v153; // [rsp+3B0h] [rbp-A0h]
  char v154; // [rsp+3C4h] [rbp-8Ch]
  _BYTE v155[64]; // [rsp+3C8h] [rbp-88h] BYREF
  const __m128i *v156; // [rsp+408h] [rbp-48h]
  const __m128i *v157; // [rsp+410h] [rbp-40h]
  __int64 v158; // [rsp+418h] [rbp-38h]

  v1 = a1;
  v142 = v144;
  v2 = *(_QWORD *)(*a1 + 128);
  v114 = 0;
  v115 = 0;
  v144[0] = v2;
  v143 = 0x1000000001LL;
  v3 = a1[3];
  v116 = 0;
  v117 = 0;
  v4 = *(_QWORD *)(v3 + 96);
  memset(v118, 0, 0x78u);
  v118[0].m128i_i64[1] = (__int64)v118[2].m128i_i64;
  v120.m128i_i64[0] = 0x100000008LL;
  v119.m128i_i64[1] = (__int64)&v121;
  v118[1].m128i_i32[0] = 8;
  v118[1].m128i_i8[12] = 1;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v120.m128i_i32[2] = 0;
  v120.m128i_i8[12] = 1;
  v121.m128i_i64[0] = v4;
  v145.m128i_i64[0] = v4;
  v119.m128i_i64[0] = 1;
  LOBYTE(v146) = 0;
  sub_103FE30((__int64)&v123, &v145);
  sub_C8CF70((__int64)&v135, v139, 8, (__int64)v118[2].m128i_i64, (__int64)v118);
  v5 = v118[6].m128i_i64[0];
  memset(&v118[6], 0, 24);
  v140 = __PAIR128__(v118[6].m128i_u64[1], v5);
  v141 = v118[7].m128i_i64[0];
  v109 = &v126;
  sub_C8CF70((__int64)&v126, v131, 8, (__int64)&v121, (__int64)&v119);
  v6 = v123;
  v123 = 0;
  v132 = v6;
  v7 = v124;
  v124 = 0;
  v133 = v7;
  v8 = v125;
  v125 = 0;
  v134 = v8;
  sub_C8CF70((__int64)&v145, v148, 8, (__int64)v131, (__int64)&v126);
  v9 = (const __m128i *)v132;
  v10 = v155;
  v132 = 0;
  v149 = v9;
  v11 = v133;
  v133 = 0;
  v150 = (const __m128i *)v11;
  v12 = v134;
  v134 = 0;
  v151 = v12;
  sub_C8CF70((__int64)v152, v155, 8, (__int64)v139, (__int64)&v135);
  v16 = (const __m128i *)*((_QWORD *)&v140 + 1);
  v156 = (const __m128i *)v140;
  v140 = 0u;
  v157 = v16;
  v17 = v141;
  v141 = 0;
  v158 = v17;
  if ( v132 )
  {
    v10 = (_BYTE *)(v134 - v132);
    j_j___libc_free_0(v132, v134 - v132);
  }
  if ( !v130 )
    _libc_free(v127, v10);
  if ( (_QWORD)v140 )
  {
    v10 = (_BYTE *)(v141 - v140);
    j_j___libc_free_0(v140, v141 - v140);
  }
  if ( !BYTE4(v138) )
    _libc_free(v136, v10);
  if ( v123 )
  {
    v10 = (_BYTE *)(v125 - v123);
    j_j___libc_free_0(v123, v125 - v123);
  }
  if ( !v120.m128i_i8[12] )
    _libc_free(v119.m128i_i64[1], v10);
  if ( v118[6].m128i_i64[0] )
  {
    v10 = (_BYTE *)(v118[7].m128i_i64[0] - v118[6].m128i_i64[0]);
    j_j___libc_free_0(v118[6].m128i_i64[0], v118[7].m128i_i64[0] - v118[6].m128i_i64[0]);
  }
  if ( !v118[1].m128i_i8[12] )
    _libc_free(v118[0].m128i_i64[1], v10);
  sub_C8CD80((__int64)&v126, (__int64)v131, (__int64)&v145, v13, v14, v15);
  v21 = v150;
  v22 = v149;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v23 = (char *)v150 - (char *)v149;
  if ( v150 == v149 )
  {
    v23 = 0;
    v25 = 0;
  }
  else
  {
    if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_191;
    v24 = sub_22077B0((char *)v150 - (char *)v149);
    v21 = v150;
    v22 = v149;
    v25 = v24;
  }
  v132 = v25;
  v133 = v25;
  v134 = v25 + v23;
  if ( v22 != v21 )
  {
    v26 = (__m128i *)v25;
    v27 = v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v27);
        v19 = v27[1].m128i_i64[0];
        v26[1].m128i_i64[0] = v19;
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (__m128i *)((char *)v26 + 24);
    }
    while ( v27 != v21 );
    v25 += 8 * ((unsigned __int64)((char *)&v27[-2].m128i_u64[1] - (char *)v22) >> 3) + 24;
  }
  v22 = (const __m128i *)&v135;
  v133 = v25;
  sub_C8CD80((__int64)&v135, (__int64)v139, (__int64)v152, v25, v19, v20);
  v30 = v157;
  v21 = v156;
  v140 = 0u;
  v141 = 0;
  v31 = (char *)v157 - (char *)v156;
  if ( v157 != v156 )
  {
    if ( v31 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v32 = sub_22077B0((char *)v157 - (char *)v156);
      v30 = v157;
      v21 = v156;
      v33 = (__m128i *)v32;
      goto LABEL_29;
    }
LABEL_191:
    sub_4261EA(v22, v21, v18);
  }
  v31 = 0;
  v33 = 0;
LABEL_29:
  *(_QWORD *)&v140 = v33;
  v34 = v33;
  *((_QWORD *)&v140 + 1) = v33;
  v141 = (__int64)v33->m128i_i64 + v31;
  if ( v21 != v30 )
  {
    v35 = v21;
    do
    {
      if ( v34 )
      {
        *v34 = _mm_loadu_si128(v35);
        v28 = v35[1].m128i_i64[0];
        v34[1].m128i_i64[0] = v28;
      }
      v35 = (const __m128i *)((char *)v35 + 24);
      v34 = (__m128i *)((char *)v34 + 24);
    }
    while ( v35 != v30 );
    v34 = (__m128i *)((char *)v33 + 8 * ((unsigned __int64)((char *)&v35[-2].m128i_u64[1] - (char *)v21) >> 3) + 24);
  }
  v36 = v133;
  v37 = v132;
  *((_QWORD *)&v140 + 1) = v34;
  v107 = 1;
  v108 = 1;
  if ( v133 - v132 == (char *)v34 - (char *)v33 )
    goto LABEL_97;
  do
  {
LABEL_36:
    v38 = *(_QWORD *)(v36 - 24);
    v39 = *(unsigned int *)(*v1 + 88);
    v40 = *(_QWORD *)v38;
    v41 = *(_QWORD *)(*v1 + 72);
    if ( (_DWORD)v39 )
    {
      v42 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v43 = (__int64 *)(v41 + 16LL * v42);
      v28 = *v43;
      if ( v40 == *v43 )
      {
LABEL_38:
        if ( v43 != (__int64 *)(v41 + 16 * v39) )
        {
          v110 = v43[1];
          if ( v110 )
          {
            v44 = v107;
            for ( i = *(_QWORD *)(v142[(unsigned int)v143 - 1] + 64LL); !(unsigned __int8)sub_B19720(v1[3], i, v40); i = v46 )
            {
              v46 = *(_QWORD *)(v142[(unsigned int)v143 - 1] + 64LL);
              if ( v46 == i )
              {
                v47 = v143 - 1;
                do
                {
                  v48 = v47;
                  LODWORD(v143) = v47--;
                  v46 = *(_QWORD *)(v142[v48 - 1] + 64LL);
                }
                while ( v46 == i );
              }
              ++v44;
            }
            v107 = v44;
            v49 = *(_QWORD *)(v110 + 8);
            if ( v49 != v110 )
            {
              v106 = v40;
              v111 = v1;
              while ( 1 )
              {
                while ( 1 )
                {
                  if ( !v49 )
                    BUG();
                  if ( *(_BYTE *)(v49 - 32) == 26 )
                    break;
                  v61 = (unsigned int)v143;
                  v62 = (unsigned int)v143 + 1LL;
                  if ( v62 > HIDWORD(v143) )
                  {
                    sub_C8D5F0((__int64)&v142, v144, v62, 8u, v28, v29);
                    v61 = (unsigned int)v143;
                  }
                  ++v108;
                  v142[v61] = v49 - 32;
                  LODWORD(v143) = v143 + 1;
                  v49 = *(_QWORD *)(v49 + 8);
                  if ( v110 == v49 )
                  {
LABEL_70:
                    v1 = v111;
                    goto LABEL_71;
                  }
                }
                v50 = *(_QWORD *)(v49 - 64);
                if ( !v50 )
                  break;
                v51 = *(_BYTE *)v50 == 27 ? *(_DWORD *)(v50 + 80) : *(_DWORD *)(v50 + 72);
                if ( *(_DWORD *)(v49 + 48) != v51 )
                  break;
LABEL_64:
                v49 = *(_QWORD *)(v49 + 8);
                if ( v110 == v49 )
                  goto LABEL_70;
              }
              v52 = *(_BYTE **)(v49 + 40);
              v118[0].m128i_i8[0] = 0;
              v53 = (unsigned __int8)*v52;
              if ( (unsigned __int8)(v53 - 34) > 0x33u )
                goto LABEL_53;
              v69 = 0x8000000000041LL;
              if ( _bittest64(&v69, (unsigned int)(v53 - 34)) )
              {
                v118[0].m128i_i8[0] = 1;
                v118[0].m128i_i64[1] = (__int64)v52;
              }
              else if ( (_BYTE)v53 != 64 )
              {
LABEL_53:
                sub_D66840(&v119, v52);
                *(__m128i *)((char *)v118 + 8) = v119;
                *(__m128i *)((char *)&v118[1] + 8) = v120;
                *(__m128i *)((char *)&v118[2] + 8) = v121;
              }
              if ( sub_103ED40((__int64)&v114, (__int64)v118, (__int64 *)&v112) )
              {
                v54 = &v112[3].m128i_i8[8];
                v55 = v112[4].m128i_i64[0];
                goto LABEL_56;
              }
              v70 = v117;
              v71 = (__int64)v112;
              ++v114;
              v72 = v116 + 1;
              v113 = v112;
              if ( 4 * ((int)v116 + 1) >= 3 * v117 )
              {
                v70 = 2 * v117;
              }
              else if ( v117 - HIDWORD(v116) - v72 > v117 >> 3 )
              {
LABEL_90:
                LODWORD(v116) = v72;
                v119.m128i_i64[0] = 0;
                v119.m128i_i64[1] = -4096;
                v120 = (__m128i)0xFFFFFFFFFFFFFFFDLL;
                v121 = 0u;
                v122 = 0;
                if ( !sub_103B1D0(v71, (__int64)&v119) )
                  --HIDWORD(v116);
                v73 = v113;
                *v113 = _mm_loadu_si128(v118);
                v54 = &v73[3].m128i_i8[8];
                v73[1] = _mm_loadu_si128(&v118[1]);
                v73[2] = _mm_loadu_si128(&v118[2]);
                v74 = v118[3].m128i_i64[0];
                v73[3].m128i_i64[1] = 0;
                v73[3].m128i_i64[0] = v74;
                v55 = 0;
                v73[4].m128i_i64[0] = 0;
                v73[4].m128i_i64[1] = 0;
                v73[5].m128i_i64[0] = 0;
                v73[5].m128i_i64[1] = 0;
                v73[6].m128i_i8[0] = 0;
LABEL_56:
                if ( v55 == v107 )
                {
                  if ( v108 != *(_QWORD *)v54 )
                    *(_QWORD *)v54 = v108;
                }
                else
                {
                  *((_QWORD *)v54 + 1) = v107;
                  v56 = *((_QWORD *)v54 + 3);
                  *(_QWORD *)v54 = v108;
                  if ( v106 != v56 && v56 && !(unsigned __int8)sub_B19720(v111[3], v56, v106) )
                  {
                    v57 = v142;
                    *((_QWORD *)v54 + 2) = 0;
                    *((_QWORD *)v54 + 3) = *(_QWORD *)(*v57 + 64LL);
                    v58 = 0;
                    goto LABEL_61;
                  }
                }
                v58 = *((_QWORD *)v54 + 2);
                if ( v54[40] )
                {
                  v60 = (unsigned int)v143 - 1LL;
LABEL_62:
                  if ( v60 - v58 <= (unsigned int)qword_4F8F988 )
                  {
                    LODWORD(v113) = qword_4F8F988;
                    v105 = v49 - 64;
                    if ( v58 < v60 )
                    {
                      v85 = v60;
                      while ( 1 )
                      {
                        v86 = (_BYTE *)v142[v85];
                        v87 = (_QWORD **)v111[2];
                        if ( *v86 == 28 )
                        {
                          v60 = v85;
                          v100 = sub_10460C0(
                                   *(_QWORD *)(v111[1] + 16),
                                   (unsigned __int8 *)(v49 - 32),
                                   (__int64 *)v111[2],
                                   &v113,
                                   0,
                                   0);
                          for ( j = v142[v85]; v100 != (unsigned __int8 *)j; j = v142[v60] )
                            --v60;
                          goto LABEL_152;
                        }
                        v88 = *(unsigned __int8 **)(v49 + 40);
                        if ( v118[0].m128i_i8[0] )
                        {
                          v119.m128i_i64[0] = 0;
                          v119.m128i_i64[1] = -1;
                          v120 = 0u;
                          v121 = 0u;
                          if ( (unsigned __int8)sub_103AFA0((__int64)v86, &v119, v88, v87) )
                            goto LABEL_140;
                        }
                        else
                        {
                          v89 = _mm_loadu_si128((const __m128i *)&v118[1].m128i_u64[1]);
                          v90 = _mm_loadu_si128((const __m128i *)&v118[2].m128i_u64[1]);
                          v119 = _mm_loadu_si128((const __m128i *)&v118[0].m128i_u64[1]);
                          v120 = v89;
                          v121 = v90;
                          if ( (unsigned __int8)sub_103AFA0((__int64)v86, &v119, v88, v87) )
                          {
LABEL_140:
                            v91 = *(_QWORD *)(v49 - 64);
                            v60 = v85;
                            j = v142[v85];
                            if ( *(_BYTE *)(v49 - 32) == 27 )
                              goto LABEL_153;
                            goto LABEL_141;
                          }
                        }
                        if ( *((_QWORD *)v54 + 2) >= --v85 )
                        {
                          v60 = v85;
                          break;
                        }
                      }
                    }
                    v96 = *((_QWORD *)v54 + 4);
                    if ( v60 >= v96 )
                    {
                      v102 = v142[v96];
                      if ( *(_BYTE *)(v49 - 32) == 27 )
                      {
                        sub_AC2B30(v105, v142[v96]);
                        if ( *(_BYTE *)v102 == 27 )
                          v103 = *(_DWORD *)(v102 + 80);
                        else
                          v103 = *(_DWORD *)(v102 + 72);
                        *(_DWORD *)(v49 + 52) = v103;
                      }
                      else
                      {
                        if ( *(_BYTE *)v102 == 27 )
                          v104 = *(_DWORD *)(v102 + 80);
                        else
                          v104 = *(_DWORD *)(v102 + 72);
                        *(_DWORD *)(v49 + 48) = v104;
                        sub_AC2B30(v105, v102);
                      }
                    }
                    else
                    {
                      j = v142[v60];
LABEL_152:
                      v91 = *(_QWORD *)(v49 - 64);
                      if ( *(_BYTE *)(v49 - 32) == 27 )
                      {
LABEL_153:
                        if ( v91 )
                        {
                          v97 = *(_QWORD *)(v49 - 56);
                          **(_QWORD **)(v49 - 48) = v97;
                          if ( v97 )
                            *(_QWORD *)(v97 + 16) = *(_QWORD *)(v49 - 48);
                        }
                        *(_QWORD *)(v49 - 64) = j;
                        if ( j )
                        {
                          v98 = *(_QWORD *)(j + 16);
                          *(_QWORD *)(v49 - 56) = v98;
                          if ( v98 )
                            *(_QWORD *)(v98 + 16) = v49 - 56;
                          *(_QWORD *)(v49 - 48) = j + 16;
                          *(_QWORD *)(j + 16) = v105;
                        }
                        if ( *(_BYTE *)j == 27 )
                          v99 = *(_DWORD *)(j + 80);
                        else
                          v99 = *(_DWORD *)(j + 72);
                        *(_DWORD *)(v49 + 52) = v99;
                      }
                      else
                      {
LABEL_141:
                        if ( *(_BYTE *)j == 27 )
                          v93 = *(_DWORD *)(j + 80);
                        else
                          v93 = *(_DWORD *)(j + 72);
                        *(_DWORD *)(v49 + 48) = v93;
                        if ( v91 )
                        {
                          v94 = *(_QWORD *)(v49 - 56);
                          **(_QWORD **)(v49 - 48) = v94;
                          if ( v94 )
                            *(_QWORD *)(v94 + 16) = *(_QWORD *)(v49 - 48);
                        }
                        *(_QWORD *)(v49 - 64) = j;
                        v95 = *(_QWORD *)(j + 16);
                        *(_QWORD *)(v49 - 56) = v95;
                        if ( v95 )
                          *(_QWORD *)(v95 + 16) = v49 - 56;
                        *(_QWORD *)(v49 - 48) = j + 16;
                        *(_QWORD *)(j + 16) = v105;
                      }
                      *((_QWORD *)v54 + 4) = v60;
                    }
                    *((_QWORD *)v54 + 2) = (unsigned int)v143 - 1LL;
                    *((_QWORD *)v54 + 3) = v106;
                  }
                  else
                  {
                    v54[40] = 0;
                  }
                  goto LABEL_64;
                }
LABEL_61:
                v59 = (unsigned int)v143;
                v54[40] = 1;
                v60 = v59 - 1;
                *((_QWORD *)v54 + 4) = v59 - 1;
                goto LABEL_62;
              }
              sub_103F130((__int64)&v114, v70);
              sub_103ED40((__int64)&v114, (__int64)v118, (__int64 *)&v113);
              v71 = (__int64)v113;
              v72 = v116 + 1;
              goto LABEL_90;
            }
LABEL_71:
            v36 = v133;
            v63 = (__int64)v109;
            v38 = *(_QWORD *)(v133 - 24);
            goto LABEL_72;
          }
        }
      }
      else
      {
        v101 = 1;
        while ( v28 != -4096 )
        {
          v29 = (unsigned int)(v101 + 1);
          v42 = (v39 - 1) & (v101 + v42);
          v43 = (__int64 *)(v41 + 16LL * v42);
          v28 = *v43;
          if ( v40 == *v43 )
            goto LABEL_38;
          v101 = v29;
        }
      }
    }
    v63 = (__int64)v109;
LABEL_72:
    while ( 2 )
    {
      if ( !*(_BYTE *)(v36 - 8) )
      {
        v64 = *(__int64 **)(v38 + 24);
        *(_BYTE *)(v36 - 8) = 1;
        *(_QWORD *)(v36 - 16) = v64;
        goto LABEL_74;
      }
      while ( 1 )
      {
        v64 = *(__int64 **)(v36 - 16);
LABEL_74:
        v65 = *(unsigned int *)(v38 + 32);
        if ( v64 == (__int64 *)(*(_QWORD *)(v38 + 24) + 8 * v65) )
          break;
        v66 = v64 + 1;
        *(_QWORD *)(v36 - 16) = v64 + 1;
        v67 = *v64;
        if ( !v130 )
          goto LABEL_93;
        v68 = v127;
        v65 = v129;
        v66 = &v127[v129];
        if ( v127 == v66 )
        {
LABEL_129:
          if ( v129 < v128 )
          {
            v109 = (__int64 *)v63;
            ++v129;
            *v66 = v67;
            ++v126;
            goto LABEL_95;
          }
LABEL_93:
          sub_C8CC70(v63, v67, (__int64)v66, v65, v28, v29);
          if ( v75 )
          {
            v109 = (__int64 *)v63;
LABEL_95:
            v119.m128i_i64[0] = v67;
            v120.m128i_i8[0] = 0;
            sub_103FE30((__int64)&v132, &v119);
            v37 = v132;
            v36 = v133;
            goto LABEL_96;
          }
        }
        else
        {
          while ( v67 != *v68 )
          {
            if ( v66 == ++v68 )
              goto LABEL_129;
          }
        }
      }
      v133 -= 24;
      v37 = v132;
      v36 = v133;
      if ( v133 != v132 )
      {
        v38 = *(_QWORD *)(v133 - 24);
        continue;
      }
      break;
    }
    v109 = (__int64 *)v63;
LABEL_96:
    v33 = (__m128i *)v140;
  }
  while ( v36 - v37 != *((_QWORD *)&v140 + 1) - (_QWORD)v140 );
LABEL_97:
  if ( v36 != v37 )
  {
    v76 = v33;
    while ( *(_QWORD *)v37 == v76->m128i_i64[0] )
    {
      v77 = *(_BYTE *)(v37 + 16);
      if ( v77 != v76[1].m128i_i8[0] || v77 && *(_QWORD *)(v37 + 8) != v76->m128i_i64[1] )
        break;
      v37 += 24;
      v76 = (__m128i *)((char *)v76 + 24);
      if ( v37 == v36 )
        goto LABEL_104;
    }
    goto LABEL_36;
  }
LABEL_104:
  v78 = v141 - (_QWORD)v33;
  if ( v33 )
    j_j___libc_free_0(v33, v78);
  if ( !BYTE4(v138) )
    _libc_free(v136, v78);
  if ( v132 )
  {
    v78 = v134 - v132;
    j_j___libc_free_0(v132, v134 - v132);
  }
  if ( !v130 )
    _libc_free(v127, v78);
  if ( v156 )
  {
    v78 = v158 - (_QWORD)v156;
    j_j___libc_free_0(v156, v158 - (_QWORD)v156);
  }
  if ( !v154 )
    _libc_free(v153, v78);
  if ( v149 )
  {
    v78 = v151 - (_QWORD)v149;
    j_j___libc_free_0(v149, v151 - (_QWORD)v149);
  }
  if ( !BYTE4(v147) )
    _libc_free(v145.m128i_i64[1], v78);
  v79 = v117;
  if ( v117 )
  {
    v80 = v115;
    v135 = 0;
    v136 = -4096;
    v137 = -3;
    v81 = v115 + 104LL * v117;
    v138 = 0;
    memset(v139, 0, 24);
    v145.m128i_i64[0] = 0;
    v145.m128i_i64[1] = -8192;
    v146 = -4;
    v147 = 0;
    memset(v148, 0, 24);
    do
    {
      while ( sub_103B1D0(v80, (__int64)&v135) )
      {
        v80 += 104;
        if ( v81 == v80 )
          goto LABEL_125;
      }
      v82 = v80;
      v80 += 104;
      sub_103B1D0(v82, (__int64)&v145);
    }
    while ( v81 != v80 );
LABEL_125:
    v79 = v117;
  }
  v83 = 104 * v79;
  result = sub_C7D6A0(v115, 104 * v79, 8);
  if ( v142 != v144 )
    return _libc_free(v142, v83);
  return result;
}
