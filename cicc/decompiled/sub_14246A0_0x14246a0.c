// Function: sub_14246A0
// Address: 0x14246a0
//
void __fastcall sub_14246A0(_QWORD *a1)
{
  __m128i *v2; // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rsi
  __int64 *v21; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rdx
  __int64 *v26; // rax
  char v27; // r8
  __int64 *v28; // rcx
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdx
  __int64 *v33; // rax
  char v34; // r8
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdx
  __int64 *v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r8
  unsigned int v42; // esi
  __int64 *v43; // rdx
  __int64 v44; // r9
  __int64 v45; // r12
  __int64 i; // rbx
  __int8 v47; // al
  __int64 v48; // rdx
  unsigned int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 v52; // rsi
  unsigned __int8 v53; // al
  char v54; // al
  __int64 v55; // r8
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // rsi
  unsigned __int64 v59; // rax
  __int64 v60; // rbx
  unsigned __int64 v61; // rbx
  bool v62; // zf
  unsigned __int64 v63; // rax
  __int64 v64; // rax
  __int64 *v65; // r13
  __int64 j; // r14
  __int64 *v67; // rax
  char v68; // dl
  __int64 v69; // rbx
  __int64 *v70; // rax
  __int64 *v71; // rsi
  __int64 *k; // rdi
  unsigned __int64 v73; // rcx
  char v74; // al
  char v75; // si
  bool v76; // al
  _QWORD *v77; // rbx
  _QWORD *v78; // r12
  _QWORD *v79; // rdi
  __int64 v80; // rax
  int v81; // esi
  int v82; // eax
  __m128i v83; // xmm5
  unsigned __int64 v84; // r13
  __int64 v85; // r10
  _QWORD *v86; // r8
  __m128i v87; // xmm1
  __m128i v88; // xmm2
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rax
  __m128i v92; // xmm6
  __m128i v93; // xmm7
  __int64 v94; // rsi
  int v95; // edx
  unsigned __int64 v96; // rcx
  __int8 v97; // al
  __int64 *v98; // rsi
  __int64 v99; // rdi
  __int64 v100; // rax
  __int64 v101; // rcx
  unsigned __int64 v102; // rax
  __int64 *v103; // rdx
  __int8 v104; // dl
  int v105; // r10d
  __int8 v106; // [rsp+1Fh] [rbp-461h]
  __int64 v107; // [rsp+20h] [rbp-460h]
  __int64 v108; // [rsp+38h] [rbp-448h]
  __int64 v109; // [rsp+40h] [rbp-440h]
  __int64 v110; // [rsp+48h] [rbp-438h]
  __int64 *v111; // [rsp+50h] [rbp-430h]
  __int64 v112; // [rsp+58h] [rbp-428h]
  __int64 v113; // [rsp+60h] [rbp-420h]
  __m128i *v114; // [rsp+68h] [rbp-418h]
  __int64 v115; // [rsp+68h] [rbp-418h]
  __int64 *v116; // [rsp+68h] [rbp-418h]
  __int64 v117; // [rsp+70h] [rbp-410h] BYREF
  _QWORD *v118; // [rsp+78h] [rbp-408h]
  __int64 v119; // [rsp+80h] [rbp-400h]
  unsigned int v120; // [rsp+88h] [rbp-3F8h]
  __m128i v121; // [rsp+90h] [rbp-3F0h] BYREF
  __m128i v122; // [rsp+A0h] [rbp-3E0h] BYREF
  __int64 v123; // [rsp+B0h] [rbp-3D0h]
  __m128i v124[8]; // [rsp+C0h] [rbp-3C0h] BYREF
  __m128i v125; // [rsp+140h] [rbp-340h] BYREF
  __m128i v126; // [rsp+150h] [rbp-330h]
  __int64 v127; // [rsp+160h] [rbp-320h]
  _QWORD v128[8]; // [rsp+168h] [rbp-318h] BYREF
  unsigned __int64 v129; // [rsp+1A8h] [rbp-2D8h] BYREF
  unsigned __int64 v130; // [rsp+1B0h] [rbp-2D0h]
  unsigned __int64 v131; // [rsp+1B8h] [rbp-2C8h]
  __int64 v132; // [rsp+1C0h] [rbp-2C0h] BYREF
  __int64 *v133; // [rsp+1C8h] [rbp-2B8h]
  __int64 *v134; // [rsp+1D0h] [rbp-2B0h]
  unsigned int v135; // [rsp+1D8h] [rbp-2A8h]
  unsigned int v136; // [rsp+1DCh] [rbp-2A4h]
  int v137; // [rsp+1E0h] [rbp-2A0h]
  _BYTE v138[64]; // [rsp+1E8h] [rbp-298h] BYREF
  unsigned __int64 v139; // [rsp+228h] [rbp-258h] BYREF
  unsigned __int64 v140; // [rsp+230h] [rbp-250h]
  unsigned __int64 v141; // [rsp+238h] [rbp-248h]
  __int64 v142; // [rsp+240h] [rbp-240h] BYREF
  __int64 v143; // [rsp+248h] [rbp-238h]
  unsigned __int64 v144; // [rsp+250h] [rbp-230h]
  __int64 v145; // [rsp+258h] [rbp-228h]
  __int64 v146; // [rsp+260h] [rbp-220h]
  _QWORD v147[8]; // [rsp+268h] [rbp-218h] BYREF
  __int64 v148; // [rsp+2A8h] [rbp-1D8h]
  unsigned __int64 v149; // [rsp+2B0h] [rbp-1D0h]
  __int64 v150; // [rsp+2B8h] [rbp-1C8h]
  __int64 *v151; // [rsp+2C0h] [rbp-1C0h] BYREF
  __int64 v152; // [rsp+2C8h] [rbp-1B8h]
  _QWORD v153[16]; // [rsp+2D0h] [rbp-1B0h] BYREF
  __int64 v154; // [rsp+350h] [rbp-130h] BYREF
  __int64 v155; // [rsp+358h] [rbp-128h]
  unsigned __int64 v156; // [rsp+360h] [rbp-120h]
  __int64 v157; // [rsp+368h] [rbp-118h]
  __int64 v158; // [rsp+370h] [rbp-110h]
  __int64 v159[8]; // [rsp+378h] [rbp-108h] BYREF
  __int64 *v160; // [rsp+3B8h] [rbp-C8h]
  __int64 *v161; // [rsp+3C0h] [rbp-C0h]
  unsigned __int64 v162; // [rsp+3C8h] [rbp-B8h]
  char v163[8]; // [rsp+3D0h] [rbp-B0h] BYREF
  __int64 v164; // [rsp+3D8h] [rbp-A8h]
  unsigned __int64 v165; // [rsp+3E0h] [rbp-A0h]
  char v166[64]; // [rsp+3F8h] [rbp-88h] BYREF
  __int64 *v167; // [rsp+438h] [rbp-48h]
  __int64 *v168; // [rsp+440h] [rbp-40h]
  __int64 v169; // [rsp+448h] [rbp-38h]

  v2 = &v125;
  v151 = v153;
  v3 = *a1;
  v117 = 0;
  v4 = *(_QWORD *)(v3 + 120);
  v118 = 0;
  v153[0] = v4;
  v152 = 0x1000000001LL;
  v5 = a1[3];
  v119 = 0;
  v120 = 0;
  v6 = *(_QWORD *)(v5 + 56);
  memset(v124, 0, sizeof(v124));
  v124[1].m128i_i32[2] = 8;
  v124[0].m128i_i64[1] = (__int64)&v124[2].m128i_i64[1];
  v124[1].m128i_i64[0] = (__int64)&v124[2].m128i_i64[1];
  v125.m128i_i64[1] = (__int64)v128;
  v126.m128i_i64[0] = (__int64)v128;
  v126.m128i_i64[1] = 0x100000008LL;
  v128[0] = v6;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  LODWORD(v127) = 0;
  v125.m128i_i64[0] = 1;
  v154 = v6;
  LOBYTE(v156) = 0;
  sub_13B8390(&v129, (__int64)&v154);
  sub_16CCEE0(&v142, v147, 8, v124);
  v7 = v124[6].m128i_i64[1];
  v124[6].m128i_i64[1] = 0;
  v148 = v7;
  v8 = v124[7].m128i_i64[0];
  v124[7].m128i_i64[0] = 0;
  v149 = v8;
  v9 = v124[7].m128i_i64[1];
  v124[7].m128i_i64[1] = 0;
  v150 = v9;
  v111 = &v132;
  sub_16CCEE0(&v132, v138, 8, &v125);
  v10 = v129;
  v129 = 0;
  v139 = v10;
  v11 = v130;
  v130 = 0;
  v140 = v11;
  v12 = v131;
  v131 = 0;
  v141 = v12;
  sub_16CCEE0(&v154, v159, 8, &v132);
  v13 = v139;
  v139 = 0;
  v160 = (__int64 *)v13;
  v14 = v140;
  v140 = 0;
  v161 = (__int64 *)v14;
  v15 = v141;
  v141 = 0;
  v162 = v15;
  sub_16CCEE0(v163, v166, 8, &v142);
  v16 = v148;
  v148 = 0;
  v167 = (__int64 *)v16;
  v17 = v149;
  v149 = 0;
  v168 = (__int64 *)v17;
  v18 = v150;
  v150 = 0;
  v169 = v18;
  if ( v139 )
    j_j___libc_free_0(v139, v141 - v139);
  if ( v134 != v133 )
    _libc_free((unsigned __int64)v134);
  if ( v148 )
    j_j___libc_free_0(v148, v150 - v148);
  if ( v144 != v143 )
    _libc_free(v144);
  if ( v129 )
    j_j___libc_free_0(v129, v131 - v129);
  if ( v126.m128i_i64[0] != v125.m128i_i64[1] )
    _libc_free(v126.m128i_u64[0]);
  if ( v124[6].m128i_i64[1] )
    j_j___libc_free_0(v124[6].m128i_i64[1], v124[7].m128i_i64[1] - v124[6].m128i_i64[1]);
  if ( v124[1].m128i_i64[0] != v124[0].m128i_i64[1] )
    _libc_free(v124[1].m128i_u64[0]);
  sub_16CCCB0(&v132, v138, &v154);
  v20 = v161;
  v21 = v160;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v22 = (char *)v161 - (char *)v160;
  if ( v161 == v160 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_205;
    v23 = sub_22077B0((char *)v161 - (char *)v160);
    v20 = v161;
    v21 = v160;
    v24 = v23;
  }
  v139 = v24;
  v140 = v24;
  v141 = v24 + v22;
  if ( v21 != v20 )
  {
    v25 = v24;
    v26 = v21;
    do
    {
      if ( v25 )
      {
        *(_QWORD *)v25 = *v26;
        v27 = *((_BYTE *)v26 + 16);
        *(_BYTE *)(v25 + 16) = v27;
        if ( v27 )
          *(_QWORD *)(v25 + 8) = v26[1];
      }
      v26 += 3;
      v25 += 24LL;
    }
    while ( v26 != v20 );
    v24 += 8 * ((unsigned __int64)((char *)(v26 - 3) - (char *)v21) >> 3) + 24;
  }
  v21 = &v142;
  v140 = v24;
  sub_16CCCB0(&v142, v147, v163);
  v28 = v168;
  v20 = v167;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v29 = (char *)v168 - (char *)v167;
  if ( v168 != v167 )
  {
    if ( v29 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v30 = sub_22077B0((char *)v168 - (char *)v167);
      v28 = v168;
      v20 = v167;
      v31 = v30;
      goto LABEL_30;
    }
LABEL_205:
    sub_4261EA(v21, v20, v19);
  }
  v29 = 0;
  v31 = 0;
LABEL_30:
  v148 = v31;
  v149 = v31;
  v150 = v31 + v29;
  if ( v28 == v20 )
  {
    v35 = v31;
  }
  else
  {
    v32 = v31;
    v33 = v20;
    do
    {
      if ( v32 )
      {
        *(_QWORD *)v32 = *v33;
        v34 = *((_BYTE *)v33 + 16);
        *(_BYTE *)(v32 + 16) = v34;
        if ( v34 )
          *(_QWORD *)(v32 + 8) = v33[1];
      }
      v33 += 3;
      v32 += 24LL;
    }
    while ( v33 != v28 );
    v35 = v31 + 8 * ((unsigned __int64)((char *)(v33 - 3) - (char *)v20) >> 3) + 24;
  }
  v36 = v140;
  v37 = v139;
  v149 = v35;
  v109 = 1;
  v110 = 1;
  if ( v140 - v139 == v35 - v31 )
    goto LABEL_89;
  while ( 1 )
  {
    do
    {
      v38 = *(__int64 **)(v36 - 24);
      v39 = *(unsigned int *)(*a1 + 80LL);
      if ( (_DWORD)v39 )
      {
        v40 = *(_QWORD *)(*a1 + 64LL);
        v41 = (unsigned int)(v39 - 1);
        v108 = *v38;
        v42 = v41 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
        v43 = (__int64 *)(v40 + 16LL * v42);
        v44 = *v43;
        if ( *v38 == *v43 )
        {
LABEL_40:
          if ( v43 != (__int64 *)(v40 + 16 * v39) )
          {
            v112 = v43[1];
            if ( v112 )
            {
              v45 = v109;
              for ( i = *(_QWORD *)(v151[(unsigned int)v152 - 1] + 64); ; i = v48 )
              {
                v47 = sub_15CC8F0(a1[3], i, v108, v38, v41);
                if ( v47 )
                  break;
                v38 = v151;
                v48 = *(_QWORD *)(v151[(unsigned int)v152 - 1] + 64);
                if ( i == v48 )
                {
                  v49 = v152 - 1;
                  do
                  {
                    v50 = v49;
                    LODWORD(v152) = v49--;
                    v48 = *(_QWORD *)(v151[v50 - 1] + 64);
                  }
                  while ( v48 == i );
                }
                ++v45;
              }
              v106 = v47;
              v109 = v45;
              v51 = *(_QWORD *)(v112 + 8);
              if ( v51 == v112 )
              {
LABEL_72:
                v36 = v140;
                v65 = v111;
                v114 = v2;
                j = *(_QWORD *)(v140 - 24);
                goto LABEL_73;
              }
              while ( 1 )
              {
LABEL_67:
                if ( !v51 )
                  BUG();
                if ( *(_BYTE *)(v51 - 16) == 21 )
                  break;
                v64 = (unsigned int)v152;
                if ( (unsigned int)v152 >= HIDWORD(v152) )
                {
                  sub_16CD150(&v151, v153, 0, 8);
                  v64 = (unsigned int)v152;
                }
                ++v110;
                v151[v64] = v51 - 32;
                LODWORD(v152) = v152 + 1;
                v51 = *(_QWORD *)(v51 + 8);
                if ( v112 == v51 )
                  goto LABEL_72;
              }
              if ( !(unsigned __int8)sub_1420060(a1[2], *(_QWORD *)(v51 + 40)) )
              {
                v52 = *(_QWORD *)(v51 + 40);
                v124[0].m128i_i8[0] = 0;
                v53 = *(_BYTE *)(v52 + 16);
                if ( v53 <= 0x17u )
                  goto LABEL_151;
                switch ( v53 )
                {
                  case 0x4Eu:
                    if ( (v52 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                      goto LABEL_151;
                    v124[0].m128i_i8[0] = 1;
                    v94 = v52 | 4;
                    break;
                  case 0x1Du:
                    if ( (v52 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                      goto LABEL_151;
                    v124[0].m128i_i8[0] = 1;
                    v94 = v52 & 0xFFFFFFFFFFFFFFFBLL;
                    break;
                  case 0x39u:
                    goto LABEL_55;
                  default:
LABEL_151:
                    switch ( *(_BYTE *)(v52 + 16) )
                    {
                      case '6':
                        sub_141EB40(&v121, (__int64 *)v52);
                        goto LABEL_153;
                      case '7':
                        sub_141EDF0(&v121, v52);
                        goto LABEL_153;
                      case ':':
                        sub_141F110(&v121, v52);
                        goto LABEL_153;
                      case ';':
                        sub_141F3C0(&v121, v52);
                        goto LABEL_153;
                      case 'R':
                        sub_141F0A0(&v121, v52);
LABEL_153:
                        v92 = _mm_loadu_si128(&v121);
                        v93 = _mm_loadu_si128(&v122);
                        v127 = v123;
                        v125 = v92;
                        v126 = v93;
                        break;
                      default:
                        break;
                    }
                    *(__m128i *)((char *)v124 + 8) = v125;
                    *(__m128i *)((char *)&v124[1] + 8) = v126;
                    v124[2].m128i_i64[1] = v127;
LABEL_55:
                    v54 = sub_1423F10((__int64)&v117, (__int64)v124, v2);
                    v56 = v125.m128i_i64[0];
                    if ( v54 )
                    {
                      v57 = *(_QWORD *)(v125.m128i_i64[0] + 56);
                      goto LABEL_57;
                    }
                    v81 = v120;
                    ++v117;
                    v82 = v119 + 1;
                    if ( 4 * ((int)v119 + 1) >= 3 * v120 )
                    {
                      v81 = 2 * v120;
                    }
                    else if ( v120 - HIDWORD(v119) - v82 > v120 >> 3 )
                    {
LABEL_128:
                      LODWORD(v119) = v82;
                      v125.m128i_i64[0] = 0;
                      v125.m128i_i64[1] = -8;
                      v126 = 0u;
                      v127 = 0;
                      v128[0] = 0;
                      if ( !(unsigned __int8)sub_1423BB0((_QWORD *)v56, (__int64)v2) )
                        --HIDWORD(v119);
                      v57 = 0;
                      *(__m128i *)v56 = _mm_loadu_si128(v124);
                      *(__m128i *)(v56 + 16) = _mm_loadu_si128(&v124[1]);
                      v83 = _mm_loadu_si128(&v124[2]);
                      *(_OWORD *)(v56 + 48) = 0;
                      *(__m128i *)(v56 + 32) = v83;
                      *(_OWORD *)(v56 + 64) = 0;
                      *(_OWORD *)(v56 + 80) = 0;
LABEL_57:
                      if ( v57 == v109 )
                      {
                        if ( v110 != *(_QWORD *)(v56 + 48) )
                          *(_QWORD *)(v56 + 48) = v110;
                      }
                      else
                      {
                        v58 = *(_QWORD *)(v56 + 72);
                        *(_QWORD *)(v56 + 56) = v109;
                        *(_QWORD *)(v56 + 48) = v110;
                        if ( v108 != v58 && v58 && !(unsigned __int8)sub_15CC8F0(a1[3], v58, v108, v109, v55) )
                        {
                          v59 = (unsigned __int64)v151;
                          *(_QWORD *)(v56 + 64) = 0;
                          *(_QWORD *)(v56 + 72) = *(_QWORD *)(*(_QWORD *)v59 + 64LL);
LABEL_62:
                          v60 = (unsigned int)v152;
                          *(_BYTE *)(v56 + 88) = 1;
                          v61 = v60 - 1;
                          v62 = *(_BYTE *)(v56 + 90) == 0;
                          *(_QWORD *)(v56 + 80) = v61;
                          if ( v62 )
                            *(_WORD *)(v56 + 89) = 257;
                          else
                            *(_BYTE *)(v56 + 89) = 1;
                          goto LABEL_64;
                        }
                      }
                      if ( !*(_BYTE *)(v56 + 88) )
                        goto LABEL_62;
                      v61 = (unsigned int)v152 - 1LL;
LABEL_64:
                      v63 = *(_QWORD *)(v56 + 64);
                      if ( v61 - v63 > (unsigned int)dword_4F99900 )
                      {
                        *(_BYTE *)(v56 + 88) = 0;
                        goto LABEL_66;
                      }
                      if ( v61 <= v63 )
                      {
LABEL_168:
                        v96 = *(_QWORD *)(v56 + 80);
                        v97 = *(_BYTE *)(v56 + 90);
                        if ( v96 > v61 )
                        {
                          v98 = &v151[v61];
LABEL_170:
                          if ( *v98 == *(_QWORD *)(*a1 + 120LL) )
                            goto LABEL_180;
                          goto LABEL_171;
                        }
                        v125.m128i_i8[1] = *(_BYTE *)(v56 + 90);
                        if ( v97 )
                          v125.m128i_i8[0] = *(_BYTE *)(v56 + 89);
                        sub_1420670(v51 - 32, v151[v96], 1, v2->m128i_i8);
                      }
                      else
                      {
                        v115 = v56;
                        v84 = v61;
                        while ( 1 )
                        {
                          v113 = v84;
                          v85 = v151[v84];
                          if ( *(_BYTE *)(v85 + 16) == 23 )
                            break;
                          v86 = (_QWORD *)a1[2];
                          if ( !v124[0].m128i_i8[0] )
                          {
                            v87 = _mm_loadu_si128((const __m128i *)&v124[0].m128i_u64[1]);
                            v88 = _mm_loadu_si128((const __m128i *)&v124[1].m128i_u64[1]);
                            v123 = v124[2].m128i_i64[1];
                            v121 = v87;
                            v122 = v88;
                            v89 = *(_QWORD *)(v85 + 72);
                            if ( *(_BYTE *)(v89 + 16) == 78 )
                            {
                              v90 = *(_QWORD *)(v89 - 24);
                              if ( !*(_BYTE *)(v90 + 16)
                                && (*(_BYTE *)(v90 + 33) & 0x20) != 0
                                && *(_DWORD *)(v90 + 36) == 116 )
                              {
                                v107 = v85;
                                v91 = *(_QWORD *)(v89 + 24 * (1LL - (*(_DWORD *)(v89 + 20) & 0xFFFFFFF)));
                                v125.m128i_i64[1] = -1;
                                v126 = 0u;
                                v125.m128i_i64[0] = v91;
                                v127 = 0;
                                if ( (unsigned __int8)sub_134CB50((__int64)v86, (__int64)v2, (__int64)&v121) == 3 )
                                {
                                  v56 = v115;
                                  v97 = *(_BYTE *)(v115 + 90);
                                  if ( v97 )
                                  {
                                    *(_BYTE *)(v115 + 89) = 3;
                                    v98 = v151;
                                    v61 = 0;
                                  }
                                  else
                                  {
                                    v98 = v151;
                                    v61 = 0;
                                    *(_WORD *)(v115 + 89) = 259;
                                    v97 = v106;
                                  }
                                  goto LABEL_170;
                                }
                                v86 = (_QWORD *)a1[2];
                                v85 = v107;
                              }
                            }
                          }
                          sub_14205E0((bool *)v2->m128i_i8, v85, *(_QWORD *)(v51 + 40), (__int64)v124, v86);
                          if ( v125.m128i_i8[0] )
                          {
                            v61 = v84;
                            v104 = v125.m128i_i8[2];
                            v56 = v115;
                            v97 = *(_BYTE *)(v115 + 90);
                            if ( v125.m128i_i8[2] )
                            {
                              *(_BYTE *)(v115 + 89) = v125.m128i_i8[1];
                              if ( !v97 )
                              {
                                *(_BYTE *)(v115 + 90) = 1;
                                v97 = v104;
                              }
                            }
                            else if ( v97 )
                            {
                              *(_BYTE *)(v115 + 90) = 0;
                              v97 = 0;
                            }
                            v98 = &v151[v113];
                            goto LABEL_170;
                          }
                          if ( *(_QWORD *)(v115 + 64) >= --v84 )
                          {
                            v61 = v84;
                            v56 = v115;
                            goto LABEL_168;
                          }
                        }
                        v61 = v84;
                        v56 = v115;
                        v116 = (__int64 *)a1[1];
                        v100 = sub_1422850(v116[1], *(_QWORD *)(v51 + 40));
                        v101 = (*(__int64 (__fastcall **)(__int64 *, __int64))(*v116 + 16))(v116, v100);
                        v98 = &v151[v113];
                        if ( v101 != v151[v113] )
                        {
                          v102 = v61 - 1;
                          v103 = &v151[v61 - 1];
                          while ( 1 )
                          {
                            v98 = v103;
                            v61 = v102;
                            --v103;
                            if ( *v98 == v101 )
                              break;
                            --v102;
                          }
                        }
                        v97 = *(_BYTE *)(v56 + 90);
                        if ( *v98 == *(_QWORD *)(*a1 + 120LL) )
                        {
LABEL_180:
                          if ( v97 )
                          {
                            *(_BYTE *)(v56 + 90) = 0;
                            v99 = v51 - 32;
                            v125.m128i_i8[1] = 0;
                          }
                          else
                          {
                            v125.m128i_i8[1] = 0;
                            v99 = v51 - 32;
                          }
                          goto LABEL_173;
                        }
LABEL_171:
                        v125.m128i_i8[1] = v97;
                        v99 = v51 - 32;
                        if ( v97 )
                          v125.m128i_i8[0] = *(_BYTE *)(v56 + 89);
LABEL_173:
                        sub_1420670(v99, *v98, 1, v2->m128i_i8);
                        *(_QWORD *)(v56 + 80) = v61;
                      }
                      *(_QWORD *)(v56 + 64) = (unsigned int)v152 - 1LL;
                      *(_QWORD *)(v56 + 72) = v108;
LABEL_66:
                      v51 = *(_QWORD *)(v51 + 8);
                      if ( v112 == v51 )
                        goto LABEL_72;
                      goto LABEL_67;
                    }
                    sub_14241E0((__int64)&v117, v81);
                    sub_1423F10((__int64)&v117, (__int64)v124, v2);
                    v56 = v125.m128i_i64[0];
                    v82 = v119 + 1;
                    goto LABEL_128;
                }
                v124[0].m128i_i64[1] = v94;
                goto LABEL_55;
              }
              v80 = *a1;
              v125.m128i_i8[1] = 0;
              sub_1420670(v51 - 32, *(_QWORD *)(v80 + 120), 1, v2->m128i_i8);
              goto LABEL_66;
            }
          }
        }
        else
        {
          v95 = 1;
          while ( v44 != -8 )
          {
            v105 = v95 + 1;
            v42 = v41 & (v95 + v42);
            v43 = (__int64 *)(v40 + 16LL * v42);
            v44 = *v43;
            if ( v108 == *v43 )
              goto LABEL_40;
            v95 = v105;
          }
        }
      }
      v114 = v2;
      v65 = v111;
      for ( j = *(_QWORD *)(v36 - 24); ; j = *(_QWORD *)(v140 - 24) )
      {
LABEL_73:
        if ( !*(_BYTE *)(v36 - 8) )
        {
          v67 = *(__int64 **)(j + 24);
          *(_BYTE *)(v36 - 8) = 1;
          *(_QWORD *)(v36 - 16) = v67;
          goto LABEL_77;
        }
LABEL_76:
        while ( 1 )
        {
          v67 = *(__int64 **)(v36 - 16);
LABEL_77:
          if ( *(__int64 **)(j + 32) == v67 )
            break;
          *(_QWORD *)(v36 - 16) = v67 + 1;
          v69 = *v67;
          v70 = v133;
          if ( v134 == v133 )
          {
            v71 = &v133[v136];
            if ( v133 != v71 )
            {
              for ( k = 0; ; k = v70++ )
              {
                while ( 1 )
                {
                  if ( v69 == *v70 )
                    goto LABEL_76;
                  if ( *v70 == -2 )
                    break;
                  if ( v71 == ++v70 )
                  {
                    if ( !k )
                      goto LABEL_145;
                    v111 = v65;
                    v2 = v114;
                    goto LABEL_86;
                  }
                }
                if ( v70 + 1 == v71 )
                {
                  v111 = v65;
                  v2 = v114;
                  k = v70;
LABEL_86:
                  *k = v69;
                  --v137;
                  ++v132;
                  goto LABEL_87;
                }
              }
            }
LABEL_145:
            if ( v136 < v135 )
            {
              v111 = v65;
              v2 = v114;
              ++v136;
              *v71 = v69;
              ++v132;
LABEL_87:
              v125.m128i_i64[0] = v69;
              v126.m128i_i8[0] = 0;
              sub_13B8390(&v139, (__int64)v2);
              v37 = v139;
              v36 = v140;
              goto LABEL_88;
            }
          }
          sub_16CCBA0(v65, v69);
          if ( v68 )
          {
            v111 = v65;
            v2 = v114;
            goto LABEL_87;
          }
        }
        v140 -= 24LL;
        v37 = v139;
        v36 = v140;
        if ( v140 == v139 )
          break;
      }
      v111 = v65;
      v2 = v114;
LABEL_88:
      v31 = v148;
    }
    while ( v36 - v37 != v149 - v148 );
LABEL_89:
    if ( v36 == v37 )
      break;
    v73 = v31;
    while ( *(_QWORD *)v37 == *(_QWORD *)v73 )
    {
      v74 = *(_BYTE *)(v37 + 16);
      v75 = *(_BYTE *)(v73 + 16);
      if ( v74 && v75 )
        v76 = *(_QWORD *)(v37 + 8) == *(_QWORD *)(v73 + 8);
      else
        v76 = v75 == v74;
      if ( !v76 )
        break;
      v37 += 24LL;
      v73 += 24LL;
      if ( v36 == v37 )
        goto LABEL_97;
    }
  }
LABEL_97:
  if ( v31 )
    j_j___libc_free_0(v31, v150 - v31);
  if ( v144 != v143 )
    _libc_free(v144);
  if ( v139 )
    j_j___libc_free_0(v139, v141 - v139);
  if ( v134 != v133 )
    _libc_free((unsigned __int64)v134);
  if ( v167 )
    j_j___libc_free_0(v167, v169 - (_QWORD)v167);
  if ( v165 != v164 )
    _libc_free(v165);
  if ( v160 )
    j_j___libc_free_0(v160, v162 - (_QWORD)v160);
  if ( v156 != v155 )
    _libc_free(v156);
  if ( v120 )
  {
    v77 = v118;
    v142 = 0;
    v143 = -8;
    v144 = 0;
    v78 = &v118[12 * v120];
    v145 = 0;
    v146 = 0;
    v147[0] = 0;
    v154 = 0;
    v155 = -16;
    v156 = 0;
    v157 = 0;
    v158 = 0;
    v159[0] = 0;
    do
    {
      while ( (unsigned __int8)sub_1423BB0(v77, (__int64)&v142) )
      {
        v77 += 12;
        if ( v78 == v77 )
          goto LABEL_118;
      }
      v79 = v77;
      v77 += 12;
      sub_1423BB0(v79, (__int64)&v154);
    }
    while ( v78 != v77 );
  }
LABEL_118:
  j___libc_free_0(v118);
  if ( v151 != v153 )
    _libc_free((unsigned __int64)v151);
}
