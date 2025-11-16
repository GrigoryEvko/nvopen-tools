// Function: sub_1F180C0
// Address: 0x1f180c0
//
__int64 __fastcall sub_1F180C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r14
  unsigned int v7; // r13d
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  unsigned __int32 v10; // r12d
  __int64 v11; // rax
  unsigned __int32 v12; // ecx
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // r8
  unsigned __int64 *v20; // rax
  __m128i *v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r11
  __int64 v24; // rbx
  int v25; // edi
  unsigned __int64 v26; // r8
  unsigned int i; // edx
  __int64 v28; // r9
  __int64 v29; // r12
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // r15
  int v33; // r10d
  unsigned __int64 v34; // rdx
  unsigned int v35; // eax
  __int64 v36; // r8
  __int64 v37; // r13
  __int64 v38; // r14
  int v39; // ecx
  __int64 v40; // rcx
  __int64 v41; // r8
  int v42; // r10d
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // rdx
  int v47; // edi
  unsigned int v48; // eax
  unsigned __int64 v49; // rax
  unsigned int v50; // r14d
  __int64 v51; // rcx
  __int64 v52; // r13
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 *v56; // rax
  int v57; // edx
  __int64 v58; // rsi
  __int64 v59; // rbx
  __int64 v60; // rax
  unsigned int v61; // ecx
  unsigned __int64 v62; // r11
  __int64 v63; // r10
  __int64 *v64; // rax
  unsigned int v65; // r15d
  unsigned int v66; // eax
  __int64 v67; // rbx
  __int64 v68; // r15
  __int64 v69; // r14
  int v70; // r8d
  int v71; // r9d
  __int64 v72; // rdx
  unsigned __int64 v73; // r10
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // rsi
  unsigned int v77; // ecx
  __int64 *v78; // rdx
  __int64 v79; // rdi
  __int64 v80; // rax
  __m128i *v81; // rax
  unsigned int v82; // eax
  unsigned int v83; // ecx
  __int64 v84; // rsi
  unsigned int v85; // edx
  __int64 v86; // r13
  unsigned int v87; // eax
  bool v88; // cf
  __int64 *v89; // rax
  __int64 *v90; // rax
  __int64 v91; // r15
  unsigned __int64 v92; // rdi
  __int64 v93; // r10
  __int64 v94; // rcx
  int v95; // r8d
  __int64 v96; // r13
  __int64 *v97; // rsi
  __int64 v98; // rdx
  __int64 v99; // r14
  unsigned int v100; // edx
  __int64 v101; // rcx
  int v102; // r11d
  __int64 v103; // rsi
  int v104; // r8d
  int v105; // r9d
  __int64 v106; // rdx
  __int64 v107; // rcx
  __int64 v108; // rax
  __int64 v109; // rsi
  unsigned int v110; // ecx
  __int64 *v111; // rdx
  __int64 v112; // rdi
  __int64 v113; // rax
  __m128i *v114; // rax
  _QWORD *v115; // rax
  __int64 v116; // rax
  _QWORD *v118; // rsi
  _QWORD *v119; // rax
  __int64 v120; // rdx
  __int64 *v121; // rax
  int v122; // edx
  int v123; // edx
  __int64 v124; // rdx
  _BYTE *v125; // r8
  size_t v126; // rdx
  __int128 v127; // [rsp-20h] [rbp-1A0h]
  __int64 v128; // [rsp+18h] [rbp-168h]
  unsigned __int8 v129; // [rsp+27h] [rbp-159h]
  const void *v130; // [rsp+28h] [rbp-158h]
  __int64 v131; // [rsp+38h] [rbp-148h]
  __int64 v132; // [rsp+40h] [rbp-140h]
  unsigned __int64 v133; // [rsp+40h] [rbp-140h]
  __int64 v134; // [rsp+48h] [rbp-138h]
  __int64 v135; // [rsp+48h] [rbp-138h]
  unsigned __int64 v136; // [rsp+48h] [rbp-138h]
  __int64 v137; // [rsp+48h] [rbp-138h]
  __int64 *v138; // [rsp+50h] [rbp-130h]
  __int64 v139; // [rsp+58h] [rbp-128h]
  unsigned __int64 v140; // [rsp+58h] [rbp-128h]
  __int64 v141; // [rsp+58h] [rbp-128h]
  __int64 v142; // [rsp+58h] [rbp-128h]
  unsigned int *v143; // [rsp+60h] [rbp-120h]
  __int64 v144; // [rsp+68h] [rbp-118h]
  __int64 v145; // [rsp+70h] [rbp-110h]
  unsigned __int64 v146; // [rsp+70h] [rbp-110h]
  int v147; // [rsp+70h] [rbp-110h]
  __int64 v148; // [rsp+78h] [rbp-108h]
  unsigned int v149; // [rsp+78h] [rbp-108h]
  __int64 v150; // [rsp+78h] [rbp-108h]
  __int64 v151; // [rsp+78h] [rbp-108h]
  __int64 v152; // [rsp+78h] [rbp-108h]
  unsigned __int64 v153; // [rsp+78h] [rbp-108h]
  __int64 v154; // [rsp+80h] [rbp-100h] BYREF
  __int64 v155; // [rsp+88h] [rbp-F8h] BYREF
  __int64 v156; // [rsp+90h] [rbp-F0h] BYREF
  _BYTE *v157; // [rsp+98h] [rbp-E8h] BYREF
  unsigned __int64 v158; // [rsp+A0h] [rbp-E0h]
  _BYTE v159[72]; // [rsp+A8h] [rbp-D8h] BYREF
  __m128i v160; // [rsp+F0h] [rbp-90h] BYREF
  __m128i src; // [rsp+100h] [rbp-80h] BYREF
  __int64 v162; // [rsp+110h] [rbp-70h]

  v6 = a1;
  v7 = *(_DWORD *)(a1 + 384);
  src.m128i_i64[0] = 0x400000000LL;
  v160.m128i_i64[0] = a1 + 200;
  v8 = *(_DWORD *)(a1 + 388);
  v160.m128i_i64[1] = (__int64)&src.m128i_i64[1];
  if ( v7 )
  {
    src.m128i_i32[0] = 1;
    v9 = &src.m128i_i64[1];
    v10 = 1;
    src.m128i_i64[1] = a1 + 208;
    v11 = v8;
    v12 = 4;
    v162 = v11;
    v13 = 0;
    while ( 1 )
    {
      v14 = &v9[2 * v13];
      v15 = *((unsigned int *)v14 + 3);
      v16 = *(_QWORD *)(*v14 + 8 * v15);
      v17 = v16 & 0x3F;
      v18 = v16 & 0xFFFFFFFFFFFFFFC0LL;
      v19 = v17 + 1;
      if ( v12 <= v10 )
      {
        v15 = (__int64)&src.m128i_i64[1];
        v153 = v17 + 1;
        sub_16CD150((__int64)&v160.m128i_i64[1], &src.m128i_u64[1], 0, 16, v19, a6);
        v9 = (__int64 *)v160.m128i_i64[1];
        v19 = v153;
      }
      v20 = (unsigned __int64 *)&v9[2 * src.m128i_u32[0]];
      *v20 = v18;
      v20[1] = v19;
      v13 = src.m128i_u32[0];
      v10 = ++src.m128i_i32[0];
      if ( v7 <= (unsigned int)v13 )
        break;
      v9 = (__int64 *)v160.m128i_i64[1];
      v12 = src.m128i_u32[1];
    }
    v21 = (__m128i *)v160.m128i_i64[1];
    v156 = v160.m128i_i64[0];
    v157 = v159;
    v158 = 0x400000000LL;
    if ( !v10 )
      goto LABEL_8;
    if ( (unsigned __int64 *)v160.m128i_i64[1] != &src.m128i_u64[1] )
    {
      v157 = (_BYTE *)v160.m128i_i64[1];
      v158 = __PAIR64__(src.m128i_u32[1], v10);
      goto LABEL_10;
    }
    v124 = v10;
    if ( v10 > 4 )
    {
      v15 = (__int64)v159;
      sub_16CD150((__int64)&v157, v159, v10, 16, v19, a6);
      v125 = v157;
      v21 = (__m128i *)v160.m128i_i64[1];
      v126 = 16LL * src.m128i_u32[0];
      if ( !v126 )
        goto LABEL_114;
      goto LABEL_113;
    }
  }
  else
  {
    src.m128i_i64[1] = a1 + 200;
    v10 = 1;
    v156 = a1 + 200;
    v162 = v8;
    src.m128i_i32[0] = 1;
    v157 = v159;
    v158 = 0x400000000LL;
    v124 = 1;
  }
  v125 = v159;
  v126 = 16 * v124;
  v21 = (__m128i *)&src.m128i_u64[1];
LABEL_113:
  v15 = (__int64)v21;
  memcpy(v125, v21, v126);
  v21 = (__m128i *)v160.m128i_i64[1];
LABEL_114:
  LODWORD(v158) = v10;
LABEL_8:
  if ( v21 != (__m128i *)&src.m128i_u64[1] )
    _libc_free((unsigned __int64)v21);
LABEL_10:
  v22 = *(_QWORD *)(*(_QWORD *)(v6 + 72) + 8LL);
  v128 = *(_QWORD *)v22 + 24LL * *(unsigned int *)(v22 + 8);
  if ( v128 != *(_QWORD *)v22 )
  {
    v138 = *(__int64 **)v22;
    v129 = 0;
    while ( 1 )
    {
      v23 = *v138;
      v143 = (unsigned int *)v138[2];
      v24 = (unsigned int)v158;
      if ( (_DWORD)v158 && *((_DWORD *)v157 + 3) < *((_DWORD *)v157 + 2) )
      {
        if ( *(_DWORD *)(v156 + 184) )
        {
          v152 = *v138;
          sub_1F17DE0((__int64)&v156, v23);
          v24 = (unsigned int)v158;
          v23 = v152;
        }
        else
        {
          v25 = *(_DWORD *)(v156 + 188);
          v26 = (unsigned __int64)&v157[16 * (unsigned int)v158 - 16];
          for ( i = *(_DWORD *)(v26 + 12); v25 != i; ++i )
          {
            v28 = *(_QWORD *)(v156 + 16LL * i + 8);
            if ( (*(_DWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v28 >> 1) & 3) > (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(v23 >> 1) & 3) )
              break;
          }
          *(_DWORD *)(v26 + 12) = i;
          v24 = (unsigned int)v158;
        }
      }
      v29 = v23;
      v30 = v6;
      while ( 1 )
      {
        v31 = v138[1];
        v154 = v31;
        if ( !(_DWORD)v24 )
        {
          v144 = v31;
          goto LABEL_23;
        }
        v144 = v31;
        if ( *((_DWORD *)v157 + 3) >= *((_DWORD *)v157 + 2) )
        {
          LODWORD(v24) = 0;
          goto LABEL_23;
        }
        v91 = v156;
        v92 = (unsigned __int64)&v157[16 * v24 - 16];
        v24 = *(unsigned int *)(v92 + 12);
        v93 = *(_QWORD *)v92;
        v94 = (v29 >> 1) & 3;
        v95 = *(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v96 = 16 * v24;
        v97 = (__int64 *)(*(_QWORD *)v92 + 16 * v24);
        v98 = *v97;
        v99 = *v97;
        if ( *(_DWORD *)(v156 + 184) )
        {
          v100 = *(_DWORD *)((v98 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v99 >> 1) & 3;
          if ( (v95 | (unsigned int)v94) >= v100 )
            goto LABEL_65;
        }
        else
        {
          v100 = *(_DWORD *)((v99 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v98 >> 1) & 3;
          if ( (v95 | (unsigned int)v94) >= v100 )
          {
LABEL_65:
            LODWORD(v24) = *(_DWORD *)(v93 + 4 * v24 + 144);
            v101 = *(_QWORD *)(v93 + v96 + 8);
            if ( (*(_DWORD *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v101 >> 1) & 3) < (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) )
            {
              v102 = *(_DWORD *)(v92 + 12) + 1;
              v154 = *(_QWORD *)(v93 + v96 + 8);
              *(_DWORD *)(v92 + 12) = v102;
              v144 = v101;
              if ( v102 == *(_DWORD *)&v157[16 * (unsigned int)v158 - 8] )
              {
                v103 = *(unsigned int *)(v91 + 184);
                if ( (_DWORD)v103 )
                {
                  v150 = v30;
                  sub_39460A0(&v157, v103);
                  v30 = v150;
                }
              }
            }
            goto LABEL_23;
          }
        }
        if ( (*(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v31 >> 1) & 3) <= v100 )
          v97 = &v154;
        LODWORD(v24) = 0;
        v154 = *v97;
        v144 = v154;
LABEL_23:
        v32 = *(_QWORD *)(v30 + 16);
        v33 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v30 + 72) + 16LL)
                        + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v30 + 72) + 64LL) + v24));
        v34 = *(unsigned int *)(v32 + 408);
        v35 = v33 & 0x7FFFFFFF;
        v36 = v33 & 0x7FFFFFFF;
        v37 = 8 * v36;
        if ( (v33 & 0x7FFFFFFFu) < (unsigned int)v34 )
        {
          v38 = *(_QWORD *)(*(_QWORD *)(v32 + 400) + 8LL * v35);
          if ( v38 )
          {
            v39 = *(_DWORD *)(v30 + 424);
            if ( !v39 )
              goto LABEL_38;
            goto LABEL_26;
          }
        }
        v50 = v35 + 1;
        if ( (unsigned int)v34 >= v35 + 1 )
          goto LABEL_36;
        v116 = v50;
        if ( v50 < v34 )
        {
          *(_DWORD *)(v32 + 408) = v50;
LABEL_36:
          v51 = *(_QWORD *)(v32 + 400);
          goto LABEL_37;
        }
        if ( v50 <= v34 )
          goto LABEL_36;
        if ( v50 > (unsigned __int64)*(unsigned int *)(v32 + 412) )
        {
          v137 = v30;
          v142 = v33 & 0x7FFFFFFF;
          v147 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v30 + 72) + 16LL)
                           + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v30 + 72) + 64LL) + v24));
          sub_16CD150(v32 + 400, (const void *)(v32 + 416), v50, 8, v36, v30);
          v34 = *(unsigned int *)(v32 + 408);
          v30 = v137;
          v36 = v142;
          v33 = v147;
          v116 = v50;
        }
        v51 = *(_QWORD *)(v32 + 400);
        v118 = (_QWORD *)(v51 + 8 * v116);
        v119 = (_QWORD *)(v51 + 8 * v34);
        v120 = *(_QWORD *)(v32 + 416);
        if ( v118 != v119 )
        {
          do
            *v119++ = v120;
          while ( v118 != v119 );
          v51 = *(_QWORD *)(v32 + 400);
        }
        *(_DWORD *)(v32 + 408) = v50;
LABEL_37:
        v145 = v30;
        v148 = v36;
        *(_QWORD *)(v51 + v37) = sub_1DBA290(v33);
        v38 = *(_QWORD *)(*(_QWORD *)(v32 + 400) + 8 * v148);
        sub_1DBB110((_QWORD *)v32, v38);
        v30 = v145;
        v39 = *(_DWORD *)(v145 + 424);
        if ( !v39 )
          goto LABEL_38;
LABEL_26:
        v40 = (unsigned int)(v39 - 1);
        v41 = *(_QWORD *)(v30 + 408);
        v42 = 1;
        v15 = *v143;
        v43 = ((((unsigned int)(37 * v15) | ((unsigned __int64)(unsigned int)(37 * v24) << 32))
              - 1
              - ((unsigned __int64)(unsigned int)(37 * v15) << 32)) >> 22)
            ^ (((unsigned int)(37 * v15) | ((unsigned __int64)(unsigned int)(37 * v24) << 32))
             - 1
             - ((unsigned __int64)(unsigned int)(37 * v15) << 32));
        v44 = ((9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13)))) >> 15)
            ^ (9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13))));
        v45 = v40 & (((v44 - 1 - (v44 << 27)) >> 31) ^ (v44 - 1 - ((_DWORD)v44 << 27)));
        v46 = v41 + 16LL * v45;
        v47 = *(_DWORD *)v46;
        if ( (_DWORD)v24 == *(_DWORD *)v46 )
          goto LABEL_29;
        do
        {
          do
          {
            if ( v47 == -1 && *(_DWORD *)(v46 + 4) == -1 )
              goto LABEL_38;
            v48 = v42 + v45;
            ++v42;
            v45 = v40 & v48;
            v46 = v41 + 16LL * v45;
            v47 = *(_DWORD *)v46;
          }
          while ( (_DWORD)v24 != *(_DWORD *)v46 );
LABEL_29:
          ;
        }
        while ( (_DWORD)v15 != *(_DWORD *)(v46 + 4) );
        v49 = *(_QWORD *)(v46 + 8);
        if ( (v49 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          src.m128i_i64[0] = v49 & 0xFFFFFFFFFFFFFFF8LL;
          v160.m128i_i64[1] = v144;
          *((_QWORD *)&v127 + 1) = v144;
          *(_QWORD *)&v127 = v29;
          v151 = v30;
          v160.m128i_i64[0] = v29;
          sub_1DB8610(v38, v15, v49 & 0xFFFFFFFFFFFFFFF8LL, v40, v41, v30, v127, v49 & 0xFFFFFFFFFFFFFFF8LL);
          v30 = v151;
          goto LABEL_33;
        }
        if ( v49 >> 2 )
        {
          v129 = 1;
          goto LABEL_33;
        }
LABEL_38:
        v139 = v30;
        v52 = 664LL * ((*(_DWORD *)(v30 + 84) != 0) & (unsigned __int8)((_DWORD)v24 != 0));
        v131 = v30 + v52 + 432;
        v53 = *(_QWORD *)(*(_QWORD *)(v30 + 16) + 272LL);
        v54 = sub_1DA9310(v53, v29);
        v30 = v139;
        v55 = v54;
        v56 = (__int64 *)(*(_QWORD *)(v53 + 392) + 16LL * *(unsigned int *)(v54 + 48));
        v57 = *(_DWORD *)((v144 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v58 = *v56;
        v146 = v144 & 0xFFFFFFFFFFFFFFF8LL;
        v59 = (v144 >> 1) & 3;
        v60 = v56[1];
        v149 = v59;
        v61 = v59 | v57;
        v155 = v60;
        if ( v29 == v58 )
        {
          v67 = v55;
          v15 = v29 & 0xFFFFFFFFFFFFFFF8LL;
          v66 = *(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v29 >> 1) & 3;
        }
        else
        {
          v29 = v60;
          v62 = v60 & 0xFFFFFFFFFFFFFFF8LL;
          v63 = v60 >> 1;
          v64 = &v155;
          v132 = v139;
          v134 = v55;
          v65 = v63 & 3;
          v140 = v62;
          if ( (v65 | *(_DWORD *)(v62 + 24)) > v61 )
            v64 = &v154;
          v15 = (__int64)sub_1DB7FE0(v38, v58, *v64);
          v30 = v132;
          v57 = *(_DWORD *)(v146 + 24);
          v66 = *(_DWORD *)(v140 + 24) | v65;
          v61 = v59 | v57;
          if ( v66 <= ((unsigned int)v59 | v57) )
          {
            *(_QWORD *)(*(_QWORD *)(v131 + 40) + 8LL * (*(_DWORD *)(v134 + 48) >> 6)) |= 1LL << *(_DWORD *)(v134 + 48);
            v121 = (__int64 *)(*(_QWORD *)(v131 + 96) + 16LL * *(unsigned int *)(v134 + 48));
            *v121 = v15;
            v15 = v144 & 0xFFFFFFFFFFFFFFF8LL;
            v121[1] = 0;
            v57 = *(_DWORD *)(v146 + 24);
            v66 = v65 | *(_DWORD *)(v140 + 24);
            v61 = v57 | v59;
          }
          v67 = *(_QWORD *)(v134 + 8);
        }
        v68 = v30 + v52 + 568;
        v130 = (const void *)(v30 + v52 + 584);
        if ( v61 > v66 )
        {
          v141 = v38;
          v69 = v30;
          while ( 1 )
          {
            while ( 1 )
            {
              v84 = v29;
              v85 = v149 | v57;
              v155 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v69 + 16) + 272LL) + 392LL)
                               + 16LL * *(unsigned int *)(v67 + 48)
                               + 8);
              v29 = v155;
              v86 = (v155 >> 1) & 3;
              v87 = v86 | *(_DWORD *)((v155 & 0xFFFFFFFFFFFFFFF8LL) + 24);
              if ( *((_QWORD *)v143 + 1) != v84 )
                break;
              v88 = v85 < v87;
              v136 = v155 & 0xFFFFFFFFFFFFFFF8LL;
              v89 = &v155;
              if ( v88 )
                v89 = &v154;
              v15 = (__int64)sub_1DB7FE0(v141, v84, *v89);
              v57 = *(_DWORD *)(v146 + 24);
              v82 = *(_DWORD *)(v136 + 24) | v86;
              v83 = v57 | v149;
              if ( (v57 | v149) < v82 )
                goto LABEL_55;
              *(_QWORD *)(*(_QWORD *)(v131 + 40) + 8LL * (*(_DWORD *)(v67 + 48) >> 6)) |= 1LL << *(_DWORD *)(v67 + 48);
              v90 = (__int64 *)(*(_QWORD *)(v131 + 96) + 16LL * *(unsigned int *)(v67 + 48));
              *v90 = v15;
              v90[1] = 0;
              v57 = *(_DWORD *)(v146 + 24);
              v67 = *(_QWORD *)(v67 + 8);
              if ( (v57 | v149) <= ((unsigned int)v86 | *(_DWORD *)(v136 + 24)) )
              {
LABEL_61:
                v30 = v69;
                goto LABEL_33;
              }
            }
            v133 = v155 & 0xFFFFFFFFFFFFFFF8LL;
            v135 = *(_QWORD *)(v69 + 40);
            if ( v85 >= v87 )
            {
              sub_1E06620(v135);
              v106 = 0;
              v73 = v133;
              v107 = *(_QWORD *)(v135 + 1312);
              v108 = *(unsigned int *)(v107 + 48);
              if ( (_DWORD)v108 )
              {
                v109 = *(_QWORD *)(v107 + 32);
                v110 = (v108 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
                v111 = (__int64 *)(v109 + 16LL * v110);
                v112 = *v111;
                if ( *v111 == v67 )
                {
LABEL_71:
                  if ( v111 != (__int64 *)(v109 + 16 * v108) )
                  {
                    v106 = v111[1];
                    goto LABEL_73;
                  }
                }
                else
                {
                  v123 = 1;
                  while ( v112 != -8 )
                  {
                    v104 = v123 + 1;
                    v110 = (v108 - 1) & (v123 + v110);
                    v111 = (__int64 *)(v109 + 16LL * v110);
                    v112 = *v111;
                    if ( *v111 == v67 )
                      goto LABEL_71;
                    v123 = v104;
                  }
                }
                v106 = 0;
              }
LABEL_73:
              v160.m128i_i64[1] = v106;
              src = 0u;
              v160.m128i_i64[0] = v141;
              v113 = *(unsigned int *)(v68 + 8);
              if ( (unsigned int)v113 >= *(_DWORD *)(v68 + 12) )
              {
                sub_16CD150(v68, v130, 0, 32, v104, v105);
                v113 = *(unsigned int *)(v68 + 8);
                v73 = v133;
              }
              v114 = (__m128i *)(*(_QWORD *)v68 + 32 * v113);
              *v114 = _mm_load_si128(&v160);
              v114[1] = _mm_load_si128(&src);
              ++*(_DWORD *)(v68 + 8);
              *(_QWORD *)(*(_QWORD *)(v131 + 40) + 8LL * (*(_DWORD *)(v67 + 48) >> 6)) |= 1LL << *(_DWORD *)(v67 + 48);
              v115 = (_QWORD *)(*(_QWORD *)(v131 + 96) + 16LL * *(unsigned int *)(v67 + 48));
              *v115 = 0;
              v115[1] = 0;
              goto LABEL_54;
            }
            sub_1E06620(v135);
            v72 = 0;
            v73 = v133;
            v74 = *(_QWORD *)(v135 + 1312);
            v75 = *(unsigned int *)(v74 + 48);
            if ( (_DWORD)v75 )
            {
              v76 = *(_QWORD *)(v74 + 32);
              v77 = (v75 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
              v78 = (__int64 *)(v76 + 16LL * v77);
              v79 = *v78;
              if ( *v78 != v67 )
              {
                v122 = 1;
                while ( v79 != -8 )
                {
                  v70 = v122 + 1;
                  v77 = (v75 - 1) & (v122 + v77);
                  v78 = (__int64 *)(v76 + 16LL * v77);
                  v79 = *v78;
                  if ( *v78 == v67 )
                    goto LABEL_49;
                  v122 = v70;
                }
LABEL_100:
                v72 = 0;
                goto LABEL_51;
              }
LABEL_49:
              if ( v78 == (__int64 *)(v76 + 16 * v75) )
                goto LABEL_100;
              v72 = v78[1];
            }
LABEL_51:
            v160.m128i_i64[1] = v72;
            v160.m128i_i64[0] = v141;
            src = (__m128i)(unsigned __int64)v144;
            v80 = *(unsigned int *)(v68 + 8);
            if ( (unsigned int)v80 >= *(_DWORD *)(v68 + 12) )
            {
              sub_16CD150(v68, v130, 0, 32, v70, v71);
              v80 = *(unsigned int *)(v68 + 8);
              v73 = v133;
            }
            v81 = (__m128i *)(*(_QWORD *)v68 + 32 * v80);
            *v81 = _mm_load_si128(&v160);
            v81[1] = _mm_load_si128(&src);
            ++*(_DWORD *)(v68 + 8);
LABEL_54:
            v15 = v144 & 0xFFFFFFFFFFFFFFF8LL;
            v57 = *(_DWORD *)(v146 + 24);
            v82 = v86 | *(_DWORD *)(v73 + 24);
            v83 = v57 | v149;
LABEL_55:
            v67 = *(_QWORD *)(v67 + 8);
            if ( v83 <= v82 )
              goto LABEL_61;
          }
        }
LABEL_33:
        v29 = v144;
        if ( v138[1] == v144 )
          break;
        v24 = (unsigned int)v158;
      }
      v138 += 3;
      v6 = v30;
      if ( (__int64 *)v128 == v138 )
        goto LABEL_86;
    }
  }
  v129 = 0;
LABEL_86:
  sub_1DC4840((_QWORD *)(v6 + 432), v15);
  if ( *(_DWORD *)(v6 + 84) )
    sub_1DC4840((_QWORD *)(v6 + 1096), v15);
  if ( v157 != v159 )
    _libc_free((unsigned __int64)v157);
  return v129;
}
