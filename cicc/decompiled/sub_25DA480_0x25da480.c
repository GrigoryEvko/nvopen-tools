// Function: sub_25DA480
// Address: 0x25da480
//
_QWORD *__fastcall sub_25DA480(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // r14
  unsigned __int64 v7; // r12
  _QWORD *v8; // rax
  _QWORD *v9; // r14
  unsigned __int64 v10; // r12
  _QWORD *v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // r14
  unsigned __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 (__fastcall *v18)(__int64); // rax
  unsigned __int64 *v19; // r12
  __int64 (__fastcall **v20)(__int64); // rbx
  __m128i *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 (__fastcall *v24)(__int64); // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char v29; // al
  __m128i *v30; // r15
  __int64 (__fastcall **v31)(__int64); // rbx
  __int64 (__fastcall *v32)(__int64); // rax
  __m128i *v33; // rdi
  __int64 (__fastcall *v34)(__int64); // rdi
  __int64 v35; // rbx
  _QWORD *m; // r14
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  char v40; // al
  __int64 v41; // rdx
  _QWORD *v42; // r14
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  char v46; // al
  __int64 v47; // rdx
  __int64 *v48; // r14
  __int64 v49; // rax
  __int64 *v50; // r15
  _QWORD *v51; // r15
  __int64 (__fastcall *v52)(__int64); // r13
  __int64 (__fastcall **v53)(__int64); // rax
  __int64 (__fastcall **v54)(__int64); // rdx
  _QWORD *v55; // r15
  __int64 (__fastcall *v56)(__int64); // r13
  __int64 (__fastcall **v57)(__int64); // rax
  __int64 (__fastcall **v58)(__int64); // rdx
  _QWORD *v59; // r15
  __int64 (__fastcall *v60)(__int64); // r13
  __int64 (__fastcall **v61)(__int64); // rax
  __int64 (__fastcall **v62)(__int64); // rdx
  _QWORD *v63; // r13
  __int64 v64; // r12
  _QWORD *v65; // rax
  _QWORD *v66; // rdx
  __int64 *v67; // r12
  __int64 *v68; // r14
  __int64 v69; // r13
  _BYTE *v70; // r12
  _QWORD **v71; // r13
  _QWORD *v72; // r14
  __int64 (__fastcall *v73)(__int64); // r12
  __int64 (__fastcall *v74)(__int64); // r13
  _QWORD *v75; // r14
  __int64 (__fastcall *v76)(__int64); // r12
  __int64 (__fastcall *v77)(__int64); // r13
  _QWORD *v78; // r14
  unsigned int v79; // eax
  __int64 v80; // rdx
  int v81; // r13d
  __int64 v82; // r12
  unsigned int v83; // eax
  __int64 v84; // r14
  __int64 v85; // r15
  __int64 v86; // rax
  _BYTE *v87; // rsi
  char v88; // al
  __int64 v89; // rdx
  _QWORD *v90; // r12
  unsigned __int64 v91; // rdi
  __int64 v92; // rsi
  int v93; // r14d
  _QWORD *v94; // r13
  unsigned int v95; // eax
  __int64 v96; // r15
  _QWORD *v97; // r12
  unsigned __int64 v98; // rdi
  unsigned int v99; // eax
  __int64 v100; // rdx
  _QWORD *v101; // rsi
  _QWORD *v102; // rdx
  __int64 *v104; // rdx
  __int64 v105; // r8
  __int64 *v106; // rax
  __m128i *v107; // rcx
  __int64 v108; // rdx
  __int64 *v109; // rax
  unsigned __int32 v110; // eax
  __int64 v111; // r13
  __int64 v112; // r9
  int v113; // r10d
  __int64 v114; // r8
  __int64 *v115; // rdx
  __int64 v116; // rdi
  __int64 v117; // rcx
  __int64 v118; // rax
  __int64 *v119; // rax
  __int64 *v120; // rdx
  __int64 *v121; // r15
  __int64 *v122; // r13
  __int64 *v123; // rax
  int v124; // eax
  int v125; // eax
  int v126; // r9d
  __int64 v127; // r10
  int v128; // edi
  int v129; // r8d
  __int64 *v130; // rdi
  unsigned int v131; // r15d
  int v132; // edx
  __int64 v133; // r12
  unsigned int v134; // eax
  _QWORD *v135; // rdi
  unsigned __int64 v136; // rdx
  unsigned __int64 v137; // rax
  _QWORD *v138; // rax
  __int64 v139; // rdx
  _QWORD *ii; // rdx
  unsigned __int64 v141; // rbx
  unsigned __int64 v142; // rdi
  unsigned __int64 v143; // rdi
  __int64 v144; // rdx
  int v145; // r12d
  unsigned int v146; // eax
  _QWORD *v147; // rdi
  unsigned __int64 v148; // rdx
  unsigned __int64 v149; // rax
  _QWORD *v150; // rax
  __int64 v151; // rcx
  _QWORD *jj; // rdx
  _QWORD *v153; // rax
  _QWORD *v154; // rax
  unsigned int v155; // r10d
  unsigned __int64 v156; // [rsp+0h] [rbp-170h]
  unsigned __int64 v157; // [rsp+8h] [rbp-168h]
  unsigned __int64 v158; // [rsp+10h] [rbp-160h]
  int v159; // [rsp+10h] [rbp-160h]
  _QWORD *j; // [rsp+18h] [rbp-158h]
  __int64 *v161; // [rsp+18h] [rbp-158h]
  _QWORD *i; // [rsp+20h] [rbp-150h]
  char v164; // [rsp+37h] [rbp-139h]
  unsigned __int64 v166; // [rsp+40h] [rbp-130h]
  _QWORD *n; // [rsp+40h] [rbp-130h]
  __int64 v168; // [rsp+48h] [rbp-128h]
  _QWORD *k; // [rsp+50h] [rbp-120h]
  __int64 v170; // [rsp+58h] [rbp-118h]
  __int64 v171; // [rsp+68h] [rbp-108h] BYREF
  _BYTE *v172; // [rsp+70h] [rbp-100h] BYREF
  _BYTE *v173; // [rsp+78h] [rbp-F8h]
  _BYTE *v174; // [rsp+80h] [rbp-F0h]
  __m128i v175; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v176; // [rsp+A0h] [rbp-D0h]
  __int64 (__fastcall *v177)(__int64); // [rsp+B0h] [rbp-C0h] BYREF
  _BYTE *v178; // [rsp+B8h] [rbp-B8h]
  __int64 (__fastcall *v179)(__int64 *); // [rsp+C0h] [rbp-B0h]
  __int64 v180; // [rsp+C8h] [rbp-A8h]
  __int64 (__fastcall *v181)(__int64); // [rsp+D0h] [rbp-A0h] BYREF
  __int64 (__fastcall *v182)(__int64); // [rsp+D8h] [rbp-98h]
  __int64 (__fastcall *v183)(__int64); // [rsp+E0h] [rbp-90h]
  __int64 v184; // [rsp+E8h] [rbp-88h]
  __m128i v185; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v186; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int64 v187; // [rsp+110h] [rbp-60h]
  unsigned __int64 v188; // [rsp+118h] [rbp-58h]
  unsigned __int64 v189; // [rsp+120h] [rbp-50h]
  unsigned __int64 v190; // [rsp+128h] [rbp-48h]

  v164 = sub_29C00F0(a3, sub_25D6AA0, &v185);
  v5 = a3;
  v6 = (_QWORD *)a3[4];
  for ( i = v5 + 3; i != v6; v6 = (_QWORD *)v6[1] )
  {
    if ( !v6 )
      BUG();
    v7 = *(v6 - 1);
    if ( v7 )
    {
      v8 = (_QWORD *)sub_22077B0(0x18u);
      if ( v8 )
        *v8 = 0;
      v8[1] = v7;
      v8[2] = v6 - 7;
      sub_24B19A0((unsigned __int64 *)(a2 + 384), 0, v8 + 1, v7, (unsigned __int64)v8);
    }
  }
  v9 = (_QWORD *)a3[2];
  for ( j = a3 + 1; j != v9; v9 = (_QWORD *)v9[1] )
  {
    if ( !v9 )
      BUG();
    v10 = *(v9 - 1);
    if ( v10 )
    {
      v11 = (_QWORD *)sub_22077B0(0x18u);
      if ( v11 )
        *v11 = 0;
      v11[1] = v10;
      v11[2] = v9 - 7;
      sub_24B19A0((unsigned __int64 *)(a2 + 384), 0, v11 + 1, v10, (unsigned __int64)v11);
    }
  }
  v12 = (_QWORD *)a3[6];
  for ( k = a3 + 5; k != v12; v12 = (_QWORD *)v12[1] )
  {
    v13 = 0;
    if ( v12 )
      v13 = (__int64)(v12 - 6);
    v14 = sub_B326A0(v13);
    if ( v14 )
    {
      v15 = (_QWORD *)sub_22077B0(0x18u);
      if ( v15 )
        *v15 = 0;
      v15[1] = v14;
      v15[2] = v13;
      sub_24B19A0((unsigned __int64 *)(a2 + 384), 0, v15 + 1, v14, (unsigned __int64)v15);
    }
  }
  sub_25DA3D0(a2, (__int64)a3);
  v16 = (__int64)a3;
  sub_BA9600(&v185, (__int64)a3);
  v168 = a2;
  v158 = v187;
  v175 = _mm_loadu_si128(&v185);
  v166 = v188;
  v176 = _mm_loadu_si128(&v186);
  v157 = v189;
  v156 = v190;
  while ( *(_OWORD *)&v175 != __PAIR128__(v166, v158) || *(_OWORD *)&v176 != __PAIR128__(v156, v157) )
  {
    v18 = sub_25AC5C0;
    v19 = (unsigned __int64 *)&v177;
    v20 = &v177;
    v178 = 0;
    v177 = sub_25AC5C0;
    v21 = &v175;
    v179 = sub_25AC5E0;
    v180 = 0;
    if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
      goto LABEL_27;
    while ( 1 )
    {
      v18 = *(__int64 (__fastcall **)(__int64))((char *)v18 + v21->m128i_i64[0] - 1);
LABEL_27:
      v22 = v18((__int64)v21);
      v23 = v22;
      if ( v22 )
        break;
      while ( 1 )
      {
        v19 += 2;
        if ( &v181 == (__int64 (__fastcall **)(__int64))v19 )
LABEL_325:
          BUG();
        v24 = v20[3];
        v18 = v20[2];
        v20 = (__int64 (__fastcall **)(__int64))v19;
        v21 = (__m128i *)((char *)&v175 + (_QWORD)v24);
        if ( ((unsigned __int8)v18 & 1) != 0 )
          break;
        v22 = v18((__int64)v21);
        v23 = v22;
        if ( v22 )
          goto LABEL_31;
      }
    }
LABEL_31:
    sub_AD0030(v22);
    if ( !sub_B2FC80(v23) )
    {
      v29 = *(_BYTE *)(v23 + 32) & 0xF;
      v25 = (v29 + 15) & 0xF;
      if ( (unsigned __int8)v25 > 2u && ((v29 + 9) & 0xFu) > 1 )
        sub_25D78D0(v168, v23, 0, v26, v27, v28);
    }
    v16 = v23;
    v30 = (__m128i *)&v181;
    v31 = &v181;
    sub_25D85D0(v168, v16, v25, v26, v27, v28);
    v32 = sub_25AC560;
    v182 = 0;
    v33 = &v175;
    v181 = sub_25AC560;
    v183 = (__int64 (__fastcall *)(__int64))sub_25AC590;
    v184 = 0;
    if ( ((unsigned __int8)sub_25AC560 & 1) != 0 )
LABEL_36:
      v32 = *(__int64 (__fastcall **)(__int64))((char *)v32 + v33->m128i_i64[0] - 1);
    while ( !(unsigned __int8)v32((__int64)v33) )
    {
      if ( &v185 == ++v30 )
        goto LABEL_325;
      v34 = v31[3];
      v32 = v31[2];
      v31 = (__int64 (__fastcall **)(__int64))v30;
      v33 = (__m128i *)((char *)&v175 + (_QWORD)v34);
      if ( ((unsigned __int8)v32 & 1) != 0 )
        goto LABEL_36;
    }
  }
  v35 = v168;
  for ( m = (_QWORD *)a3[6]; k != m; m = (_QWORD *)m[1] )
  {
    if ( !m )
    {
      sub_AD0030(0);
      BUG();
    }
    sub_AD0030((__int64)(m - 6));
    v40 = *(_BYTE *)(m - 2) & 0xF;
    v41 = (v40 + 9) & 0xF;
    if ( (unsigned __int8)v41 > 1u && ((v40 + 15) & 0xFu) > 2 )
      sub_25D78D0(v168, (__int64)(m - 6), 0, v37, v38, v39);
    v16 = (__int64)(m - 6);
    sub_25D85D0(v168, (__int64)(m - 6), v41, v37, v38, v39);
  }
  v42 = (_QWORD *)a3[8];
  for ( n = a3 + 7; a3 + 7 != v42; v42 = (_QWORD *)v42[1] )
  {
    if ( !v42 )
    {
      sub_AD0030(0);
      BUG();
    }
    sub_AD0030((__int64)(v42 - 7));
    v46 = *(_BYTE *)(v42 - 3) & 0xF;
    v47 = (v46 + 15) & 0xF;
    if ( (unsigned __int8)v47 > 2u && ((v46 + 9) & 0xFu) > 1 )
      sub_25D78D0(v168, (__int64)(v42 - 7), 0, v43, v44, v45);
    v16 = (__int64)(v42 - 7);
    sub_25D85D0(v168, (__int64)(v42 - 7), v47, v43, v44, v45);
  }
  v48 = *(__int64 **)(v168 + 16);
  if ( *(_BYTE *)(v168 + 36) )
    v49 = *(unsigned int *)(v168 + 28);
  else
    v49 = *(unsigned int *)(v168 + 24);
  v50 = &v48[v49];
  if ( v48 == v50 )
  {
LABEL_59:
    v185.m128i_i32[3] = 8;
    v170 = v168 + 8;
    v185.m128i_i64[0] = (__int64)&v186;
LABEL_60:
    v185.m128i_i32[2] = 0;
    goto LABEL_61;
  }
  while ( (unsigned __int64)*v48 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( ++v48 == v50 )
      goto LABEL_59;
  }
  v170 = v168 + 8;
  v185.m128i_i64[0] = (__int64)&v186;
  v185.m128i_i64[1] = 0x800000000LL;
  if ( v48 == v50 )
    goto LABEL_60;
  v104 = v48;
  v105 = 0;
  while ( 1 )
  {
    v106 = v104 + 1;
    if ( v50 == v104 + 1 )
      break;
    while ( 1 )
    {
      v104 = v106;
      if ( (unsigned __int64)*v106 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v50 == ++v106 )
        goto LABEL_207;
    }
    ++v105;
    if ( v50 == v106 )
      goto LABEL_208;
  }
LABEL_207:
  ++v105;
LABEL_208:
  v107 = &v186;
  if ( v105 > 8 )
  {
    v16 = (__int64)&v186;
    v159 = v105;
    sub_C8D5F0((__int64)&v185, &v186, v105, 8u, v105, v17);
    LODWORD(v105) = v159;
    v107 = (__m128i *)(v185.m128i_i64[0] + 8LL * v185.m128i_u32[2]);
  }
  v108 = *v48;
  do
  {
    v109 = v48 + 1;
    v107->m128i_i64[0] = v108;
    v107 = (__m128i *)((char *)v107 + 8);
    if ( v50 == v48 + 1 )
      break;
    while ( 1 )
    {
      v108 = *v109;
      v48 = v109;
      if ( (unsigned __int64)*v109 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v50 == ++v109 )
        goto LABEL_214;
    }
  }
  while ( v50 != v109 );
LABEL_214:
  v185.m128i_i32[2] += v105;
  v110 = v185.m128i_u32[2];
  if ( v185.m128i_i32[2] )
  {
    while ( 2 )
    {
      v16 = *(unsigned int *)(v168 + 320);
      v111 = *(_QWORD *)(v185.m128i_i64[0] + 8LL * v110 - 8);
      v185.m128i_i32[2] = v110 - 1;
      if ( (_DWORD)v16 )
      {
        v112 = (unsigned int)(v16 - 1);
        v113 = 1;
        v114 = *(_QWORD *)(v168 + 304);
        v115 = 0;
        LODWORD(v116) = v112 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
        v117 = v114 + 72LL * (unsigned int)v116;
        v118 = *(_QWORD *)v117;
        if ( v111 == *(_QWORD *)v117 )
        {
LABEL_217:
          v119 = *(__int64 **)(v117 + 16);
          v120 = (__int64 *)(v117 + 8);
          if ( !*(_BYTE *)(v117 + 36) )
          {
            v121 = &v119[*(unsigned int *)(v117 + 24)];
            goto LABEL_221;
          }
          goto LABEL_247;
        }
        while ( v118 != -4096 )
        {
          if ( !v115 && v118 == -8192 )
            v115 = (__int64 *)v117;
          v116 = (unsigned int)v112 & ((_DWORD)v116 + v113);
          v117 = v114 + 72 * v116;
          v118 = *(_QWORD *)v117;
          if ( v111 == *(_QWORD *)v117 )
            goto LABEL_217;
          ++v113;
        }
        v124 = *(_DWORD *)(v168 + 312);
        if ( !v115 )
          v115 = (__int64 *)v117;
        ++*(_QWORD *)(v168 + 296);
        v125 = v124 + 1;
        if ( 4 * v125 < (unsigned int)(3 * v16) )
        {
          v117 = (unsigned int)(v16 - *(_DWORD *)(v168 + 316) - v125);
          if ( (unsigned int)v117 <= (unsigned int)v16 >> 3 )
          {
            sub_25D83A0(v168 + 296, v16);
            v129 = *(_DWORD *)(v168 + 320);
            if ( !v129 )
            {
LABEL_323:
              ++*(_DWORD *)(v168 + 312);
              BUG();
            }
            v114 = (unsigned int)(v129 - 1);
            v112 = *(_QWORD *)(v168 + 304);
            v117 = 1;
            v130 = 0;
            v131 = v114 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
            v115 = (__int64 *)(v112 + 72LL * v131);
            v16 = *v115;
            v125 = *(_DWORD *)(v168 + 312) + 1;
            if ( v111 != *v115 )
            {
              while ( v16 != -4096 )
              {
                if ( !v130 && v16 == -8192 )
                  v130 = v115;
                v155 = v117 + 1;
                v117 = (unsigned int)v114 & (v131 + (_DWORD)v117);
                v131 = v117;
                v115 = (__int64 *)(v112 + 72LL * (unsigned int)v117);
                v16 = *v115;
                if ( v111 == *v115 )
                  goto LABEL_244;
                v117 = v155;
              }
              if ( v130 )
                v115 = v130;
            }
          }
          goto LABEL_244;
        }
      }
      else
      {
        ++*(_QWORD *)(v168 + 296);
      }
      v16 = (unsigned int)(2 * v16);
      sub_25D83A0(v168 + 296, v16);
      v126 = *(_DWORD *)(v168 + 320);
      if ( !v126 )
        goto LABEL_323;
      v112 = (unsigned int)(v126 - 1);
      v127 = *(_QWORD *)(v168 + 304);
      v117 = (unsigned int)v112 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
      v115 = (__int64 *)(v127 + 72 * v117);
      v114 = *v115;
      v125 = *(_DWORD *)(v168 + 312) + 1;
      if ( v111 != *v115 )
      {
        v128 = 1;
        v16 = 0;
        while ( v114 != -4096 )
        {
          if ( !v16 && v114 == -8192 )
            v16 = (__int64)v115;
          v117 = (unsigned int)v112 & ((_DWORD)v117 + v128);
          v115 = (__int64 *)(v127 + 72 * v117);
          v114 = *v115;
          if ( v111 == *v115 )
            goto LABEL_244;
          ++v128;
        }
        if ( v16 )
          v115 = (__int64 *)v16;
      }
LABEL_244:
      *(_DWORD *)(v168 + 312) = v125;
      if ( *v115 != -4096 )
        --*(_DWORD *)(v168 + 316);
      v119 = v115 + 5;
      *v115 = v111;
      v120 = v115 + 1;
      *v120 = 0;
      v120[1] = (__int64)v119;
      v120[2] = 4;
      *((_DWORD *)v120 + 6) = 0;
      *((_BYTE *)v120 + 28) = 1;
LABEL_247:
      v121 = &v119[*((unsigned int *)v120 + 5)];
      while ( 1 )
      {
LABEL_221:
        if ( v121 == v119 )
          goto LABEL_222;
        v16 = *v119;
        v122 = v119;
        if ( (unsigned __int64)*v119 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v119;
      }
LABEL_224:
      if ( v121 == v122 || (sub_25D78D0(v168, v16, v185.m128i_i64, v117, v114, v112), v123 = v122 + 1, v122 + 1 == v121) )
      {
LABEL_222:
        v110 = v185.m128i_u32[2];
        if ( !v185.m128i_i32[2] )
          break;
      }
      else
      {
        do
        {
          v16 = *v123;
          v122 = v123;
          if ( (unsigned __int64)*v123 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_224;
          ++v123;
        }
        while ( v121 != v123 );
        v110 = v185.m128i_u32[2];
        if ( !v185.m128i_i32[2] )
          break;
      }
      continue;
    }
  }
LABEL_61:
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v51 = (_QWORD *)a3[2];
  if ( j != v51 )
  {
    while ( 2 )
    {
      v52 = (__int64 (__fastcall *)(__int64))(v51 - 7);
      if ( !v51 )
        v52 = 0;
      if ( *(_BYTE *)(v168 + 36) )
      {
        v53 = *(__int64 (__fastcall ***)(__int64))(v168 + 16);
        v54 = &v53[*(unsigned int *)(v168 + 28)];
        if ( v53 == v54 )
        {
LABEL_159:
          v181 = v52;
          v16 = (__int64)v173;
          if ( v173 == v174 )
          {
            sub_24400C0((__int64)&v172, v173, &v181);
          }
          else
          {
            if ( v173 )
            {
              *(_QWORD *)v173 = v52;
              v16 = (__int64)v173;
            }
            v16 += 8;
            v173 = (_BYTE *)v16;
          }
          if ( !sub_B2FC80((__int64)v52) )
          {
            v16 = 0;
            v161 = (__int64 *)*((_QWORD *)v52 - 4);
            sub_B30160((__int64)v52, 0);
            if ( (unsigned __int8)sub_29DCFA0(v161) )
              sub_ACFDF0(v161, 0, v89);
          }
        }
        else
        {
          while ( v52 != *v53 )
          {
            if ( v54 == ++v53 )
              goto LABEL_159;
          }
        }
      }
      else
      {
        v16 = (__int64)v52;
        if ( !sub_C8CA60(v170, (__int64)v52) )
          goto LABEL_159;
      }
      v51 = (_QWORD *)v51[1];
      if ( a3 + 1 == v51 )
        break;
      continue;
    }
  }
  v175 = 0u;
  v176.m128i_i64[0] = 0;
  v55 = (_QWORD *)a3[4];
  if ( i != v55 )
  {
    while ( 2 )
    {
      v56 = (__int64 (__fastcall *)(__int64))(v55 - 7);
      if ( !v55 )
        v56 = 0;
      if ( *(_BYTE *)(v168 + 36) )
      {
        v57 = *(__int64 (__fastcall ***)(__int64))(v168 + 16);
        v58 = &v57[*(unsigned int *)(v168 + 28)];
        if ( v57 == v58 )
        {
LABEL_151:
          v181 = v56;
          v16 = v175.m128i_i64[1];
          if ( v175.m128i_i64[1] == v176.m128i_i64[0] )
          {
            sub_24147A0((__int64)&v175, (_BYTE *)v175.m128i_i64[1], &v181);
          }
          else
          {
            if ( v175.m128i_i64[1] )
            {
              *(_QWORD *)v175.m128i_i64[1] = v56;
              v16 = v175.m128i_i64[1];
            }
            v16 += 8;
            v175.m128i_i64[1] = v16;
          }
          if ( !sub_B2FC80((__int64)v56) )
          {
            v16 = 0;
            sub_B2CA40((__int64)v56, 0);
            v88 = *((_BYTE *)v56 + 32);
            *((_BYTE *)v56 + 32) = v88 & 0xF0;
            if ( (v88 & 0x30) != 0 )
              *((_BYTE *)v56 + 33) |= 0x40u;
          }
        }
        else
        {
          while ( v56 != *v57 )
          {
            if ( v58 == ++v57 )
              goto LABEL_151;
          }
        }
      }
      else
      {
        v16 = (__int64)v56;
        if ( !sub_C8CA60(v170, (__int64)v56) )
          goto LABEL_151;
      }
      v55 = (_QWORD *)v55[1];
      if ( i == v55 )
        break;
      continue;
    }
  }
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v59 = (_QWORD *)a3[6];
  if ( k != v59 )
  {
    while ( 2 )
    {
      v60 = (__int64 (__fastcall *)(__int64))(v59 - 6);
      if ( !v59 )
        v60 = 0;
      if ( *(_BYTE *)(v168 + 36) )
      {
        v61 = *(__int64 (__fastcall ***)(__int64))(v168 + 16);
        v62 = &v61[*(unsigned int *)(v168 + 28)];
        if ( v61 == v62 )
        {
LABEL_145:
          v181 = v60;
          v87 = v178;
          if ( v178 == (char *)v179 )
          {
            sub_25D8080((__int64)&v177, v178, &v181);
          }
          else
          {
            if ( v178 )
            {
              *(_QWORD *)v178 = v60;
              v87 = v178;
            }
            v178 = v87 + 8;
          }
          v16 = 0;
          sub_B303B0((__int64)v60, 0);
        }
        else
        {
          while ( v60 != *v61 )
          {
            if ( v62 == ++v61 )
              goto LABEL_145;
          }
        }
      }
      else
      {
        v16 = (__int64)v60;
        if ( !sub_C8CA60(v170, (__int64)v60) )
          goto LABEL_145;
      }
      v59 = (_QWORD *)v59[1];
      if ( k == v59 )
        break;
      continue;
    }
  }
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v63 = (_QWORD *)a3[8];
  if ( n != v63 )
  {
    while ( 2 )
    {
      v64 = (__int64)(v63 - 7);
      if ( !v63 )
        v64 = 0;
      if ( *(_BYTE *)(v168 + 36) )
      {
        v65 = *(_QWORD **)(v168 + 16);
        v66 = &v65[*(unsigned int *)(v168 + 28)];
        if ( v65 == v66 )
        {
LABEL_136:
          v171 = v64;
          v16 = (__int64)v182;
          if ( v182 == v183 )
          {
            sub_25D8210((__int64)&v181, v182, &v171);
          }
          else
          {
            if ( v182 )
            {
              *(_QWORD *)v182 = v64;
              v16 = (__int64)v182;
            }
            v16 += 8;
            v182 = (__int64 (__fastcall *)(__int64))v16;
          }
          if ( *(_QWORD *)(v64 - 32) )
          {
            v86 = *(_QWORD *)(v64 - 24);
            **(_QWORD **)(v64 - 16) = v86;
            if ( v86 )
              *(_QWORD *)(v86 + 16) = *(_QWORD *)(v64 - 16);
          }
          *(_QWORD *)(v64 - 32) = 0;
        }
        else
        {
          while ( v64 != *v65 )
          {
            if ( v66 == ++v65 )
              goto LABEL_136;
          }
        }
      }
      else
      {
        v16 = v64;
        if ( !sub_C8CA60(v170, v64) )
          goto LABEL_136;
      }
      v63 = (_QWORD *)v63[1];
      if ( n == v63 )
        break;
      continue;
    }
  }
  v67 = (__int64 *)v175.m128i_i64[1];
  v68 = (__int64 *)v175.m128i_i64[0];
  if ( v175.m128i_i64[1] != v175.m128i_i64[0] )
  {
    do
    {
      v69 = *v68;
      if ( *(_QWORD *)(*v68 + 16) )
      {
        sub_E02AA0(*v68);
        v16 = sub_AC9EC0(*(__int64 ***)(v69 + 8));
        sub_BD84E0(v69, v16);
      }
      ++v68;
      sub_AD0030(v69);
      sub_B30810((_QWORD *)v69);
    }
    while ( v67 != v68 );
    v164 = 1;
  }
  v70 = v173;
  v71 = (_QWORD **)v172;
  if ( v173 != v172 )
  {
    do
    {
      v72 = *v71++;
      sub_AD0030((__int64)v72);
      sub_B30810(v72);
    }
    while ( v70 != (_BYTE *)v71 );
    v164 = 1;
  }
  v73 = (__int64 (__fastcall *)(__int64))v178;
  v74 = v177;
  if ( v178 != (char *)v177 )
  {
    do
    {
      v75 = *(_QWORD **)v74;
      v74 = (__int64 (__fastcall *)(__int64))((char *)v74 + 8);
      sub_AD0030((__int64)v75);
      sub_B30810(v75);
    }
    while ( v73 != v74 );
    v164 = 1;
  }
  v76 = v182;
  v77 = v181;
  if ( v182 != v181 )
  {
    do
    {
      v78 = *(_QWORD **)v77;
      v77 = (__int64 (__fastcall *)(__int64))((char *)v77 + 8);
      sub_AD0030((__int64)v78);
      sub_B30810(v78);
    }
    while ( v76 != v77 );
    v164 = 1;
  }
  ++*(_QWORD *)(v168 + 8);
  if ( *(_BYTE *)(v168 + 36) )
  {
LABEL_116:
    *(_QWORD *)(v168 + 28) = 0;
  }
  else
  {
    v79 = 4 * (*(_DWORD *)(v168 + 28) - *(_DWORD *)(v168 + 32));
    v80 = *(unsigned int *)(v168 + 24);
    if ( v79 < 0x20 )
      v79 = 32;
    if ( v79 >= (unsigned int)v80 )
    {
      memset(*(void **)(v168 + 16), -1, 8 * v80);
      goto LABEL_116;
    }
    sub_C8C990(v170, v16);
  }
  sub_25D7B50(v168 + 328);
  v81 = *(_DWORD *)(v168 + 312);
  ++*(_QWORD *)(v168 + 296);
  if ( v81 || *(_DWORD *)(v168 + 316) )
  {
    v82 = *(_QWORD *)(v168 + 304);
    v83 = 4 * v81;
    v84 = 72LL * *(unsigned int *)(v168 + 320);
    if ( (unsigned int)(4 * v81) < 0x40 )
      v83 = 64;
    v85 = v82 + v84;
    if ( v83 < *(_DWORD *)(v168 + 320) )
    {
      do
      {
        if ( *(_QWORD *)v82 != -8192 && *(_QWORD *)v82 != -4096 && !*(_BYTE *)(v82 + 36) )
          _libc_free(*(_QWORD *)(v82 + 16));
        v82 += 72;
      }
      while ( v85 != v82 );
      v132 = *(_DWORD *)(v168 + 320);
      if ( v81 )
      {
        v133 = 64;
        if ( v81 != 1 )
        {
          _BitScanReverse(&v134, v81 - 1);
          v133 = (unsigned int)(1 << (33 - (v134 ^ 0x1F)));
          if ( (int)v133 < 64 )
            v133 = 64;
        }
        v135 = *(_QWORD **)(v168 + 304);
        if ( (_DWORD)v133 == v132 )
        {
          *(_QWORD *)(v168 + 312) = 0;
          v153 = &v135[9 * v133];
          do
          {
            if ( v135 )
              *v135 = -4096;
            v135 += 9;
          }
          while ( v153 != v135 );
        }
        else
        {
          sub_C7D6A0((__int64)v135, v84, 8);
          v136 = ((((((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v133 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v133 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v133 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v133 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 16;
          v137 = (v136
                | (((((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v133 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v133 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v133 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v133 / 3u + 1) | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v133 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v133 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(v168 + 320) = v137;
          v138 = (_QWORD *)sub_C7D670(72 * v137, 8);
          v139 = *(unsigned int *)(v168 + 320);
          *(_QWORD *)(v168 + 312) = 0;
          *(_QWORD *)(v168 + 304) = v138;
          for ( ii = &v138[9 * v139]; ii != v138; v138 += 9 )
          {
            if ( v138 )
              *v138 = -4096;
          }
        }
      }
      else
      {
        if ( !v132 )
          goto LABEL_166;
        sub_C7D6A0(*(_QWORD *)(v168 + 304), v84, 8);
        *(_QWORD *)(v168 + 304) = 0;
        *(_QWORD *)(v168 + 312) = 0;
        *(_DWORD *)(v168 + 320) = 0;
      }
    }
    else
    {
      for ( ; v82 != v85; v82 += 72 )
      {
        if ( *(_QWORD *)v82 != -4096 )
        {
          if ( *(_QWORD *)v82 != -8192 && !*(_BYTE *)(v82 + 36) )
            _libc_free(*(_QWORD *)(v82 + 16));
          *(_QWORD *)v82 = -4096;
        }
      }
LABEL_166:
      *(_QWORD *)(v168 + 312) = 0;
    }
  }
  v90 = *(_QWORD **)(v168 + 400);
  while ( v90 )
  {
    v91 = (unsigned __int64)v90;
    v90 = (_QWORD *)*v90;
    j_j___libc_free_0(v91);
  }
  v92 = 0;
  memset(*(void **)(v168 + 384), 0, 8LL * *(_QWORD *)(v168 + 392));
  v93 = *(_DWORD *)(v168 + 456);
  ++*(_QWORD *)(v168 + 440);
  *(_QWORD *)(v168 + 408) = 0;
  *(_QWORD *)(v168 + 400) = 0;
  if ( v93 || *(_DWORD *)(v168 + 460) )
  {
    v94 = *(_QWORD **)(v168 + 448);
    v95 = 4 * v93;
    v96 = 136LL * *(unsigned int *)(v168 + 464);
    if ( (unsigned int)(4 * v93) < 0x40 )
      v95 = 64;
    v97 = &v94[(unsigned __int64)v96 / 8];
    if ( v95 >= *(_DWORD *)(v168 + 464) )
    {
      while ( v94 != v97 )
      {
        if ( *v94 != -4096 )
        {
          if ( *v94 != -8192 )
          {
            sub_25D6BE0(v94[13]);
            v98 = v94[1];
            if ( (_QWORD *)v98 != v94 + 3 )
              _libc_free(v98);
          }
          *v94 = -4096;
        }
        v94 += 17;
      }
    }
    else
    {
      do
      {
        if ( *v94 != -8192 && *v94 != -4096 )
        {
          v141 = v94[13];
          while ( v141 )
          {
            sub_25D6BE0(*(_QWORD *)(v141 + 24));
            v142 = v141;
            v141 = *(_QWORD *)(v141 + 16);
            v92 = 48;
            j_j___libc_free_0(v142);
          }
          v143 = v94[1];
          if ( (_QWORD *)v143 != v94 + 3 )
            _libc_free(v143);
        }
        v94 += 17;
      }
      while ( v94 != v97 );
      v35 = v168;
      v144 = *(unsigned int *)(v168 + 464);
      if ( v93 )
      {
        v145 = 64;
        if ( v93 != 1 )
        {
          _BitScanReverse(&v146, v93 - 1);
          v145 = 1 << (33 - (v146 ^ 0x1F));
          if ( v145 < 64 )
            v145 = 64;
        }
        v147 = *(_QWORD **)(v168 + 448);
        if ( (_DWORD)v144 == v145 )
        {
          *(_QWORD *)(v168 + 456) = 0;
          v154 = &v147[17 * v144];
          do
          {
            if ( v147 )
              *v147 = -4096;
            v147 += 17;
          }
          while ( v154 != v147 );
        }
        else
        {
          sub_C7D6A0((__int64)v147, v96, 8);
          v92 = 8;
          v148 = ((((((((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                    | (4 * v145 / 3u + 1)
                    | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                  | (4 * v145 / 3u + 1)
                  | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                  | (4 * v145 / 3u + 1)
                  | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 4)
                | (((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                | (4 * v145 / 3u + 1)
                | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 16;
          v149 = (v148
                | (((((((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                    | (4 * v145 / 3u + 1)
                    | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                  | (4 * v145 / 3u + 1)
                  | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                  | (4 * v145 / 3u + 1)
                  | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 4)
                | (((4 * v145 / 3u + 1) | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1)) >> 2)
                | (4 * v145 / 3u + 1)
                | ((unsigned __int64)(4 * v145 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(v168 + 464) = v149;
          v150 = (_QWORD *)sub_C7D670(136 * v149, 8);
          v151 = *(unsigned int *)(v168 + 464);
          *(_QWORD *)(v168 + 456) = 0;
          *(_QWORD *)(v168 + 448) = v150;
          for ( jj = &v150[17 * v151]; jj != v150; v150 += 17 )
          {
            if ( v150 )
              *v150 = -4096;
          }
        }
        goto LABEL_183;
      }
      if ( (_DWORD)v144 )
      {
        v92 = v96;
        sub_C7D6A0(*(_QWORD *)(v168 + 448), v96, 8);
        *(_QWORD *)(v168 + 448) = 0;
        *(_QWORD *)(v168 + 456) = 0;
        *(_DWORD *)(v168 + 464) = 0;
        goto LABEL_183;
      }
    }
    *(_QWORD *)(v35 + 456) = 0;
  }
LABEL_183:
  ++*(_QWORD *)(v35 + 472);
  if ( *(_BYTE *)(v35 + 500) )
  {
LABEL_188:
    *(_QWORD *)(v35 + 492) = 0;
  }
  else
  {
    v99 = 4 * (*(_DWORD *)(v35 + 492) - *(_DWORD *)(v35 + 496));
    v100 = *(unsigned int *)(v35 + 488);
    if ( v99 < 0x20 )
      v99 = 32;
    if ( (unsigned int)v100 <= v99 )
    {
      memset(*(void **)(v35 + 480), -1, 8 * v100);
      goto LABEL_188;
    }
    sub_C8C990(v35 + 472, v92);
  }
  v101 = a1 + 4;
  v102 = a1 + 10;
  if ( v164 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v101;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v102;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[2] = 0x100000002LL;
    a1[1] = v101;
    a1[6] = 0;
    a1[7] = v102;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    a1[4] = &qword_4F82400;
  }
  if ( v181 )
    j_j___libc_free_0((unsigned __int64)v181);
  if ( v177 )
    j_j___libc_free_0((unsigned __int64)v177);
  if ( v175.m128i_i64[0] )
    j_j___libc_free_0(v175.m128i_u64[0]);
  if ( v172 )
    j_j___libc_free_0((unsigned __int64)v172);
  if ( (__m128i *)v185.m128i_i64[0] != &v186 )
    _libc_free(v185.m128i_u64[0]);
  return a1;
}
