// Function: sub_35D98D0
// Address: 0x35d98d0
//
__int64 __fastcall sub_35D98D0(_QWORD *a1, __int64 *a2, unsigned int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 v7; // r12
  unsigned int v8; // r13d
  __int64 *v9; // r15
  __int64 v10; // r12
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned int v13; // ecx
  bool v14; // dl
  unsigned int v15; // r13d
  __int64 *v16; // rbx
  __int64 *v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int32 v21; // edx
  unsigned __int64 v23; // r15
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int); // rax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  __int8 v30; // cl
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // r13
  bool v34; // al
  unsigned __int32 v35; // edx
  __int64 v36; // rsi
  char *v37; // r12
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int32 v40; // r14d
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // rdx
  __int32 v44; // r13d
  __int64 v45; // r14
  __int64 v46; // rdi
  char *v47; // r15
  __m128i *v48; // r8
  __int64 v49; // r9
  __int64 i; // rax
  __int64 v51; // r13
  unsigned __int64 v52; // rsi
  __int64 v53; // r10
  __int64 v54; // rax
  unsigned int v55; // ecx
  char *v56; // r14
  __int64 v57; // rsi
  unsigned __int32 v58; // eax
  __int64 v59; // r15
  __int64 v60; // rsi
  unsigned __int32 v61; // eax
  __int32 v62; // r8d
  int v63; // eax
  __int64 v64; // rcx
  int v65; // eax
  int v66; // eax
  __int64 v67; // rcx
  unsigned int v68; // eax
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // r13
  unsigned __int64 v72; // rax
  __int64 *v73; // r13
  __int64 *v74; // rdi
  __int64 *v75; // r10
  __m128i *v76; // r15
  __m128i *v77; // r13
  __int64 v78; // rsi
  __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // r8
  __int64 v82; // rdx
  __int64 v83; // r9
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  __m128i *v86; // rcx
  __m128i *v87; // rax
  __int64 v88; // r8
  __int64 v89; // r14
  unsigned int v90; // r15d
  unsigned int v91; // eax
  unsigned __int64 v92; // r13
  unsigned __int64 v93; // r9
  int v94; // r12d
  char *v95; // rdi
  __int64 v96; // rdx
  bool v97; // al
  __int64 v98; // r8
  __int64 v99; // rax
  __int64 v100; // r13
  int v101; // eax
  __int64 *v102; // rsi
  unsigned __int32 v103; // edx
  unsigned __int64 v104; // r13
  __int64 v105; // rsi
  unsigned __int64 v106; // rdi
  unsigned __int64 v107; // rax
  bool v108; // cf
  unsigned __int64 v109; // rax
  unsigned __int64 v110; // r15
  __int64 v111; // rax
  __int64 v112; // r15
  __int64 *v113; // rcx
  unsigned int v114; // r11d
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // r9
  unsigned __int64 v118; // r10
  __int32 v119; // esi
  __int64 v120; // r15
  unsigned __int64 v121; // r12
  unsigned int v122; // eax
  __int64 v123; // rax
  unsigned int v124; // eax
  unsigned int v125; // eax
  unsigned __int64 v126; // r12
  unsigned __int64 v127; // rdi
  unsigned __int64 v128; // rdi
  char v129; // [rsp+17h] [rbp-239h]
  int v130; // [rsp+18h] [rbp-238h]
  unsigned __int64 v131; // [rsp+18h] [rbp-238h]
  __int64 v132; // [rsp+20h] [rbp-230h]
  __int64 v133; // [rsp+28h] [rbp-228h]
  unsigned int v136; // [rsp+38h] [rbp-218h]
  __int64 v137; // [rsp+40h] [rbp-210h]
  unsigned __int64 v138; // [rsp+40h] [rbp-210h]
  __int64 v139; // [rsp+40h] [rbp-210h]
  unsigned int v140; // [rsp+48h] [rbp-208h]
  unsigned int v141; // [rsp+48h] [rbp-208h]
  __int64 v143; // [rsp+58h] [rbp-1F8h]
  __int64 v144; // [rsp+58h] [rbp-1F8h]
  unsigned __int64 *v146; // [rsp+68h] [rbp-1E8h]
  __int64 v147; // [rsp+68h] [rbp-1E8h]
  char *v148; // [rsp+68h] [rbp-1E8h]
  unsigned __int32 v149; // [rsp+68h] [rbp-1E8h]
  __int64 v150; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 v151; // [rsp+68h] [rbp-1E8h]
  unsigned int v152; // [rsp+68h] [rbp-1E8h]
  unsigned int v153; // [rsp+68h] [rbp-1E8h]
  __int64 v154; // [rsp+68h] [rbp-1E8h]
  __int64 v155; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 v157; // [rsp+70h] [rbp-1E0h]
  unsigned __int64 v158; // [rsp+70h] [rbp-1E0h]
  unsigned int v159; // [rsp+70h] [rbp-1E0h]
  unsigned int v160; // [rsp+78h] [rbp-1D8h]
  char *v161; // [rsp+78h] [rbp-1D8h]
  __int64 v162; // [rsp+78h] [rbp-1D8h]
  unsigned int v163; // [rsp+78h] [rbp-1D8h]
  __int64 v164; // [rsp+78h] [rbp-1D8h]
  __int64 v165; // [rsp+78h] [rbp-1D8h]
  unsigned __int64 v166; // [rsp+80h] [rbp-1D0h]
  unsigned __int32 v167; // [rsp+80h] [rbp-1D0h]
  bool v168; // [rsp+80h] [rbp-1D0h]
  unsigned __int64 v169; // [rsp+80h] [rbp-1D0h]
  unsigned int v170; // [rsp+88h] [rbp-1C8h]
  unsigned __int64 v171; // [rsp+90h] [rbp-1C0h] BYREF
  unsigned int v172; // [rsp+98h] [rbp-1B8h]
  unsigned __int64 v173; // [rsp+A0h] [rbp-1B0h] BYREF
  unsigned __int32 v174; // [rsp+A8h] [rbp-1A8h]
  __int64 v175; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned __int32 v176; // [rsp+B8h] [rbp-198h]
  __int64 v177; // [rsp+C0h] [rbp-190h] BYREF
  unsigned int v178; // [rsp+C8h] [rbp-188h]
  void *src; // [rsp+D0h] [rbp-180h] BYREF
  __m128i *v180; // [rsp+D8h] [rbp-178h]
  __m128i *v181; // [rsp+E0h] [rbp-170h]
  void *v182; // [rsp+F0h] [rbp-160h] BYREF
  __int64 v183; // [rsp+F8h] [rbp-158h]
  _BYTE v184[48]; // [rsp+100h] [rbp-150h] BYREF
  int v185; // [rsp+130h] [rbp-120h]
  char *v186; // [rsp+140h] [rbp-110h] BYREF
  __int64 v187; // [rsp+148h] [rbp-108h]
  _BYTE v188[96]; // [rsp+150h] [rbp-100h] BYREF
  __m128i v189; // [rsp+1B0h] [rbp-A0h] BYREF
  __int64 v190; // [rsp+1C0h] [rbp-90h] BYREF
  int v191; // [rsp+1C8h] [rbp-88h]

  v6 = a2;
  v7 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1[13] + 8LL) + 104LL) - *(_QWORD *)(*(_QWORD *)(a1[13] + 8LL) + 96LL)) >> 3;
  v182 = v184;
  v8 = (unsigned int)(v7 + 63) >> 6;
  v183 = 0x600000000LL;
  if ( v8 > 6 )
  {
    sub_C8D5F0((__int64)&v182, v184, v8, 8u, a5, a6);
    memset(v182, 0, 8LL * v8);
    LODWORD(v183) = (unsigned int)(v7 + 63) >> 6;
    v9 = (__int64 *)v182;
  }
  else
  {
    if ( v8 && 8LL * v8 )
      memset(v184, 0, 8LL * v8);
    LODWORD(v183) = (unsigned int)(v7 + 63) >> 6;
    v9 = (__int64 *)v184;
  }
  v185 = v7;
  v10 = *a2;
  if ( a3 > a4 )
  {
    v11 = 0;
    v137 = a3;
  }
  else
  {
    v11 = 0;
    v137 = a3;
    v12 = 40LL * a3;
    do
    {
      v13 = *(_DWORD *)(*(_QWORD *)(v10 + v12 + 24) + 24LL);
      v9[v13 >> 6] |= 1LL << v13;
      v10 = *a2;
      v9 = (__int64 *)v182;
      v14 = *(_QWORD *)(*a2 + v12 + 8) != *(_QWORD *)(*a2 + v12 + 16);
      v12 += 40;
      v11 += v14 + 1;
    }
    while ( 8 * (5LL * a4 + 5) != v12 );
  }
  v15 = 0;
  if ( &v9[(unsigned int)v183] != v9 )
  {
    v16 = v9;
    v17 = &v9[(unsigned int)v183];
    do
    {
      v18 = *v16++;
      v15 += sub_39FAC40(v18);
    }
    while ( v17 != v16 );
    v6 = a2;
  }
  v133 = 40 * v137;
  v19 = *(_QWORD *)(v10 + 40 * v137 + 8);
  v172 = *(_DWORD *)(v19 + 32);
  if ( v172 > 0x40 )
  {
    sub_C43780((__int64)&v171, (const void **)(v19 + 24));
    v10 = *v6;
  }
  else
  {
    v171 = *(_QWORD *)(v19 + 24);
  }
  v132 = 40LL * a4;
  v20 = *(_QWORD *)(v10 + v132 + 16);
  v174 = *(_DWORD *)(v20 + 32);
  if ( v174 > 0x40 )
    sub_C43780((__int64)&v173, (const void **)(v20 + 24));
  else
    v173 = *(_QWORD *)(v20 + 24);
  v166 = (unsigned int)sub_AE2980(a1[12], 0)[3];
  v189.m128i_i32[2] = v174;
  if ( v174 > 0x40 )
    sub_C43780((__int64)&v189, (const void **)&v173);
  else
    v189.m128i_i64[0] = v173;
  sub_C46B40((__int64)&v189, (__int64 *)&v171);
  v21 = v189.m128i_u32[2];
  v189.m128i_i32[2] = 0;
  LODWORD(v187) = v21;
  v186 = (char *)v189.m128i_i64[0];
  if ( v21 > 0x40 )
  {
    v146 = (unsigned __int64 *)v189.m128i_i64[0];
    if ( v21 - (unsigned int)sub_C444A0((__int64)&v186) > 0x40 )
    {
      v23 = -1;
    }
    else
    {
      v23 = *v146;
      if ( *v146 == -1 )
      {
        j_j___libc_free_0_0((unsigned __int64)v146);
        if ( v189.m128i_i32[2] <= 0x40u )
          goto LABEL_19;
LABEL_34:
        if ( v189.m128i_i64[0] )
          j_j___libc_free_0_0(v189.m128i_u64[0]);
        goto LABEL_36;
      }
      ++v23;
    }
    if ( !v146 )
      goto LABEL_36;
    j_j___libc_free_0_0((unsigned __int64)v146);
    if ( v189.m128i_i32[2] <= 0x40u )
      goto LABEL_36;
    goto LABEL_34;
  }
  if ( v189.m128i_i64[0] == -1 )
  {
LABEL_19:
    v15 = 0;
    goto LABEL_20;
  }
  v23 = v189.m128i_i64[0] + 1;
LABEL_36:
  if ( v166 < v23 )
    goto LABEL_19;
  if ( (v15 != 1 || v11 <= 2) && (v15 != 2 || v11 <= 4) )
  {
    LOBYTE(v15) = v11 > 5 && v15 == 3;
    if ( !(_BYTE)v15 )
      goto LABEL_20;
  }
  v176 = 1;
  v175 = 0;
  v24 = a1[10];
  v25 = a1[12];
  v178 = 1;
  v177 = 0;
  v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v24 + 32LL);
  if ( v26 == sub_2D42F30 )
  {
    v27 = sub_AE2980(v25, 0)[1];
    switch ( v27 )
    {
      case 1:
        v28 = 2;
        break;
      case 2:
        v28 = 3;
        break;
      case 4:
        v28 = 4;
        break;
      case 8:
        v28 = 5;
        break;
      case 16:
        v28 = 6;
        break;
      case 32:
        v28 = 7;
        break;
      case 64:
        v28 = 8;
        break;
      case 128:
        v28 = 9;
        break;
      default:
LABEL_263:
        BUG();
    }
  }
  else
  {
    v28 = (unsigned __int16)v26(v24, v25, 0);
    if ( (unsigned __int16)v28 <= 1u || (unsigned __int16)(v28 - 504) <= 7u )
      goto LABEL_263;
  }
  v29 = 16LL * (v28 - 1);
  v30 = byte_444C4A0[v29 + 8];
  v31 = *(_QWORD *)&byte_444C4A0[v29];
  v189.m128i_i8[8] = v30;
  v189.m128i_i64[0] = v31;
  v130 = sub_CA1930(&v189);
  v32 = a3 + 1;
  v160 = a3 + 1;
  if ( v32 > a4 )
  {
LABEL_76:
    v129 = 1;
  }
  else
  {
    v33 = 8 * (5 * v32 - 5);
    while ( 1 )
    {
      v36 = *(_QWORD *)(*v6 + v33 + 16);
      LODWORD(v187) = *(_DWORD *)(v36 + 32);
      if ( (unsigned int)v187 > 0x40 )
        sub_C43780((__int64)&v186, (const void **)(v36 + 24));
      else
        v186 = *(char **)(v36 + 24);
      v33 += 40;
      sub_C46A40((__int64)&v186, 1);
      v35 = v187;
      v37 = v186;
      LODWORD(v187) = 0;
      v38 = *v6;
      v189.m128i_i32[2] = v35;
      v189.m128i_i64[0] = (__int64)v186;
      v39 = *(_QWORD *)(v38 + v33 + 8);
      if ( *(_DWORD *)(v39 + 32) > 0x40u )
      {
        v167 = v35;
        v34 = sub_C43C50(v39 + 24, (const void **)&v189);
        v35 = v167;
      }
      else
      {
        v34 = *(_QWORD *)(v39 + 24) == (_QWORD)v186;
      }
      if ( v35 > 0x40 )
      {
        if ( v37 )
        {
          v168 = v34;
          j_j___libc_free_0_0((unsigned __int64)v37);
          v34 = v168;
          if ( (unsigned int)v187 > 0x40 )
          {
            if ( v186 )
            {
              j_j___libc_free_0_0((unsigned __int64)v186);
              v34 = v168;
            }
          }
        }
      }
      if ( !v34 )
        break;
      if ( ++v32 > a4 )
        goto LABEL_76;
    }
    v129 = 0;
  }
  v40 = v172;
  v41 = v171;
  v42 = 1LL << ((unsigned __int8)v172 - 1);
  if ( v172 > 0x40 )
  {
    if ( (*(_QWORD *)(v171 + 8LL * ((v172 - 1) >> 6)) & v42) != 0 )
      goto LABEL_81;
    v97 = v40 == (unsigned int)sub_C444A0((__int64)&v171);
  }
  else
  {
    if ( (v42 & v171) != 0 )
    {
LABEL_79:
      if ( v176 <= 0x40 && v40 <= 0x40 )
      {
        v175 = v41;
        v176 = v40;
LABEL_82:
        v189.m128i_i32[2] = v174;
        if ( v174 > 0x40 )
          sub_C43780((__int64)&v189, (const void **)&v173);
        else
          v189.m128i_i64[0] = v173;
        sub_C46B40((__int64)&v189, (__int64 *)&v171);
        v44 = v189.m128i_i32[2];
        v189.m128i_i32[2] = 0;
        v45 = v189.m128i_i64[0];
        if ( v178 > 0x40 && v177 )
        {
          j_j___libc_free_0_0(v177);
          v177 = v45;
          v178 = v44;
          if ( v189.m128i_i32[2] > 0x40u && v189.m128i_i64[0] )
            j_j___libc_free_0_0(v189.m128i_u64[0]);
        }
        else
        {
          v177 = v189.m128i_i64[0];
          v178 = v44;
        }
        goto LABEL_89;
      }
LABEL_81:
      sub_C43990((__int64)&v175, (__int64)&v171);
      goto LABEL_82;
    }
    v97 = v171 == 0;
  }
  if ( v97 )
    goto LABEL_79;
  v98 = v130;
  if ( v174 > 0x40 )
  {
    v149 = v174;
    v169 = v173;
    v100 = *(_QWORD *)(v173 + 8LL * ((v174 - 1) >> 6)) & (1LL << ((unsigned __int8)v174 - 1));
    if ( v100 )
    {
      v101 = sub_C44500((__int64)&v173);
      v98 = v130;
      v102 = (__int64 *)v169;
      v103 = v149;
    }
    else
    {
      v101 = sub_C444A0((__int64)&v173);
      v103 = v149;
      v102 = (__int64 *)v169;
      v98 = v130;
    }
    if ( v103 + 1 - v101 > 0x40 )
    {
      if ( !v100 )
        goto LABEL_79;
      goto LABEL_163;
    }
    v99 = *v102;
  }
  else
  {
    v99 = 0;
    if ( v174 )
      v99 = (__int64)(v173 << (64 - (unsigned __int8)v174)) >> (64 - (unsigned __int8)v174);
  }
  if ( v98 <= v99 )
    goto LABEL_79;
LABEL_163:
  v189.m128i_i32[2] = v40;
  if ( v40 > 0x40 )
    sub_C43690((__int64)&v189, 0, 0);
  else
    v189.m128i_i64[0] = 0;
  if ( v176 > 0x40 && v175 )
    j_j___libc_free_0_0(v175);
  v175 = v189.m128i_i64[0];
  v176 = v189.m128i_u32[2];
  if ( v178 <= 0x40 && v174 <= 0x40 )
  {
    v43 = v173;
    v178 = v174;
    v129 = 0;
    v177 = v173;
  }
  else
  {
    sub_C43990((__int64)&v177, (__int64)&v173);
    v129 = 0;
  }
LABEL_89:
  src = 0;
  v180 = 0;
  v181 = 0;
  if ( a3 > a4 )
  {
    v170 = 0;
    v186 = v188;
    v187 = 0x300000000LL;
  }
  else
  {
    v46 = *v6;
    v47 = 0;
    v48 = 0;
    v170 = 0;
    v49 = *v6;
    for ( i = v137; ; i = v160++ )
    {
      v51 = 40 * i;
      v52 = 0xAAAAAAAAAAAAAAABLL * (((char *)v48 - v47) >> 3);
      if ( v52 )
      {
        v53 = v46 + v51;
        v54 = 0;
        v55 = 0;
        while ( 1 )
        {
          v56 = &v47[24 * v54];
          if ( *((_QWORD *)v56 + 1) == *(_QWORD *)(v46 + v51 + 24) )
            break;
          v54 = ++v55;
          if ( v55 >= v52 )
          {
            v69 = 24LL * v55;
            if ( v55 == v52 )
              goto LABEL_118;
            v56 = &v47[24 * v55];
            break;
          }
        }
      }
      else
      {
        v69 = 0;
LABEL_118:
        v70 = *(_QWORD *)(v46 + v51 + 24);
        v189.m128i_i64[0] = 0;
        v190 = 0;
        v189.m128i_i64[1] = v70;
        if ( v48 == v181 )
        {
          sub_35D9720((unsigned __int64 *)&src, v48, &v189);
          v47 = (char *)src;
          v49 = *v6;
        }
        else
        {
          if ( v48 )
          {
            *v48 = _mm_loadu_si128(&v189);
            v47 = (char *)src;
            v48[1].m128i_i64[0] = v190;
            v48 = v180;
            v49 = *v6;
          }
          v180 = (__m128i *)((char *)v48 + 24);
        }
        v56 = &v47[v69];
        v53 = v49 + v51;
      }
      v57 = *(_QWORD *)(v53 + 8);
      v189.m128i_i32[2] = *(_DWORD *)(v57 + 32);
      if ( v189.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v189, (const void **)(v57 + 24));
      else
        v189.m128i_i64[0] = *(_QWORD *)(v57 + 24);
      sub_C46B40((__int64)&v189, &v175);
      v58 = v189.m128i_u32[2];
      LODWORD(v59) = v189.m128i_i32[0];
      v189.m128i_i32[2] = 0;
      if ( v58 > 0x40 )
      {
        v59 = *(_QWORD *)v189.m128i_i64[0];
        j_j___libc_free_0_0(v189.m128i_u64[0]);
        if ( v189.m128i_i32[2] > 0x40u )
        {
          if ( v189.m128i_i64[0] )
            j_j___libc_free_0_0(v189.m128i_u64[0]);
        }
      }
      v60 = *(_QWORD *)(*v6 + v51 + 16);
      v189.m128i_i32[2] = *(_DWORD *)(v60 + 32);
      if ( v189.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v189, (const void **)(v60 + 24));
      else
        v189.m128i_i64[0] = *(_QWORD *)(v60 + 24);
      sub_C46B40((__int64)&v189, &v175);
      v61 = v189.m128i_u32[2];
      v62 = v189.m128i_i32[0];
      v189.m128i_i32[2] = 0;
      if ( v61 > 0x40 )
      {
        v147 = *(_QWORD *)v189.m128i_i64[0];
        j_j___libc_free_0_0(v189.m128i_u64[0]);
        v62 = v147;
        if ( v189.m128i_i32[2] > 0x40u )
        {
          if ( v189.m128i_i64[0] )
          {
            j_j___libc_free_0_0(v189.m128i_u64[0]);
            v62 = v147;
          }
        }
      }
      v63 = *((_DWORD *)v56 + 4) + 1 - v59;
      *(_QWORD *)v56 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v59 + 63 - v62) << v59;
      v64 = *((unsigned int *)v56 + 5);
      v65 = v62 + v63;
      v48 = v180;
      v47 = (char *)src;
      *((_DWORD *)v56 + 4) = v65;
      v66 = *((_DWORD *)v56 + 5) + *(_DWORD *)(*v6 + v51 + 32);
      if ( (unsigned __int64)*(unsigned int *)(*v6 + v51 + 32) + v64 > 0x80000000 )
        v66 = 0x80000000;
      *((_DWORD *)v56 + 5) = v66;
      v46 = *v6;
      v67 = *(unsigned int *)(*v6 + v51 + 32);
      v49 = *v6;
      v68 = v170 + *(_DWORD *)(*v6 + v51 + 32);
      v43 = v160;
      if ( v67 + (unsigned __int64)v170 > 0x80000000 )
        v68 = 0x80000000;
      v170 = v68;
      if ( a4 < v160 )
        break;
    }
    v186 = v188;
    v187 = 0x300000000LL;
    if ( v47 != (char *)v48 )
    {
      v161 = (char *)v48;
      v71 = (char *)v48 - v47;
      _BitScanReverse64(&v72, 0xAAAAAAAAAAAAAAABLL * (((char *)v48 - v47) >> 3));
      sub_35D8050((__int64)v47, v48, 2LL * (int)(63 - (v72 ^ 0x3F)), v67, (__int64)v48, v49);
      if ( v71 <= 384 )
      {
        sub_35D7D80(v47, v161);
      }
      else
      {
        v73 = (__int64 *)(v47 + 384);
        sub_35D7D80(v47, v47 + 384);
        if ( v161 != v47 + 384 )
        {
          do
          {
            v74 = v73;
            v73 += 3;
            sub_35D7D20(v74);
          }
          while ( v75 != v73 );
        }
      }
      v76 = (__m128i *)src;
      v77 = v180;
      if ( src != v180 )
      {
        do
        {
          v78 = *(_QWORD *)(a5 + 40);
          v79 = *(_QWORD *)(a1[13] + 8LL);
          v189.m128i_i8[8] = 0;
          v80 = sub_2E7AAE0(v79, v78, v189.m128i_i64[0], 0);
          v82 = v76->m128i_i64[1];
          v83 = v80;
          LODWORD(v80) = v76[1].m128i_i32[1];
          v189.m128i_i64[0] = v76->m128i_i64[0];
          v191 = v80;
          v84 = (unsigned int)v187;
          v190 = v82;
          v85 = (unsigned int)v187 + 1LL;
          v189.m128i_i64[1] = v83;
          if ( v85 > HIDWORD(v187) )
          {
            if ( v186 > (char *)&v189 || (v148 = v186, &v189 >= (__m128i *)&v186[32 * (unsigned int)v187]) )
            {
              sub_C8D5F0((__int64)&v186, v188, v85, 0x20u, v81, (__int64)v186);
              v43 = (__int64)v186;
              v84 = (unsigned int)v187;
              v86 = &v189;
            }
            else
            {
              sub_C8D5F0((__int64)&v186, v188, v85, 0x20u, v81, (__int64)v186);
              v43 = (__int64)v186;
              v84 = (unsigned int)v187;
              v86 = (__m128i *)&v186[(char *)&v189 - v148];
            }
          }
          else
          {
            v43 = (__int64)v186;
            v86 = &v189;
          }
          v76 = (__m128i *)((char *)v76 + 24);
          v87 = (__m128i *)(v43 + 32 * v84);
          *v87 = _mm_loadu_si128(v86);
          v87[1] = _mm_loadu_si128(v86 + 1);
          LODWORD(v187) = v187 + 1;
        }
        while ( v77 != v76 );
      }
    }
  }
  v88 = **(_QWORD **)(a5 - 8);
  v89 = a1[8];
  if ( v89 != a1[9] )
  {
    v90 = v176;
    v91 = v178;
    v189.m128i_i64[0] = (__int64)&v190;
    v189.m128i_i64[1] = 0x300000000LL;
    v92 = v175;
    v176 = 0;
    v93 = v177;
    v178 = 0;
    if ( (_DWORD)v187 )
    {
      v151 = v177;
      v159 = v91;
      v165 = v88;
      sub_35D7BC0((__int64)&v189, &v186, v43, (__int64)&v190, v88, v177);
      v93 = v151;
      v91 = v159;
      v88 = v165;
    }
    if ( v89 )
    {
      *(_DWORD *)(v89 + 24) = v91;
      *(_QWORD *)(v89 + 32) = v88;
      *(_BYTE *)(v89 + 47) = v129;
      *(_QWORD *)(v89 + 64) = v89 + 80;
      *(_DWORD *)(v89 + 8) = v90;
      *(_QWORD *)v89 = v92;
      *(_QWORD *)(v89 + 16) = v93;
      *(_DWORD *)(v89 + 40) = 0;
      *(_WORD *)(v89 + 44) = 1;
      *(_BYTE *)(v89 + 46) = 0;
      *(_QWORD *)(v89 + 48) = 0;
      *(_QWORD *)(v89 + 56) = 0;
      *(_QWORD *)(v89 + 72) = 0x300000000LL;
      if ( v189.m128i_i32[2] )
        sub_35D7BC0(v89 + 64, (char **)&v189, v43, (__int64)&v190, 1, v189.m128i_u32[2]);
      *(_BYTE *)(v89 + 184) = 0;
      *(_DWORD *)(v89 + 180) = -1;
      *(_DWORD *)(v89 + 176) = v170;
      if ( (__int64 *)v189.m128i_i64[0] != &v190 )
        _libc_free(v189.m128i_u64[0]);
    }
    else
    {
      if ( (__int64 *)v189.m128i_i64[0] != &v190 )
      {
        v157 = v93;
        v163 = v91;
        _libc_free(v189.m128i_u64[0]);
        v93 = v157;
        v91 = v163;
      }
      if ( v91 > 0x40 && v93 )
        j_j___libc_free_0_0(v93);
      if ( v90 > 0x40 && v92 )
        j_j___libc_free_0_0(v92);
    }
    v162 = a1[8];
    a1[8] = v162 + 192;
    v94 = -1431655765 * ((v162 + 192 - a1[7]) >> 6) - 1;
    goto LABEL_140;
  }
  v104 = a1[7];
  v105 = v89 - v104;
  v106 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v89 - v104) >> 6);
  if ( v106 == 0xAAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v107 = 1;
  if ( v106 )
    v107 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v89 - v104) >> 6);
  v108 = __CFADD__(v106, v107);
  v109 = v106 + v107;
  if ( v108 )
  {
    v110 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v109 )
    {
      v158 = 0;
      v112 = 192;
      v164 = 0;
      goto LABEL_194;
    }
    if ( v109 > 0xAAAAAAAAAAAAAALL )
      v109 = 0xAAAAAAAAAAAAAALL;
    v110 = 192 * v109;
  }
  v150 = **(_QWORD **)(a5 - 8);
  v111 = sub_22077B0(v110);
  v88 = v150;
  v105 = v89 - v104;
  v164 = v111;
  v158 = v111 + v110;
  v112 = v111 + 192;
LABEL_194:
  v113 = &v190;
  v114 = v176;
  v189.m128i_i64[1] = 0x300000000LL;
  v115 = v178;
  v189.m128i_i64[0] = (__int64)&v190;
  v116 = v105 + v164;
  v117 = v175;
  v118 = v177;
  v176 = 0;
  v178 = 0;
  if ( (_DWORD)v187 )
  {
    v131 = v177;
    v136 = v115;
    v139 = v175;
    v141 = v114;
    v154 = v88;
    sub_35D7BC0((__int64)&v189, &v186, v115, (__int64)&v190, v88, v175);
    v113 = &v190;
    v118 = v131;
    v115 = v136;
    v117 = v139;
    v114 = v141;
    v116 = v105 + v164;
    v88 = v154;
  }
  if ( v116 )
  {
    *(_DWORD *)(v116 + 24) = v115;
    v115 = 1;
    *(_QWORD *)(v116 + 64) = v116 + 80;
    v119 = v189.m128i_i32[2];
    *(_BYTE *)(v116 + 47) = v129;
    *(_DWORD *)(v116 + 8) = v114;
    *(_QWORD *)v116 = v117;
    *(_QWORD *)(v116 + 16) = v118;
    *(_QWORD *)(v116 + 32) = v88;
    *(_DWORD *)(v116 + 40) = 0;
    *(_WORD *)(v116 + 44) = 1;
    *(_BYTE *)(v116 + 46) = 0;
    *(_QWORD *)(v116 + 48) = 0;
    *(_QWORD *)(v116 + 56) = 0;
    *(_QWORD *)(v116 + 72) = 0x300000000LL;
    if ( v119 )
    {
      v155 = v116;
      sub_35D7BC0(v116 + 64, (char **)&v189, 1, (__int64)&v190, v88, v117);
      v113 = &v190;
      v116 = v155;
    }
    *(_BYTE *)(v116 + 184) = 0;
    *(_DWORD *)(v116 + 180) = -1;
    *(_DWORD *)(v116 + 176) = v170;
    if ( (__int64 *)v189.m128i_i64[0] != &v190 )
      _libc_free(v189.m128i_u64[0]);
  }
  else
  {
    if ( (__int64 *)v189.m128i_i64[0] != &v190 )
    {
      v138 = v118;
      v140 = v115;
      v143 = v117;
      v152 = v114;
      _libc_free(v189.m128i_u64[0]);
      v118 = v138;
      v115 = v140;
      v117 = v143;
      v114 = v152;
    }
    if ( (unsigned int)v115 > 0x40 && v118 )
    {
      v144 = v117;
      v153 = v114;
      j_j___libc_free_0_0(v118);
      v117 = v144;
      v114 = v153;
    }
    if ( v114 > 0x40 && v117 )
      j_j___libc_free_0_0(v117);
  }
  if ( v89 == v104 )
  {
    v94 = 0;
    goto LABEL_227;
  }
  v120 = v164;
  v121 = v104;
  while ( 1 )
  {
    if ( !v120 )
      goto LABEL_208;
    v124 = *(_DWORD *)(v121 + 8);
    *(_DWORD *)(v120 + 8) = v124;
    if ( v124 <= 0x40 )
    {
      *(_QWORD *)v120 = *(_QWORD *)v121;
      v122 = *(_DWORD *)(v121 + 24);
      *(_DWORD *)(v120 + 24) = v122;
      if ( v122 > 0x40 )
        goto LABEL_213;
    }
    else
    {
      sub_C43780(v120, (const void **)v121);
      v125 = *(_DWORD *)(v121 + 24);
      *(_DWORD *)(v120 + 24) = v125;
      if ( v125 > 0x40 )
      {
LABEL_213:
        sub_C43780(v120 + 16, (const void **)(v121 + 16));
        goto LABEL_205;
      }
    }
    *(_QWORD *)(v120 + 16) = *(_QWORD *)(v121 + 16);
LABEL_205:
    *(_QWORD *)(v120 + 32) = *(_QWORD *)(v121 + 32);
    *(_DWORD *)(v120 + 40) = *(_DWORD *)(v121 + 40);
    *(_WORD *)(v120 + 44) = *(_WORD *)(v121 + 44);
    *(_BYTE *)(v120 + 46) = *(_BYTE *)(v121 + 46);
    *(_BYTE *)(v120 + 47) = *(_BYTE *)(v121 + 47);
    *(_QWORD *)(v120 + 48) = *(_QWORD *)(v121 + 48);
    v123 = *(_QWORD *)(v121 + 56);
    *(_DWORD *)(v120 + 72) = 0;
    *(_QWORD *)(v120 + 56) = v123;
    *(_QWORD *)(v120 + 64) = v120 + 80;
    *(_DWORD *)(v120 + 76) = 3;
    if ( *(_DWORD *)(v121 + 72) )
      sub_35D7AE0(v120 + 64, v121 + 64, v115, (__int64)v113, v88, v117);
    *(_DWORD *)(v120 + 176) = *(_DWORD *)(v121 + 176);
    *(_DWORD *)(v120 + 180) = *(_DWORD *)(v121 + 180);
    *(_BYTE *)(v120 + 184) = *(_BYTE *)(v121 + 184);
LABEL_208:
    v121 += 192LL;
    if ( v89 == v121 )
      break;
    v120 += 192;
  }
  v112 = v120 + 384;
  v126 = v104;
  do
  {
    v127 = *(_QWORD *)(v126 + 64);
    if ( v127 != v126 + 80 )
      _libc_free(v127);
    if ( *(_DWORD *)(v126 + 24) > 0x40u )
    {
      v128 = *(_QWORD *)(v126 + 16);
      if ( v128 )
        j_j___libc_free_0_0(v128);
    }
    if ( *(_DWORD *)(v126 + 8) > 0x40u && *(_QWORD *)v126 )
      j_j___libc_free_0_0(*(_QWORD *)v126);
    v126 += 192LL;
  }
  while ( v89 != v126 );
  v94 = -1431655765 * ((v112 - v164) >> 6) - 1;
LABEL_227:
  if ( v104 )
    j_j___libc_free_0(v104);
  a1[7] = v164;
  a1[8] = v112;
  a1[9] = v158;
LABEL_140:
  v95 = v186;
  v96 = *(_QWORD *)(*v6 + v132 + 16);
  *(_QWORD *)(a6 + 8) = *(_QWORD *)(*v6 + v133 + 8);
  *(_DWORD *)a6 = 2;
  *(_QWORD *)(a6 + 16) = v96;
  *(_DWORD *)(a6 + 24) = v94;
  *(_DWORD *)(a6 + 32) = v170;
  if ( v95 != v188 )
    _libc_free((unsigned __int64)v95);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  if ( v178 > 0x40 && v177 )
    j_j___libc_free_0_0(v177);
  if ( v176 > 0x40 && v175 )
    j_j___libc_free_0_0(v175);
  v15 = 1;
LABEL_20:
  if ( v174 > 0x40 && v173 )
    j_j___libc_free_0_0(v173);
  if ( v172 > 0x40 && v171 )
    j_j___libc_free_0_0(v171);
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  return v15;
}
