// Function: sub_14B5970
// Address: 0x14b5970
//
void __fastcall sub_14B5970(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rbx
  int *v19; // rax
  int v20; // eax
  int v21; // eax
  char v22; // al
  int v23; // eax
  char v24; // al
  int v25; // eax
  char v26; // al
  int v27; // eax
  char v28; // al
  int v29; // eax
  char v30; // al
  int v31; // eax
  char v32; // bl
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rdx
  int v36; // eax
  int v37; // eax
  char v38; // al
  int v39; // eax
  int v40; // eax
  __int64 *v41; // rsi
  int v42; // eax
  __int64 v43; // rdi
  __int64 v44; // rax
  char v45; // al
  __int64 v46; // rax
  int v47; // eax
  int v48; // eax
  char v49; // al
  __int64 v50; // rax
  int v51; // eax
  int v52; // eax
  __int64 *v53; // rsi
  int v54; // eax
  char v55; // al
  __int64 v56; // rax
  int v57; // eax
  int v58; // eax
  char v59; // al
  int v60; // eax
  int v61; // eax
  int v62; // eax
  int v63; // eax
  __int64 *v64; // rsi
  int v65; // eax
  char v66; // al
  int v67; // eax
  int v68; // eax
  int v69; // eax
  int v70; // eax
  char v71; // al
  int v72; // eax
  __int64 v73; // rax
  char v74; // al
  int v75; // eax
  __int64 v76; // rdi
  char v77; // al
  int v78; // eax
  unsigned int v79; // ebx
  unsigned __int64 v80; // rax
  unsigned int v81; // r15d
  unsigned __int64 *v82; // rsi
  unsigned __int64 v83; // rax
  char v84; // al
  int v85; // eax
  unsigned int v86; // r15d
  unsigned __int64 v87; // rax
  _QWORD *v88; // rcx
  int v89; // eax
  __int64 v90; // rax
  __int64 v91; // rax
  _QWORD *v92; // rcx
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  _QWORD *v97; // r14
  __int64 v98; // rax
  __m128i v99; // xmm0
  __m128i v100; // xmm1
  __int64 v101; // rsi
  unsigned int v102; // r15d
  const __m128i *v103; // r12
  __m128i *v104; // r13
  __m128i *v105; // r12
  __m128i *v106; // rdi
  __int32 v107; // r15d
  bool v108; // al
  unsigned int v109; // ebx
  unsigned __int64 v110; // rax
  unsigned __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // rax
  __m128i *v117; // r13
  const __m128i *v118; // rbx
  const __m128i *v119; // r13
  const __m128i *v120; // rdi
  __int64 v121; // [rsp+0h] [rbp-480h]
  unsigned int v122; // [rsp+8h] [rbp-478h]
  __int64 v123; // [rsp+8h] [rbp-478h]
  unsigned int v125; // [rsp+18h] [rbp-468h]
  __int64 v127; // [rsp+20h] [rbp-460h]
  __int64 v129; // [rsp+38h] [rbp-448h]
  int v130; // [rsp+4Ch] [rbp-434h] BYREF
  __int64 v131; // [rsp+50h] [rbp-430h] BYREF
  __int64 v132; // [rsp+58h] [rbp-428h] BYREF
  __int64 v133; // [rsp+60h] [rbp-420h] BYREF
  int v134; // [rsp+68h] [rbp-418h]
  __int64 v135[2]; // [rsp+70h] [rbp-410h] BYREF
  __int64 v136[2]; // [rsp+80h] [rbp-400h] BYREF
  unsigned __int64 v137; // [rsp+90h] [rbp-3F0h] BYREF
  unsigned int v138; // [rsp+98h] [rbp-3E8h]
  char v139; // [rsp+9Ch] [rbp-3E4h]
  unsigned __int64 v140; // [rsp+A0h] [rbp-3E0h] BYREF
  __m128i v141; // [rsp+A8h] [rbp-3D8h]
  unsigned __int64 v142; // [rsp+B8h] [rbp-3C8h]
  __int64 *v143; // [rsp+C0h] [rbp-3C0h]
  __m128i v144; // [rsp+C8h] [rbp-3B8h]
  __int64 v145; // [rsp+D8h] [rbp-3A8h]
  char v146; // [rsp+E0h] [rbp-3A0h]
  __m128i *v147; // [rsp+E8h] [rbp-398h] BYREF
  __int64 v148; // [rsp+F0h] [rbp-390h]
  _BYTE v149[352]; // [rsp+F8h] [rbp-388h] BYREF
  char v150; // [rsp+258h] [rbp-228h]
  int v151; // [rsp+25Ch] [rbp-224h]
  __int64 v152; // [rsp+260h] [rbp-220h]
  __int64 v153; // [rsp+270h] [rbp-210h] BYREF
  __int64 v154; // [rsp+278h] [rbp-208h] BYREF
  unsigned __int64 v155; // [rsp+280h] [rbp-200h]
  __m128i v156; // [rsp+288h] [rbp-1F8h] BYREF
  __m128i *v157; // [rsp+298h] [rbp-1E8h]
  __int64 *v158; // [rsp+2A0h] [rbp-1E0h]
  __m128i v159; // [rsp+2A8h] [rbp-1D8h] BYREF
  __int64 v160; // [rsp+2B8h] [rbp-1C8h]
  char v161; // [rsp+2C0h] [rbp-1C0h]
  const __m128i *v162; // [rsp+2C8h] [rbp-1B8h]
  unsigned int v163; // [rsp+2D0h] [rbp-1B0h]
  char v164; // [rsp+2D8h] [rbp-1A8h] BYREF
  char v165; // [rsp+438h] [rbp-48h]
  int v166; // [rsp+43Ch] [rbp-44h]
  __int64 v167; // [rsp+440h] [rbp-40h]

  v5 = *(_QWORD *)(a4 + 8);
  if ( !v5 )
    return;
  v6 = *(_QWORD *)(a4 + 16);
  if ( !v6 )
    return;
  v125 = *(_DWORD *)(a2 + 8);
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15 )
  {
    LODWORD(v153) = 1;
    v110 = sub_14C9BB0(a1, &v153, 1, v6, *(_QWORD *)(a4 + 24), v5);
    if ( (_DWORD)v110 )
    {
      v111 = HIDWORD(v110);
      v112 = 0xFFFFFFFFLL;
      if ( v111 )
      {
        _BitScanReverse((unsigned int *)&v111, v111);
        v112 = 31 - ((unsigned int)v111 ^ 0x1F);
      }
      sub_14A9D90(a2, 0, v112);
    }
    v5 = *(_QWORD *)(a4 + 8);
  }
  v7 = a1;
  v129 = sub_14AA3D0(v5, a1);
  v127 = v129 + 32 * v8;
  if ( v127 == v129 )
  {
LABEL_17:
    v17 = a2 + 16;
    goto LABEL_18;
  }
  v9 = a1;
  while ( 1 )
  {
    v10 = *(_QWORD *)(v129 + 16);
    if ( !v10 )
      goto LABEL_16;
    v11 = *(unsigned int *)(a4 + 104);
    if ( (_DWORD)v11 )
    {
      v12 = 8 * v11;
      v13 = *(_QWORD **)(a4 + 40);
      v7 = (__int64)&v13[(unsigned __int64)v12 / 8];
      v14 = v12 >> 3;
      v15 = v12 >> 5;
      if ( v15 )
      {
        v16 = &v13[4 * v15];
        while ( v10 != *v13 )
        {
          if ( v10 == v13[1] )
          {
            ++v13;
            break;
          }
          if ( v10 == v13[2] )
          {
            v13 += 2;
            break;
          }
          if ( v10 == v13[3] )
          {
            v13 += 3;
            break;
          }
          v13 += 4;
          if ( v13 == v16 )
          {
            v14 = (v7 - (__int64)v13) >> 3;
            goto LABEL_22;
          }
        }
LABEL_15:
        if ( (_QWORD *)v7 != v13 )
          goto LABEL_16;
        goto LABEL_25;
      }
LABEL_22:
      if ( v14 != 2 )
      {
        if ( v14 != 3 )
        {
          if ( v14 != 1 )
            goto LABEL_25;
          goto LABEL_99;
        }
        if ( v10 == *v13 )
          goto LABEL_15;
        ++v13;
      }
      if ( v10 == *v13 )
        goto LABEL_15;
      ++v13;
LABEL_99:
      if ( v10 != *v13 )
        goto LABEL_25;
      goto LABEL_15;
    }
LABEL_25:
    v18 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
    if ( v18
      && v9 == v18
      && (unsigned __int8)sub_14AFF20(*(_QWORD *)(v129 + 16), *(_QWORD *)(a4 + 16), *(_QWORD *)(a4 + 24)) )
    {
      sub_14A9DE0(a2);
      sub_14A9CE0(a2 + 16);
      return;
    }
    v7 = v18;
    v153 = v9;
    if ( sub_13D1F50(&v153, v18) )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14A9CE0(a2);
        sub_14A9DE0(a2 + 16);
        return;
      }
    }
    v19 = (int *)sub_16D40F0(qword_4FBB370);
    if ( v19 )
      v20 = *v19;
    else
      v20 = qword_4FBB370[2];
    if ( a3 == v20 )
      goto LABEL_16;
    v154 = v9;
    v153 = (__int64)&v130;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v131;
    if ( *(_BYTE *)(v18 + 16) == 75 )
    {
      v35 = *(_QWORD *)(v18 - 48);
      if ( !v35 )
        BUG();
      v7 = *(_QWORD *)(v18 - 24);
      if ( v9 != v35 )
      {
        v36 = *(unsigned __int8 *)(v35 + 16);
        if ( (unsigned __int8)v36 <= 0x17u )
        {
          if ( (_BYTE)v36 != 5 )
            goto LABEL_110;
          if ( *(_WORD *)(v35 + 18) != 45 )
            goto LABEL_274;
        }
        else if ( (_BYTE)v36 != 69 )
        {
LABEL_108:
          v37 = v36 - 24;
LABEL_109:
          if ( v37 != 47 )
            goto LABEL_110;
          v88 = (*(_BYTE *)(v35 + 23) & 0x40) != 0
              ? *(_QWORD **)(v35 - 8)
              : (_QWORD *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
          if ( v9 != *v88 )
            goto LABEL_110;
          goto LABEL_247;
        }
        if ( (*(_BYTE *)(v35 + 23) & 0x40) != 0 )
          v92 = *(_QWORD **)(v35 - 8);
        else
          v92 = (_QWORD *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
        if ( v9 != *v92 )
        {
          if ( (unsigned __int8)v36 > 0x17u )
            goto LABEL_108;
LABEL_274:
          v37 = *(unsigned __int16 *)(v35 + 18);
          goto LABEL_109;
        }
      }
LABEL_247:
      if ( v7 )
      {
        v131 = *(_QWORD *)(v18 - 24);
LABEL_249:
        v89 = *(unsigned __int16 *)(v18 + 18);
        BYTE1(v89) &= ~0x80u;
        v130 = v89;
        if ( v89 == 32 )
        {
          v7 = *(_QWORD *)(a4 + 16);
          if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
          {
            sub_14AA4E0((__int64)&v137, v125);
            sub_14A9140((__int64)&v153, a4, v10);
            sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
            if ( v157 != &v159 )
              _libc_free((unsigned __int64)v157);
            sub_14A9220(a2, (__int64 *)&v137);
            v73 = a2;
LABEL_254:
            v76 = v73 + 16;
            v7 = (__int64)&v140;
LABEL_255:
            sub_14A9220(v76, (__int64 *)v7);
            sub_135E100((__int64 *)&v140);
            sub_135E100((__int64 *)&v137);
            goto LABEL_16;
          }
        }
        goto LABEL_33;
      }
LABEL_110:
      v123 = *(_QWORD *)(v18 - 48);
      if ( !(unsigned __int8)sub_14B2C30(&v154, v7) )
        goto LABEL_33;
      v131 = v123;
      goto LABEL_249;
    }
LABEL_33:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v157 = (__m128i *)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_34;
    v38 = sub_14B2D00(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v38 && v7 )
    {
      v157->m128i_i64[0] = v7;
    }
    else
    {
      if ( !(unsigned __int8)sub_14B2D00(&v154, v7) )
        goto LABEL_34;
      v44 = *(_QWORD *)(v18 - 48);
      if ( !v44 )
        goto LABEL_34;
      v157->m128i_i64[0] = v44;
    }
    v39 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v39) &= ~0x80u;
    *(_DWORD *)v153 = v39;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v135);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v40 = v134;
        v134 = 0;
        LODWORD(v154) = v40;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v41 = v136;
LABEL_122:
        sub_13A38D0((__int64)&v133, (__int64)v41);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v42 = v134;
        v134 = 0;
        v7 = (__int64)&v153;
        LODWORD(v154) = v42;
        v153 = v133;
        v43 = a2 + 16;
LABEL_123:
        sub_14A9220(v43, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
LABEL_124:
        sub_135E100((__int64 *)&v140);
        sub_135E100((__int64 *)&v137);
        sub_135E100(v136);
        sub_135E100(v135);
        goto LABEL_16;
      }
    }
LABEL_34:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v158 = &v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_35;
    v45 = sub_14B4650(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v45 && v7 )
    {
      *v158 = v7;
    }
    else
    {
      if ( !sub_14B4650(&v154, v7) )
        goto LABEL_35;
      v46 = *(_QWORD *)(v18 - 48);
      if ( !v46 )
        goto LABEL_35;
      *v158 = v46;
    }
    v47 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v47) &= ~0x80u;
    *(_DWORD *)v153 = v47;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v136);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v48 = v134;
        v134 = 0;
        LODWORD(v154) = v48;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v41 = v135;
        goto LABEL_122;
      }
    }
LABEL_35:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v157 = (__m128i *)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_36;
    v49 = sub_14B3150(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v49 && v7 )
    {
      v157->m128i_i64[0] = v7;
    }
    else
    {
      if ( !(unsigned __int8)sub_14B3150(&v154, v7) )
        goto LABEL_36;
      v50 = *(_QWORD *)(v18 - 48);
      if ( !v50 )
        goto LABEL_36;
      v157->m128i_i64[0] = v50;
    }
    v51 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v51) &= ~0x80u;
    *(_DWORD *)v153 = v51;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v135);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v52 = v134;
        v134 = 0;
        LODWORD(v154) = v52;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v53 = v136;
        goto LABEL_150;
      }
    }
LABEL_36:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v158 = &v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_37;
    v55 = sub_14B4BB0(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v55 && v7 )
    {
      *v158 = v7;
    }
    else
    {
      if ( !sub_14B4BB0(&v154, v7) )
        goto LABEL_37;
      v56 = *(_QWORD *)(v18 - 48);
      if ( !v56 )
        goto LABEL_37;
      *v158 = v56;
    }
    v57 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v57) &= ~0x80u;
    *(_DWORD *)v153 = v57;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v136);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v58 = v134;
        v134 = 0;
        LODWORD(v154) = v58;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v53 = v135;
LABEL_150:
        sub_13A38D0((__int64)&v133, (__int64)v53);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v54 = v134;
        v7 = (__int64)&v153;
        v134 = 0;
        LODWORD(v154) = v54;
        v153 = v133;
        sub_14A9220(a2 + 16, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        goto LABEL_124;
      }
    }
LABEL_37:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v157 = (__m128i *)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_38;
    v59 = sub_14B35A0(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v59 && v7 )
    {
      v157->m128i_i64[0] = v7;
    }
    else
    {
      if ( !(unsigned __int8)sub_14B35A0(&v154, v7) )
        goto LABEL_38;
      v90 = *(_QWORD *)(v18 - 48);
      if ( !v90 )
        goto LABEL_38;
      v157->m128i_i64[0] = v90;
    }
    v60 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v60) &= ~0x80u;
    *(_DWORD *)v153 = v60;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v135);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v61 = v134;
        v134 = 0;
        LODWORD(v154) = v61;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        sub_13A38D0((__int64)&v133, (__int64)v136);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v62 = v134;
        v134 = 0;
        LODWORD(v154) = v62;
        v153 = v133;
        v121 = a2 + 16;
        sub_14A9220(a2 + 16, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        sub_13A38D0((__int64)&v133, (__int64)v136);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v63 = v134;
        v134 = 0;
        LODWORD(v154) = v63;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v64 = v135;
        goto LABEL_172;
      }
    }
LABEL_38:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v132;
    v158 = &v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_39;
    v66 = sub_14B5110(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v66 && v7 )
    {
      *v158 = v7;
    }
    else
    {
      if ( !sub_14B5110(&v154, v7) )
        goto LABEL_39;
      v91 = *(_QWORD *)(v18 - 48);
      if ( !v91 )
        goto LABEL_39;
      *v158 = v91;
    }
    v67 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v67) &= ~0x80u;
    *(_DWORD *)v153 = v67;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)v135, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, v135, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v132, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        sub_13A38D0((__int64)&v133, (__int64)v136);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v68 = v134;
        v134 = 0;
        LODWORD(v154) = v68;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        sub_13A38D0((__int64)&v133, (__int64)v135);
        sub_14A9240((__int64)&v133, (__int64 *)&v137);
        v69 = v134;
        v134 = 0;
        LODWORD(v154) = v69;
        v153 = v133;
        v121 = a2 + 16;
        sub_14A9220(a2 + 16, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        sub_13A38D0((__int64)&v133, (__int64)v135);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v70 = v134;
        v134 = 0;
        LODWORD(v154) = v70;
        v153 = v133;
        sub_14A9220(a2, &v153);
        sub_135E100(&v153);
        sub_135E100(&v133);
        v64 = v136;
LABEL_172:
        sub_13A38D0((__int64)&v133, (__int64)v64);
        sub_14A9240((__int64)&v133, (__int64 *)&v140);
        v65 = v134;
        v134 = 0;
        v7 = (__int64)&v153;
        LODWORD(v154) = v65;
        v43 = v121;
        v153 = v133;
        goto LABEL_123;
      }
    }
LABEL_39:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)v135;
    v157 = (__m128i *)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_40;
    v71 = sub_14B39F0(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v71 && v7 )
    {
      v157->m128i_i64[0] = v7;
    }
    else
    {
      if ( !(unsigned __int8)sub_14B39F0(&v154, v7) )
        goto LABEL_40;
      v93 = *(_QWORD *)(v18 - 48);
      if ( !v93 )
        goto LABEL_40;
      v157->m128i_i64[0] = v93;
    }
    v72 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v72) &= ~0x80u;
    *(_DWORD *)v153 = v72;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        if ( (unsigned __int64)v125 > v135[0] )
        {
          sub_14AA4E0((__int64)&v137, v125);
          sub_14A9140((__int64)&v153, a4, v10);
          sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
          if ( v157 != &v159 )
            _libc_free((unsigned __int64)v157);
          if ( v138 > 0x40 )
          {
            sub_16A8110(&v137, LODWORD(v135[0]));
          }
          else if ( LODWORD(v135[0]) == v138 )
          {
            v137 = 0;
          }
          else
          {
            v137 >>= SLOBYTE(v135[0]);
          }
          sub_14A9220(a2, (__int64 *)&v137);
          if ( v141.m128i_i32[0] > 0x40u )
          {
            sub_16A8110(&v140, LODWORD(v135[0]));
          }
          else if ( LODWORD(v135[0]) == v141.m128i_i32[0] )
          {
            v140 = 0;
          }
          else
          {
            v140 >>= SLOBYTE(v135[0]);
          }
          v73 = a2;
          goto LABEL_254;
        }
      }
    }
LABEL_40:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)v135;
    v158 = &v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_41;
    v74 = sub_14B5490(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v74 && v7 )
    {
      *v158 = v7;
    }
    else
    {
      if ( !sub_14B5490(&v154, v7) )
        goto LABEL_41;
      v94 = *(_QWORD *)(v18 - 48);
      if ( !v94 )
        goto LABEL_41;
      *v158 = v94;
    }
    v75 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v75) &= ~0x80u;
    *(_DWORD *)v153 = v75;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        if ( (unsigned __int64)v125 > v135[0] )
        {
          sub_14AA4E0((__int64)&v137, v125);
          sub_14A9140((__int64)&v153, a4, v10);
          sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
          if ( v157 != &v159 )
            _libc_free((unsigned __int64)v157);
          if ( v141.m128i_i32[0] > 0x40u )
          {
            sub_16A8110(&v140, LODWORD(v135[0]));
          }
          else if ( LODWORD(v135[0]) == v141.m128i_i32[0] )
          {
            v140 = 0;
          }
          else
          {
            v140 >>= SLOBYTE(v135[0]);
          }
          sub_14A9220(a2, (__int64 *)&v140);
          if ( v138 > 0x40 )
          {
            sub_16A8110(&v137, LODWORD(v135[0]));
          }
          else if ( LODWORD(v135[0]) == v138 )
          {
            v137 = 0;
          }
          else
          {
            v137 >>= SLOBYTE(v135[0]);
          }
          v7 = (__int64)&v137;
          v76 = a2 + 16;
          goto LABEL_255;
        }
      }
    }
LABEL_41:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)v135;
    v157 = (__m128i *)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_42;
    v77 = sub_14B3C60(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v77 && v7 )
    {
      v157->m128i_i64[0] = v7;
    }
    else
    {
      if ( !(unsigned __int8)sub_14B3C60(&v154, v7) )
        goto LABEL_42;
      v95 = *(_QWORD *)(v18 - 48);
      if ( !v95 )
        goto LABEL_42;
      v157->m128i_i64[0] = v95;
    }
    v78 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v78) &= ~0x80u;
    *(_DWORD *)v153 = v78;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        if ( (unsigned __int64)v125 > v135[0] )
        {
          sub_14AA4E0((__int64)&v137, v125);
          sub_14A9140((__int64)&v153, a4, v10);
          sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
          if ( v157 != &v159 )
            _libc_free((unsigned __int64)v157);
          v79 = v135[0];
          sub_13A38D0((__int64)&v153, (__int64)&v137);
          if ( (unsigned int)v154 > 0x40 )
          {
            sub_16A7DC0(&v153, v79);
          }
          else
          {
            v80 = 0;
            if ( v79 != (_DWORD)v154 )
              v80 = (v153 << v79) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v154);
            v153 = v80;
          }
          sub_14A9220(a2, &v153);
          sub_135E100(&v153);
          v81 = v135[0];
          v82 = &v140;
          goto LABEL_226;
        }
      }
    }
LABEL_42:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)v135;
    v158 = &v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_43;
    v84 = sub_14B5860(&v154, *(_QWORD *)(v18 - 48));
    v7 = *(_QWORD *)(v18 - 24);
    if ( v84 && v7 )
    {
      *v158 = v7;
    }
    else
    {
      if ( !sub_14B5860(&v154, v7) )
        goto LABEL_43;
      v96 = *(_QWORD *)(v18 - 48);
      if ( !v96 )
        goto LABEL_43;
      *v158 = v96;
    }
    v85 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v85) &= ~0x80u;
    *(_DWORD *)v153 = v85;
    if ( v130 == 32 )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        if ( (unsigned __int64)v125 > v135[0] )
        {
          sub_14AA4E0((__int64)&v137, v125);
          sub_14A9140((__int64)&v153, a4, v10);
          sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
          if ( v157 != &v159 )
            _libc_free((unsigned __int64)v157);
          v86 = v135[0];
          sub_13A38D0((__int64)&v153, (__int64)&v140);
          if ( (unsigned int)v154 > 0x40 )
          {
            sub_16A7DC0(&v153, v86);
          }
          else
          {
            v87 = 0;
            if ( v86 != (_DWORD)v154 )
              v87 = (v153 << v86) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v154);
            v153 = v87;
          }
          sub_14A9220(a2, &v153);
          sub_135E100(&v153);
          v81 = v135[0];
          v82 = &v137;
LABEL_226:
          sub_13A38D0((__int64)&v153, (__int64)v82);
          if ( (unsigned int)v154 > 0x40 )
          {
            sub_16A7DC0(&v153, v81);
          }
          else
          {
            v83 = 0;
            if ( v81 != (_DWORD)v154 )
              v83 = (v153 << v81) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v154);
            v153 = v83;
          }
          v7 = (__int64)&v153;
          sub_14A9220(a2 + 16, &v153);
          sub_135E100(&v153);
          sub_135E100((__int64 *)&v140);
          sub_135E100((__int64 *)&v137);
          goto LABEL_16;
        }
      }
    }
LABEL_43:
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v131;
    if ( *(_BYTE *)(v18 + 16) != 75 )
      goto LABEL_16;
    if ( (unsigned __int8)sub_14B2C30(&v154, *(_QWORD *)(v18 - 48))
      && *(_QWORD *)(v18 - 24)
      && (v131 = *(_QWORD *)(v18 - 24),
          v21 = *(unsigned __int16 *)(v18 + 18),
          BYTE1(v21) &= ~0x80u,
          v130 = v21,
          v21 == 39) )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        v7 = v138 - 1;
        if ( !sub_13D0200((__int64 *)&v137, v7) )
          goto LABEL_91;
LABEL_319:
        v7 = (unsigned int)(*(_DWORD *)(a2 + 8) - 1);
        sub_14A9D60((__int64 *)a2, v7);
        goto LABEL_91;
      }
      v22 = *(_BYTE *)(v18 + 16);
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
      if ( v22 != 75 )
        goto LABEL_16;
    }
    else
    {
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
    }
    if ( !(unsigned __int8)sub_14B2C30(&v154, *(_QWORD *)(v18 - 48))
      || !*(_QWORD *)(v18 - 24)
      || (v131 = *(_QWORD *)(v18 - 24),
          v23 = *(unsigned __int16 *)(v18 + 18),
          BYTE1(v23) &= ~0x80u,
          v130 = v23,
          v23 != 38) )
    {
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
      goto LABEL_58;
    }
    v7 = *(_QWORD *)(a4 + 16);
    if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
    {
      sub_14AA4E0((__int64)&v137, v125);
      sub_14A9140((__int64)&v153, a4, v10);
      sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
      if ( v157 != &v159 )
        _libc_free((unsigned __int64)v157);
      v107 = v141.m128i_i32[0];
      if ( v141.m128i_i32[0] <= 0x40u )
        v108 = 0xFFFFFFFFFFFFFFFFLL >> (64 - v141.m128i_i8[0]) == v140;
      else
        v108 = v107 == (unsigned int)sub_16A58F0(&v140);
      if ( !v108 )
      {
        v7 = v138 - 1;
        if ( !sub_13D0200((__int64 *)&v137, v7) )
          goto LABEL_91;
      }
      goto LABEL_319;
    }
    v24 = *(_BYTE *)(v18 + 16);
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v131;
    if ( v24 != 75 )
      goto LABEL_16;
LABEL_58:
    if ( (unsigned __int8)sub_14B2C30(&v154, *(_QWORD *)(v18 - 48))
      && *(_QWORD *)(v18 - 24)
      && (v131 = *(_QWORD *)(v18 - 24),
          v25 = *(unsigned __int16 *)(v18 + 18),
          BYTE1(v25) &= ~0x80u,
          v130 = v25,
          v25 == 41) )
    {
      v7 = *(_QWORD *)(a4 + 16);
      if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
      {
        sub_14AA4E0((__int64)&v137, v125);
        sub_14A9140((__int64)&v153, a4, v10);
        sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
        if ( v157 != &v159 )
          _libc_free((unsigned __int64)v157);
        goto LABEL_322;
      }
      v26 = *(_BYTE *)(v18 + 16);
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
      if ( v26 != 75 )
        goto LABEL_16;
    }
    else
    {
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
    }
    if ( !(unsigned __int8)sub_14B2C30(&v154, *(_QWORD *)(v18 - 48))
      || !*(_QWORD *)(v18 - 24)
      || (v131 = *(_QWORD *)(v18 - 24),
          v27 = *(unsigned __int16 *)(v18 + 18),
          BYTE1(v27) &= ~0x80u,
          v130 = v27,
          v27 != 40) )
    {
      v153 = (__int64)&v130;
      v154 = v9;
      v155 = v9;
      v156.m128i_i64[0] = v9;
      v156.m128i_i64[1] = (__int64)&v131;
      goto LABEL_72;
    }
    v7 = *(_QWORD *)(a4 + 16);
    if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
    {
      sub_14AA4E0((__int64)&v137, v125);
      sub_14A9140((__int64)&v153, a4, v10);
      sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
      if ( v157 != &v159 )
        _libc_free((unsigned __int64)v157);
      v109 = v138;
      if ( v138 <= 0x40 )
      {
        if ( v137 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v138) )
          goto LABEL_323;
      }
      else if ( v109 == (unsigned int)sub_16A58F0(&v137) )
      {
        goto LABEL_323;
      }
LABEL_322:
      v7 = (unsigned int)(v141.m128i_i32[0] - 1);
      if ( !sub_13D0200((__int64 *)&v140, v7) )
        goto LABEL_91;
LABEL_323:
      v7 = (unsigned int)(*(_DWORD *)(a2 + 24) - 1);
      sub_14A9D60((__int64 *)(a2 + 16), v7);
      goto LABEL_91;
    }
    v28 = *(_BYTE *)(v18 + 16);
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v131;
    if ( v28 != 75 )
      goto LABEL_16;
LABEL_72:
    if ( !(unsigned __int8)sub_14B2C30(&v154, *(_QWORD *)(v18 - 48)) )
      break;
    if ( !*(_QWORD *)(v18 - 24) )
      break;
    v131 = *(_QWORD *)(v18 - 24);
    v29 = *(unsigned __int16 *)(v18 + 18);
    BYTE1(v29) &= ~0x80u;
    v130 = v29;
    if ( v29 != 37 )
      break;
    v7 = *(_QWORD *)(a4 + 16);
    if ( (unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
    {
      sub_14AA4E0((__int64)&v137, v125);
      sub_14A9140((__int64)&v153, a4, v10);
      sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
      if ( v157 != &v159 )
        _libc_free((unsigned __int64)v157);
      goto LABEL_308;
    }
    v30 = *(_BYTE *)(v18 + 16);
    v153 = (__int64)&v130;
    v154 = v9;
    v155 = v9;
    v156.m128i_i64[0] = v9;
    v156.m128i_i64[1] = (__int64)&v131;
    if ( v30 == 75 )
      goto LABEL_79;
LABEL_16:
    v129 += 32;
    if ( v127 == v129 )
      goto LABEL_17;
  }
  v153 = (__int64)&v130;
  v154 = v9;
  v155 = v9;
  v156.m128i_i64[0] = v9;
  v156.m128i_i64[1] = (__int64)&v131;
LABEL_79:
  v7 = *(_QWORD *)(v18 - 48);
  if ( !(unsigned __int8)sub_14B2C30(&v154, v7) )
    goto LABEL_16;
  if ( !*(_QWORD *)(v18 - 24) )
    goto LABEL_16;
  v131 = *(_QWORD *)(v18 - 24);
  v31 = *(unsigned __int16 *)(v18 + 18);
  BYTE1(v31) &= ~0x80u;
  v130 = v31;
  if ( v31 != 36 )
    goto LABEL_16;
  v7 = *(_QWORD *)(a4 + 16);
  if ( !(unsigned __int8)sub_14AFF20(v10, v7, *(_QWORD *)(a4 + 24)) )
    goto LABEL_16;
  sub_14AA4E0((__int64)&v137, v125);
  sub_14A9140((__int64)&v153, a4, v10);
  v7 = (__int64)&v137;
  sub_14B86A0(v131, &v137, (unsigned int)(a3 + 1), &v153);
  if ( v157 != &v159 )
    _libc_free((unsigned __int64)v157);
  if ( v138 > 0x40 )
  {
    v122 = v138;
    if ( v122 == (unsigned int)sub_16A58F0(&v137) )
      goto LABEL_347;
LABEL_87:
    sub_14A9140((__int64)&v153, a4, v10);
    v32 = sub_14BDDF0(v131, 0, (unsigned int)(a3 + 1), &v153);
    if ( v157 != &v159 )
      _libc_free((unsigned __int64)v157);
    if ( v32 )
    {
      v33 = sub_13D05A0((__int64)&v137);
      v34 = *(unsigned int *)(a2 + 8);
      v7 = (unsigned int)(v34 - 1 - v33);
      sub_14A9D90(a2, v7, v34);
LABEL_91:
      sub_135E100((__int64 *)&v140);
      sub_135E100((__int64 *)&v137);
      goto LABEL_16;
    }
LABEL_308:
    v7 = *(_DWORD *)(a2 + 8) - (unsigned int)sub_13D05A0((__int64)&v137);
    sub_14A9D90(a2, v7, *(unsigned int *)(a2 + 8));
    goto LABEL_91;
  }
  if ( v137 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v138) )
    goto LABEL_87;
LABEL_347:
  v17 = a2 + 16;
  sub_14A9CE0(a2);
  sub_14A9CE0(a2 + 16);
  sub_135E100((__int64 *)&v140);
  sub_135E100((__int64 *)&v137);
LABEL_18:
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    if ( (*(_QWORD *)(a2 + 16) & *(_QWORD *)a2) != 0 )
      goto LABEL_288;
    return;
  }
  v7 = v17;
  if ( !(unsigned __int8)sub_16A59B0(a2, v17) )
    return;
LABEL_288:
  sub_14A9DE0(a2);
  sub_14A9DE0(v17);
  v97 = *(_QWORD **)(a4 + 32);
  if ( v97 )
  {
    v98 = sub_15E0530(*v97);
    if ( sub_1602790(v98)
      || (v113 = sub_15E0530(*v97),
          v116 = sub_16033E0(v113, v7, v114, v115),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v116 + 48LL))(v116)) )
    {
      sub_15CA700(&v153, "value-tracking", "BadAssumption", 13, *(_QWORD *)(a4 + 16));
      sub_15CAB20(
        &v153,
        "Detected conflicting code assumptions. Program may have undefined behavior, or compiler may have internal error.",
        112);
      v99 = _mm_loadu_si128(&v156);
      v100 = _mm_loadu_si128(&v159);
      v138 = v154;
      v141 = v99;
      v139 = BYTE4(v154);
      v144 = v100;
      v140 = v155;
      v142 = (unsigned __int64)v157;
      v137 = (unsigned __int64)&unk_49ECF68;
      v143 = v158;
      v146 = v161;
      if ( v161 )
        v145 = v160;
      v101 = v163;
      v147 = (__m128i *)v149;
      v102 = v163;
      v148 = 0x400000000LL;
      if ( v163 )
      {
        if ( v163 > 4uLL )
        {
          sub_14B3F20((__int64)&v147, v163);
          v117 = v147;
          v101 = v163;
        }
        else
        {
          v117 = (__m128i *)v149;
        }
        v118 = v162;
        v103 = (const __m128i *)((char *)v162 + 88 * v101);
        if ( v162 != v103 )
        {
          do
          {
            if ( v117 )
            {
              v117->m128i_i64[0] = (__int64)v117[1].m128i_i64;
              sub_14A9090(v117->m128i_i64, v118->m128i_i64[0], v118->m128i_i64[0] + v118->m128i_i64[1]);
              v117[2].m128i_i64[0] = (__int64)v117[3].m128i_i64;
              sub_14A9090(v117[2].m128i_i64, (_BYTE *)v118[2].m128i_i64[0], v118[2].m128i_i64[0] + v118[2].m128i_i64[1]);
              v117[4] = _mm_loadu_si128(v118 + 4);
              v117[5].m128i_i64[0] = v118[5].m128i_i64[0];
            }
            v118 = (const __m128i *)((char *)v118 + 88);
            v117 = (__m128i *)((char *)v117 + 88);
          }
          while ( v103 != v118 );
          v119 = v162;
          LODWORD(v148) = v102;
          v103 = (const __m128i *)((char *)v162 + 88 * v163);
          v150 = v165;
          v151 = v166;
          v152 = v167;
          v137 = (unsigned __int64)&unk_49ECFF8;
          v153 = (__int64)&unk_49ECF68;
          if ( v103 != v162 )
          {
            do
            {
              v103 = (const __m128i *)((char *)v103 - 88);
              v120 = (const __m128i *)v103[2].m128i_i64[0];
              if ( v120 != &v103[3] )
                j_j___libc_free_0(v120, v103[3].m128i_i64[0] + 1);
              if ( (const __m128i *)v103->m128i_i64[0] != &v103[1] )
                j_j___libc_free_0(v103->m128i_i64[0], v103[1].m128i_i64[0] + 1);
            }
            while ( v119 != v103 );
            v103 = v162;
          }
          goto LABEL_295;
        }
        LODWORD(v148) = v102;
      }
      else
      {
        v103 = v162;
      }
      v150 = v165;
      v151 = v166;
      v152 = v167;
      v137 = (unsigned __int64)&unk_49ECFF8;
LABEL_295:
      if ( v103 != (const __m128i *)&v164 )
        _libc_free((unsigned __int64)v103);
      sub_143AA50(v97, (__int64)&v137);
      v104 = v147;
      v137 = (unsigned __int64)&unk_49ECF68;
      v105 = (__m128i *)((char *)v147 + 88 * (unsigned int)v148);
      if ( v147 != v105 )
      {
        do
        {
          v105 = (__m128i *)((char *)v105 - 88);
          v106 = (__m128i *)v105[2].m128i_i64[0];
          if ( v106 != &v105[3] )
            j_j___libc_free_0(v106, v105[3].m128i_i64[0] + 1);
          if ( (__m128i *)v105->m128i_i64[0] != &v105[1] )
            j_j___libc_free_0(v105->m128i_i64[0], v105[1].m128i_i64[0] + 1);
        }
        while ( v104 != v105 );
        v105 = v147;
      }
      if ( v105 != (__m128i *)v149 )
        _libc_free((unsigned __int64)v105);
    }
  }
}
