// Function: sub_38F7FC0
// Address: 0x38f7fc0
//
__int64 __fastcall sub_38F7FC0(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r15
  unsigned __int8 **v5; // rbx
  unsigned int v6; // ecx
  __int64 v7; // r8
  __int64 v8; // r9
  _DWORD *v9; // rdx
  __m128i *v10; // r14
  __int64 v11; // r13
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  void *v15; // r15
  __m128i *v16; // rbx
  __int64 v17; // r14
  const void *v18; // r13
  int v19; // eax
  __m128i *v20; // rsi
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  unsigned int v25; // r12d
  _DWORD *v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // r13
  __m128i v29; // xmm1
  bool v30; // cc
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  _DWORD *v34; // rax
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  int *v37; // rdx
  int v38; // eax
  unsigned __int64 v39; // rcx
  _DWORD *v40; // r12
  int v41; // eax
  unsigned __int64 v42; // r13
  __m128i v43; // xmm0
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  int *v47; // rax
  unsigned __int64 v48; // rdi
  __int64 v49; // r15
  unsigned __int64 v50; // rbx
  unsigned __int64 v51; // r13
  unsigned __int64 v52; // rdi
  __m128i *v53; // r14
  unsigned __int64 v54; // r15
  __int64 v55; // rbx
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // rdi
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rdx
  unsigned __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rax
  unsigned __int64 v70; // rcx
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // rdx
  __int64 v73; // rcx
  unsigned __int64 v74; // r15
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rax
  __int64 v79; // r8
  __int64 v80; // r9
  unsigned int v81; // edx
  __int64 v82; // r12
  __int64 v83; // rax
  unsigned __int64 v84; // r13
  __m128i v85; // xmm3
  unsigned __int64 v86; // rdi
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  unsigned __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // rdx
  unsigned __int64 v94; // rdx
  __int64 v95; // r12
  __int64 v96; // rdx
  unsigned int v97; // ecx
  __int64 v98; // r12
  int v99; // eax
  __int64 v100; // rsi
  __int64 v101; // rax
  unsigned __int8 *v102; // r15
  bool v103; // r14
  unsigned __int64 v104; // r13
  unsigned __int8 v105; // al
  __int64 i; // r12
  __int64 v107; // rbx
  bool v108; // r10
  __m128i *v109; // rcx
  const __m128i *v110; // r8
  const __m128i *v111; // rax
  __int64 v112; // rdi
  __m128i *v113; // r14
  unsigned __int64 v114; // rbx
  __int64 v115; // r13
  unsigned __int64 v116; // r12
  unsigned __int64 v117; // rdi
  unsigned __int64 v118; // rax
  unsigned __int64 v119; // r14
  int v120; // r12d
  unsigned __int64 v121; // r13
  int v122; // ebx
  __int64 v123; // r12
  unsigned __int64 v124; // r8
  unsigned __int64 v125; // r13
  unsigned __int8 *v126; // rsi
  __int64 v127; // rbx
  int v128; // r12d
  size_t v129; // r13
  const __m128i *v130; // rbx
  __int64 v131; // r12
  __int64 v132; // rcx
  int v133; // eax
  int v134; // r9d
  unsigned __int64 v135; // [rsp+8h] [rbp-1E8h]
  const __m128i *v136; // [rsp+10h] [rbp-1E0h]
  __int64 v137; // [rsp+18h] [rbp-1D8h]
  __int64 v138; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v139; // [rsp+20h] [rbp-1D0h]
  bool v140; // [rsp+28h] [rbp-1C8h]
  __int64 v141; // [rsp+28h] [rbp-1C8h]
  int v142; // [rsp+30h] [rbp-1C0h]
  char v143; // [rsp+37h] [rbp-1B9h]
  __int64 v144; // [rsp+38h] [rbp-1B8h]
  __int64 v145; // [rsp+40h] [rbp-1B0h]
  __int64 v146; // [rsp+40h] [rbp-1B0h]
  unsigned __int64 v147; // [rsp+40h] [rbp-1B0h]
  __m128i *v148; // [rsp+48h] [rbp-1A8h]
  __int64 v149; // [rsp+48h] [rbp-1A8h]
  __int64 v150; // [rsp+48h] [rbp-1A8h]
  __int64 v151; // [rsp+50h] [rbp-1A0h]
  int v152; // [rsp+50h] [rbp-1A0h]
  _QWORD *v153; // [rsp+50h] [rbp-1A0h]
  unsigned __int8 *v155; // [rsp+60h] [rbp-190h] BYREF
  size_t v156; // [rsp+68h] [rbp-188h]
  __int64 v157; // [rsp+70h] [rbp-180h] BYREF
  __int64 v158; // [rsp+78h] [rbp-178h]
  const __m128i *v159; // [rsp+80h] [rbp-170h] BYREF
  __m128i *v160; // [rsp+88h] [rbp-168h]
  const __m128i *v161; // [rsp+90h] [rbp-160h]
  _QWORD v162[2]; // [rsp+A0h] [rbp-150h] BYREF
  __int16 v163; // [rsp+B0h] [rbp-140h]
  __m128i v164; // [rsp+C0h] [rbp-130h] BYREF
  __int16 v165; // [rsp+D0h] [rbp-120h]
  __m128i v166; // [rsp+E0h] [rbp-110h] BYREF
  __int16 v167; // [rsp+F0h] [rbp-100h]
  __m128i v168; // [rsp+100h] [rbp-F0h] BYREF
  __int16 v169; // [rsp+110h] [rbp-E0h]
  __m128i *v170; // [rsp+120h] [rbp-D0h] BYREF
  __m128i v171; // [rsp+128h] [rbp-C8h]
  unsigned __int64 v172; // [rsp+138h] [rbp-B8h] BYREF
  unsigned int v173; // [rsp+140h] [rbp-B0h]
  const char *v174; // [rsp+150h] [rbp-A0h] BYREF
  __m128i v175; // [rsp+158h] [rbp-98h]
  unsigned __int64 v176; // [rsp+168h] [rbp-88h] BYREF
  unsigned int v177; // [rsp+170h] [rbp-80h]
  void *s2[2]; // [rsp+180h] [rbp-70h] BYREF
  unsigned __int64 v179; // [rsp+190h] [rbp-60h] BYREF
  __int64 v180; // [rsp+198h] [rbp-58h]
  const __m128i *v181; // [rsp+1A0h] [rbp-50h]
  __m128i *v182; // [rsp+1A8h] [rbp-48h]
  const __m128i *v183; // [rsp+1B0h] [rbp-40h]

  v4 = a1;
  v5 = &v155;
  v155 = 0;
  v156 = 0;
  if ( (unsigned __int8)sub_38F0EE0(a1, (__int64 *)&v155, a3, a4) )
  {
    s2[0] = "expected identifier in '.macro' directive";
    LOWORD(v179) = 259;
    return (unsigned int)sub_3909CF0(a1, s2, 0, 0, v7, v8);
  }
  v9 = *(_DWORD **)(a1 + 152);
  if ( *v9 == 25 )
  {
    sub_38EB180(a1);
    v9 = *(_DWORD **)(a1 + 152);
  }
  v159 = 0;
  v10 = (__m128i *)s2;
  v160 = 0;
  v161 = 0;
  if ( *v9 != 9 )
  {
    v145 = a1 + 144;
    v11 = a1;
    while ( 1 )
    {
      s2[0] = 0;
      s2[1] = 0;
      v179 = 0;
      v180 = 0;
      v181 = 0;
      LOWORD(v182) = 0;
      if ( (unsigned __int8)sub_38F0EE0(v11, v10->m128i_i64, (__int64)v9, v6) )
        break;
      v14 = (__int64)v159;
      v6 = (unsigned int)v160;
      if ( v160 == v159 )
        goto LABEL_12;
      v15 = s2[1];
      v151 = (__int64)v5;
      v16 = v160;
      v148 = v10;
      v17 = v11;
      v18 = s2[0];
      do
      {
        if ( *(void **)(v14 + 8) == v15 && (!v15 || !memcmp(*(const void **)v14, v18, (size_t)v15)) )
        {
          v49 = v17;
          v166.m128i_i64[0] = (__int64)"macro '";
          v168.m128i_i64[0] = (__int64)&v166;
          v12 = 1282;
          v13 = 770;
          v168.m128i_i64[1] = (__int64)"' has multiple parameters named '";
          v170 = &v168;
          v174 = (const char *)&v170;
          v167 = 1283;
          v166.m128i_i64[1] = v151;
          v169 = 770;
          v171.m128i_i64[0] = (__int64)v148;
          v171.m128i_i16[4] = 1282;
          v175.m128i_i64[0] = (__int64)"'";
          v175.m128i_i16[4] = 770;
LABEL_69:
          v25 = sub_3909CF0(v49, &v174, 0, 0, v12, v13);
          goto LABEL_70;
        }
        v14 += 48;
      }
      while ( v16 != (__m128i *)v14 );
      v11 = v17;
      v5 = (unsigned __int8 **)v151;
      v10 = v148;
LABEL_12:
      v19 = **(_DWORD **)(v11 + 152);
      if ( v19 == 10 )
      {
        sub_38EB180(v11);
        v157 = 0;
        v158 = 0;
        v95 = sub_3909290(v145);
        if ( (unsigned __int8)sub_38F0EE0(v11, &v157, v96, v97) )
        {
          v162[1] = v10;
          v170 = (__m128i *)"'";
          v162[0] = "missing parameter qualifier for '";
          v164.m128i_i64[0] = (__int64)v162;
          v164.m128i_i64[1] = (__int64)"' in macro '";
          v171.m128i_i16[4] = 259;
          v167 = 261;
          v166.m128i_i64[0] = (__int64)v5;
          v163 = 1283;
          v165 = 770;
          sub_14EC200(&v168, &v164, &v166);
          sub_14EC200((__m128i *)&v174, &v168, (const __m128i *)&v170);
          v25 = sub_3909790(v11, v95, &v174, 0, 0);
        }
        else
        {
          v6 = v158;
          if ( v158 == 3 )
          {
            if ( *(_WORD *)v157 == 25970 && *(_BYTE *)(v157 + 2) == 113 )
            {
              LOBYTE(v182) = 1;
              goto LABEL_180;
            }
          }
          else if ( v158 == 6 && *(_DWORD *)v157 == 1634886006 && *(_WORD *)(v157 + 4) == 26482 )
          {
            BYTE1(v182) = 1;
LABEL_180:
            v19 = **(_DWORD **)(v11 + 152);
            goto LABEL_13;
          }
          v165 = 773;
          v164.m128i_i64[1] = (__int64)" is not a valid parameter qualifier for '";
          v166.m128i_i64[0] = (__int64)&v164;
          v167 = 1282;
          v168.m128i_i64[0] = (__int64)&v166;
          v168.m128i_i64[1] = (__int64)"' in macro '";
          v169 = 770;
          v170 = &v168;
          v171.m128i_i16[4] = 1282;
          v174 = (const char *)&v170;
          v175.m128i_i16[4] = 770;
          v164.m128i_i64[0] = (__int64)&v157;
          v166.m128i_i64[1] = (__int64)v10;
          v171.m128i_i64[0] = (__int64)v5;
          v175.m128i_i64[0] = (__int64)"'";
          v25 = sub_3909790(v11, v95, &v174, 0, 0);
        }
LABEL_70:
        v50 = v180;
        v51 = v179;
        if ( v180 != v179 )
        {
          do
          {
            if ( *(_DWORD *)(v51 + 32) > 0x40u )
            {
              v52 = *(_QWORD *)(v51 + 24);
              if ( v52 )
                j_j___libc_free_0_0(v52);
            }
            v51 += 40LL;
          }
          while ( v50 != v51 );
          v51 = v179;
        }
        if ( v51 )
          j_j___libc_free_0(v51);
        goto LABEL_78;
      }
LABEL_13:
      if ( v19 != 27 )
        goto LABEL_14;
      sub_38EB180(v11);
      v74 = sub_3909290(v145);
      v25 = sub_38F6040(v11, &v179, 0, v75, v76, v77);
      if ( (_BYTE)v25 )
        goto LABEL_70;
      if ( (_BYTE)v182 )
      {
        v166.m128i_i64[0] = (__int64)"pointless default value for required parameter '";
        v168.m128i_i64[0] = (__int64)&v166;
        v168.m128i_i64[1] = (__int64)"' in macro '";
        v170 = &v168;
        v171.m128i_i16[4] = 1282;
        v174 = (const char *)&v170;
        v175.m128i_i64[0] = (__int64)"'";
        v167 = 1283;
        v166.m128i_i64[1] = (__int64)v10;
        v169 = 770;
        v171.m128i_i64[0] = (__int64)v5;
        v175.m128i_i16[4] = 770;
        sub_38E4170((_QWORD *)v11, v74, (__int64)&v174, 0, 0);
      }
LABEL_14:
      v20 = v160;
      if ( v160 == v161 )
      {
        sub_38EA310((__int64 *)&v159, v160, v10);
      }
      else
      {
        if ( v160 )
        {
          *v160 = _mm_loadu_si128((const __m128i *)s2);
          v20[1].m128i_i64[0] = v179;
          v20[1].m128i_i64[1] = v180;
          v20[2].m128i_i64[0] = (__int64)v181;
          v181 = 0;
          v180 = 0;
          v179 = 0;
          v20[2].m128i_i16[4] = (__int16)v182;
          v20 = v160;
        }
        v160 = v20 + 3;
      }
      if ( **(_DWORD **)(v11 + 152) == 25 )
        sub_38EB180(v11);
      v21 = v180;
      v22 = v179;
      if ( v180 != v179 )
      {
        do
        {
          if ( *(_DWORD *)(v22 + 32) > 0x40u )
          {
            v23 = *(_QWORD *)(v22 + 24);
            if ( v23 )
              j_j___libc_free_0_0(v23);
          }
          v22 += 40LL;
        }
        while ( v21 != v22 );
        v22 = v179;
      }
      if ( v22 )
        j_j___libc_free_0(v22);
      v9 = *(_DWORD **)(v11 + 152);
      if ( *v9 == 9 )
      {
        v4 = v11;
        goto LABEL_33;
      }
      if ( v160 != v159 && v160[-1].m128i_i8[9] )
      {
        v175.m128i_i64[0] = (__int64)v160[-3].m128i_i64;
        s2[0] = &v174;
        v175.m128i_i16[4] = 1283;
        v174 = "Vararg parameter '";
        s2[1] = "' should be last one in the list of parameters.";
        LOWORD(v179) = 770;
        v24 = sub_3909290(v11 + 144);
        v25 = sub_3909790(v11, v24, v10, 0, 0);
        goto LABEL_78;
      }
    }
    v49 = v11;
    v174 = "expected identifier in '.macro' directive";
    v175.m128i_i16[4] = 259;
    goto LABEL_69;
  }
LABEL_33:
  v26 = v9 + 10;
  v149 = v4 + 144;
  v146 = v4 + 152;
  v27 = *(unsigned int *)(v4 + 160);
  *(_BYTE *)(v4 + 258) = 1;
  v28 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v27 - 40) >> 3);
  if ( (unsigned __int64)(40 * v27) > 0x28 )
  {
    do
    {
      v29 = _mm_loadu_si128((const __m128i *)(v26 + 2));
      v30 = *(v26 - 2) <= 0x40u;
      *(v26 - 10) = *v26;
      *((__m128i *)v26 - 2) = v29;
      if ( !v30 )
      {
        v31 = *((_QWORD *)v26 - 2);
        if ( v31 )
          j_j___libc_free_0_0(v31);
      }
      v32 = *((_QWORD *)v26 + 3);
      v26 += 10;
      *((_QWORD *)v26 - 7) = v32;
      LODWORD(v32) = *(v26 - 2);
      *(v26 - 2) = 0;
      *(v26 - 12) = v32;
      --v28;
    }
    while ( v28 );
    LODWORD(v27) = *(_DWORD *)(v4 + 160);
    v9 = *(_DWORD **)(v4 + 152);
  }
  v33 = (unsigned int)(v27 - 1);
  *(_DWORD *)(v4 + 160) = v33;
  v34 = &v9[10 * v33];
  if ( v34[8] > 0x40u )
  {
    v35 = *((_QWORD *)v34 + 3);
    if ( v35 )
      j_j___libc_free_0_0(v35);
  }
  if ( !*(_DWORD *)(v4 + 160) )
  {
    sub_392C2E0(v10, v149);
    sub_38E90E0(v146, *(_QWORD *)(v4 + 152), (unsigned __int64)v10);
    if ( (unsigned int)v181 > 0x40 )
    {
      if ( v180 )
        j_j___libc_free_0_0(v180);
    }
  }
  v171 = 0u;
  v173 = 1;
  v172 = 0;
  v36 = sub_3909460(v4);
  LODWORD(v174) = *(_DWORD *)v36;
  v175 = _mm_loadu_si128((const __m128i *)(v36 + 8));
  v177 = *(_DWORD *)(v36 + 32);
  if ( v177 > 0x40 )
    sub_16A4FD0((__int64)&v176, (const void **)(v36 + 24));
  else
    v176 = *(_QWORD *)(v36 + 24);
  v152 = 0;
  while ( 1 )
  {
    v37 = *(int **)(v4 + 152);
    v38 = *v37;
    if ( *v37 != 1 )
      break;
LABEL_53:
    v39 = *(unsigned int *)(v4 + 160);
    *(_BYTE *)(v4 + 258) = 0;
    v40 = v37 + 10;
    v41 = v39;
    v39 *= 40LL;
    v42 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v39 - 40) >> 3);
    if ( v39 > 0x28 )
    {
      do
      {
        v43 = _mm_loadu_si128((const __m128i *)(v40 + 2));
        v30 = *(v40 - 2) <= 0x40u;
        *(v40 - 10) = *v40;
        *((__m128i *)v40 - 2) = v43;
        if ( !v30 )
        {
          v44 = *((_QWORD *)v40 - 2);
          if ( v44 )
            j_j___libc_free_0_0(v44);
        }
        v45 = *((_QWORD *)v40 + 3);
        v40 += 10;
        *((_QWORD *)v40 - 7) = v45;
        LODWORD(v45) = *(v40 - 2);
        *(v40 - 2) = 0;
        *(v40 - 12) = v45;
        --v42;
      }
      while ( v42 );
      v41 = *(_DWORD *)(v4 + 160);
      v37 = *(int **)(v4 + 152);
    }
    v46 = (unsigned int)(v41 - 1);
    *(_DWORD *)(v4 + 160) = v46;
    v47 = &v37[10 * v46];
    if ( (unsigned int)v47[8] > 0x40 )
    {
      v48 = *((_QWORD *)v47 + 3);
      if ( v48 )
        j_j___libc_free_0_0(v48);
    }
    if ( !*(_DWORD *)(v4 + 160) )
    {
      sub_392C2E0(v10, v149);
      sub_38E90E0(v146, *(_QWORD *)(v4 + 152), (unsigned __int64)v10);
      if ( (unsigned int)v181 > 0x40 )
      {
        if ( v180 )
          j_j___libc_free_0_0(v180);
      }
    }
  }
  while ( 1 )
  {
    if ( !v38 )
    {
      s2[0] = "no matching '.endmacro' in definition";
      LOWORD(v179) = 259;
LABEL_126:
      v25 = sub_3909790(v4, a2, v10, 0, 0);
      goto LABEL_127;
    }
    if ( v38 == 2 )
      break;
LABEL_52:
    sub_38F0630(v4);
    v37 = *(int **)(v4 + 152);
    v38 = *v37;
    if ( *v37 == 1 )
      goto LABEL_53;
  }
  v59 = sub_3909460(v4);
  if ( *(_DWORD *)v59 == 2 )
  {
    v63 = *(_QWORD *)(v59 + 8);
    v62 = *(_QWORD *)(v59 + 16);
  }
  else
  {
    v60 = *(_QWORD *)(v59 + 16);
    if ( !v60 )
      goto LABEL_103;
    v61 = v60 - 1;
    if ( v60 == 1 )
      v61 = 1;
    if ( v61 > v60 )
      v61 = *(_QWORD *)(v59 + 16);
    v62 = v61 - 1;
    v63 = *(_QWORD *)(v59 + 8) + 1LL;
  }
  if ( v62 != 5 || *(_DWORD *)v63 != 1684956462 || *(_BYTE *)(v63 + 4) != 109 )
  {
LABEL_103:
    v64 = sub_3909460(v4);
    if ( *(_DWORD *)v64 == 2 )
    {
      v68 = *(_QWORD *)(v64 + 8);
      v67 = *(_QWORD *)(v64 + 16);
    }
    else
    {
      v65 = *(_QWORD *)(v64 + 16);
      if ( !v65 )
        goto LABEL_112;
      v66 = v65 - 1;
      if ( v65 == 1 )
        v66 = 1;
      if ( v66 > v65 )
        v66 = *(_QWORD *)(v64 + 16);
      v67 = v66 - 1;
      v68 = *(_QWORD *)(v64 + 8) + 1LL;
    }
    if ( v67 == 9 && *(_QWORD *)v68 == 0x7263616D646E652ELL && *(_BYTE *)(v68 + 8) == 111 )
      goto LABEL_134;
LABEL_112:
    v69 = sub_3909460(v4);
    if ( *(_DWORD *)v69 == 2 )
    {
      v73 = *(_QWORD *)(v69 + 8);
      v72 = *(_QWORD *)(v69 + 16);
    }
    else
    {
      v70 = *(_QWORD *)(v69 + 16);
      if ( !v70 )
        goto LABEL_52;
      v71 = v70 - 1;
      if ( v70 == 1 )
        v71 = 1;
      if ( v71 > v70 )
        v71 = *(_QWORD *)(v69 + 16);
      v72 = v71 - 1;
      v73 = *(_QWORD *)(v69 + 8) + 1LL;
    }
    if ( v72 == 6 && *(_DWORD *)v73 == 1667329326 )
      v152 += *(_WORD *)(v73 + 4) == 28530;
    goto LABEL_52;
  }
LABEL_134:
  if ( v152 )
  {
    --v152;
    goto LABEL_52;
  }
  v78 = sub_3909460(v4);
  LODWORD(v170) = *(_DWORD *)v78;
  v171 = _mm_loadu_si128((const __m128i *)(v78 + 8));
  if ( v173 <= 0x40 && (v81 = *(_DWORD *)(v78 + 32), v81 <= 0x40) )
  {
    v100 = *(_QWORD *)(v78 + 24);
    v173 = *(_DWORD *)(v78 + 32);
    v172 = v100 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v81);
  }
  else
  {
    sub_16A51C0((__int64)&v172, v78 + 24);
  }
  v82 = *(_QWORD *)(v4 + 152);
  v83 = *(unsigned int *)(v4 + 160);
  *(_BYTE *)(v4 + 258) = *(_DWORD *)v82 == 9;
  v84 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v83 - 40) >> 3);
  if ( (unsigned __int64)(40 * v83) > 0x28 )
  {
    do
    {
      v85 = _mm_loadu_si128((const __m128i *)(v82 + 48));
      v30 = *(_DWORD *)(v82 + 32) <= 0x40u;
      *(_DWORD *)v82 = *(_DWORD *)(v82 + 40);
      *(__m128i *)(v82 + 8) = v85;
      if ( !v30 )
      {
        v86 = *(_QWORD *)(v82 + 24);
        if ( v86 )
          j_j___libc_free_0_0(v86);
      }
      v87 = *(_QWORD *)(v82 + 64);
      v82 += 40;
      *(_QWORD *)(v82 - 16) = v87;
      LODWORD(v87) = *(_DWORD *)(v82 + 32);
      *(_DWORD *)(v82 + 32) = 0;
      *(_DWORD *)(v82 - 8) = v87;
      --v84;
    }
    while ( v84 );
    LODWORD(v83) = *(_DWORD *)(v4 + 160);
    v82 = *(_QWORD *)(v4 + 152);
  }
  v88 = (unsigned int)(v83 - 1);
  *(_DWORD *)(v4 + 160) = v88;
  v89 = v82 + 40 * v88;
  if ( *(_DWORD *)(v89 + 32) > 0x40u )
  {
    v90 = *(_QWORD *)(v89 + 24);
    if ( v90 )
      j_j___libc_free_0_0(v90);
  }
  if ( !*(_DWORD *)(v4 + 160) )
  {
    sub_392C2E0(v10, v149);
    sub_38E90E0(v146, *(_QWORD *)(v4 + 152), (unsigned __int64)v10);
    if ( (unsigned int)v181 > 0x40 )
    {
      if ( v180 )
        j_j___libc_free_0_0(v180);
    }
  }
  if ( **(_DWORD **)(v4 + 152) != 9 )
  {
    v91 = v171.m128i_i64[1];
    v92 = v171.m128i_i64[0];
    if ( (_DWORD)v170 != 2 )
    {
      v93 = 0;
      if ( v171.m128i_i64[1] )
      {
        v94 = v171.m128i_i64[1] - 1;
        if ( v171.m128i_i64[1] == 1 )
          v94 = 1;
        if ( v94 > v171.m128i_i64[1] )
          v94 = v171.m128i_u64[1];
        v91 = 1;
        v93 = v94 - 1;
      }
      v92 = v91 + v171.m128i_i64[0];
      v91 = v93;
    }
    v166.m128i_i64[1] = v91;
    v168.m128i_i64[0] = (__int64)"unexpected token in '";
    v166.m128i_i64[0] = v92;
    v168.m128i_i64[1] = (__int64)&v166;
    v169 = 1283;
    s2[0] = &v168;
    LOWORD(v179) = 770;
    s2[1] = "' directive";
    v25 = sub_3909CF0(v4, v10, 0, 0, v79, v80);
    goto LABEL_127;
  }
  v98 = *(_QWORD *)(v4 + 320);
  v99 = sub_16D1B30((__int64 *)(v98 + 1488), v155, v156);
  if ( v99 != -1 && 8LL * *(unsigned int *)(v98 + 1496) != 8LL * v99 )
  {
    v168.m128i_i64[1] = (__int64)v5;
    v169 = 1283;
    v168.m128i_i64[0] = (__int64)"macro '";
    s2[0] = &v168;
    s2[1] = "' is already defined";
    LOWORD(v179) = 770;
    goto LABEL_126;
  }
  v144 = sub_39092A0(&v174);
  v101 = sub_39092A0(&v170);
  v147 = v101 - v144;
  v136 = v159;
  v142 = -1431655765 * (v160 - v159);
  if ( v142 && v147 )
  {
    v143 = 0;
    v153 = (_QWORD *)v4;
    v150 = (__int64)v10;
    v102 = (unsigned __int8 *)v144;
    v103 = 0;
    v104 = v101 - v144;
    while ( 2 )
    {
      v105 = *v102;
      for ( i = 0; ; ++i )
      {
        v107 = i + 1;
        if ( v105 == 92 )
          break;
        if ( v105 == 36 )
        {
          if ( v107 == v104 )
            goto LABEL_195;
          v105 = v102[v107];
          if ( v105 == 110 || v105 == 36 )
          {
            if ( i == v104 )
              goto LABEL_195;
            if ( v105 != 36 )
              v103 = v105 == 110 || v105 == 36;
            goto LABEL_236;
          }
          if ( (unsigned int)v105 - 48 <= 9 )
          {
            if ( i == v104 )
              goto LABEL_195;
            v103 = 1;
LABEL_236:
            v124 = i + 2;
            goto LABEL_231;
          }
        }
        else
        {
          if ( v107 == v104 )
            goto LABEL_195;
          v105 = v102[v107];
        }
      }
      if ( v107 == v104 || i == v104 )
        break;
      v137 = i + 1;
      v118 = (unsigned int)(i + 1);
      v140 = v103;
      v119 = v104;
      v138 = i;
      v120 = i + 1;
      do
      {
        v121 = v118;
        v122 = v102[v118];
        if ( !isalnum(v122) )
        {
          if ( (unsigned __int8)(v122 - 36) > 0x3Bu )
            break;
          v132 = 0x800000000000401LL;
          if ( !_bittest64(&v132, (unsigned int)(v122 - 36)) )
            break;
        }
        v118 = (unsigned int)++v120;
      }
      while ( v119 != v120 );
      v123 = v138;
      v124 = v121;
      v125 = v119;
      v103 = v140;
      v139 = v125;
      v141 = v123;
      v126 = &v102[v137];
      v127 = v123;
      v128 = 0;
      v129 = v124 + ~v127;
      v130 = v136;
      while ( 1 )
      {
        if ( v129 == v130->m128i_i64[1] )
        {
          if ( !v129 )
            break;
          v135 = v124;
          v133 = memcmp((const void *)v130->m128i_i64[0], v126, v129);
          v124 = v135;
          if ( !v133 )
            break;
        }
        ++v128;
        v130 += 3;
        if ( v142 == v128 )
        {
          v131 = v141;
          v104 = v139;
          goto LABEL_221;
        }
      }
      v134 = v128;
      v131 = v141;
      v104 = v139;
      if ( v134 != v142 )
      {
        v143 = 1;
        goto LABEL_231;
      }
LABEL_221:
      if ( *v126 == 40 && v102[v131 + 2] == 41 )
        v124 = v131 + 3;
LABEL_231:
      if ( v104 >= v124 )
      {
        v102 += v124;
        v104 -= v124;
        if ( v104 )
          continue;
      }
      break;
    }
LABEL_195:
    v108 = v103;
    v4 = (__int64)v153;
    v10 = (__m128i *)v150;
    if ( !v143 && v108 )
    {
      s2[0] = "macro defined with named parameters which are not used in macro body, possible positional parameter found "
              "in body which will have no effect";
      LOWORD(v179) = 259;
      sub_38E4170(v153, a2, v150, 0, 0);
    }
  }
  v109 = v160;
  v160 = 0;
  v110 = v159;
  v111 = v161;
  v159 = 0;
  v179 = v144;
  v182 = v109;
  v112 = *(_QWORD *)(v4 + 320);
  s2[1] = (void *)v156;
  v180 = v147;
  v161 = 0;
  s2[0] = v155;
  v181 = v110;
  v183 = v111;
  sub_38E8150(v112, v155, v156, v10);
  v113 = v182;
  v114 = (unsigned __int64)v181;
  if ( v182 != v181 )
  {
    do
    {
      v115 = *(_QWORD *)(v114 + 24);
      v116 = *(_QWORD *)(v114 + 16);
      if ( v115 != v116 )
      {
        do
        {
          if ( *(_DWORD *)(v116 + 32) > 0x40u )
          {
            v117 = *(_QWORD *)(v116 + 24);
            if ( v117 )
              j_j___libc_free_0_0(v117);
          }
          v116 += 40LL;
        }
        while ( v115 != v116 );
        v116 = *(_QWORD *)(v114 + 16);
      }
      if ( v116 )
        j_j___libc_free_0(v116);
      v114 += 48LL;
    }
    while ( v113 != (__m128i *)v114 );
  }
  if ( v181 )
    j_j___libc_free_0((unsigned __int64)v181);
  v25 = 0;
LABEL_127:
  if ( v177 > 0x40 && v176 )
    j_j___libc_free_0_0(v176);
  if ( v173 > 0x40 && v172 )
    j_j___libc_free_0_0(v172);
LABEL_78:
  v53 = v160;
  v54 = (unsigned __int64)v159;
  if ( v160 != v159 )
  {
    do
    {
      v55 = *(_QWORD *)(v54 + 24);
      v56 = *(_QWORD *)(v54 + 16);
      if ( v55 != v56 )
      {
        do
        {
          if ( *(_DWORD *)(v56 + 32) > 0x40u )
          {
            v57 = *(_QWORD *)(v56 + 24);
            if ( v57 )
              j_j___libc_free_0_0(v57);
          }
          v56 += 40LL;
        }
        while ( v55 != v56 );
        v56 = *(_QWORD *)(v54 + 16);
      }
      if ( v56 )
        j_j___libc_free_0(v56);
      v54 += 48LL;
    }
    while ( v53 != (__m128i *)v54 );
    v54 = (unsigned __int64)v159;
  }
  if ( v54 )
    j_j___libc_free_0(v54);
  return v25;
}
