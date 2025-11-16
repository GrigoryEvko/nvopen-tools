// Function: sub_EBE070
// Address: 0xebe070
//
__int64 __fastcall sub_EBE070(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r15
  _DWORD *v3; // rdx
  _DWORD *v4; // r13
  __int64 v5; // r12
  __m128i *v6; // rbx
  const void *v7; // r14
  void *v8; // r13
  unsigned int v9; // r12d
  char *v10; // rbx
  char *v11; // r13
  __int64 v12; // rdi
  __m128i *v13; // r14
  const __m128i *v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdi
  unsigned __int64 v19; // rsi
  int v20; // ecx
  unsigned __int64 v21; // rbx
  __m128i v22; // xmm1
  bool v23; // cc
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  _DWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  int *v30; // rdx
  int v31; // eax
  unsigned __int64 v32; // rcx
  _DWORD *v33; // rbx
  int v34; // eax
  unsigned __int64 v35; // r13
  __m128i v36; // xmm0
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  int *v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rax
  int v61; // eax
  __m128i *v62; // rsi
  char *v63; // rbx
  char *v64; // r12
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  unsigned __int64 v71; // r13
  __int64 v72; // r12
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  unsigned int v85; // edx
  __int64 v86; // rbx
  __int64 v87; // rax
  unsigned __int64 v88; // r13
  __m128i v89; // xmm2
  __int64 v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdi
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rax
  size_t v101; // r12
  const void *v102; // r13
  __int64 v103; // rbx
  int v104; // eax
  int v105; // eax
  __int64 v106; // rax
  unsigned __int64 v107; // r13
  char *v108; // r14
  unsigned __int8 v109; // al
  __int64 i; // r12
  bool v111; // dl
  __int64 v112; // rbx
  __int64 v113; // rdi
  _QWORD *v114; // r12
  _QWORD *j; // rbx
  __int64 *v116; // r14
  __int64 *v117; // rbx
  __int64 v118; // r13
  __int64 v119; // r12
  __int64 v120; // rdi
  unsigned __int64 v121; // rax
  int v122; // r15d
  char *v123; // r12
  unsigned __int64 v124; // r14
  int v125; // ebx
  unsigned __int64 v126; // r15
  char *v127; // rsi
  int v128; // r13d
  size_t v129; // r12
  const __m128i *v130; // rbx
  __int64 v131; // r12
  int v132; // r10d
  __int64 v133; // rcx
  bool v134; // zf
  char v135; // al
  unsigned __int64 v136; // [rsp+8h] [rbp-2D8h]
  __int64 v137; // [rsp+10h] [rbp-2D0h]
  __int64 v138; // [rsp+18h] [rbp-2C8h]
  char *v139; // [rsp+20h] [rbp-2C0h]
  __m128i *v140; // [rsp+28h] [rbp-2B8h]
  __int64 v141; // [rsp+30h] [rbp-2B0h]
  const __m128i *v142; // [rsp+38h] [rbp-2A8h]
  int v143; // [rsp+40h] [rbp-2A0h]
  char v144; // [rsp+46h] [rbp-29Ah]
  char v145; // [rsp+47h] [rbp-299h]
  __int64 v146; // [rsp+48h] [rbp-298h]
  __int64 v147; // [rsp+48h] [rbp-298h]
  int v149; // [rsp+58h] [rbp-288h]
  _QWORD *v150; // [rsp+58h] [rbp-288h]
  __m128i v151; // [rsp+60h] [rbp-280h] BYREF
  __m128i v152; // [rsp+70h] [rbp-270h] BYREF
  const __m128i *v153; // [rsp+80h] [rbp-260h] BYREF
  __m128i *v154; // [rsp+88h] [rbp-258h]
  const __m128i *v155; // [rsp+90h] [rbp-250h]
  __m128i v156; // [rsp+A0h] [rbp-240h] BYREF
  const char *v157; // [rsp+B0h] [rbp-230h]
  __int16 v158; // [rsp+C0h] [rbp-220h]
  __m128i v159; // [rsp+D0h] [rbp-210h] BYREF
  __int16 v160; // [rsp+F0h] [rbp-1F0h]
  __m128i v161[3]; // [rsp+100h] [rbp-1E0h] BYREF
  __m128i v162[2]; // [rsp+130h] [rbp-1B0h] BYREF
  char v163; // [rsp+150h] [rbp-190h]
  char v164; // [rsp+151h] [rbp-18Fh]
  __m128i v165[3]; // [rsp+160h] [rbp-180h] BYREF
  __m128i v166; // [rsp+190h] [rbp-150h] BYREF
  __m128i v167; // [rsp+1A0h] [rbp-140h]
  __int16 v168; // [rsp+1B0h] [rbp-130h]
  __m128i *v169; // [rsp+1C0h] [rbp-120h] BYREF
  __m128i v170; // [rsp+1C8h] [rbp-118h]
  __int64 v171; // [rsp+1D8h] [rbp-108h] BYREF
  unsigned int v172; // [rsp+1E0h] [rbp-100h]
  char *v173; // [rsp+1F0h] [rbp-F0h] BYREF
  _BYTE v174[24]; // [rsp+1F8h] [rbp-E8h] BYREF
  unsigned int v175; // [rsp+210h] [rbp-D0h]
  __m128i v176; // [rsp+220h] [rbp-C0h] BYREF
  __m128i v177; // [rsp+230h] [rbp-B0h]
  __int16 v178; // [rsp+240h] [rbp-A0h]
  void *s2[2]; // [rsp+250h] [rbp-90h] BYREF
  char *v180; // [rsp+260h] [rbp-80h] BYREF
  char *v181; // [rsp+268h] [rbp-78h]
  const __m128i *v182; // [rsp+270h] [rbp-70h]
  __m128i *v183; // [rsp+278h] [rbp-68h]
  const __m128i *v184; // [rsp+280h] [rbp-60h]
  _QWORD *v185; // [rsp+288h] [rbp-58h]
  _QWORD *v186; // [rsp+290h] [rbp-50h]
  __int64 v187; // [rsp+298h] [rbp-48h]
  char v188; // [rsp+2A0h] [rbp-40h]
  int v189; // [rsp+2A4h] [rbp-3Ch]

  v2 = a1;
  v151 = 0u;
  if ( (unsigned __int8)sub_EB61F0(a1, v151.m128i_i64) )
  {
    s2[0] = "expected identifier in '.macro' directive";
    LOWORD(v182) = 259;
    return (unsigned int)sub_ECE0E0(a1, s2, 0, 0);
  }
  v3 = *(_DWORD **)(a1 + 48);
  v4 = v3;
  if ( *v3 == 26 )
  {
    sub_EABFE0(a1);
    v3 = *(_DWORD **)(a1 + 48);
    v4 = v3;
  }
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v146 = a1 + 40;
  if ( *v3 != 9 )
  {
    while ( 1 )
    {
      s2[0] = 0;
      LOWORD(v183) = 0;
      s2[1] = 0;
      v180 = 0;
      v181 = 0;
      v182 = 0;
      if ( (unsigned __int8)sub_EB61F0(v2, (__int64 *)s2) )
        break;
      v5 = (__int64)v153;
      v6 = v154;
      if ( v153 != v154 )
      {
        v7 = s2[0];
        v8 = s2[1];
        while ( v8 != *(void **)(v5 + 8) || v8 && memcmp(*(const void **)v5, v7, (size_t)v8) )
        {
          v5 += 48;
          if ( v6 == (__m128i *)v5 )
            goto LABEL_93;
        }
        v168 = 1283;
        v166.m128i_i64[0] = (__int64)"macro '";
        LOWORD(v172) = 770;
        v167 = v151;
        *(_QWORD *)&v174[8] = v7;
        v169 = &v166;
        v170.m128i_i64[1] = (__int64)"' has multiple parameters named '";
        v173 = (char *)&v169;
        v176.m128i_i64[0] = (__int64)&v173;
        *(_QWORD *)&v174[16] = v8;
        LOWORD(v175) = 1282;
        v177.m128i_i64[0] = (__int64)"'";
        v178 = 770;
        goto LABEL_13;
      }
LABEL_93:
      v61 = **(_DWORD **)(v2 + 48);
      if ( v61 == 10 )
      {
        sub_EABFE0(v2);
        v152 = 0u;
        v72 = sub_ECD690(v146);
        if ( (unsigned __int8)sub_EB61F0(v2, v152.m128i_i64) )
        {
          v168 = 1283;
          v166.m128i_i64[0] = (__int64)"missing parameter qualifier for '";
          v167 = *(__m128i *)s2;
          v169 = &v166;
          v170.m128i_i64[1] = (__int64)"' in macro '";
          LOWORD(v172) = 770;
          v173 = (char *)&v169;
          *(__m128i *)&v174[8] = v151;
          LOWORD(v175) = 1282;
          v176.m128i_i64[0] = (__int64)&v173;
          v177.m128i_i64[0] = (__int64)"'";
          v178 = 770;
          v9 = sub_ECDA70(v2, v72, &v176, 0, 0);
          goto LABEL_14;
        }
        if ( v152.m128i_i64[1] == 3 )
        {
          if ( *(_WORD *)v152.m128i_i64[0] != 25970 || *(_BYTE *)(v152.m128i_i64[0] + 2) != 113 )
            goto LABEL_140;
          LOBYTE(v183) = 1;
        }
        else
        {
          if ( v152.m128i_i64[1] != 6
            || *(_DWORD *)v152.m128i_i64[0] != 1634886006
            || *(_WORD *)(v152.m128i_i64[0] + 4) != 26482 )
          {
LABEL_140:
            v156 = v152;
            v173 = "'";
            v166 = v151;
            v157 = " is not a valid parameter qualifier for '";
            v162[0].m128i_i64[0] = (__int64)"' in macro '";
            v159 = *(__m128i *)s2;
            LOWORD(v175) = 259;
            v168 = 261;
            v164 = 1;
            v163 = 3;
            v160 = 261;
            v158 = 773;
            sub_9C6370(v161, &v156, &v159, (__int64)s2[1], v73, v74);
            sub_9C6370(v165, v161, v162, v75, v76, v77);
            sub_9C6370((__m128i *)&v169, v165, &v166, v78, v79, v80);
            sub_9C6370(&v176, (const __m128i *)&v169, (const __m128i *)&v173, v81, v82, v83);
            v9 = sub_ECDA70(v2, v72, &v176, 0, 0);
            goto LABEL_14;
          }
          BYTE1(v183) = 1;
        }
        v61 = **(_DWORD **)(v2 + 48);
      }
      if ( v61 == 28 )
      {
        sub_EABFE0(v2);
        v71 = sub_ECD690(v146);
        v9 = sub_EBC400(v2, (__int64 *)&v180, 0);
        if ( (_BYTE)v9 )
          goto LABEL_14;
        if ( (_BYTE)v183 )
        {
          v166.m128i_i64[0] = (__int64)"pointless default value for required parameter '";
          v168 = 1283;
          v167 = *(__m128i *)s2;
          LOWORD(v172) = 770;
          v169 = &v166;
          v170.m128i_i64[1] = (__int64)"' in macro '";
          v173 = (char *)&v169;
          LOWORD(v175) = 1282;
          *(__m128i *)&v174[8] = v151;
          v178 = 770;
          v176.m128i_i64[0] = (__int64)&v173;
          v177.m128i_i64[0] = (__int64)"'";
          sub_EA8060((_QWORD *)v2, v71, (__int64)&v176, 0, 0);
        }
      }
      v62 = v154;
      if ( v154 == v155 )
      {
        sub_EA9CE0((__int64 *)&v153, v154, (const __m128i *)s2);
      }
      else
      {
        if ( v154 )
        {
          *v154 = _mm_loadu_si128((const __m128i *)s2);
          v62[1].m128i_i64[0] = (__int64)v180;
          v62[1].m128i_i64[1] = (__int64)v181;
          v62[2].m128i_i64[0] = (__int64)v182;
          v182 = 0;
          v181 = 0;
          v180 = 0;
          v62[2].m128i_i16[4] = (__int16)v183;
          v62 = v154;
        }
        v154 = v62 + 3;
      }
      if ( **(_DWORD **)(v2 + 48) == 26 )
        sub_EABFE0(v2);
      v63 = v181;
      v64 = v180;
      if ( v181 != v180 )
      {
        do
        {
          if ( *((_DWORD *)v64 + 8) > 0x40u )
          {
            v65 = *((_QWORD *)v64 + 3);
            if ( v65 )
              j_j___libc_free_0_0(v65);
          }
          v64 += 40;
        }
        while ( v63 != v64 );
        v64 = v180;
      }
      if ( v64 )
        j_j___libc_free_0(v64, (char *)v182 - v64);
      v3 = *(_DWORD **)(v2 + 48);
      v4 = v3;
      if ( *v3 == 9 )
        goto LABEL_36;
      if ( v153 != v154 && v154[-1].m128i_i8[9] )
      {
        v178 = 1283;
        v176.m128i_i64[0] = (__int64)"vararg parameter '";
        v177.m128i_i64[0] = v154[-3].m128i_i64[0];
        v66 = v154[-3].m128i_i64[1];
        LOWORD(v182) = 770;
        v177.m128i_i64[1] = v66;
        s2[0] = &v176;
        v180 = "' should be the last parameter";
        v67 = sub_ECD690(v2 + 40);
        v9 = sub_ECDA70(v2, v67, s2, 0, 0);
        goto LABEL_22;
      }
    }
    v176.m128i_i64[0] = (__int64)"expected identifier in '.macro' directive";
    v178 = 259;
LABEL_13:
    v9 = sub_ECE0E0(v2, &v176, 0, 0);
LABEL_14:
    v10 = v181;
    v11 = v180;
    if ( v181 != v180 )
    {
      do
      {
        if ( *((_DWORD *)v11 + 8) > 0x40u )
        {
          v12 = *((_QWORD *)v11 + 3);
          if ( v12 )
            j_j___libc_free_0_0(v12);
        }
        v11 += 40;
      }
      while ( v10 != v11 );
      v11 = v180;
    }
    if ( v11 )
      j_j___libc_free_0(v11, (char *)v182 - v11);
    goto LABEL_22;
  }
LABEL_36:
  v19 = *(unsigned int *)(v2 + 56);
  *(_BYTE *)(v2 + 155) = 1;
  v20 = v19;
  v147 = v2 + 48;
  v19 *= 40LL;
  v21 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v19 - 40) >> 3);
  if ( v19 > 0x28 )
  {
    do
    {
      v22 = _mm_loadu_si128((const __m128i *)v4 + 3);
      v23 = v4[8] <= 0x40u;
      *v4 = v4[10];
      *(__m128i *)(v4 + 2) = v22;
      if ( !v23 )
      {
        v24 = *((_QWORD *)v4 + 3);
        if ( v24 )
          j_j___libc_free_0_0(v24);
      }
      v25 = *((_QWORD *)v4 + 8);
      v4 += 10;
      *((_QWORD *)v4 - 2) = v25;
      LODWORD(v25) = v4[8];
      v4[8] = 0;
      *(v4 - 2) = v25;
      --v21;
    }
    while ( v21 );
    v20 = *(_DWORD *)(v2 + 56);
    v3 = *(_DWORD **)(v2 + 48);
  }
  v26 = (unsigned int)(v20 - 1);
  *(_DWORD *)(v2 + 56) = v26;
  v27 = &v3[10 * v26];
  if ( v27[8] > 0x40u )
  {
    v28 = *((_QWORD *)v27 + 3);
    if ( v28 )
      j_j___libc_free_0_0(v28);
  }
  if ( !*(_DWORD *)(v2 + 56) )
  {
    sub_1097F60(s2, v2 + 40);
    sub_EAA0A0(v147, *(_QWORD *)(v2 + 48), (unsigned __int64)s2, v68, v69, v70);
    if ( (unsigned int)v182 > 0x40 )
    {
      if ( v181 )
        j_j___libc_free_0_0(v181);
    }
  }
  LODWORD(v169) = 0;
  v170 = 0u;
  v172 = 1;
  v171 = 0;
  v29 = sub_ECD7B0(v2);
  LODWORD(v173) = *(_DWORD *)v29;
  *(__m128i *)v174 = _mm_loadu_si128((const __m128i *)(v29 + 8));
  v175 = *(_DWORD *)(v29 + 32);
  if ( v175 > 0x40 )
    sub_C43780((__int64)&v174[16], (const void **)(v29 + 24));
  else
    *(_QWORD *)&v174[16] = *(_QWORD *)(v29 + 24);
  v149 = 0;
  while ( 1 )
  {
    v30 = *(int **)(v2 + 48);
    v31 = *v30;
    if ( *v30 != 1 )
      break;
LABEL_55:
    v32 = *(unsigned int *)(v2 + 56);
    *(_BYTE *)(v2 + 155) = 0;
    v33 = v30 + 10;
    v34 = v32;
    v32 *= 40LL;
    v35 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v32 - 40) >> 3);
    if ( v32 > 0x28 )
    {
      do
      {
        v36 = _mm_loadu_si128((const __m128i *)(v33 + 2));
        v23 = *(v33 - 2) <= 0x40u;
        *(v33 - 10) = *v33;
        *((__m128i *)v33 - 2) = v36;
        if ( !v23 )
        {
          v37 = *((_QWORD *)v33 - 2);
          if ( v37 )
            j_j___libc_free_0_0(v37);
        }
        v38 = *((_QWORD *)v33 + 3);
        v33 += 10;
        *((_QWORD *)v33 - 7) = v38;
        LODWORD(v38) = *(v33 - 2);
        *(v33 - 2) = 0;
        *(v33 - 12) = v38;
        --v35;
      }
      while ( v35 );
      v34 = *(_DWORD *)(v2 + 56);
      v30 = *(int **)(v2 + 48);
    }
    v39 = (unsigned int)(v34 - 1);
    *(_DWORD *)(v2 + 56) = v39;
    v40 = &v30[10 * v39];
    if ( (unsigned int)v40[8] > 0x40 )
    {
      v41 = *((_QWORD *)v40 + 3);
      if ( v41 )
        j_j___libc_free_0_0(v41);
    }
    if ( !*(_DWORD *)(v2 + 56) )
    {
      sub_1097F60(s2, v2 + 40);
      sub_EAA0A0(v147, *(_QWORD *)(v2 + 48), (unsigned __int64)s2, v42, v43, v44);
      if ( (unsigned int)v182 > 0x40 )
      {
        if ( v181 )
          j_j___libc_free_0_0(v181);
      }
    }
  }
  while ( 1 )
  {
    if ( !v31 )
    {
      s2[0] = "no matching '.endmacro' in definition";
      LOWORD(v182) = 259;
LABEL_114:
      v9 = sub_ECDA70(v2, a2, s2, 0, 0);
      goto LABEL_115;
    }
    if ( v31 != 2 )
    {
      if ( v31 == 8 )
      {
        v60 = sub_ECD690(v2 + 40);
        sub_EB0E30(v2, v60, 1);
      }
      goto LABEL_54;
    }
    v45 = sub_ECD7B0(v2);
    if ( *(_DWORD *)v45 == 2 )
      break;
    v46 = *(_QWORD *)(v45 + 16);
    if ( v46 )
    {
      v47 = v46 - 1;
      if ( !v47 )
        v47 = 1;
      v48 = *(_QWORD *)(v45 + 8) + 1LL;
      if ( v47 == 6 )
        goto LABEL_88;
    }
LABEL_73:
    v49 = sub_ECD7B0(v2);
    if ( *(_DWORD *)v49 == 2 )
    {
      v54 = *(_QWORD *)(v49 + 8);
      v53 = *(_QWORD *)(v49 + 16);
    }
    else
    {
      v50 = *(_QWORD *)(v49 + 16);
      v51 = *(_QWORD *)(v49 + 8);
      if ( !v50 )
        goto LABEL_79;
      v52 = v50 - 1;
      if ( !v52 )
        v52 = 1;
      v53 = v52 - 1;
      v54 = v51 + 1;
    }
    if ( v53 == 9 && *(_QWORD *)v54 == 0x7263616D646E652ELL && *(_BYTE *)(v54 + 8) == 111 )
      goto LABEL_90;
LABEL_79:
    v55 = sub_ECD7B0(v2);
    if ( *(_DWORD *)v55 == 2 )
    {
      v58 = *(_QWORD *)(v55 + 8);
      v59 = *(_QWORD *)(v55 + 16);
LABEL_84:
      if ( v59 == 6 && *(_DWORD *)v58 == 1667329326 )
        v149 += *(_WORD *)(v58 + 4) == 28530;
      goto LABEL_54;
    }
    v56 = *(_QWORD *)(v55 + 8);
    if ( *(_QWORD *)(v55 + 16) )
    {
      v57 = *(_QWORD *)(v55 + 16) - 1LL;
      if ( !v57 )
        v57 = 1;
      v58 = v56 + 1;
      v59 = v57 - 1;
      goto LABEL_84;
    }
LABEL_54:
    sub_EB4E00(v2);
    v30 = *(int **)(v2 + 48);
    v31 = *v30;
    if ( *v30 == 1 )
      goto LABEL_55;
  }
  v48 = *(_QWORD *)(v45 + 8);
  if ( *(_QWORD *)(v45 + 16) != 5 )
    goto LABEL_73;
LABEL_88:
  if ( *(_DWORD *)v48 != 1684956462 || *(_BYTE *)(v48 + 4) != 109 )
    goto LABEL_73;
LABEL_90:
  if ( v149 )
  {
    --v149;
    goto LABEL_54;
  }
  v84 = sub_ECD7B0(v2);
  LODWORD(v169) = *(_DWORD *)v84;
  v170 = _mm_loadu_si128((const __m128i *)(v84 + 8));
  if ( v172 <= 0x40 && (v85 = *(_DWORD *)(v84 + 32), v85 <= 0x40) )
  {
    v106 = *(_QWORD *)(v84 + 24);
    v172 = v85;
    v171 = v106;
  }
  else
  {
    sub_C43990((__int64)&v171, v84 + 24);
  }
  v86 = *(_QWORD *)(v2 + 48);
  v87 = *(unsigned int *)(v2 + 56);
  *(_BYTE *)(v2 + 155) = *(_DWORD *)v86 == 9;
  v88 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v87 - 40) >> 3);
  if ( (unsigned __int64)(40 * v87) > 0x28 )
  {
    do
    {
      v89 = _mm_loadu_si128((const __m128i *)(v86 + 48));
      v23 = *(_DWORD *)(v86 + 32) <= 0x40u;
      *(_DWORD *)v86 = *(_DWORD *)(v86 + 40);
      *(__m128i *)(v86 + 8) = v89;
      if ( !v23 )
      {
        v90 = *(_QWORD *)(v86 + 24);
        if ( v90 )
          j_j___libc_free_0_0(v90);
      }
      v91 = *(_QWORD *)(v86 + 64);
      v86 += 40;
      *(_QWORD *)(v86 - 16) = v91;
      LODWORD(v91) = *(_DWORD *)(v86 + 32);
      *(_DWORD *)(v86 + 32) = 0;
      *(_DWORD *)(v86 - 8) = v91;
      --v88;
    }
    while ( v88 );
    LODWORD(v87) = *(_DWORD *)(v2 + 56);
    v86 = *(_QWORD *)(v2 + 48);
  }
  v92 = (unsigned int)(v87 - 1);
  *(_DWORD *)(v2 + 56) = v92;
  v93 = v86 + 40 * v92;
  if ( *(_DWORD *)(v93 + 32) > 0x40u )
  {
    v94 = *(_QWORD *)(v93 + 24);
    if ( v94 )
      j_j___libc_free_0_0(v94);
  }
  if ( !*(_DWORD *)(v2 + 56) )
  {
    sub_1097F60(s2, v2 + 40);
    sub_EAA0A0(v147, *(_QWORD *)(v2 + 48), (unsigned __int64)s2, v95, v96, v97);
    if ( (unsigned int)v182 > 0x40 )
    {
      if ( v181 )
        j_j___libc_free_0_0(v181);
    }
  }
  if ( **(_DWORD **)(v2 + 48) != 9 )
  {
    v98 = v170.m128i_i64[1];
    v99 = v170.m128i_i64[0];
    if ( (_DWORD)v169 != 2 && v170.m128i_i64[1] )
    {
      v100 = v170.m128i_i64[1] - 1;
      if ( v170.m128i_i64[1] == 1 )
        v100 = 1;
      v99 = v170.m128i_i64[0] + 1;
      v98 = v100 - 1;
    }
    v177.m128i_i64[1] = v98;
    v178 = 1283;
    v176.m128i_i64[0] = (__int64)"unexpected token in '";
    v177.m128i_i64[0] = v99;
    s2[0] = &v176;
    LOWORD(v182) = 770;
    v180 = "' directive";
    v9 = sub_ECE0E0(v2, s2, 0, 0);
    goto LABEL_115;
  }
  v101 = v151.m128i_u64[1];
  v102 = (const void *)v151.m128i_i64[0];
  v103 = *(_QWORD *)(v2 + 224);
  v104 = sub_C92610();
  v105 = sub_C92860((__int64 *)(v103 + 2384), v102, v101, v104);
  if ( v105 != -1 && 8LL * *(unsigned int *)(v103 + 2392) != 8LL * v105 )
  {
    v178 = 1283;
    v176.m128i_i64[0] = (__int64)"macro '";
    LOWORD(v182) = 770;
    v177 = v151;
    s2[0] = &v176;
    v180 = "' is already defined";
    goto LABEL_114;
  }
  v139 = (char *)sub_ECD6A0(&v173);
  v141 = sub_ECD6A0(&v169) - (_QWORD)v139;
  v142 = v153;
  v140 = v154;
  v143 = -1431655765 * (v154 - v153);
  if ( v143 )
  {
    v107 = v141;
    if ( v141 )
    {
      v145 = 0;
      v144 = 0;
      v150 = (_QWORD *)v2;
      v108 = v139;
      while ( 2 )
      {
        v109 = *v108;
        for ( i = 0; ; ++i )
        {
          v112 = i + 1;
          if ( v109 == 92 )
            break;
          if ( v109 == 36 )
          {
            if ( v112 == v107 )
              goto LABEL_184;
            v109 = v108[v112];
            v111 = v109 == 110 || v109 == 36;
            if ( v111 )
            {
              if ( i == v107 )
                goto LABEL_184;
              v134 = v109 == 36;
              v135 = v145;
              if ( !v134 )
                v135 = v111;
              v145 = v135;
              goto LABEL_231;
            }
            if ( (unsigned int)v109 - 48 <= 9 )
            {
              if ( i == v107 )
                goto LABEL_184;
              v145 = 1;
LABEL_231:
              v126 = i + 2;
              goto LABEL_219;
            }
          }
          else
          {
            if ( v112 == v107 )
              goto LABEL_184;
            v109 = v108[v112];
          }
        }
        if ( v112 == v107 || i == v107 )
          break;
        v137 = i + 1;
        v121 = (unsigned int)(i + 1);
        v138 = i;
        v122 = i + 1;
        v123 = v108;
        do
        {
          v124 = v121;
          v125 = (unsigned __int8)v123[v121];
          if ( !isalnum(v125) )
          {
            if ( (unsigned __int8)(v125 - 36) > 0x3Bu )
              break;
            v133 = 0x800000000000401LL;
            if ( !_bittest64(&v133, (unsigned int)(v125 - 36)) )
              break;
          }
          v121 = (unsigned int)++v122;
        }
        while ( v122 != v107 );
        v126 = v124;
        v108 = v123;
        v127 = &v123[v137];
        v136 = v107;
        v128 = 0;
        v129 = ~v138 + v126;
        v130 = v142;
        while ( v129 != v130->m128i_i64[1] || v129 && memcmp((const void *)v130->m128i_i64[0], v127, v129) )
        {
          ++v128;
          v130 += 3;
          if ( v143 == v128 )
          {
            v131 = v138;
            v107 = v136;
            goto LABEL_226;
          }
        }
        v131 = v138;
        v132 = v128;
        v107 = v136;
        if ( v132 != v143 )
        {
          v144 = 1;
          goto LABEL_219;
        }
LABEL_226:
        if ( *v127 == 40 && v108[v131 + 2] == 41 )
          v126 = v131 + 3;
LABEL_219:
        if ( v126 <= v107 )
        {
          v108 += v126;
          v107 -= v126;
          if ( v107 )
            continue;
        }
        break;
      }
LABEL_184:
      v2 = (__int64)v150;
      if ( !v144 && v145 )
      {
        s2[0] = "macro defined with named parameters which are not used in macro body, possible positional parameter foun"
                "d in body which will have no effect";
        LOWORD(v182) = 259;
        sub_EA8060(v150, a2, (__int64)s2, 0, 0);
        v142 = v153;
        v140 = v154;
      }
    }
  }
  v154 = 0;
  v188 = 0;
  v180 = v139;
  v184 = v155;
  v181 = (char *)v141;
  v113 = *(_QWORD *)(v2 + 224);
  *(__m128i *)s2 = v151;
  v182 = v142;
  v155 = 0;
  v183 = v140;
  v153 = 0;
  v185 = 0;
  v186 = 0;
  v187 = 0;
  v189 = 0;
  sub_EA7B20(v113, (const void *)v151.m128i_i64[0], v151.m128i_u64[1], (const __m128i *)s2);
  v114 = v186;
  for ( j = v185; v114 != j; j += 4 )
  {
    if ( (_QWORD *)*j != j + 2 )
      j_j___libc_free_0(*j, j[2] + 1LL);
  }
  if ( v185 )
    j_j___libc_free_0(v185, v187 - (_QWORD)v185);
  v116 = (__int64 *)v183;
  v117 = (__int64 *)v182;
  if ( v183 != v182 )
  {
    do
    {
      v118 = v117[3];
      v119 = v117[2];
      if ( v118 != v119 )
      {
        do
        {
          if ( *(_DWORD *)(v119 + 32) > 0x40u )
          {
            v120 = *(_QWORD *)(v119 + 24);
            if ( v120 )
              j_j___libc_free_0_0(v120);
          }
          v119 += 40;
        }
        while ( v118 != v119 );
        v119 = v117[2];
      }
      if ( v119 )
        j_j___libc_free_0(v119, v117[4] - v119);
      v117 += 6;
    }
    while ( v116 != v117 );
  }
  if ( v182 )
    j_j___libc_free_0(v182, (char *)v184 - (char *)v182);
  v9 = 0;
LABEL_115:
  if ( v175 > 0x40 && *(_QWORD *)&v174[16] )
    j_j___libc_free_0_0(*(_QWORD *)&v174[16]);
  if ( v172 > 0x40 && v171 )
    j_j___libc_free_0_0(v171);
LABEL_22:
  v13 = v154;
  v14 = v153;
  if ( v154 != v153 )
  {
    do
    {
      v15 = v14[1].m128i_i64[1];
      v16 = v14[1].m128i_i64[0];
      if ( v15 != v16 )
      {
        do
        {
          if ( *(_DWORD *)(v16 + 32) > 0x40u )
          {
            v17 = *(_QWORD *)(v16 + 24);
            if ( v17 )
              j_j___libc_free_0_0(v17);
          }
          v16 += 40;
        }
        while ( v15 != v16 );
        v16 = v14[1].m128i_i64[0];
      }
      if ( v16 )
        j_j___libc_free_0(v16, v14[2].m128i_i64[0] - v16);
      v14 += 3;
    }
    while ( v13 != v14 );
    v14 = v153;
  }
  if ( v14 )
    j_j___libc_free_0(v14, (char *)v155 - (char *)v14);
  return v9;
}
