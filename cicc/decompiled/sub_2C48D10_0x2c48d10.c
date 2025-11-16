// Function: sub_2C48D10
// Address: 0x2c48d10
//
__int64 __fastcall sub_2C48D10(__int64 a1, __int64 a2)
{
  int v4; // eax
  unsigned int v5; // edx
  __int64 v6; // r14
  _QWORD *v7; // rax
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __m128i v10; // xmm0
  unsigned int v11; // r12d
  __int64 v13; // r8
  __int64 *v14; // r10
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // rax
  __m128i *v20; // rdx
  __m128i v21; // xmm0
  _QWORD *v22; // rax
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rdx
  __int64 *v26; // r8
  __int64 v27; // r9
  __int64 v28; // r10
  __int64 *v29; // r11
  _QWORD *v30; // rdi
  _QWORD *v31; // rsi
  __int64 v32; // r8
  __int64 v33; // rax
  _QWORD *v34; // rax
  __m128i *v35; // rdx
  __m128i v36; // xmm0
  __m128i v37; // xmm0
  _QWORD *v38; // rax
  __m128i *v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rdi
  __int64 v42; // rbx
  __int64 v43; // r8
  unsigned int v44; // ecx
  _QWORD *v45; // rax
  __m128i *v46; // rdx
  __m128i v47; // xmm0
  _QWORD *v48; // rax
  __m128i *v49; // rdx
  __m128i v50; // xmm0
  _QWORD *v51; // rax
  __m128i *v52; // rdx
  __m128i v53; // xmm0
  char v54; // dl
  __int64 v55; // r12
  unsigned int v56; // esi
  __int64 v57; // rcx
  int v58; // r13d
  __int64 v59; // r11
  int v60; // r10d
  _QWORD *v61; // rax
  __int64 v62; // rdi
  _DWORD *v63; // rax
  __int64 v64; // rbx
  int v65; // eax
  __int64 v66; // rax
  __int64 v67; // rsi
  _QWORD *v68; // rax
  _QWORD *v69; // r8
  char *v70; // rax
  _QWORD *v71; // rdi
  __m128i *v72; // rax
  __m128i v73; // xmm0
  unsigned int v74; // r14d
  int v75; // esi
  __int64 v76; // rdi
  __int64 v77; // rcx
  __int64 *v78; // rax
  __int64 v79; // rbx
  __int64 *v80; // r12
  __int64 *v81; // r13
  __int64 v82; // rbx
  unsigned int v83; // edx
  __int64 v84; // rdx
  __int64 v85; // rdi
  unsigned int v86; // r10d
  unsigned int v87; // esi
  __int64 v88; // rax
  __int64 v89; // r11
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rbx
  unsigned int v93; // r10d
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // rax
  __int64 v97; // r15
  void *v98; // rax
  __m128i *v99; // rcx
  __int64 *v100; // rax
  unsigned int v101; // r9d
  __int64 *v102; // rdx
  __int64 v103; // r10
  int v104; // r11d
  int v105; // edx
  unsigned int v106; // esi
  __int64 v107; // r10
  int v108; // edx
  __int64 v109; // rax
  __int64 v110; // rbx
  __int64 v111; // rdi
  unsigned int v112; // r11d
  int v113; // ebx
  unsigned int v114; // r10d
  __int64 v115; // rdx
  __m128i *v116; // rdx
  unsigned int v117; // eax
  __int64 v118; // rax
  int v119; // eax
  int v120; // r11d
  int v121; // eax
  int v122; // ebx
  _QWORD *v123; // r8
  char *v124; // rax
  _QWORD *v125; // r8
  char *v126; // rax
  __int64 v127; // rsi
  int v128; // edi
  __int64 v129; // r10
  unsigned int v130; // esi
  __int64 v131; // r10
  int v132; // r8d
  __int64 *v133; // rdi
  __int64 *v134; // rsi
  unsigned int v135; // r15d
  int v136; // edi
  __int64 v137; // r8
  _QWORD *v138; // r8
  char *v139; // rax
  int v140; // edi
  __int64 v141; // rsi
  _QWORD *v142; // rdi
  __m128i *v143; // rax
  __m128i v144; // xmm0
  _QWORD *v145; // r8
  char *v146; // rax
  __int64 *v147; // r13
  __int64 v148; // rbx
  __int64 *v149; // r12
  __int64 i; // rbx
  signed __int64 v151; // rax
  _QWORD *v152; // r8
  char *v153; // rax
  int v154; // edi
  __int64 v155; // [rsp+0h] [rbp-C0h]
  __int64 v156; // [rsp+0h] [rbp-C0h]
  __int64 v157; // [rsp+0h] [rbp-C0h]
  __int64 v158; // [rsp+0h] [rbp-C0h]
  __int64 v159; // [rsp+0h] [rbp-C0h]
  __int64 *v160; // [rsp+10h] [rbp-B0h]
  unsigned int v161; // [rsp+24h] [rbp-9Ch]
  __int64 v162; // [rsp+28h] [rbp-98h]
  __int64 *v163; // [rsp+30h] [rbp-90h]
  __int64 v164; // [rsp+38h] [rbp-88h]
  __int64 v165; // [rsp+38h] [rbp-88h]
  __int64 v166; // [rsp+38h] [rbp-88h]
  unsigned int v167; // [rsp+40h] [rbp-80h]
  __int64 v168; // [rsp+48h] [rbp-78h]
  __int64 v169; // [rsp+48h] [rbp-78h]
  __int64 v170; // [rsp+50h] [rbp-70h]
  __int64 v171; // [rsp+58h] [rbp-68h] BYREF
  __int64 v172; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v173; // [rsp+68h] [rbp-58h] BYREF
  __int64 v174; // [rsp+70h] [rbp-50h] BYREF
  __int64 v175; // [rsp+78h] [rbp-48h]
  __int64 v176; // [rsp+80h] [rbp-40h]
  unsigned int v177; // [rsp+88h] [rbp-38h]

  v4 = *(unsigned __int8 *)(a2 + 8);
  v171 = a2;
  v5 = *(_DWORD *)(a2 + 88);
  if ( v4 == 1 || v4 == 2 )
  {
    if ( v5 > 1 || *(_QWORD *)(a2 + 48) && sub_2BF0A20(a2) && !*(_BYTE *)(*(_QWORD *)(a2 + 48) + 128LL) )
    {
      if ( !sub_2BF0AD0(a2) )
      {
LABEL_4:
        v7 = sub_CB72A0();
        v8 = (__m128i *)v7[4];
        if ( v7[3] - (_QWORD)v8 <= 0x46u )
        {
          sub_CB6200((__int64)v7, "Block has multiple successors but doesn't have a proper branch recipe!\n", 0x47u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_43A1510);
          v8[4].m128i_i32[0] = 1885954917;
          v8[4].m128i_i16[2] = 8549;
          *v8 = si128;
          v10 = _mm_load_si128((const __m128i *)&xmmword_43A1520);
          v8[4].m128i_i8[6] = 10;
          v8[1] = v10;
          v8[2] = _mm_load_si128((const __m128i *)&xmmword_43A1530);
          v8[3] = _mm_load_si128((const __m128i *)&xmmword_43A1540);
          v7[4] += 71LL;
        }
        return 0;
      }
    }
    else if ( sub_2BF0AD0(a2) )
    {
      v22 = sub_CB72A0();
      v23 = (__m128i *)v22[4];
      if ( v22[3] - (_QWORD)v23 <= 0x19u )
      {
        sub_CB6200((__int64)v22, "Unexpected branch recipe!\n", 0x1Au);
      }
      else
      {
        v24 = _mm_load_si128((const __m128i *)&xmmword_43A1550);
        qmemcpy(&v23[1], "h recipe!\n", 10);
        *v23 = v24;
        v22[4] += 26LL;
      }
      return 0;
    }
    v6 = a2;
  }
  else
  {
    v6 = 0;
    if ( v5 > 1 )
      goto LABEL_4;
  }
  v11 = sub_2C489D0(a2 + 80);
  if ( (_BYTE)v11 )
  {
    v38 = sub_CB72A0();
    v39 = (__m128i *)v38[4];
    if ( v38[3] - (_QWORD)v39 <= 0x29u )
    {
      sub_CB6200((__int64)v38, "Multiple instances of the same successor.\n", 0x2Au);
    }
    else
    {
      v40 = _mm_load_si128((const __m128i *)&xmmword_43A1560);
      qmemcpy(&v39[2], "uccessor.\n", 10);
      *v39 = v40;
      v39[1] = _mm_load_si128((const __m128i *)&xmmword_43A1570);
      v38[4] += 42LL;
    }
    return 0;
  }
  v13 = *(_QWORD *)(a2 + 80);
  v14 = &v171;
  if ( v13 != v13 + 8LL * *(unsigned int *)(a2 + 88) )
  {
    do
    {
      v15 = *(_QWORD **)(*(_QWORD *)v13 + 56LL);
      v16 = &v15[*(unsigned int *)(*(_QWORD *)v13 + 64LL)];
      if ( v16 == sub_2C47A50(v15, (__int64)v16, v14) )
      {
        v19 = sub_CB72A0();
        v20 = (__m128i *)v19[4];
        if ( v19[3] - (_QWORD)v20 <= 0x19u )
        {
          sub_CB6200((__int64)v19, "Missing predecessor link.\n", 0x1Au);
        }
        else
        {
          v21 = _mm_load_si128((const __m128i *)&xmmword_43A1580);
          qmemcpy(&v20[1], "sor link.\n", 10);
          *v20 = v21;
          v19[4] += 26LL;
        }
        return v11;
      }
      v13 = v17 + 8;
    }
    while ( v18 != v13 );
  }
  if ( (unsigned __int8)sub_2C489D0(a2 + 56) )
  {
    v51 = sub_CB72A0();
    v52 = (__m128i *)v51[4];
    if ( v51[3] - (_QWORD)v52 <= 0x2Bu )
    {
      sub_CB6200((__int64)v51, "Multiple instances of the same predecessor.\n", 0x2Cu);
    }
    else
    {
      v53 = _mm_load_si128((const __m128i *)&xmmword_43A1560);
      qmemcpy(&v52[2], "redecessor.\n", 12);
      *v52 = v53;
      v52[1] = _mm_load_si128((const __m128i *)&xmmword_43A1590);
      v51[4] += 44LL;
    }
    return v11;
  }
  v26 = *(__int64 **)(a2 + 56);
  v27 = (__int64)&v26[*(unsigned int *)(a2 + 64)];
  if ( (__int64 *)v27 != v26 )
  {
    v28 = *(_QWORD *)(a2 + 48);
    v29 = &v171;
    do
    {
      v33 = *v26;
      if ( *(_QWORD *)(*v26 + 48) != v28 )
      {
        v34 = sub_CB72A0();
        v35 = (__m128i *)v34[4];
        if ( v34[3] - (_QWORD)v35 <= 0x26u )
        {
          sub_CB6200((__int64)v34, "Predecessor is not in the same region.\n", 0x27u);
        }
        else
        {
          v36 = _mm_load_si128((const __m128i *)&xmmword_43A15A0);
          v35[2].m128i_i32[0] = 1869178725;
          v35[2].m128i_i16[2] = 11886;
          *v35 = v36;
          v37 = _mm_load_si128((const __m128i *)&xmmword_43A15B0);
          v35[2].m128i_i8[6] = 10;
          v35[1] = v37;
          v34[4] += 39LL;
        }
        return 0;
      }
      v30 = *(_QWORD **)(v33 + 80);
      v31 = &v30[*(unsigned int *)(v33 + 88)];
      if ( v31 == sub_2C47A50(v30, (__int64)v31, v29) )
      {
        v48 = sub_CB72A0();
        v49 = (__m128i *)v48[4];
        if ( v48[3] - (_QWORD)v49 <= 0x17u )
        {
          sub_CB6200((__int64)v48, "Missing successor link.\n", 0x18u);
        }
        else
        {
          v50 = _mm_load_si128((const __m128i *)&xmmword_43A15C0);
          v49[1].m128i_i64[0] = 0xA2E6B6E696C2072LL;
          *v49 = v50;
          v48[4] += 24LL;
        }
        return 0;
      }
      v26 = (__int64 *)(v32 + 8);
    }
    while ( (__int64 *)v27 != v26 );
  }
  if ( !v6 )
    return 1;
  v41 = *(_QWORD *)(v6 + 48);
  v42 = *(_QWORD *)(v6 + 120);
  v43 = 0;
  v170 = v6 + 112;
  if ( v41 && !*(_BYTE *)(v41 + 128) )
    LOBYTE(v43) = v6 == sub_2BF04B0(v41);
  if ( v42 != v170 )
  {
    v44 = 0;
    do
    {
      if ( !v42 )
        BUG();
      v25 = *(unsigned __int8 *)(v42 - 16);
      if ( (unsigned int)(v25 - 27) > 9 )
      {
        if ( v44 > 1 )
          goto LABEL_101;
        v25 = v6 + 112;
        if ( v170 != v42 )
        {
          while ( 1 )
          {
            if ( (unsigned int)*(unsigned __int8 *)(v42 - 16) - 27 <= 9 )
            {
              v45 = sub_CB72A0();
              v46 = (__m128i *)v45[4];
              if ( v45[3] - (_QWORD)v46 <= 0x29u )
              {
                sub_CB6200((__int64)v45, "Found phi-like recipe after non-phi recipe", 0x2Au);
              }
              else
              {
                v47 = _mm_load_si128((const __m128i *)&xmmword_43A1600);
                qmemcpy(&v46[2], "phi recipe", 10);
                *v46 = v47;
                v46[1] = _mm_load_si128((const __m128i *)&xmmword_43A1610);
                v45[4] += 42LL;
              }
              return 0;
            }
            v42 = *(_QWORD *)(v42 + 8);
            if ( v170 == v42 )
              break;
            if ( !v42 )
              BUG();
          }
        }
        goto LABEL_64;
      }
      if ( (_BYTE)v25 == 30 )
      {
        ++v44;
        if ( !(_BYTE)v43 )
        {
LABEL_214:
          v125 = sub_CB72A0();
          v126 = (char *)v125[4];
          if ( v125[3] - (_QWORD)v126 <= 0x29u )
          {
            sub_CB6200((__int64)v125, "Found header PHI recipe in non-header VPBB", 0x2Au);
          }
          else
          {
            qmemcpy(v126, "Found header PHI recipe in non-header VPBB", 0x2Au);
            v125[4] += 42LL;
          }
          return 0;
        }
      }
      else if ( (_BYTE)v43 )
      {
        if ( (_BYTE)v25 != 27 && (unsigned int)v25 <= 0x1C )
        {
          v123 = sub_CB72A0();
          v124 = (char *)v123[4];
          if ( v123[3] - (_QWORD)v124 <= 0x29u )
          {
            sub_CB6200((__int64)v123, "Found non-header PHI recipe in header VPBB", 0x2Au);
          }
          else
          {
            qmemcpy(v124, "Found non-header PHI recipe in header VPBB", 0x2Au);
            v123[4] += 42LL;
          }
          return 0;
        }
      }
      else if ( (unsigned int)v25 > 0x1C )
      {
        goto LABEL_214;
      }
      v42 = *(_QWORD *)(v42 + 8);
    }
    while ( v170 != v42 );
    if ( v44 > 1 )
    {
LABEL_101:
      v71 = sub_CB72A0();
      v72 = (__m128i *)v71[4];
      if ( v71[3] - (_QWORD)v72 <= 0x39u )
      {
        sub_CB6200((__int64)v71, "There should be no more than one VPActiveLaneMaskPHIRecipe", 0x3Au);
      }
      else
      {
        v73 = _mm_load_si128((const __m128i *)&xmmword_43A15D0);
        qmemcpy(&v72[3], "kPHIRecipe", 10);
        *v72 = v73;
        v72[1] = _mm_load_si128((const __m128i *)&xmmword_43A15E0);
        v72[2] = _mm_load_si128((const __m128i *)&xmmword_43A15F0);
        v71[4] += 58LL;
      }
      return 0;
    }
  }
LABEL_64:
  v174 = 0;
  v55 = *(_QWORD *)(v6 + 120);
  v56 = 0;
  v57 = 0;
  v175 = 0;
  v58 = 0;
  v176 = 0;
  v177 = 0;
  if ( v55 == v170 )
    goto LABEL_88;
  v168 = v6;
  v59 = v6 + 112;
  while ( 1 )
  {
    v64 = v55 - 24;
    if ( !v55 )
      v64 = 0;
    if ( v56 )
    {
      v27 = v56 - 1;
      v60 = 1;
      v25 = 0;
      v43 = (unsigned int)v27 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
      v61 = (_QWORD *)(v57 + 16 * v43);
      v62 = *v61;
      if ( v64 == *v61 )
      {
LABEL_67:
        v63 = v61 + 1;
        goto LABEL_68;
      }
      while ( v62 != -4096 )
      {
        if ( !v25 && v62 == -8192 )
          v25 = (__int64)v61;
        v43 = (unsigned int)v27 & (v60 + (_DWORD)v43);
        v61 = (_QWORD *)(v57 + 16LL * (unsigned int)v43);
        v62 = *v61;
        if ( v64 == *v61 )
          goto LABEL_67;
        ++v60;
      }
      if ( !v25 )
        v25 = (__int64)v61;
      ++v174;
      v65 = v176 + 1;
      if ( 4 * ((int)v176 + 1) < 3 * v56 )
      {
        v57 = v56 >> 3;
        if ( v56 - (v65 + HIDWORD(v176)) <= (unsigned int)v57 )
        {
          v166 = v59;
          sub_2C47F20((__int64)&v174, v56);
          if ( !v177 )
          {
LABEL_328:
            LODWORD(v176) = v176 + 1;
            BUG();
          }
          v27 = v175;
          v57 = 0;
          v74 = (v177 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
          v59 = v166;
          v75 = 1;
          v65 = v176 + 1;
          v25 = v175 + 16LL * v74;
          v76 = *(_QWORD *)v25;
          if ( v64 != *(_QWORD *)v25 )
          {
            while ( v76 != -4096 )
            {
              if ( v76 == -8192 && !v57 )
                v57 = v25;
              v43 = (unsigned int)(v75 + 1);
              v74 = (v177 - 1) & (v75 + v74);
              v25 = v175 + 16LL * v74;
              v76 = *(_QWORD *)v25;
              if ( v64 == *(_QWORD *)v25 )
                goto LABEL_76;
              ++v75;
            }
            if ( v57 )
              v25 = v57;
          }
        }
        goto LABEL_76;
      }
    }
    else
    {
      ++v174;
    }
    v164 = v59;
    sub_2C47F20((__int64)&v174, 2 * v56);
    if ( !v177 )
      goto LABEL_328;
    v59 = v164;
    v57 = (v177 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
    v65 = v176 + 1;
    v25 = v175 + 16 * v57;
    v43 = *(_QWORD *)v25;
    if ( v64 != *(_QWORD *)v25 )
    {
      v140 = 1;
      v141 = 0;
      while ( v43 != -4096 )
      {
        if ( !v141 && v43 == -8192 )
          v141 = v25;
        v27 = (unsigned int)(v140 + 1);
        v57 = (v177 - 1) & (v140 + (_DWORD)v57);
        v25 = v175 + 16LL * (unsigned int)v57;
        v43 = *(_QWORD *)v25;
        if ( v64 == *(_QWORD *)v25 )
          goto LABEL_76;
        ++v140;
      }
      if ( v141 )
        v25 = v141;
    }
LABEL_76:
    LODWORD(v176) = v65;
    if ( *(_QWORD *)v25 != -4096 )
      --HIDWORD(v176);
    *(_QWORD *)v25 = v64;
    v63 = (_DWORD *)(v25 + 8);
    *(_DWORD *)(v25 + 8) = 0;
LABEL_68:
    *v63 = v58;
    v55 = *(_QWORD *)(v55 + 8);
    if ( v55 == v59 )
      break;
    v57 = v175;
    v56 = v177;
    ++v58;
  }
  v6 = v168;
  v165 = *(_QWORD *)(v168 + 120);
  if ( v165 == v170 )
    goto LABEL_88;
  v162 = a1;
  v167 = ((unsigned int)v168 >> 9) ^ ((unsigned int)v168 >> 4);
  while ( 2 )
  {
    if ( !v165 )
      BUG();
    v25 = *(unsigned __int8 *)(v165 - 16);
    v169 = v165 - 24;
    if ( (_BYTE)v25 == 3 )
    {
      if ( *(_BYTE *)(v6 + 8) != 2 )
      {
        v142 = sub_CB72A0();
        v143 = (__m128i *)v142[4];
        if ( v142[3] - (_QWORD)v143 <= 0x10u )
        {
          sub_CB6200((__int64)v142, "VPIRInstructions ", 0x11u);
        }
        else
        {
          v144 = _mm_load_si128((const __m128i *)&xmmword_43A1620);
          v143[1].m128i_i8[0] = 32;
          *v143 = v144;
          v142[4] += 17LL;
        }
        v145 = sub_CB72A0();
        v146 = (char *)v145[4];
        if ( v145[3] - (_QWORD)v146 <= 0x18u )
        {
          sub_CB6200((__int64)v145, "not in a VPIRBasicBlock!\n", 0x19u);
        }
        else
        {
          qmemcpy(v146, "not in a VPIRBasicBlock!\n", 0x19u);
          v145[4] += 25LL;
        }
        goto LABEL_150;
      }
      v66 = *(_QWORD *)(v165 - 8);
      v57 = v66 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v66 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_121;
      goto LABEL_86;
    }
    v66 = *(_QWORD *)(v165 - 8);
    v57 = v66 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v66 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_84;
LABEL_121:
    if ( (v66 & 4) != 0 )
    {
      v78 = *(__int64 **)v57;
      v77 = *(unsigned int *)(v57 + 8);
    }
    else
    {
      v77 = 1;
      v78 = (__int64 *)(v165 - 8);
    }
    v57 = (__int64)&v78[v77];
    v160 = (__int64 *)v57;
    if ( (__int64 *)v57 == v78 )
    {
LABEL_84:
      if ( (_BYTE)v25 != 4 || *(_BYTE *)(v165 + 136) != 74 )
        goto LABEL_86;
      v172 = v165 - 24;
      v147 = *(__int64 **)(v165 + 88);
      v148 = *(unsigned int *)(v165 + 96);
      v149 = &v147[v148];
      v173 = &v172;
      for ( i = (v148 * 8) >> 5; i; --i )
      {
        if ( !(unsigned __int8)sub_2C47CA0(&v173, *v147, v25, v57, v43) )
          goto LABEL_291;
        if ( !(unsigned __int8)sub_2C47CA0(&v173, v147[1], v25, v57, v43) )
        {
          ++v147;
          goto LABEL_291;
        }
        if ( !(unsigned __int8)sub_2C47CA0(&v173, v147[2], v25, v57, v43) )
        {
          v147 += 2;
          goto LABEL_291;
        }
        if ( !(unsigned __int8)sub_2C47CA0(&v173, v147[3], v25, v57, v43) )
        {
          v147 += 3;
          goto LABEL_291;
        }
        v147 += 4;
      }
      v151 = (char *)v149 - (char *)v147;
      if ( (char *)v149 - (char *)v147 != 16 )
      {
        if ( v151 != 24 )
        {
          if ( v151 != 8 )
            goto LABEL_86;
          goto LABEL_281;
        }
        if ( !(unsigned __int8)sub_2C47CA0(&v173, *v147, v25, v57, v43) )
          goto LABEL_291;
        ++v147;
      }
      if ( !(unsigned __int8)sub_2C47CA0(&v173, *v147, v25, v57, v43) )
        goto LABEL_291;
      ++v147;
LABEL_281:
      if ( !(unsigned __int8)sub_2C47CA0(&v173, *v147, v25, v57, v43) )
      {
LABEL_291:
        if ( v149 != v147 )
        {
          v152 = sub_CB72A0();
          v153 = (char *)v152[4];
          if ( v152[3] - (_QWORD)v153 <= 0x21u )
          {
            sub_CB6200((__int64)v152, "EVL VPValue is not used correctly\n", 0x22u);
          }
          else
          {
            qmemcpy(v153, "EVL VPValue is not used correctly\n", 0x22u);
            v152[4] += 34LL;
          }
          goto LABEL_150;
        }
      }
LABEL_86:
      v165 = *(_QWORD *)(v165 + 8);
      if ( v165 != v170 )
        continue;
      a1 = v162;
LABEL_88:
      if ( *(_BYTE *)(v6 + 8) != 2 )
        goto LABEL_57;
      v67 = *(_QWORD *)(v6 + 128);
      if ( *(_BYTE *)(a1 + 44) )
      {
        v68 = *(_QWORD **)(a1 + 24);
        v25 = *(unsigned int *)(a1 + 36);
        v57 = (__int64)&v68[v25];
        if ( v68 != (_QWORD *)v57 )
        {
          while ( v67 != *v68 )
          {
            if ( (_QWORD *)v57 == ++v68 )
              goto LABEL_259;
          }
          goto LABEL_94;
        }
LABEL_259:
        if ( (unsigned int)v25 < *(_DWORD *)(a1 + 32) )
        {
          *(_DWORD *)(a1 + 36) = v25 + 1;
          *(_QWORD *)v57 = v67;
          ++*(_QWORD *)(a1 + 16);
          goto LABEL_57;
        }
      }
      sub_C8CC70(a1 + 16, v67, v25, v57, v43, v27);
      if ( v54 )
      {
LABEL_57:
        sub_C7D6A0(v175, 16LL * v177, 8);
        return 1;
      }
LABEL_94:
      v69 = sub_CB72A0();
      v70 = (char *)v69[4];
      if ( v69[3] - (_QWORD)v70 <= 0x34u )
      {
        sub_CB6200((__int64)v69, "Same IR basic block used by multiple wrapper blocks!\n", 0x35u);
      }
      else
      {
        qmemcpy(v70, "Same IR basic block used by multiple wrapper blocks!\n", 0x35u);
        v69[4] += 53LL;
      }
      goto LABEL_150;
    }
    break;
  }
  v163 = v78;
  v161 = ((unsigned int)v169 >> 9) ^ ((unsigned int)v169 >> 4);
  while ( 1 )
  {
    v79 = *v163;
    if ( !sub_2BFD6A0(*(_QWORD *)(v162 + 8), *v163) )
    {
      v138 = sub_CB72A0();
      v139 = (char *)v138[4];
      if ( v138[3] - (_QWORD)v139 <= 0x1Cu )
      {
        sub_CB6200((__int64)v138, "Failed to infer scalar type!\n", 0x1Du);
      }
      else
      {
        qmemcpy(v139, "Failed to infer scalar type!\n", 0x1Du);
        v138[4] += 29LL;
      }
      goto LABEL_150;
    }
    v80 = *(__int64 **)(v79 + 16);
    v57 = v162;
    v81 = &v80[*(unsigned int *)(v79 + 24)];
    if ( v80 != v81 )
      break;
LABEL_171:
    if ( v160 == ++v163 )
    {
      v25 = *(unsigned __int8 *)(v165 - 16);
      goto LABEL_84;
    }
  }
  while ( 2 )
  {
    v82 = *v80;
    if ( !*v80 )
      goto LABEL_170;
    v83 = *(unsigned __int8 *)(v82 - 32);
    if ( (unsigned __int8)v83 > 0x1Cu )
    {
      if ( v83 > 0x24 )
        goto LABEL_130;
      goto LABEL_170;
    }
    if ( (unsigned __int8)(v83 - 27) <= 1u || *(_BYTE *)(v82 - 32) == 3 && **(_BYTE **)(v82 + 56) == 84 )
      goto LABEL_170;
LABEL_130:
    v84 = *(_QWORD *)(v82 + 40);
    if ( v84 != v6 )
    {
      v85 = *(_QWORD *)v57;
      v86 = *(_DWORD *)(*(_QWORD *)v57 + 112LL);
      v43 = *(_QWORD *)(*(_QWORD *)v57 + 96LL);
      if ( v86 )
      {
        v27 = v86 - 1;
        v87 = v27 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
        v88 = v43 + 16LL * v87;
        v89 = *(_QWORD *)v88;
        if ( v84 == *(_QWORD *)v88 )
        {
LABEL_133:
          v90 = v43 + 16LL * v86;
          if ( v90 != v88 )
          {
            v91 = *(unsigned int *)(v88 + 8);
            if ( *(_DWORD *)(v85 + 32) > (unsigned int)v91 )
            {
              v92 = *(_QWORD *)(*(_QWORD *)(v85 + 24) + 8 * v91);
LABEL_136:
              v93 = v27 & v167;
              v94 = v43 + 16LL * ((unsigned int)v27 & v167);
              v95 = *(_QWORD *)v94;
              if ( v6 == *(_QWORD *)v94 )
              {
LABEL_137:
                if ( v94 != v90 )
                {
                  v96 = *(unsigned int *)(v94 + 8);
                  if ( *(_DWORD *)(v85 + 32) > (unsigned int)v96 )
                  {
                    v97 = *(_QWORD *)(*(_QWORD *)(v85 + 24) + 8 * v96);
                    if ( v97 != v92 && v92 )
                    {
                      if ( !v97 )
                        goto LABEL_148;
                      if ( *(_QWORD *)(v92 + 8) != v97 )
                      {
                        if ( *(_QWORD *)(v97 + 8) == v92 || *(_DWORD *)(v97 + 16) >= *(_DWORD *)(v92 + 16) )
                          goto LABEL_148;
                        if ( *(_BYTE *)(v85 + 136) )
                        {
                          if ( *(_DWORD *)(v92 + 72) < *(_DWORD *)(v97 + 72)
                            || *(_DWORD *)(v92 + 76) > *(_DWORD *)(v97 + 76) )
                          {
                            goto LABEL_148;
                          }
                        }
                        else
                        {
                          v117 = *(_DWORD *)(v85 + 140) + 1;
                          *(_DWORD *)(v85 + 140) = v117;
                          if ( v117 > 0x20 )
                          {
                            v156 = v57;
                            sub_2BF23E0(v85);
                            if ( *(_DWORD *)(v92 + 72) < *(_DWORD *)(v97 + 72) )
                              goto LABEL_148;
                            v57 = v156;
                            if ( *(_DWORD *)(v92 + 76) > *(_DWORD *)(v97 + 76) )
                              goto LABEL_148;
                          }
                          else
                          {
                            do
                            {
                              v118 = v92;
                              v92 = *(_QWORD *)(v92 + 8);
                            }
                            while ( v92 && *(_DWORD *)(v97 + 16) <= *(_DWORD *)(v92 + 16) );
                            if ( v118 != v97 )
                            {
LABEL_148:
                              v98 = sub_CB72A0();
                              v99 = (__m128i *)*((_QWORD *)v98 + 4);
                              if ( *((_QWORD *)v98 + 3) - (_QWORD)v99 > 0xFu )
                              {
                                *v99 = _mm_load_si128((const __m128i *)&xmmword_43A1630);
                                *((_QWORD *)v98 + 4) += 16LL;
                                goto LABEL_150;
                              }
                              goto LABEL_205;
                            }
                          }
                        }
                      }
                    }
                    goto LABEL_170;
                  }
                }
              }
              else
              {
                v119 = 1;
                while ( v95 != -4096 )
                {
                  v120 = v119 + 1;
                  v93 = v27 & (v119 + v93);
                  v94 = v43 + 16LL * v93;
                  v95 = *(_QWORD *)v94;
                  if ( v6 == *(_QWORD *)v94 )
                    goto LABEL_137;
                  v119 = v120;
                }
              }
              if ( v92 )
                goto LABEL_148;
LABEL_170:
              if ( v81 == ++v80 )
                goto LABEL_171;
              continue;
            }
LABEL_188:
            v92 = 0;
            goto LABEL_136;
          }
LABEL_187:
          v27 = v86 - 1;
          if ( !v86 )
            goto LABEL_170;
          goto LABEL_188;
        }
        v121 = 1;
        while ( v89 != -4096 )
        {
          v122 = v121 + 1;
          v87 = v27 & (v121 + v87);
          v88 = v43 + 16LL * v87;
          v89 = *(_QWORD *)v88;
          if ( v84 == *(_QWORD *)v88 )
            goto LABEL_133;
          v121 = v122;
        }
      }
      v90 = v43 + 16LL * v86;
      goto LABEL_187;
    }
    break;
  }
  v106 = v177;
  v110 = v82 - 40;
  if ( !v177 )
  {
    ++v174;
LABEL_235:
    v158 = v57;
    sub_2C47F20((__int64)&v174, 2 * v177);
    if ( !v177 )
      goto LABEL_327;
    v130 = (v177 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
    v105 = v176 + 1;
    v57 = v158;
    v100 = (__int64 *)(v175 + 16LL * v130);
    v131 = *v100;
    if ( v110 != *v100 )
    {
      v132 = 1;
      v133 = 0;
      while ( v131 != -4096 )
      {
        if ( !v133 && v131 == -8192 )
          v133 = v100;
        v130 = (v177 - 1) & (v132 + v130);
        v100 = (__int64 *)(v175 + 16LL * v130);
        v131 = *v100;
        if ( v110 == *v100 )
          goto LABEL_161;
        ++v132;
      }
      if ( v133 )
        v100 = v133;
    }
    goto LABEL_161;
  }
  v43 = v177 - 1;
  v111 = v175;
  v104 = 1;
  v100 = 0;
  v101 = v43 & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
  v102 = (__int64 *)(v175 + 16LL * v101);
  v103 = *v102;
  if ( v110 == *v102 )
  {
LABEL_179:
    v112 = *((_DWORD *)v102 + 2);
    goto LABEL_180;
  }
  while ( v103 != -4096 )
  {
    if ( v103 == -8192 && !v100 )
      v100 = v102;
    v101 = v43 & (v104 + v101);
    v102 = (__int64 *)(v175 + 16LL * v101);
    v103 = *v102;
    if ( v110 == *v102 )
      goto LABEL_179;
    ++v104;
  }
  if ( !v100 )
    v100 = v102;
  ++v174;
  v105 = v176 + 1;
  if ( 4 * ((int)v176 + 1) >= 3 * v177 )
    goto LABEL_235;
  if ( v177 - HIDWORD(v176) - v105 <= v177 >> 3 )
  {
    v159 = v57;
    sub_2C47F20((__int64)&v174, v177);
    if ( !v177 )
    {
LABEL_327:
      LODWORD(v176) = v176 + 1;
      BUG();
    }
    v134 = 0;
    v135 = (v177 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
    v136 = 1;
    v105 = v176 + 1;
    v57 = v159;
    v100 = (__int64 *)(v175 + 16LL * v135);
    v137 = *v100;
    if ( *v100 != v110 )
    {
      while ( v137 != -4096 )
      {
        if ( !v134 && v137 == -8192 )
          v134 = v100;
        v135 = (v177 - 1) & (v136 + v135);
        v100 = (__int64 *)(v175 + 16LL * v135);
        v137 = *v100;
        if ( v110 == *v100 )
          goto LABEL_161;
        ++v136;
      }
      if ( v134 )
        v100 = v134;
    }
  }
LABEL_161:
  LODWORD(v176) = v105;
  if ( *v100 != -4096 )
    --HIDWORD(v176);
  *v100 = v110;
  *((_DWORD *)v100 + 2) = 0;
  v106 = v177;
  if ( !v177 )
  {
    ++v174;
    goto LABEL_165;
  }
  v111 = v175;
  v43 = v177 - 1;
  v112 = 0;
LABEL_180:
  v113 = 1;
  v109 = 0;
  v114 = v43 & v161;
  v115 = v111 + 16LL * ((unsigned int)v43 & v161);
  v27 = *(_QWORD *)v115;
  if ( v169 != *(_QWORD *)v115 )
  {
    while ( v27 != -4096 )
    {
      if ( v27 == -8192 && !v109 )
        v109 = v115;
      v114 = v43 & (v113 + v114);
      v115 = v111 + 16LL * v114;
      v27 = *(_QWORD *)v115;
      if ( v169 == *(_QWORD *)v115 )
        goto LABEL_181;
      ++v113;
    }
    if ( !v109 )
      v109 = v115;
    ++v174;
    v108 = v176 + 1;
    if ( 4 * ((int)v176 + 1) >= 3 * v106 )
    {
LABEL_165:
      v155 = v57;
      sub_2C47F20((__int64)&v174, 2 * v106);
      if ( !v177 )
        goto LABEL_323;
      v27 = v177 - 1;
      LODWORD(v107) = v27 & v161;
      v108 = v176 + 1;
      v57 = v155;
      v109 = v175 + 16LL * ((unsigned int)v27 & v161);
      v43 = *(_QWORD *)v109;
      if ( *(_QWORD *)v109 == v169 )
        goto LABEL_167;
      v154 = 1;
      v127 = 0;
      while ( v43 != -4096 )
      {
        if ( !v127 && v43 == -8192 )
          v127 = v109;
        v107 = (unsigned int)v27 & ((_DWORD)v107 + v154);
        v109 = v175 + 16 * v107;
        v43 = *(_QWORD *)v109;
        if ( v169 == *(_QWORD *)v109 )
          goto LABEL_167;
        ++v154;
      }
    }
    else
    {
      v43 = v106 - (v108 + HIDWORD(v176));
      if ( (unsigned int)v43 > v106 >> 3 )
        goto LABEL_167;
      v157 = v57;
      sub_2C47F20((__int64)&v174, v106);
      if ( !v177 )
      {
LABEL_323:
        LODWORD(v176) = v176 + 1;
        BUG();
      }
      v43 = v177 - 1;
      v127 = 0;
      v128 = 1;
      LODWORD(v129) = v43 & v161;
      v108 = v176 + 1;
      v57 = v157;
      v109 = v175 + 16LL * ((unsigned int)v43 & v161);
      v27 = *(_QWORD *)v109;
      if ( v169 == *(_QWORD *)v109 )
        goto LABEL_167;
      while ( v27 != -4096 )
      {
        if ( v27 == -8192 && !v127 )
          v127 = v109;
        v129 = (unsigned int)v43 & ((_DWORD)v129 + v128);
        v109 = v175 + 16 * v129;
        v27 = *(_QWORD *)v109;
        if ( v169 == *(_QWORD *)v109 )
          goto LABEL_167;
        ++v128;
      }
    }
    if ( v127 )
      v109 = v127;
LABEL_167:
    LODWORD(v176) = v108;
    if ( *(_QWORD *)v109 != -4096 )
      --HIDWORD(v176);
    *(_DWORD *)(v109 + 8) = 0;
    *(_QWORD *)v109 = v169;
    goto LABEL_170;
  }
LABEL_181:
  if ( v112 >= *(_DWORD *)(v115 + 8) )
    goto LABEL_170;
  v98 = sub_CB72A0();
  v116 = (__m128i *)*((_QWORD *)v98 + 4);
  if ( *((_QWORD *)v98 + 3) - (_QWORD)v116 > 0xFu )
  {
    *v116 = _mm_load_si128((const __m128i *)&xmmword_43A1630);
    *((_QWORD *)v98 + 4) += 16LL;
    goto LABEL_150;
  }
LABEL_205:
  sub_CB6200((__int64)v98, "Use before def!\n", 0x10u);
LABEL_150:
  sub_C7D6A0(v175, 16LL * v177, 8);
  return 0;
}
