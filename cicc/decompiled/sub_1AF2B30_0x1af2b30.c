// Function: sub_1AF2B30
// Address: 0x1af2b30
//
__int64 __fastcall sub_1AF2B30(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 v11; // rbx
  int v12; // r12d
  unsigned __int64 v13; // rsi
  const __m128i *v14; // rsi
  __int64 v15; // r15
  _QWORD *v16; // rax
  int v17; // ebx
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r8
  int v21; // r13d
  unsigned __int64 v22; // rax
  int v23; // edx
  const __m128i *v24; // rsi
  _QWORD *v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 k; // r15
  char v31; // r9
  __int64 v32; // rax
  char v33; // r8
  unsigned int v34; // edi
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r14
  __int64 v40; // r13
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 *v45; // r12
  __int64 *v46; // rbx
  _QWORD *v47; // rax
  __int64 v48; // rsi
  __int64 *v49; // rdi
  __int64 *v50; // rcx
  __int64 *v51; // rax
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // r13
  _QWORD *v55; // rax
  __int64 v56; // r12
  __int64 v57; // rbx
  __int64 *v58; // r14
  unsigned __int64 v59; // rcx
  __int64 v60; // rax
  __int64 *v61; // rax
  int v62; // ebx
  unsigned __int64 v63; // rax
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 v66; // r12
  __int64 v67; // r13
  _QWORD *v68; // rax
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  _QWORD *v71; // rax
  const __m128i *v72; // rsi
  __int64 i; // rbx
  __int64 v74; // r13
  _QWORD *v75; // rax
  _QWORD *v76; // rcx
  __int64 v77; // r13
  int v78; // r8d
  int v79; // r9d
  __int64 v80; // rbx
  __int64 v81; // r12
  signed __int64 v82; // r12
  __int64 **v83; // rbx
  _QWORD *v84; // rax
  __int32 v85; // r12d
  __int64 j; // r15
  unsigned int v87; // ecx
  unsigned int v88; // esi
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 v93; // r10
  __int64 v94; // r11
  __int64 v95; // rdx
  __int64 v96; // r13
  unsigned int v97; // edi
  _QWORD *v98; // rax
  __int64 v99; // rcx
  __int64 v100; // r12
  __int64 v101; // r14
  __int64 v102; // rbx
  __int64 v103; // rdi
  __int64 v104; // r13
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  int v109; // eax
  __int64 v110; // rax
  int v111; // edx
  __int64 v112; // rdx
  __int64 *v113; // rax
  __int64 v114; // rdi
  unsigned __int64 v115; // rdx
  __int64 v116; // rdx
  __int64 v117; // rdx
  __int64 v118; // r9
  __int64 v119; // rax
  __int64 v120; // rsi
  __int64 v121; // rdi
  _QWORD *v122; // rdx
  __int64 v123; // r11
  unsigned int v124; // r10d
  __int64 *v125; // rdi
  __int64 v126; // r12
  __int64 v127; // r9
  __int64 v128; // rdi
  unsigned __int64 v129; // rcx
  __int64 v130; // rcx
  _QWORD *v131; // rdx
  int v132; // eax
  int v133; // edi
  int v134; // r13d
  __int64 v135; // r13
  unsigned int v136; // ebx
  __int64 v137; // r12
  __int64 v138; // rdi
  __int64 v139; // rdx
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // r9
  int v143; // eax
  __int64 v144; // rax
  int v145; // edx
  __int64 v146; // rdx
  __int64 *v147; // rax
  __int64 v148; // rdi
  unsigned __int64 v149; // rdx
  __int64 v150; // rdx
  __int64 v151; // rdx
  __int64 v152; // rdi
  __int64 v153; // rbx
  unsigned int v154; // ecx
  __int64 v155; // r9
  int v156; // r8d
  _QWORD *v157; // rdi
  _QWORD *v158; // rsi
  int v159; // edi
  unsigned int v160; // r14d
  __int64 v161; // r8
  __int64 v162; // r12
  __int64 v163; // r15
  __int64 v164; // r14
  __int64 v165; // rdx
  __int64 v166; // rbx
  __int64 v167; // rdx
  __int64 v168; // r10
  __int64 v169; // rax
  char v170; // r9
  unsigned int v171; // esi
  __int64 v172; // rdx
  __int64 v173; // rax
  __int64 v174; // rcx
  __int64 v175; // rcx
  __int64 v176; // rax
  __int64 v177; // [rsp+0h] [rbp-170h]
  __int64 v178; // [rsp+0h] [rbp-170h]
  unsigned __int64 v179; // [rsp+18h] [rbp-158h]
  __int64 v180; // [rsp+18h] [rbp-158h]
  __int64 v181; // [rsp+18h] [rbp-158h]
  __int64 v182; // [rsp+18h] [rbp-158h]
  __int64 v184; // [rsp+28h] [rbp-148h]
  __int64 v185; // [rsp+28h] [rbp-148h]
  int v186; // [rsp+28h] [rbp-148h]
  __int64 v187; // [rsp+28h] [rbp-148h]
  __int64 v188; // [rsp+28h] [rbp-148h]
  __int64 v189; // [rsp+28h] [rbp-148h]
  __int64 v191; // [rsp+38h] [rbp-138h]
  __int64 v192; // [rsp+38h] [rbp-138h]
  __int64 v193; // [rsp+48h] [rbp-128h] BYREF
  const __m128i *v194; // [rsp+50h] [rbp-120h] BYREF
  __m128i *v195; // [rsp+58h] [rbp-118h]
  const __m128i *v196; // [rsp+60h] [rbp-110h]
  __int64 v197; // [rsp+70h] [rbp-100h] BYREF
  __int64 v198; // [rsp+78h] [rbp-F8h]
  __int64 v199; // [rsp+80h] [rbp-F0h]
  unsigned int v200; // [rsp+88h] [rbp-E8h]
  __m128i v201; // [rsp+90h] [rbp-E0h] BYREF
  __int64 *v202; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v203; // [rsp+A8h] [rbp-C8h]
  int v204; // [rsp+B0h] [rbp-C0h]
  _BYTE v205[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v10 = *(_QWORD *)(sub_157EBA0(a1) - 24);
  v193 = v10;
  if ( a1 == v10 )
    return 0;
  if ( sub_157F0B0(v10) )
  {
LABEL_3:
    if ( !sub_157F0B0(v193) )
    {
      for ( i = *(_QWORD *)(a1 + 48); ; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        if ( *(_BYTE *)(i - 8) != 77 )
          goto LABEL_4;
        v74 = *(_QWORD *)(i - 16);
        if ( v74 )
          break;
LABEL_129:
        ;
      }
      while ( 1 )
      {
        v75 = sub_1648700(v74);
        if ( *((_BYTE *)v75 + 16) != 77 )
          return 0;
        v76 = (*((_BYTE *)v75 + 23) & 0x40) != 0 ? (_QWORD *)*(v75 - 1) : &v75[-3 * (*((_DWORD *)v75 + 5) & 0xFFFFFFF)];
        if ( a1 != v76[3 * *((unsigned int *)v75 + 14) + 1 + -1431655765 * (unsigned int)((v74 - (__int64)v76) >> 3)] )
          return 0;
        v74 = *(_QWORD *)(v74 + 8);
        if ( !v74 )
          goto LABEL_129;
      }
    }
LABEL_4:
    v194 = 0;
    v195 = 0;
    v196 = 0;
    if ( a2 )
    {
      v11 = *(_QWORD *)(a1 + 8);
      if ( v11 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v11) + 16) - 25) > 9u )
        {
          v11 = *(_QWORD *)(v11 + 8);
          if ( !v11 )
            goto LABEL_109;
        }
        v12 = 0;
        while ( 1 )
        {
          v11 = *(_QWORD *)(v11 + 8);
          if ( !v11 )
            break;
          while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v11) + 16) - 25) <= 9u )
          {
            v11 = *(_QWORD *)(v11 + 8);
            ++v12;
            if ( !v11 )
              goto LABEL_11;
          }
        }
LABEL_11:
        v13 = (unsigned int)(2 * v12 + 3);
      }
      else
      {
LABEL_109:
        v13 = 1;
      }
      sub_1953AE0(&v194, v13);
      v14 = v195;
      v201.m128i_i64[0] = a1;
      v201.m128i_i64[1] = v193 | 4;
      if ( v195 == v196 )
      {
        sub_17F2860(&v194, v195, &v201);
      }
      else
      {
        if ( v195 )
        {
          a5 = _mm_loadu_si128(&v201);
          *v195 = a5;
          v14 = v195;
        }
        v195 = (__m128i *)&v14[1];
      }
      v15 = *(_QWORD *)(a1 + 8);
      if ( v15 )
      {
        while ( 1 )
        {
          v16 = sub_1648700(v15);
          if ( (unsigned __int8)(*((_BYTE *)v16 + 16) - 25) <= 9u )
            break;
          v15 = *(_QWORD *)(v15 + 8);
          if ( !v15 )
            goto LABEL_74;
        }
LABEL_24:
        v24 = v195;
        v201.m128i_i64[0] = v16[5];
        v201.m128i_i64[1] = a1 | 4;
        if ( v195 == v196 )
        {
          sub_17F2860(&v194, v195, &v201);
        }
        else
        {
          if ( v195 )
          {
            a3 = (__m128)_mm_loadu_si128(&v201);
            *v195 = (__m128i)a3;
            v24 = v195;
          }
          v195 = (__m128i *)&v24[1];
        }
        v25 = sub_1648700(v15);
        v26 = sub_157EBA0(v25[5]);
        v20 = v26;
        if ( v26 )
        {
          v17 = sub_15F4D60(v26);
          v18 = sub_1648700(v15)[5];
          v19 = sub_157EBA0(v18);
          v20 = v19;
          if ( v19 )
          {
            v179 = v19;
            v21 = sub_15F4D60(v19);
            v22 = sub_157EBA0(v18);
            v20 = v179;
          }
          else
          {
            v22 = 0;
            v21 = 0;
          }
        }
        else
        {
          v22 = 0;
          v17 = 0;
          v21 = 0;
        }
        v201.m128i_i64[0] = v22;
        v201.m128i_i32[2] = 0;
        v202 = (__int64 *)v20;
        LODWORD(v203) = v21;
        sub_1AED640((__int64)&v201, &v193);
        if ( v17 == v23 )
        {
          v71 = sub_1648700(v15);
          v72 = v195;
          v201.m128i_i64[0] = v71[5];
          v201.m128i_i64[1] = v193 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v195 == v196 )
          {
            sub_17F2860(&v194, v195, &v201);
          }
          else
          {
            if ( v195 )
            {
              a4 = _mm_loadu_si128(&v201);
              *v195 = a4;
              v72 = v195;
            }
            v195 = (__m128i *)&v72[1];
          }
        }
        while ( 1 )
        {
          v15 = *(_QWORD *)(v15 + 8);
          if ( !v15 )
            break;
          v16 = sub_1648700(v15);
          if ( (unsigned __int8)(*((_BYTE *)v16 + 16) - 25) <= 9u )
            goto LABEL_24;
        }
      }
    }
LABEL_74:
    v52 = v193;
    v53 = *(_QWORD *)(v193 + 48);
    if ( !v53 )
      BUG();
    if ( *(_BYTE *)(v53 - 8) != 77 )
      goto LABEL_76;
    v77 = *(_QWORD *)(a1 + 8);
    if ( v77 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v77) + 16) - 25) > 9u )
      {
        v77 = *(_QWORD *)(v77 + 8);
        if ( !v77 )
          goto LABEL_204;
      }
      v80 = v77;
      v81 = 0;
      v201.m128i_i64[0] = (__int64)&v202;
      v201.m128i_i64[1] = 0x1000000000LL;
      while ( 1 )
      {
        v80 = *(_QWORD *)(v80 + 8);
        if ( !v80 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v80) + 16) - 25) <= 9u )
        {
          v80 = *(_QWORD *)(v80 + 8);
          ++v81;
          if ( !v80 )
            goto LABEL_139;
        }
      }
LABEL_139:
      v82 = v81 + 1;
      if ( v82 > 16 )
      {
        sub_16CD150((__int64)&v201, &v202, v82, 8, v78, v79);
        v83 = (__int64 **)(v201.m128i_i64[0] + 8LL * v201.m128i_u32[2]);
      }
      else
      {
        v83 = &v202;
      }
      v84 = sub_1648700(v77);
LABEL_144:
      if ( v83 )
        *v83 = (__int64 *)v84[5];
      while ( 1 )
      {
        v77 = *(_QWORD *)(v77 + 8);
        if ( !v77 )
          break;
        v84 = sub_1648700(v77);
        if ( (unsigned __int8)(*((_BYTE *)v84 + 16) - 25) <= 9u )
        {
          ++v83;
          goto LABEL_144;
        }
      }
      v52 = v193;
      v85 = v201.m128i_i32[2] + v82;
    }
    else
    {
LABEL_204:
      v85 = 0;
      v201.m128i_i32[3] = 16;
      v201.m128i_i64[0] = (__int64)&v202;
    }
    v201.m128i_i32[2] = v85;
    for ( j = *(_QWORD *)(v52 + 48); ; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      v192 = j - 24;
      if ( *(_BYTE *)(j - 8) != 77 )
      {
        if ( (__int64 **)v201.m128i_i64[0] != &v202 )
          _libc_free(v201.m128i_u64[0]);
        v52 = v193;
LABEL_76:
        if ( sub_157F0B0(v52) )
        {
          v54 = a1 + 40;
          v55 = (_QWORD *)sub_157EBA0(a1);
          sub_15F20C0(v55);
          v56 = v193;
          v57 = sub_157ED20(v193);
          if ( a1 + 40 != (*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v58 = *(__int64 **)(a1 + 48);
            if ( v54 != v57 + 24 )
            {
              if ( v56 + 40 != v54 )
                sub_157EA80(v56 + 40, v54, (__int64)v58, v54);
              if ( (__int64 *)v54 != v58 )
              {
                v59 = *(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v58 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v54;
                *(_QWORD *)(a1 + 40) = *(_QWORD *)(a1 + 40) & 7LL | *v58 & 0xFFFFFFFFFFFFFFF8LL;
                v60 = *(_QWORD *)(v57 + 24);
                *(_QWORD *)(v59 + 8) = v57 + 24;
                v60 &= 0xFFFFFFFFFFFFFFF8LL;
                *v58 = v60 | *v58 & 7;
                *(_QWORD *)(v60 + 8) = v58;
                *(_QWORD *)(v57 + 24) = v59 | *(_QWORD *)(v57 + 24) & 7LL;
              }
            }
          }
        }
        else
        {
          v69 = *(_QWORD *)(a1 + 48);
          if ( !v69 )
LABEL_308:
            BUG();
          while ( *(_BYTE *)(v69 - 8) == 77 )
          {
            sub_15F20C0((_QWORD *)(v69 - 24));
            v69 = *(_QWORD *)(a1 + 48);
            if ( !v69 )
              goto LABEL_308;
          }
        }
        v61 = (__int64 *)sub_157E9C0(a1);
        v62 = sub_1602B80(v61, "llvm.loop", 9u);
        v63 = sub_157EBA0(a1);
        if ( v63 && (*(_QWORD *)(v63 + 48) || *(__int16 *)(v63 + 18) < 0) )
        {
          v66 = sub_1625790(v63, v62);
          if ( v66 )
          {
            v67 = *(_QWORD *)(a1 + 8);
            if ( v67 )
            {
              while ( 1 )
              {
                v68 = sub_1648700(v67);
                if ( (unsigned __int8)(*((_BYTE *)v68 + 16) - 25) <= 9u )
                  break;
                v67 = *(_QWORD *)(v67 + 8);
                if ( !v67 )
                  goto LABEL_89;
              }
LABEL_102:
              v70 = sub_157EBA0(v68[5]);
              sub_1625C10(v70, v62, v66);
              while ( 1 )
              {
                v67 = *(_QWORD *)(v67 + 8);
                if ( !v67 )
                  break;
                v68 = sub_1648700(v67);
                if ( (unsigned __int8)(*((_BYTE *)v68 + 16) - 25) <= 9u )
                  goto LABEL_102;
              }
            }
          }
        }
LABEL_89:
        sub_164D160(a1, v193, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v64, v65, a9, a10);
        if ( (*(_BYTE *)(v193 + 23) & 0x20) == 0 )
          sub_164B7C0(v193, a1);
        if ( a2 )
        {
          sub_15CD5A0(a2, a1);
          sub_15CD9D0(a2, v194->m128i_i64, v195 - v194);
        }
        else
        {
          sub_157F980(a1);
        }
        if ( v194 )
          j_j___libc_free_0(v194, (char *)v196 - (char *)v194);
        return 1;
      }
      v87 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
      if ( v87 )
      {
        v88 = 0;
        v89 = 24LL * *(unsigned int *)(j + 32) + 8;
        while ( 1 )
        {
          v90 = j - 24 - 24LL * v87;
          if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
            v90 = *(_QWORD *)(j - 32);
          if ( a1 == *(_QWORD *)(v90 + v89) )
            break;
          ++v88;
          v89 += 8;
          if ( v87 == v88 )
            goto LABEL_202;
        }
      }
      else
      {
LABEL_202:
        v88 = -1;
      }
      v91 = 0;
      v92 = sub_15F5350(v192, v88, 0);
      v93 = j - 24;
      v197 = 0;
      v180 = v92;
      v198 = 0;
      v199 = 0;
      v200 = 0;
      v94 = 8LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
      if ( (*(_DWORD *)(j - 4) & 0xFFFFFFF) != 0 )
        break;
LABEL_166:
      if ( *(_BYTE *)(v180 + 16) == 77 && a1 == *(_QWORD *)(v180 + 40) )
      {
        v135 = v180;
        v136 = *(_DWORD *)(v180 + 20) & 0xFFFFFFF;
        if ( v136 )
        {
          v137 = 0;
          v187 = 8LL * v136;
          do
          {
            if ( (*(_BYTE *)(v135 + 23) & 0x40) != 0 )
              v138 = *(_QWORD *)(v135 - 8);
            else
              v138 = v135 - 24LL * (*(_DWORD *)(v135 + 20) & 0xFFFFFFF);
            v153 = *(_QWORD *)(v137 + v138 + 24LL * *(unsigned int *)(v135 + 56) + 8);
            v141 = sub_1AF2830(*(_QWORD *)(v138 + 3 * v137), v153, (__int64)&v197);
            v143 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
            if ( v143 == *(_DWORD *)(j + 32) )
            {
              v182 = v141;
              sub_15F55D0(v192, v153, v139, v140, v141, v142);
              v141 = v182;
              v143 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
            }
            v144 = (v143 + 1) & 0xFFFFFFF;
            v145 = v144 | *(_DWORD *)(j - 4) & 0xF0000000;
            *(_DWORD *)(j - 4) = v145;
            if ( (v145 & 0x40000000) != 0 )
              v146 = *(_QWORD *)(j - 32);
            else
              v146 = v192 - 24 * v144;
            v147 = (__int64 *)(v146 + 24LL * (unsigned int)(v144 - 1));
            if ( *v147 )
            {
              v148 = v147[1];
              v149 = v147[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v149 = v148;
              if ( v148 )
                *(_QWORD *)(v148 + 16) = *(_QWORD *)(v148 + 16) & 3LL | v149;
            }
            *v147 = v141;
            if ( v141 )
            {
              v150 = *(_QWORD *)(v141 + 8);
              v147[1] = v150;
              if ( v150 )
                *(_QWORD *)(v150 + 16) = (unsigned __int64)(v147 + 1) | *(_QWORD *)(v150 + 16) & 3LL;
              v147[2] = (v141 + 8) | v147[2] & 3;
              *(_QWORD *)(v141 + 8) = v147;
            }
            v151 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
            if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
              v152 = *(_QWORD *)(j - 32);
            else
              v152 = v192 - 24 * v151;
            v137 += 8;
            *(_QWORD *)(v152 + 8LL * (unsigned int)(v151 - 1) + 24LL * *(unsigned int *)(j + 32) + 8) = v153;
          }
          while ( v137 != v187 );
        }
      }
      else if ( v201.m128i_i32[2] )
      {
        v185 = 8LL * v201.m128i_u32[2];
        v101 = v180;
        v102 = 0;
        do
        {
          v104 = *(_QWORD *)(v201.m128i_i64[0] + v102);
          v107 = sub_1AF2830(v101, v104, (__int64)&v197);
          v109 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
          if ( v109 == *(_DWORD *)(j + 32) )
          {
            v181 = v107;
            sub_15F55D0(v192, v104, v105, v106, v107, v108);
            v107 = v181;
            v109 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
          }
          v110 = (v109 + 1) & 0xFFFFFFF;
          v111 = v110 | *(_DWORD *)(j - 4) & 0xF0000000;
          *(_DWORD *)(j - 4) = v111;
          if ( (v111 & 0x40000000) != 0 )
            v112 = *(_QWORD *)(j - 32);
          else
            v112 = v192 - 24 * v110;
          v113 = (__int64 *)(v112 + 24LL * (unsigned int)(v110 - 1));
          if ( *v113 )
          {
            v114 = v113[1];
            v115 = v113[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v115 = v114;
            if ( v114 )
              *(_QWORD *)(v114 + 16) = *(_QWORD *)(v114 + 16) & 3LL | v115;
          }
          *v113 = v107;
          if ( v107 )
          {
            v116 = *(_QWORD *)(v107 + 8);
            v113[1] = v116;
            if ( v116 )
              *(_QWORD *)(v116 + 16) = (unsigned __int64)(v113 + 1) | *(_QWORD *)(v116 + 16) & 3LL;
            v113[2] = (v107 + 8) | v113[2] & 3;
            *(_QWORD *)(v107 + 8) = v113;
          }
          v117 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
          if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
            v103 = *(_QWORD *)(j - 32);
          else
            v103 = v192 - 24 * v117;
          v102 += 8;
          *(_QWORD *)(v103 + 8LL * (unsigned int)(v117 - 1) + 24LL * *(unsigned int *)(j + 32) + 8) = v104;
        }
        while ( v185 != v102 );
      }
      v118 = v198;
      if ( (*(_DWORD *)(j - 4) & 0xFFFFFFF) != 0 )
      {
        v119 = 0;
        v120 = 8LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
        while ( 1 )
        {
          if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
          {
            v121 = *(_QWORD *)(j - 32);
            v122 = (_QWORD *)(v121 + 3 * v119);
            if ( *(_BYTE *)(*v122 + 16LL) != 9 )
              goto LABEL_188;
          }
          else
          {
            v121 = v192 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
            v122 = (_QWORD *)(v121 + 3 * v119);
            if ( *(_BYTE *)(*v122 + 16LL) != 9 )
              goto LABEL_188;
          }
          if ( !v200 )
            goto LABEL_188;
          v123 = *(_QWORD *)(v119 + v121 + 24LL * *(unsigned int *)(j + 32) + 8);
          v124 = (v200 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
          v125 = (__int64 *)(v118 + 16LL * v124);
          v126 = *v125;
          if ( v123 != *v125 )
          {
            v133 = 1;
            while ( v126 != -8 )
            {
              v134 = v133 + 1;
              v124 = (v200 - 1) & (v133 + v124);
              v125 = (__int64 *)(v118 + 16LL * v124);
              v126 = *v125;
              if ( v123 == *v125 )
                goto LABEL_193;
              v133 = v134;
            }
            goto LABEL_188;
          }
LABEL_193:
          if ( v125 == (__int64 *)(v118 + 16LL * v200) )
          {
LABEL_188:
            v119 += 8;
            if ( v120 == v119 )
              break;
          }
          else
          {
            v127 = v122[1];
            v128 = v125[1];
            v129 = v122[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v129 = v127;
            if ( v127 )
              *(_QWORD *)(v127 + 16) = *(_QWORD *)(v127 + 16) & 3LL | v129;
            *v122 = v128;
            if ( v128 )
            {
              v130 = *(_QWORD *)(v128 + 8);
              v122[1] = v130;
              if ( v130 )
                *(_QWORD *)(v130 + 16) = (unsigned __int64)(v122 + 1) | *(_QWORD *)(v130 + 16) & 3LL;
              v122[2] = (v128 + 8) | v122[2] & 3LL;
              *(_QWORD *)(v128 + 8) = v122;
            }
            v119 += 8;
            v118 = v198;
            if ( v120 == v119 )
              break;
          }
        }
      }
      j___libc_free_0(v118);
    }
    while ( 1 )
    {
      if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
        v95 = *(_QWORD *)(j - 32);
      else
        v95 = v93 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
      v100 = *(_QWORD *)(v95 + 24LL * *(unsigned int *)(j + 32) + 8 + v91);
      v96 = *(_QWORD *)(v95 + 3 * v91);
      if ( *(_BYTE *)(v96 + 16) != 9 )
      {
        if ( !v200 )
        {
          ++v197;
          goto LABEL_241;
        }
        v97 = (v200 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
        v98 = (_QWORD *)(v198 + 16LL * v97);
        v99 = *v98;
        if ( v100 != *v98 )
        {
          v186 = 1;
          v131 = 0;
          while ( v99 != -8 )
          {
            if ( v99 != -16 || v131 )
              v98 = v131;
            v97 = (v200 - 1) & (v186 + v97);
            v99 = *(_QWORD *)(v198 + 16LL * v97);
            if ( v100 == v99 )
              goto LABEL_163;
            ++v186;
            v131 = v98;
            v98 = (_QWORD *)(v198 + 16LL * v97);
          }
          if ( !v131 )
            v131 = v98;
          ++v197;
          v132 = v199 + 1;
          if ( 4 * ((int)v199 + 1) < 3 * v200 )
          {
            if ( v200 - HIDWORD(v199) - v132 <= v200 >> 3 )
            {
              v178 = v93;
              v189 = v94;
              sub_141A900((__int64)&v197, v200);
              if ( !v200 )
              {
LABEL_311:
                LODWORD(v199) = v199 + 1;
                BUG();
              }
              v158 = 0;
              v159 = 1;
              v160 = (v200 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
              v94 = v189;
              v132 = v199 + 1;
              v93 = v178;
              v131 = (_QWORD *)(v198 + 16LL * v160);
              v161 = *v131;
              if ( v100 != *v131 )
              {
                while ( v161 != -8 )
                {
                  if ( v161 == -16 && !v158 )
                    v158 = v131;
                  v160 = (v200 - 1) & (v159 + v160);
                  v131 = (_QWORD *)(v198 + 16LL * v160);
                  v161 = *v131;
                  if ( v100 == *v131 )
                    goto LABEL_211;
                  ++v159;
                }
                if ( v158 )
                  v131 = v158;
              }
            }
            goto LABEL_211;
          }
LABEL_241:
          v177 = v93;
          v188 = v94;
          sub_141A900((__int64)&v197, 2 * v200);
          if ( !v200 )
            goto LABEL_311;
          v94 = v188;
          v93 = v177;
          v154 = (v200 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
          v132 = v199 + 1;
          v131 = (_QWORD *)(v198 + 16LL * v154);
          v155 = *v131;
          if ( *v131 != v100 )
          {
            v156 = 1;
            v157 = 0;
            while ( v155 != -8 )
            {
              if ( !v157 && v155 == -16 )
                v157 = v131;
              v154 = (v200 - 1) & (v156 + v154);
              v131 = (_QWORD *)(v198 + 16LL * v154);
              v155 = *v131;
              if ( v100 == *v131 )
                goto LABEL_211;
              ++v156;
            }
            if ( v157 )
              v131 = v157;
          }
LABEL_211:
          LODWORD(v199) = v132;
          if ( *v131 != -8 )
            --HIDWORD(v199);
          *v131 = v100;
          v131[1] = v96;
        }
      }
LABEL_163:
      v91 += 8;
      if ( v94 == v91 )
        goto LABEL_166;
    }
  }
  v27 = *(_QWORD *)(a1 + 8);
  if ( v27 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v27) + 16) - 25) > 9u )
    {
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        goto LABEL_31;
    }
    v45 = (__int64 *)v205;
    v201.m128i_i64[0] = 0;
    v201.m128i_i64[1] = (__int64)v205;
    v46 = (__int64 *)v205;
    v202 = (__int64 *)v205;
    v203 = 16;
    v204 = 0;
    v47 = sub_1648700(v27);
LABEL_64:
    v48 = v47[5];
    if ( v46 != v45 )
      goto LABEL_61;
    v49 = &v46[HIDWORD(v203)];
    if ( v49 != v46 )
    {
      v50 = 0;
      v51 = v46;
      do
      {
        if ( v48 == *v51 )
          goto LABEL_62;
        if ( *v51 == -2 )
          v50 = v51;
        ++v51;
      }
      while ( v49 != v51 );
      if ( v50 )
      {
        *v50 = v48;
        v46 = v202;
        --v204;
        v45 = (__int64 *)v201.m128i_i64[1];
        ++v201.m128i_i64[0];
        goto LABEL_62;
      }
    }
    if ( HIDWORD(v203) >= (unsigned int)v203 )
    {
LABEL_61:
      sub_16CCBA0((__int64)&v201, v48);
      v46 = v202;
      v45 = (__int64 *)v201.m128i_i64[1];
      goto LABEL_62;
    }
    ++HIDWORD(v203);
    *v49 = v48;
    v45 = (__int64 *)v201.m128i_i64[1];
    ++v201.m128i_i64[0];
    v46 = v202;
LABEL_62:
    while ( 1 )
    {
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        break;
      v47 = sub_1648700(v27);
      if ( (unsigned __int8)(*((_BYTE *)v47 + 16) - 25) <= 9u )
        goto LABEL_64;
    }
  }
  else
  {
LABEL_31:
    v201.m128i_i64[0] = 0;
    v201.m128i_i64[1] = (__int64)v205;
    v202 = (__int64 *)v205;
    v203 = 16;
    v204 = 0;
  }
  v28 = *(_QWORD *)(v10 + 48);
  v29 = a1;
  for ( k = v28; ; k = *(_QWORD *)(k + 8) )
  {
    if ( !k )
      BUG();
    if ( *(_BYTE *)(k - 8) != 77 )
    {
      if ( (__int64 *)v201.m128i_i64[1] != v202 )
        _libc_free((unsigned __int64)v202);
      goto LABEL_3;
    }
    v31 = *(_BYTE *)(k - 1);
    v32 = 0x17FFFFFFE8LL;
    v191 = k - 24;
    v33 = v31 & 0x40;
    v34 = *(_DWORD *)(k - 4) & 0xFFFFFFF;
    if ( v34 )
    {
      v35 = 24LL * *(unsigned int *)(k + 32) + 8;
      v36 = 0;
      do
      {
        v37 = k - 24 - 24LL * v34;
        if ( v33 )
          v37 = *(_QWORD *)(k - 32);
        if ( v29 == *(_QWORD *)(v37 + v35) )
        {
          v32 = 24 * v36;
          goto LABEL_42;
        }
        ++v36;
        v35 += 8;
      }
      while ( v34 != (_DWORD)v36 );
      v32 = 0x17FFFFFFE8LL;
      if ( v33 )
      {
LABEL_96:
        v38 = *(_QWORD *)(k - 32);
        goto LABEL_44;
      }
    }
    else
    {
LABEL_42:
      if ( v33 )
        goto LABEL_96;
    }
    v38 = v191 - 24LL * v34;
LABEL_44:
    v39 = *(_QWORD *)(v38 + v32);
    if ( *(_BYTE *)(v39 + 16) != 77 || v29 != *(_QWORD *)(v39 + 40) )
      break;
    if ( v34 )
    {
      v162 = k;
      v163 = *(_QWORD *)(v38 + v32);
      v164 = 0;
      while ( 1 )
      {
        if ( (v31 & 0x40) != 0 )
          v165 = *(_QWORD *)(v162 - 32);
        else
          v165 = v191 - 24LL * (*(_DWORD *)(v162 - 4) & 0xFFFFFFF);
        v166 = *(_QWORD *)(v164 + v165 + 24LL * *(unsigned int *)(v162 + 32) + 8);
        if ( sub_183E920((__int64)&v201, v166) )
        {
          if ( (*(_BYTE *)(v162 - 1) & 0x40) != 0 )
            v167 = *(_QWORD *)(v162 - 32);
          else
            v167 = v191 - 24LL * (*(_DWORD *)(v162 - 4) & 0xFFFFFFF);
          v168 = *(_QWORD *)(v167 + 3 * v164);
          v169 = 0x17FFFFFFE8LL;
          v170 = *(_BYTE *)(v163 + 23) & 0x40;
          v171 = *(_DWORD *)(v163 + 20) & 0xFFFFFFF;
          if ( v171 )
          {
            v172 = 24LL * *(unsigned int *)(v163 + 56) + 8;
            v173 = 0;
            do
            {
              v174 = v163 - 24LL * v171;
              if ( v170 )
                v174 = *(_QWORD *)(v163 - 8);
              if ( v166 == *(_QWORD *)(v174 + v172) )
              {
                v169 = 24 * v173;
                goto LABEL_272;
              }
              ++v173;
              v172 += 8;
            }
            while ( v171 != (_DWORD)v173 );
            v169 = 0x17FFFFFFE8LL;
          }
LABEL_272:
          v175 = v170 ? *(_QWORD *)(v163 - 8) : v163 - 24LL * v171;
          v176 = *(_QWORD *)(v175 + v169);
          if ( v168 != v176 && *(_BYTE *)(v176 + 16) != 9 && *(_BYTE *)(v168 + 16) != 9 )
            goto LABEL_55;
        }
        v164 += 8;
        if ( 8LL * v34 == v164 )
          break;
        v31 = *(_BYTE *)(v162 - 1);
      }
      k = v162;
    }
LABEL_131:
    ;
  }
  if ( !v34 )
    goto LABEL_131;
  v184 = v29;
  v40 = 0;
  if ( (*(_BYTE *)(k - 1) & 0x40) != 0 )
  {
LABEL_47:
    v41 = *(_QWORD *)(k - 32);
    goto LABEL_48;
  }
  while ( 1 )
  {
    v41 = v191 - 24LL * (*(_DWORD *)(k - 4) & 0xFFFFFFF);
LABEL_48:
    if ( sub_183E920((__int64)&v201, *(_QWORD *)(v40 + v41 + 24LL * *(unsigned int *)(k + 32) + 8)) )
    {
      v42 = (*(_BYTE *)(k - 1) & 0x40) != 0 ? *(_QWORD *)(k - 32) : v191 - 24LL * (*(_DWORD *)(k - 4) & 0xFFFFFFF);
      v43 = *(_QWORD *)(v42 + 3 * v40);
      if ( (!v43 || v39 != v43) && *(_BYTE *)(v39 + 16) != 9 && *(_BYTE *)(v43 + 16) != 9 )
        break;
    }
    v40 += 8;
    if ( 8LL * v34 == v40 )
    {
      v29 = v184;
      goto LABEL_131;
    }
    if ( (*(_BYTE *)(k - 1) & 0x40) != 0 )
      goto LABEL_47;
  }
LABEL_55:
  if ( (__int64 *)v201.m128i_i64[1] != v202 )
    _libc_free((unsigned __int64)v202);
  return 0;
}
