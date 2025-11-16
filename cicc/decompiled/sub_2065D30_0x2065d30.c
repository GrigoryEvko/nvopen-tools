// Function: sub_2065D30
// Address: 0x2065d30
//
void __fastcall sub_2065D30(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r14
  __m128i *v11; // rsi
  int v12; // r8d
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // r15
  unsigned int v18; // esi
  __int64 v19; // r9
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r12
  unsigned int v26; // esi
  __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29; // r9
  unsigned int v30; // edi
  __int64 *v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r12
  __int64 v34; // rax
  void *v35; // rbx
  __int64 v36; // r15
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rax
  __m128i *v40; // rax
  unsigned __int32 v41; // edx
  _QWORD *v42; // rax
  __int64 v43; // rax
  __m128i v44; // xmm2
  _QWORD *v45; // rax
  _QWORD *v46; // rcx
  int v47; // eax
  int v48; // edx
  __int64 v49; // rax
  _QWORD *v50; // rax
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdi
  unsigned int v54; // r9d
  __int64 v55; // rsi
  int v56; // r11d
  _QWORD *v57; // r10
  int v58; // eax
  int v59; // eax
  __int64 v60; // rdi
  _QWORD *v61; // r11
  unsigned int v62; // r14d
  int v63; // r10d
  __int64 v64; // rsi
  int v65; // eax
  unsigned int v66; // eax
  __int64 v67; // r11
  unsigned int v68; // esi
  __int64 v69; // rdx
  unsigned int v70; // r15d
  __int64 v71; // r12
  __int64 v72; // rbx
  unsigned int v73; // edi
  __int64 v74; // rax
  __int64 v75; // rcx
  unsigned int v76; // edx
  __int64 v77; // rax
  unsigned __int32 v78; // ecx
  __int32 v79; // edx
  __int64 v80; // rdi
  int v81; // r14d
  __int64 v82; // r8
  unsigned __int32 v83; // ecx
  __int64 v84; // rax
  __int64 v85; // rdi
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // r13
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  int v94; // r9d
  __int64 *v95; // rax
  __int64 v96; // rcx
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // rbx
  __m128i *v99; // rax
  __int64 v100; // rsi
  int v101; // edx
  __int64 *v102; // rbx
  int v103; // r12d
  const __m128i *v104; // rdi
  __m128i *v105; // rax
  const __m128i *v106; // rsi
  __int64 *v107; // rcx
  const __m128i *v108; // rbx
  __m128i *v109; // r15
  const __m128i *v110; // rdx
  int v111; // r15d
  int v112; // eax
  int v113; // eax
  int v114; // edi
  __int64 v115; // rsi
  __int64 v116; // rax
  int v117; // r11d
  __int64 v118; // r10
  int v119; // r10d
  __int64 v120; // r9
  int v121; // eax
  int v122; // eax
  __int64 v123; // rdi
  int v124; // r10d
  __int64 v125; // rsi
  int v126; // r10d
  unsigned __int32 v127; // r14d
  __int64 v128; // rdi
  __int64 v129; // rsi
  int v130; // r15d
  __int64 v131; // rdx
  __int32 v132; // ecx
  __int64 v133; // rax
  __int64 v134; // r14
  int v135; // edi
  __int64 v136; // rsi
  int v137; // esi
  __int64 v138; // r14
  __int64 v139; // rax
  __int64 v140; // rdi
  __int64 *v141; // r15
  int v142; // r9d
  __int64 v143; // r10
  __int64 *v144; // [rsp+0h] [rbp-1C0h]
  int v145; // [rsp+8h] [rbp-1B8h]
  unsigned int v146; // [rsp+Ch] [rbp-1B4h]
  int v147; // [rsp+Ch] [rbp-1B4h]
  int v148; // [rsp+Ch] [rbp-1B4h]
  unsigned int v149; // [rsp+10h] [rbp-1B0h]
  __int64 v150; // [rsp+18h] [rbp-1A8h]
  const __m128i *v151; // [rsp+18h] [rbp-1A8h]
  __int64 v152; // [rsp+20h] [rbp-1A0h]
  __int8 *v153; // [rsp+20h] [rbp-1A0h]
  __int64 v154; // [rsp+20h] [rbp-1A0h]
  __int64 v155; // [rsp+20h] [rbp-1A0h]
  __int64 *v156; // [rsp+20h] [rbp-1A0h]
  __int64 v157; // [rsp+20h] [rbp-1A0h]
  __int64 v158; // [rsp+20h] [rbp-1A0h]
  __int64 v159; // [rsp+20h] [rbp-1A0h]
  __int64 v161; // [rsp+30h] [rbp-190h]
  __int128 v162; // [rsp+30h] [rbp-190h]
  __int64 v163; // [rsp+68h] [rbp-158h] BYREF
  const __m128i *v164; // [rsp+70h] [rbp-150h] BYREF
  __m128i *v165; // [rsp+78h] [rbp-148h]
  const __m128i *v166; // [rsp+80h] [rbp-140h]
  void *src[2]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v168[2]; // [rsp+A0h] [rbp-120h] BYREF
  int v169[4]; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v170; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v171; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v172; // [rsp+E0h] [rbp-E0h]

  v5 = a1;
  v7 = *(_QWORD *)(a1 + 712);
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v152 = *(_QWORD *)(v7 + 32);
  sub_20553A0(&v164, ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1);
  v150 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 == 1 )
    goto LABEL_24;
  v8 = 0;
  do
  {
    while ( 1 )
    {
      if ( (_DWORD)v8 == -2 )
      {
        v13 = 24;
        v12 = 0;
      }
      else
      {
        v12 = v8 + 1;
        v13 = 24LL * (unsigned int)(2 * v8 + 3);
      }
      v14 = *(_BYTE *)(a2 + 23) & 0x40;
      if ( v14 )
        v15 = *(_QWORD *)(a2 - 8);
      else
        v15 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v16 = *(_QWORD *)(v15 + v13);
      v17 = *(_QWORD *)(a1 + 712);
      v18 = *(_DWORD *)(v17 + 72);
      if ( !v18 )
      {
        ++*(_QWORD *)(v17 + 48);
LABEL_70:
        v147 = v12;
        sub_1D52F30(v17 + 48, 2 * v18);
        v51 = *(_DWORD *)(v17 + 72);
        if ( !v51 )
          goto LABEL_263;
        v52 = v51 - 1;
        v53 = *(_QWORD *)(v17 + 56);
        v12 = v147;
        v54 = v52 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v48 = *(_DWORD *)(v17 + 64) + 1;
        v46 = (_QWORD *)(v53 + 16LL * v54);
        v55 = *v46;
        if ( v16 != *v46 )
        {
          v56 = 1;
          v57 = 0;
          while ( v55 != -8 )
          {
            if ( v55 == -16 && !v57 )
              v57 = v46;
            v54 = v52 & (v56 + v54);
            v46 = (_QWORD *)(v53 + 16LL * v54);
            v55 = *v46;
            if ( v16 == *v46 )
              goto LABEL_60;
            ++v56;
          }
          if ( v57 )
            v46 = v57;
        }
        goto LABEL_60;
      }
      v19 = *(_QWORD *)(v17 + 56);
      v146 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (_QWORD *)(v19 + 16LL * v146);
      v21 = *v20;
      if ( v16 == *v20 )
      {
        v22 = v20[1];
        goto LABEL_17;
      }
      v145 = 1;
      v46 = 0;
      while ( v21 != -8 )
      {
        if ( v46 || v21 != -16 )
          v20 = v46;
        v146 = (v18 - 1) & (v146 + v145);
        v144 = (__int64 *)(v19 + 16LL * v146);
        v21 = *v144;
        if ( v16 == *v144 )
        {
          v22 = v144[1];
          goto LABEL_17;
        }
        ++v145;
        v46 = v20;
        v20 = (_QWORD *)(v19 + 16LL * v146);
      }
      v47 = *(_DWORD *)(v17 + 64);
      if ( !v46 )
        v46 = v20;
      ++*(_QWORD *)(v17 + 48);
      v48 = v47 + 1;
      if ( 4 * (v47 + 1) >= 3 * v18 )
        goto LABEL_70;
      if ( v18 - *(_DWORD *)(v17 + 68) - v48 <= v18 >> 3 )
      {
        v148 = v12;
        sub_1D52F30(v17 + 48, v18);
        v58 = *(_DWORD *)(v17 + 72);
        if ( !v58 )
        {
LABEL_263:
          ++*(_DWORD *)(v17 + 64);
          BUG();
        }
        v59 = v58 - 1;
        v60 = *(_QWORD *)(v17 + 56);
        v61 = 0;
        v62 = v59 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v12 = v148;
        v63 = 1;
        v48 = *(_DWORD *)(v17 + 64) + 1;
        v46 = (_QWORD *)(v60 + 16LL * v62);
        v64 = *v46;
        if ( v16 != *v46 )
        {
          while ( v64 != -8 )
          {
            if ( v61 || v64 != -16 )
              v46 = v61;
            v62 = v59 & (v63 + v62);
            v64 = *(_QWORD *)(v60 + 16LL * v62);
            if ( v16 == v64 )
            {
              v46 = (_QWORD *)(v60 + 16LL * v62);
              goto LABEL_60;
            }
            ++v63;
            v61 = v46;
            v46 = (_QWORD *)(v60 + 16LL * v62);
          }
          if ( v61 )
            v46 = v61;
        }
      }
LABEL_60:
      *(_DWORD *)(v17 + 64) = v48;
      if ( *v46 != -8 )
        --*(_DWORD *)(v17 + 68);
      *v46 = v16;
      v22 = 0;
      v46[1] = 0;
      v14 = *(_BYTE *)(a2 + 23) & 0x40;
LABEL_17:
      ++v8;
      v9 = v14 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v10 = *(_QWORD *)(v9 + 24LL * (unsigned int)(2 * v8));
      if ( v152 )
        LODWORD(src[0]) = sub_1377370(v152, *(_QWORD *)(a2 + 40), v12);
      else
        sub_16AF710(src, 1u, (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1);
      v170.m128i_i64[1] = v10;
      v171.m128i_i64[0] = v10;
      v11 = v165;
      v170.m128i_i32[0] = 0;
      v171.m128i_i64[1] = v22;
      LODWORD(v172) = src[0];
      if ( v165 != v166 )
        break;
      sub_205F070(&v164, v165, &v170);
      if ( v150 == v8 )
        goto LABEL_23;
    }
    if ( v165 )
    {
      *v165 = _mm_loadu_si128(&v170);
      v11[1] = _mm_loadu_si128(&v171);
      v11[2].m128i_i64[0] = v172;
      v11 = v165;
    }
    v165 = (__m128i *)((char *)v11 + 40);
  }
  while ( v150 != v8 );
LABEL_23:
  v5 = a1;
LABEL_24:
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v23 = *(_QWORD *)(a2 - 8);
  else
    v23 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v24 = *(_QWORD *)(v5 + 712);
  v25 = *(_QWORD *)(v23 + 24);
  v26 = *(_DWORD *)(v24 + 72);
  if ( !v26 )
  {
    ++*(_QWORD *)(v24 + 48);
    goto LABEL_149;
  }
  v27 = *(_QWORD *)(v24 + 56);
  v28 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
  v29 = v28;
  v30 = (v26 - 1) & v28;
  v31 = (__int64 *)(v27 + 16LL * v30);
  v32 = *v31;
  if ( v25 == *v31 )
  {
    v33 = v31[1];
    goto LABEL_29;
  }
  v111 = 1;
  v28 = 0;
  while ( v32 != -8 )
  {
    if ( v32 != -16 || v28 )
      v31 = (__int64 *)v28;
    v28 = (unsigned int)(v111 + 1);
    v30 = (v26 - 1) & (v111 + v30);
    v141 = (__int64 *)(v27 + 16LL * v30);
    v32 = *v141;
    if ( v25 == *v141 )
    {
      v33 = v141[1];
      goto LABEL_29;
    }
    v111 = v28;
    v28 = (__int64)v31;
    v31 = (__int64 *)(v27 + 16LL * v30);
  }
  if ( !v28 )
    v28 = (__int64)v31;
  v112 = *(_DWORD *)(v24 + 64);
  ++*(_QWORD *)(v24 + 48);
  v32 = (unsigned int)(v112 + 1);
  if ( 4 * (int)v32 >= 3 * v26 )
  {
LABEL_149:
    sub_1D52F30(v24 + 48, 2 * v26);
    v113 = *(_DWORD *)(v24 + 72);
    if ( v113 )
    {
      v114 = v113 - 1;
      v115 = *(_QWORD *)(v24 + 56);
      LODWORD(v116) = (v113 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v32 = (unsigned int)(*(_DWORD *)(v24 + 64) + 1);
      v28 = v115 + 16LL * (unsigned int)v116;
      v27 = *(_QWORD *)v28;
      if ( v25 != *(_QWORD *)v28 )
      {
        v117 = 1;
        v118 = 0;
        while ( v27 != -8 )
        {
          if ( !v118 && v27 == -16 )
            v118 = v28;
          v29 = (unsigned int)(v117 + 1);
          v116 = v114 & (unsigned int)(v116 + v117);
          v28 = v115 + 16 * v116;
          v27 = *(_QWORD *)v28;
          if ( v25 == *(_QWORD *)v28 )
            goto LABEL_144;
          ++v117;
        }
        if ( v118 )
          v28 = v118;
      }
      goto LABEL_144;
    }
    goto LABEL_260;
  }
  if ( v26 - *(_DWORD *)(v24 + 68) - (unsigned int)v32 <= v26 >> 3 )
  {
    sub_1D52F30(v24 + 48, v26);
    v121 = *(_DWORD *)(v24 + 72);
    if ( v121 )
    {
      v122 = v121 - 1;
      v27 = 0;
      v123 = *(_QWORD *)(v24 + 56);
      v124 = 1;
      v29 = v122 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v32 = (unsigned int)(*(_DWORD *)(v24 + 64) + 1);
      v28 = v123 + 16 * v29;
      v125 = *(_QWORD *)v28;
      if ( v25 != *(_QWORD *)v28 )
      {
        while ( v125 != -8 )
        {
          if ( v125 == -16 && !v27 )
            v27 = v28;
          v29 = v122 & (unsigned int)(v29 + v124);
          v28 = v123 + 16 * v29;
          v125 = *(_QWORD *)v28;
          if ( v25 == *(_QWORD *)v28 )
            goto LABEL_144;
          ++v124;
        }
        if ( v27 )
          v28 = v27;
      }
      goto LABEL_144;
    }
LABEL_260:
    ++*(_DWORD *)(v24 + 64);
    BUG();
  }
LABEL_144:
  *(_DWORD *)(v24 + 64) = v32;
  if ( *(_QWORD *)v28 != -8 )
    --*(_DWORD *)(v24 + 68);
  *(_QWORD *)v28 = v25;
  v33 = 0;
  *(_QWORD *)(v28 + 8) = 0;
LABEL_29:
  sub_205A4F0(a3, v5, (__int64)&v164, v28, v32, v27, v29);
  if ( !(unsigned int)sub_1700720(*(_QWORD *)(v5 + 544)) )
    goto LABEL_33;
  v34 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( *(_BYTE *)(sub_157ED60(*(_QWORD *)(v34 + 24)) + 16) != 31 || v165 == v164 )
    goto LABEL_33;
  v170 = 0u;
  v65 = *(_DWORD *)(a2 + 20);
  v171.m128i_i64[0] = 0;
  v171.m128i_i32[2] = 0;
  v66 = (v65 & 0xFFFFFFFu) >> 1;
  v163 = 0;
  v67 = v66 - 1;
  if ( v66 == 1 )
    goto LABEL_124;
  v161 = v5;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  while ( 2 )
  {
    v77 = 24;
    if ( (_DWORD)v71 != -2 )
      v77 = 24LL * (unsigned int)(2 * v71 + 3);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v72 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v77);
      if ( !v68 )
        goto LABEL_97;
    }
    else
    {
      v72 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) + v77);
      if ( !v68 )
      {
LABEL_97:
        ++v170.m128i_i64[0];
        goto LABEL_98;
      }
    }
    v73 = (v68 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
    v74 = v69 + 16LL * v73;
    v75 = *(_QWORD *)v74;
    if ( *(_QWORD *)v74 == v72 )
    {
      v76 = *(_DWORD *)(v74 + 8) + 1;
      goto LABEL_90;
    }
    v119 = 1;
    v120 = 0;
    while ( v75 != -8 )
    {
      if ( v75 != -16 || v120 )
        v74 = v120;
      v142 = v119 + 1;
      v73 = (v68 - 1) & (v119 + v73);
      v143 = v69 + 16LL * v73;
      v75 = *(_QWORD *)v143;
      if ( v72 == *(_QWORD *)v143 )
      {
        v74 = v69 + 16LL * v73;
        v76 = *(_DWORD *)(v143 + 8) + 1;
        goto LABEL_90;
      }
      v119 = v142;
      v120 = v74;
      v74 = v69 + 16LL * v73;
    }
    if ( v120 )
      v74 = v120;
    ++v170.m128i_i64[0];
    v79 = v171.m128i_i32[0] + 1;
    if ( 4 * (v171.m128i_i32[0] + 1) >= 3 * v68 )
    {
LABEL_98:
      v154 = v67;
      sub_137BFC0((__int64)&v170, 2 * v68);
      if ( !v171.m128i_i32[2] )
        goto LABEL_261;
      v67 = v154;
      v78 = (v171.m128i_i32[2] - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v79 = v171.m128i_i32[0] + 1;
      v74 = v170.m128i_i64[1] + 16LL * v78;
      v80 = *(_QWORD *)v74;
      if ( *(_QWORD *)v74 != v72 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -8 )
        {
          if ( !v82 && v80 == -16 )
            v82 = v74;
          v78 = (v171.m128i_i32[2] - 1) & (v81 + v78);
          v74 = v170.m128i_i64[1] + 16LL * v78;
          v80 = *(_QWORD *)v74;
          if ( v72 == *(_QWORD *)v74 )
            goto LABEL_162;
          ++v81;
        }
        if ( v82 )
          v74 = v82;
      }
    }
    else if ( v68 - (v79 + v171.m128i_i32[1]) <= v68 >> 3 )
    {
      v157 = v67;
      sub_137BFC0((__int64)&v170, v68);
      if ( !v171.m128i_i32[2] )
      {
LABEL_261:
        ++v171.m128i_i32[0];
        BUG();
      }
      v126 = 1;
      v127 = (v171.m128i_i32[2] - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v67 = v157;
      v79 = v171.m128i_i32[0] + 1;
      v128 = 0;
      v74 = v170.m128i_i64[1] + 16LL * v127;
      v129 = *(_QWORD *)v74;
      if ( *(_QWORD *)v74 != v72 )
      {
        while ( v129 != -8 )
        {
          if ( v129 == -16 && !v128 )
            v128 = v74;
          v127 = (v171.m128i_i32[2] - 1) & (v126 + v127);
          v74 = v170.m128i_i64[1] + 16LL * v127;
          v129 = *(_QWORD *)v74;
          if ( v72 == *(_QWORD *)v74 )
            goto LABEL_162;
          ++v126;
        }
        if ( v128 )
          v74 = v128;
      }
    }
LABEL_162:
    v171.m128i_i32[0] = v79;
    if ( *(_QWORD *)v74 != -8 )
      --v171.m128i_i32[1];
    *(_QWORD *)v74 = v72;
    v76 = 1;
    *(_DWORD *)(v74 + 8) = 0;
LABEL_90:
    *(_DWORD *)(v74 + 8) = v76;
    if ( v70 < v76 )
    {
      if ( v171.m128i_i32[2] )
      {
        v83 = (v171.m128i_i32[2] - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v84 = v170.m128i_i64[1] + 16LL * v83;
        v85 = *(_QWORD *)v84;
        if ( v72 == *(_QWORD *)v84 )
        {
LABEL_112:
          v70 = *(_DWORD *)(v84 + 8);
LABEL_113:
          v163 = v72;
          goto LABEL_91;
        }
        v130 = 1;
        v131 = 0;
        while ( v85 != -8 )
        {
          if ( !v131 && v85 == -16 )
            v131 = v84;
          v83 = (v171.m128i_i32[2] - 1) & (v130 + v83);
          v84 = v170.m128i_i64[1] + 16LL * v83;
          v85 = *(_QWORD *)v84;
          if ( v72 == *(_QWORD *)v84 )
            goto LABEL_112;
          ++v130;
        }
        if ( !v131 )
          v131 = v84;
        ++v170.m128i_i64[0];
        v132 = v171.m128i_i32[0] + 1;
        if ( 4 * (v171.m128i_i32[0] + 1) < (unsigned int)(3 * v171.m128i_i32[2]) )
        {
          if ( v171.m128i_i32[2] - v171.m128i_i32[1] - v132 <= (unsigned __int32)v171.m128i_i32[2] >> 3 )
          {
            v159 = v67;
            sub_137BFC0((__int64)&v170, v171.m128i_i32[2]);
            if ( !v171.m128i_i32[2] )
            {
LABEL_262:
              ++v171.m128i_i32[0];
              BUG();
            }
            v137 = 1;
            LODWORD(v138) = (v171.m128i_i32[2] - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
            v67 = v159;
            v132 = v171.m128i_i32[0] + 1;
            v139 = 0;
            v131 = v170.m128i_i64[1] + 16LL * (unsigned int)v138;
            v140 = *(_QWORD *)v131;
            if ( v72 != *(_QWORD *)v131 )
            {
              while ( v140 != -8 )
              {
                if ( !v139 && v140 == -16 )
                  v139 = v131;
                v138 = (v171.m128i_i32[2] - 1) & (unsigned int)(v138 + v137);
                v131 = v170.m128i_i64[1] + 16 * v138;
                v140 = *(_QWORD *)v131;
                if ( v72 == *(_QWORD *)v131 )
                  goto LABEL_184;
                ++v137;
              }
              if ( v139 )
                v131 = v139;
            }
          }
          goto LABEL_184;
        }
      }
      else
      {
        ++v170.m128i_i64[0];
      }
      v158 = v67;
      sub_137BFC0((__int64)&v170, 2 * v171.m128i_i32[2]);
      if ( !v171.m128i_i32[2] )
        goto LABEL_262;
      v67 = v158;
      LODWORD(v133) = (v171.m128i_i32[2] - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
      v132 = v171.m128i_i32[0] + 1;
      v131 = v170.m128i_i64[1] + 16LL * (unsigned int)v133;
      v134 = *(_QWORD *)v131;
      if ( *(_QWORD *)v131 != v72 )
      {
        v135 = 1;
        v136 = 0;
        while ( v134 != -8 )
        {
          if ( v134 == -16 && !v136 )
            v136 = v131;
          v133 = (v171.m128i_i32[2] - 1) & (unsigned int)(v133 + v135);
          v131 = v170.m128i_i64[1] + 16 * v133;
          v134 = *(_QWORD *)v131;
          if ( v72 == *(_QWORD *)v131 )
            goto LABEL_184;
          ++v135;
        }
        if ( v136 )
          v131 = v136;
      }
LABEL_184:
      v171.m128i_i32[0] = v132;
      if ( *(_QWORD *)v131 != -8 )
        --v171.m128i_i32[1];
      *(_QWORD *)v131 = v72;
      v70 = 0;
      *(_DWORD *)(v131 + 8) = 0;
      goto LABEL_113;
    }
LABEL_91:
    if ( v67 != ++v71 )
    {
      v69 = v170.m128i_i64[1];
      v68 = v171.m128i_u32[2];
      continue;
    }
    break;
  }
  v5 = v161;
LABEL_124:
  v33 = sub_1FE1990(*(_QWORD *)(v5 + 712) + 48LL, &v163)[1];
  src[0] = 0;
  src[1] = 0;
  v168[0] = 0;
  sub_20553A0((const __m128i **)src, 0xCCCCCCCCCCCCCCCDLL * (((char *)v165 - (char *)v164) >> 3));
  v104 = v164;
  v105 = (__m128i *)src[1];
  v106 = (const __m128i *)v168[0];
  v107 = &v163;
  if ( v165 != v164 )
  {
    v108 = v164;
    v109 = v165;
    do
    {
      if ( v108[1].m128i_i64[1] != v33 )
      {
        if ( v106 == v105 )
        {
          v156 = v107;
          sub_205B0B0((const __m128i **)src, v106, v108);
          v105 = (__m128i *)src[1];
          v106 = (const __m128i *)v168[0];
          v107 = v156;
        }
        else
        {
          if ( v105 )
          {
            *v105 = _mm_loadu_si128(v108);
            v105[1] = _mm_loadu_si128(v108 + 1);
            v105[2].m128i_i64[0] = v108[2].m128i_i64[0];
            v105 = (__m128i *)src[1];
            v106 = (const __m128i *)v168[0];
          }
          v105 = (__m128i *)((char *)v105 + 40);
          src[1] = v105;
        }
      }
      v108 = (const __m128i *)((char *)v108 + 40);
    }
    while ( v109 != v108 );
    v104 = v164;
  }
  v110 = v166;
  v165 = v105;
  v166 = v106;
  v164 = (const __m128i *)src[0];
  src[0] = 0;
  src[1] = 0;
  v168[0] = 0;
  if ( v104 )
  {
    j_j___libc_free_0(v104, (char *)v110 - (char *)v104);
    if ( src[0] )
      j_j___libc_free_0(src[0], v168[0] - (unsigned __int64)src[0]);
  }
  j___libc_free_0(v170.m128i_i64[1]);
LABEL_33:
  LODWORD(v163) = 0;
  v35 = (void *)sub_2092C30(v5);
  v36 = *(_QWORD *)(*(_QWORD *)(v5 + 712) + 784LL);
  if ( v165 == v164 )
  {
    sub_1DD8FE0(*(_QWORD *)(*(_QWORD *)(v5 + 712) + 784LL), v33, -1);
    if ( v33 != sub_2054600(v5, v36) )
    {
      v90 = *(_QWORD *)(v5 + 552);
      *(_QWORD *)&v162 = sub_1D2A490((_QWORD *)v90, v33, v86, v87, v88, v89);
      *((_QWORD *)&v162 + 1) = v91;
      v95 = sub_2051DF0((__int64 *)v5, *(double *)a3.m128i_i64, a4, a5, v33, v91, v92, v93, v94);
      v170.m128i_i64[0] = 0;
      v96 = (__int64)v95;
      v98 = v97;
      v99 = *(__m128i **)v5;
      v170.m128i_i32[2] = *(_DWORD *)(v5 + 536);
      if ( v99 )
      {
        if ( &v170 != &v99[3] )
        {
          v100 = v99[3].m128i_i64[0];
          v170.m128i_i64[0] = v100;
          if ( v100 )
          {
            v155 = v96;
            sub_1623A60((__int64)&v170, v100, 2);
            v96 = v155;
          }
        }
      }
      v102 = sub_1D332F0((__int64 *)v90, 188, (__int64)&v170, 1, 0, 0, *(double *)a3.m128i_i64, a4, a5, v96, v98, v162);
      v103 = v101;
      if ( v102 )
      {
        nullsub_686();
        *(_QWORD *)(v90 + 176) = v102;
        *(_DWORD *)(v90 + 184) = v103;
        sub_1D23870();
      }
      else
      {
        *(_QWORD *)(v90 + 176) = 0;
        *(_DWORD *)(v90 + 184) = v101;
      }
      if ( v170.m128i_i64[0] )
        sub_161E7C0((__int64)&v170, v170.m128i_i64[0]);
    }
  }
  else
  {
    sub_20650A0((_QWORD *)v5, &v164, a2, v33);
    sub_205E850((_QWORD *)v5, (__int64 *)&v164, a2);
    v170.m128i_i64[0] = (__int64)&v171;
    v170.m128i_i64[1] = 0x400000000LL;
    v151 = v164;
    v153 = &v165[-3].m128i_i8[8];
    v37 = sub_2052E90(v5, (__int64)v35, v33);
    if ( (_DWORD)v163 )
    {
      v49 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v149 = v37;
      src[0] = *(void **)(v49 + 24);
      v50 = sub_2060320(*(_QWORD *)(v5 + 712) + 48LL, (__int64 *)src);
      v37 = v149;
      if ( v50[1] == v33 )
        v37 = sub_2044050(v149, v163);
    }
    src[0] = v35;
    v168[1] = 0;
    src[1] = (void *)v151;
    *(_QWORD *)v169 = 0;
    v168[0] = (__int64)v153;
    v39 = v170.m128i_u32[2];
    v169[2] = v37;
    if ( v170.m128i_i32[2] >= (unsigned __int32)v170.m128i_i32[3] )
    {
      sub_16CD150((__int64)&v170, &v171, 0, 48, v37, v38);
      v39 = v170.m128i_u32[2];
    }
    v40 = (__m128i *)(v170.m128i_i64[0] + 48 * v39);
    *v40 = _mm_loadu_si128((const __m128i *)src);
    v40[1] = _mm_loadu_si128((const __m128i *)v168);
    v40[2] = _mm_loadu_si128((const __m128i *)v169);
    v41 = v170.m128i_i32[2] + 1;
    for ( v170.m128i_i32[2] = v41; v170.m128i_i32[2]; v41 = v170.m128i_u32[2] )
    {
      while ( 1 )
      {
        v43 = v170.m128i_i64[0] + 48LL * v41;
        *(__m128i *)src = _mm_loadu_si128((const __m128i *)(v43 - 48));
        *(__m128i *)v168 = _mm_loadu_si128((const __m128i *)(v43 - 32));
        v44 = _mm_loadu_si128((const __m128i *)(v43 - 16));
        v170.m128i_i32[2] = v41 - 1;
        *(__m128i *)v169 = v44;
        if ( -858993459 * (unsigned int)((signed __int64)(v168[0] - (unsigned __int64)src[1]) >> 3) + 1 <= 3
          || !(unsigned int)sub_1700720(*(_QWORD *)(v5 + 544))
          || (unsigned __int8)sub_1560180(**(_QWORD **)(v33 + 56) + 112LL, 17) )
        {
          break;
        }
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v45 = *(_QWORD **)(a2 - 8);
        else
          v45 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        sub_20920A0(v5, &v170, src, *v45, v36);
        v41 = v170.m128i_u32[2];
        if ( !v170.m128i_i32[2] )
          goto LABEL_49;
      }
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v42 = *(_QWORD **)(a2 - 8);
      else
        v42 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      sub_20912B0(v5, *v42, v36, v33, v37, v38, (__int64)src[0], src[1], v168[0], v168[1], v169[0], v169[2]);
    }
LABEL_49:
    if ( (__m128i *)v170.m128i_i64[0] != &v171 )
      _libc_free(v170.m128i_u64[0]);
  }
  if ( v164 )
    j_j___libc_free_0(v164, (char *)v166 - (char *)v164);
}
