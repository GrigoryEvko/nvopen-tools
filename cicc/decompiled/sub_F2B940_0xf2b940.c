// Function: sub_F2B940
// Address: 0xf2b940
//
__int64 __fastcall sub_F2B940(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r15
  _BYTE **v3; // rdx
  unsigned __int8 v4; // al
  __int64 v5; // rbx
  unsigned int v6; // r12d
  unsigned __int64 v7; // r13
  __int64 v8; // r13
  __m128i v9; // xmm6
  __m128i v10; // xmm2
  unsigned __int64 v11; // xmm4_8
  __int64 v12; // rax
  unsigned int v13; // r14d
  unsigned __int64 v14; // r8
  _BYTE *v15; // rbx
  __int64 v16; // r14
  unsigned int v17; // eax
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int8 *v30; // rbx
  __int64 v31; // r14
  unsigned int v32; // eax
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r15
  _BYTE **v41; // rdx
  __int64 v42; // r13
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  unsigned int v54; // esi
  unsigned __int64 v55; // rsi
  unsigned int v56; // eax
  __int64 v57; // rbx
  __int64 v58; // r13
  __int64 v59; // rbx
  __int64 v60; // r15
  unsigned int v61; // r12d
  unsigned __int64 v62; // rax
  __int64 v63; // rdi
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdi
  unsigned int v67; // r8d
  unsigned int v68; // eax
  __int64 v69; // rdi
  unsigned int v70; // eax
  unsigned int v71; // r11d
  int v72; // edx
  __int64 v73; // rsi
  __int64 v74; // rdx
  _QWORD *v75; // rax
  __int64 v76; // rbx
  __int64 v77; // r10
  unsigned int v78; // r11d
  __int64 v79; // rax
  unsigned int v80; // r14d
  __int64 v81; // r13
  __int64 v82; // rbx
  __int64 v83; // r15
  __int64 *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rsi
  __int64 v88; // rsi
  __int64 v89; // rsi
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // r14
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // rcx
  __int64 v102; // rax
  unsigned int v103; // ebx
  __int64 v104; // r14
  unsigned int v105; // eax
  _BYTE *v106; // rdi
  unsigned int v107; // eax
  __int64 v108; // r12
  _BYTE *v109; // rdi
  unsigned int v110; // eax
  _BYTE *v111; // rdi
  unsigned int v112; // eax
  _BYTE *v113; // rdi
  __int64 v122; // rcx
  __int64 v123; // rax
  unsigned int v124; // eax
  __int64 v125; // rbx
  unsigned int v126; // r8d
  unsigned __int64 v127; // rdx
  __int64 v128; // r14
  __int64 v129; // r12
  __int64 v130; // r14
  __int64 v131; // rdi
  __int64 v132; // rsi
  __int64 v133; // rdi
  __int64 v134; // rsi
  unsigned __int64 v135; // rsi
  bool v136; // zf
  __int64 v137; // rax
  __int64 *v138; // rax
  __int64 v139; // rax
  __int64 v140; // rbx
  __int64 v141; // rcx
  __int64 v142; // rcx
  __int64 v143; // rbx
  __int64 v144; // rcx
  unsigned int v145; // eax
  const void **v146; // rsi
  __int64 v147; // r14
  __int64 v148; // r12
  __int64 v149; // rdx
  unsigned int v150; // esi
  __int64 v151; // rax
  _BYTE *v152; // rdx
  unsigned int v153; // eax
  _BYTE *v154; // rdx
  unsigned int v155; // eax
  _BYTE *v156; // rdx
  unsigned int v157; // eax
  unsigned int v160; // edx
  unsigned int v163; // edx
  unsigned int v166; // edx
  __int64 v167; // rdx
  __int64 v168; // rbx
  __int64 v169; // r14
  __int64 v170; // rdx
  unsigned int v171; // esi
  unsigned int v173; // [rsp+10h] [rbp-120h]
  __int64 v174; // [rsp+10h] [rbp-120h]
  __int64 v175; // [rsp+18h] [rbp-118h]
  __int64 v176; // [rsp+18h] [rbp-118h]
  _BYTE **v177; // [rsp+20h] [rbp-110h]
  __int64 v178; // [rsp+20h] [rbp-110h]
  __int64 v179; // [rsp+28h] [rbp-108h]
  __int64 v180; // [rsp+28h] [rbp-108h]
  int v182; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v183; // [rsp+3Ch] [rbp-F4h]
  unsigned int v184; // [rsp+3Ch] [rbp-F4h]
  unsigned int v185; // [rsp+3Ch] [rbp-F4h]
  char v186; // [rsp+3Ch] [rbp-F4h]
  __int64 v187; // [rsp+40h] [rbp-F0h]
  __int64 v188; // [rsp+40h] [rbp-F0h]
  unsigned int v189; // [rsp+40h] [rbp-F0h]
  __int64 v190; // [rsp+40h] [rbp-F0h]
  unsigned __int8 *v191; // [rsp+48h] [rbp-E8h]
  __int64 v192; // [rsp+48h] [rbp-E8h]
  __int64 v193; // [rsp+50h] [rbp-E0h]
  __int64 v194; // [rsp+50h] [rbp-E0h]
  __int64 v195; // [rsp+50h] [rbp-E0h]
  __int64 v196; // [rsp+50h] [rbp-E0h]
  __int64 v197; // [rsp+50h] [rbp-E0h]
  unsigned int v198; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v199; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v200; // [rsp+68h] [rbp-C8h]
  __int64 v201; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v202; // [rsp+78h] [rbp-B8h]
  unsigned int v203[8]; // [rsp+80h] [rbp-B0h] BYREF
  __int16 v204; // [rsp+A0h] [rbp-90h]
  __m128i v205[2]; // [rsp+B0h] [rbp-80h] BYREF
  unsigned __int64 v206; // [rsp+D0h] [rbp-60h]
  __int64 v207; // [rsp+D8h] [rbp-58h]
  __m128i v208; // [rsp+E0h] [rbp-50h]
  __int64 v209; // [rsp+F0h] [rbp-40h]

  v2 = a2;
  v177 = *(_BYTE ***)(a2 - 8);
  v3 = v177;
  v191 = *v177;
  v4 = **v177;
  v183 = v4;
  if ( v4 == 42 )
  {
    v193 = *((_QWORD *)v191 - 8);
    if ( !v193 )
      goto LABEL_9;
    v30 = (unsigned __int8 *)*((_QWORD *)v191 - 4);
    if ( *v30 != 17 )
      goto LABEL_9;
    v31 = 0;
    v32 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
    v33 = v32 - 1;
    if ( v32 != 1 )
    {
      while ( 1 )
      {
        v34 = sub_AD57F0((__int64)v3[4 * (unsigned int)(2 * ++v31)], v30, 0, 0);
        v35 = 32LL * (unsigned int)(2 * v31) + *(_QWORD *)(a2 - 8);
        if ( *(_QWORD *)v35 )
        {
          v36 = *(_QWORD *)(v35 + 8);
          **(_QWORD **)(v35 + 16) = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(_QWORD *)(v35 + 16);
        }
        *(_QWORD *)v35 = v34;
        if ( v34 )
        {
          v37 = *(_QWORD *)(v34 + 16);
          *(_QWORD *)(v35 + 8) = v37;
          if ( v37 )
            *(_QWORD *)(v37 + 16) = v35 + 8;
          *(_QWORD *)(v35 + 16) = v34 + 16;
          *(_QWORD *)(v34 + 16) = v35;
        }
        if ( v33 == v31 )
          break;
        v3 = *(_BYTE ***)(a2 - 8);
      }
      v8 = a2;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        goto LABEL_28;
      goto LABEL_53;
    }
LABEL_27:
    v8 = a2;
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
LABEL_28:
      v23 = *(_QWORD *)(a2 - 8);
      v24 = *(_QWORD *)v23;
      goto LABEL_29;
    }
LABEL_53:
    v23 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v24 = *(_QWORD *)v23;
LABEL_29:
    if ( v24 )
    {
      v25 = *(_QWORD *)(v23 + 8);
      **(_QWORD **)(v23 + 16) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = *(_QWORD *)(v23 + 16);
    }
    *(_QWORD *)v23 = v193;
    v26 = *(_QWORD *)(v193 + 16);
    *(_QWORD *)(v23 + 8) = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = v23 + 8;
    *(_QWORD *)(v23 + 16) = v193 + 16;
    *(_QWORD *)(v193 + 16) = v23;
    if ( *(_BYTE *)v24 > 0x1Cu )
    {
      v205[0].m128i_i64[0] = v24;
      v27 = a1[2].m128i_i64[1] + 2096;
      sub_F200C0(v27, v205[0].m128i_i64);
      v28 = *(_QWORD *)(v24 + 16);
      if ( v28 )
      {
        if ( !*(_QWORD *)(v28 + 8) )
        {
          v205[0].m128i_i64[0] = *(_QWORD *)(v28 + 24);
          sub_F200C0(v27, v205[0].m128i_i64);
        }
      }
    }
    return v8;
  }
  if ( v4 == 44 )
  {
    v15 = (_BYTE *)*((_QWORD *)v191 - 8);
    if ( *v15 != 17 )
      goto LABEL_9;
    v193 = *((_QWORD *)v191 - 4);
    if ( !v193 )
      goto LABEL_9;
    v16 = 0;
    v17 = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
    v18 = v17 - 1;
    if ( v17 != 1 )
    {
      while ( 1 )
      {
        v19 = sub_AD57F0((__int64)v15, v3[4 * (unsigned int)(2 * ++v16)], 0, 0);
        v20 = 32LL * (unsigned int)(2 * v16) + *(_QWORD *)(a2 - 8);
        if ( *(_QWORD *)v20 )
        {
          v21 = *(_QWORD *)(v20 + 8);
          **(_QWORD **)(v20 + 16) = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = *(_QWORD *)(v20 + 16);
        }
        *(_QWORD *)v20 = v19;
        if ( v19 )
        {
          v22 = *(_QWORD *)(v19 + 16);
          *(_QWORD *)(v20 + 8) = v22;
          if ( v22 )
            *(_QWORD *)(v22 + 16) = v20 + 8;
          *(_QWORD *)(v20 + 16) = v19 + 16;
          *(_QWORD *)(v19 + 16) = v20;
        }
        if ( v18 == v16 )
          break;
        v3 = *(_BYTE ***)(a2 - 8);
      }
    }
    goto LABEL_27;
  }
  if ( v4 != 54 )
  {
    if ( v4 <= 0x1Cu || v4 != 68 && v4 != 69 )
      goto LABEL_9;
    v176 = *((_QWORD *)v191 - 4);
    if ( !v176 )
      goto LABEL_9;
    LOBYTE(v199) = v4 == 68;
    v173 = sub_BCB060(*(_QWORD *)(v176 + 8));
    v203[0] = v173;
    v179 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
    v38 = v179 >> 2;
    if ( v179 >> 2 )
    {
      v39 = 4 * v38;
      v38 = 0;
      v188 = v39;
      while ( 1 )
      {
        v205[0].m128i_i64[1] = v38;
        v205[0].m128i_i64[0] = a2;
        if ( !sub_F07E00(&v199, v203, v205) )
        {
          v2 = a2;
          goto LABEL_65;
        }
        v40 = v38 + 1;
        v205[0].m128i_i64[1] = v38 + 1;
        if ( !sub_F07E00(&v199, v203, v205) )
          break;
        v40 = v38 + 2;
        v205[0].m128i_i64[1] = v38 + 2;
        if ( !sub_F07E00(&v199, v203, v205) )
          break;
        v40 = v38 + 3;
        v205[0].m128i_i64[1] = v38 + 3;
        if ( !sub_F07E00(&v199, v203, v205) )
          break;
        v38 += 4;
        if ( v38 == v188 )
        {
          v2 = a2;
          v102 = v179 - v38;
          goto LABEL_186;
        }
      }
      v97 = v40;
      v2 = a2;
      v38 = v97;
LABEL_65:
      if ( v179 != v38 )
        goto LABEL_9;
      v8 = v2;
LABEL_67:
      v194 = v8;
      v41 = v177;
      v42 = 0;
      while ( 1 )
      {
        sub_C44740((__int64)v205, (char **)v41[4 * (unsigned int)(2 * ++v42)] + 3, v173);
        v43 = (__int64 *)sub_BD5C60(v2);
        v44 = sub_ACCFD0(v43, (__int64)v205);
        v45 = 32LL * (unsigned int)(2 * v42) + *(_QWORD *)(v2 - 8);
        if ( *(_QWORD *)v45 )
        {
          v46 = *(_QWORD *)(v45 + 8);
          **(_QWORD **)(v45 + 16) = v46;
          if ( v46 )
            *(_QWORD *)(v46 + 16) = *(_QWORD *)(v45 + 16);
        }
        *(_QWORD *)v45 = v44;
        if ( v44 )
        {
          v47 = *(_QWORD *)(v44 + 16);
          *(_QWORD *)(v45 + 8) = v47;
          if ( v47 )
            *(_QWORD *)(v47 + 16) = v45 + 8;
          *(_QWORD *)(v45 + 16) = v44 + 16;
          *(_QWORD *)(v44 + 16) = v45;
        }
        if ( v205[0].m128i_i32[2] > 0x40u && v205[0].m128i_i64[0] )
          j_j___libc_free_0_0(v205[0].m128i_i64[0]);
        if ( v179 == v42 )
          break;
        v41 = *(_BYTE ***)(v2 - 8);
      }
      v8 = v194;
LABEL_81:
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
        v48 = *(_QWORD *)(v2 - 8);
      else
        v48 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
      v49 = *(_QWORD *)v48;
      if ( *(_QWORD *)v48 )
      {
        v50 = *(_QWORD *)(v48 + 8);
        **(_QWORD **)(v48 + 16) = v50;
        if ( v50 )
          *(_QWORD *)(v50 + 16) = *(_QWORD *)(v48 + 16);
      }
      *(_QWORD *)v48 = v176;
      v51 = *(_QWORD *)(v176 + 16);
      *(_QWORD *)(v48 + 8) = v51;
      if ( v51 )
        *(_QWORD *)(v51 + 16) = v48 + 8;
      *(_QWORD *)(v48 + 16) = v176 + 16;
      *(_QWORD *)(v176 + 16) = v48;
      if ( *(_BYTE *)v49 <= 0x1Cu )
        return v8;
LABEL_89:
      v205[0].m128i_i64[0] = v49;
      v52 = a1[2].m128i_i64[1] + 2096;
      sub_F200C0(v52, v205[0].m128i_i64);
      v53 = *(_QWORD *)(v49 + 16);
      if ( v53 && !*(_QWORD *)(v53 + 8) )
      {
        v205[0].m128i_i64[0] = *(_QWORD *)(v53 + 24);
        sub_F200C0(v52, v205[0].m128i_i64);
      }
      return v8;
    }
    v102 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
LABEL_186:
    if ( v102 != 2 )
    {
      if ( v102 != 3 )
      {
        if ( v102 != 1 )
          goto LABEL_189;
LABEL_214:
        v205[0].m128i_i64[0] = v2;
        v205[0].m128i_i64[1] = v38;
        if ( sub_F07E00(&v199, v203, v205) )
          goto LABEL_189;
        goto LABEL_215;
      }
      v205[0].m128i_i64[0] = v2;
      v205[0].m128i_i64[1] = v38;
      if ( !sub_F07E00(&v199, v203, v205) )
      {
LABEL_215:
        if ( v38 != v179 )
          goto LABEL_9;
LABEL_189:
        v8 = v2;
        if ( !v179 )
          goto LABEL_81;
        goto LABEL_67;
      }
      ++v38;
    }
    v205[0].m128i_i64[0] = v2;
    v205[0].m128i_i64[1] = v38;
    if ( sub_F07E00(&v199, v203, v205) )
    {
      ++v38;
      goto LABEL_214;
    }
    goto LABEL_215;
  }
  v187 = *((_QWORD *)v191 - 8);
  if ( !v187 )
    goto LABEL_9;
  v5 = *((_QWORD *)v191 - 4);
  if ( *(_BYTE *)v5 != 17 )
    goto LABEL_9;
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0(v5 + 24) > 0x40 )
      goto LABEL_9;
    v7 = **(_QWORD **)(v5 + 24);
  }
  else
  {
    v7 = *(_QWORD *)(v5 + 24);
  }
  v175 = *(_QWORD *)(v187 + 8);
  if ( (unsigned int)sub_BCB060(v175) <= v7 )
    goto LABEL_9;
  v182 = *(_DWORD *)(a2 + 4);
  v174 = ((v182 & 0x7FFFFFFu) >> 1) - 1;
  v197 = v174 >> 2;
  if ( !(v174 >> 2) )
    goto LABEL_299;
  v103 = 2;
  v104 = 0;
  v197 = 4 * (v174 >> 2);
  do
  {
    v108 = v104 + 1;
    v113 = v177[4 * v103];
    v105 = *((_DWORD *)v113 + 8);
    if ( v105 > 0x40 )
    {
      v105 = sub_C44590((__int64)(v113 + 24));
    }
    else
    {
      _RSI = *((_QWORD *)v113 + 3);
      __asm { tzcnt   rcx, rsi }
      if ( !_RSI )
        LODWORD(_RCX) = 64;
      if ( v105 > (unsigned int)_RCX )
        v105 = _RCX;
    }
    if ( v7 > v105 )
    {
      v2 = a2;
      v108 = v104;
      goto LABEL_243;
    }
    v106 = v177[4 * v103 + 8];
    v107 = *((_DWORD *)v106 + 8);
    if ( v107 <= 0x40 )
    {
      _RSI = *((_QWORD *)v106 + 3);
      __asm { tzcnt   rcx, rsi }
      if ( !_RSI )
        LODWORD(_RCX) = 64;
      if ( v107 > (unsigned int)_RCX )
        v107 = _RCX;
    }
    else
    {
      v107 = sub_C44590((__int64)(v106 + 24));
    }
    if ( v7 > v107 )
    {
LABEL_242:
      v2 = a2;
LABEL_243:
      if ( v108 == v174 )
        goto LABEL_244;
      goto LABEL_9;
    }
    v108 = v104 + 3;
    v109 = v177[4 * v103 + 16];
    v110 = *((_DWORD *)v109 + 8);
    if ( v110 <= 0x40 )
    {
      _RSI = *((_QWORD *)v109 + 3);
      __asm { tzcnt   rcx, rsi }
      if ( !_RSI )
        LODWORD(_RCX) = 64;
      if ( v110 > (unsigned int)_RCX )
        v110 = _RCX;
    }
    else
    {
      v110 = sub_C44590((__int64)(v109 + 24));
    }
    if ( v7 > v110 )
    {
      v2 = a2;
      v108 = v104 + 2;
      goto LABEL_243;
    }
    v104 += 4;
    v111 = v177[4 * (unsigned int)(2 * v104)];
    v112 = *((_DWORD *)v111 + 8);
    if ( v112 <= 0x40 )
    {
      _RSI = *((_QWORD *)v111 + 3);
      __asm { tzcnt   rcx, rsi }
      if ( !_RSI )
        LODWORD(_RCX) = 64;
      if ( v112 > (unsigned int)_RCX )
        v112 = _RCX;
    }
    else
    {
      v112 = sub_C44590((__int64)(v111 + 24));
    }
    if ( v7 > v112 )
      goto LABEL_242;
    v103 += 8;
  }
  while ( v104 != v197 );
  v2 = a2;
LABEL_299:
  v151 = v174 - v197;
  if ( v174 - v197 == 2 )
    goto LABEL_314;
  if ( v151 != 3 )
  {
    if ( v151 == 1 )
      goto LABEL_302;
    goto LABEL_244;
  }
  v154 = v177[4 * (unsigned int)(2 * (v197 + 1))];
  v155 = *((_DWORD *)v154 + 8);
  if ( v155 <= 0x40 )
  {
    _RDX = *((_QWORD *)v154 + 3);
    __asm { tzcnt   rcx, rdx }
    v136 = _RDX == 0;
    v166 = 64;
    if ( !v136 )
      v166 = _RCX;
    if ( v155 > v166 )
      v155 = v166;
  }
  else
  {
    v155 = sub_C44590((__int64)(v154 + 24));
  }
  if ( v7 > v155 )
    goto LABEL_305;
LABEL_314:
  v156 = v177[4 * (unsigned int)(2 * (++v197 + 1))];
  v157 = *((_DWORD *)v156 + 8);
  if ( v157 <= 0x40 )
  {
    _RDX = *((_QWORD *)v156 + 3);
    __asm { tzcnt   rcx, rdx }
    v136 = _RDX == 0;
    v160 = 64;
    if ( !v136 )
      v160 = _RCX;
    if ( v157 > v160 )
      v157 = v160;
  }
  else
  {
    v157 = sub_C44590((__int64)(v156 + 24));
  }
  if ( v7 > v157 )
    goto LABEL_305;
  ++v197;
LABEL_302:
  v152 = v177[4 * (unsigned int)(2 * v197 + 2)];
  v153 = *((_DWORD *)v152 + 8);
  if ( v153 <= 0x40 )
  {
    _RDX = *((_QWORD *)v152 + 3);
    __asm { tzcnt   rcx, rdx }
    v136 = _RDX == 0;
    v163 = 64;
    if ( !v136 )
      v163 = _RCX;
    if ( v153 > v163 )
      v153 = v163;
  }
  else
  {
    v153 = sub_C44590((__int64)(v152 + 24));
  }
  if ( v7 > v153 )
  {
LABEL_305:
    v108 = v197;
    goto LABEL_243;
  }
LABEL_244:
  if ( (v191[1] & 2) != 0 || ((v191[1] >> 1) & 2) != 0 )
  {
LABEL_259:
    v130 = 0;
    if ( (v182 & 0x7FFFFFFu) >> 1 == 1 )
    {
LABEL_290:
      v122 = v187;
      return sub_F20660((__int64)a1, v2, 0, v122);
    }
    while ( 1 )
    {
      v143 = 32LL * (unsigned int)(2 * ++v130);
      v136 = (v191[1] & 4) == 0;
      v144 = *(_QWORD *)(*(_QWORD *)(v2 - 8) + v143);
      v145 = *(_DWORD *)(v144 + 32);
      v146 = (const void **)(v144 + 24);
      v205[0].m128i_i32[2] = v145;
      if ( v136 )
      {
        if ( v145 <= 0x40 )
        {
          v205[0].m128i_i64[0] = *(_QWORD *)(v144 + 24);
LABEL_284:
          if ( (_DWORD)v7 == v145 )
            v205[0].m128i_i64[0] = 0;
          else
            v205[0].m128i_i64[0] = (unsigned __int64)v205[0].m128i_i64[0] >> v7;
          goto LABEL_270;
        }
        sub_C43780((__int64)v205, v146);
        v145 = v205[0].m128i_u32[2];
        if ( v205[0].m128i_i32[2] <= 0x40u )
          goto LABEL_284;
        sub_C482E0((__int64)v205, v7);
      }
      else
      {
        if ( v145 <= 0x40 )
        {
          v131 = *(_QWORD *)(v144 + 24);
LABEL_263:
          v132 = 0;
          if ( v145 )
            v132 = v131 << (64 - (unsigned __int8)v145) >> (64 - (unsigned __int8)v145);
          v133 = v132 >> 63;
          v134 = v132 >> v7;
          if ( (_DWORD)v7 == v145 )
            v134 = v133;
          v135 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v145) & v134;
          v136 = v145 == 0;
          v137 = 0;
          if ( !v136 )
            v137 = v135;
          v205[0].m128i_i64[0] = v137;
          goto LABEL_270;
        }
        sub_C43780((__int64)v205, v146);
        v145 = v205[0].m128i_u32[2];
        if ( v205[0].m128i_i32[2] <= 0x40u )
        {
          v131 = v205[0].m128i_i64[0];
          goto LABEL_263;
        }
        sub_C44B70((__int64)v205, v7);
      }
LABEL_270:
      v138 = (__int64 *)sub_BD5C60(v2);
      v139 = sub_ACCFD0(v138, (__int64)v205);
      v140 = *(_QWORD *)(v2 - 8) + v143;
      if ( *(_QWORD *)v140 )
      {
        v141 = *(_QWORD *)(v140 + 8);
        **(_QWORD **)(v140 + 16) = v141;
        if ( v141 )
          *(_QWORD *)(v141 + 16) = *(_QWORD *)(v140 + 16);
      }
      *(_QWORD *)v140 = v139;
      if ( v139 )
      {
        v142 = *(_QWORD *)(v139 + 16);
        *(_QWORD *)(v140 + 8) = v142;
        if ( v142 )
          *(_QWORD *)(v142 + 16) = v140 + 8;
        *(_QWORD *)(v140 + 16) = v139 + 16;
        *(_QWORD *)(v139 + 16) = v140;
      }
      if ( v205[0].m128i_i32[2] > 0x40u && v205[0].m128i_i64[0] )
        j_j___libc_free_0_0(v205[0].m128i_i64[0]);
      if ( ((v182 & 0x7FFFFFFu) >> 1) - 1 == v130 )
        goto LABEL_290;
    }
  }
  v123 = *((_QWORD *)v191 + 2);
  if ( v123 && !*(_QWORD *)(v123 + 8) )
  {
    v124 = sub_BCB060(v175);
    v204 = 257;
    v200 = v124;
    v125 = a1[2].m128i_i64[0];
    v126 = v124 - v7;
    if ( v124 > 0x40 )
    {
      v186 = v124;
      v198 = v124 - v7;
      sub_C43690((__int64)&v199, 0, 0);
      LOBYTE(v124) = v186;
      v126 = v198;
    }
    else
    {
      v199 = 0;
    }
    if ( v126 )
    {
      if ( v126 > 0x40 )
      {
        sub_C43C90(&v199, 0, v126);
      }
      else
      {
        v127 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v7 + 64 - (unsigned __int8)v124);
        if ( v200 > 0x40 )
          *(_QWORD *)v199 |= v127;
        else
          v199 |= v127;
      }
    }
    v128 = sub_AD8D80(*(_QWORD *)(v187 + 8), (__int64)&v199);
    v129 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v125 + 80) + 16LL))(
             *(_QWORD *)(v125 + 80),
             28,
             v187,
             v128);
    if ( !v129 )
    {
      LOWORD(v206) = 257;
      v129 = sub_B504D0(28, v187, v128, (__int64)v205, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v125 + 88) + 16LL))(
        *(_QWORD *)(v125 + 88),
        v129,
        v203,
        *(_QWORD *)(v125 + 56),
        *(_QWORD *)(v125 + 64));
      v167 = 16LL * *(unsigned int *)(v125 + 8);
      v168 = *(_QWORD *)v125;
      v169 = v168 + v167;
      while ( v169 != v168 )
      {
        v170 = *(_QWORD *)(v168 + 8);
        v171 = *(_DWORD *)v168;
        v168 += 16;
        sub_B99FD0(v129, v171, v170);
      }
    }
    if ( v200 > 0x40 && v199 )
      j_j___libc_free_0_0(v199);
    v187 = v129;
    v182 = *(_DWORD *)(v2 + 4);
    goto LABEL_259;
  }
LABEL_9:
  v8 = v2;
  if ( v183 != 86 )
  {
LABEL_10:
    v9 = _mm_loadu_si128(a1 + 9);
    v10 = _mm_loadu_si128(a1 + 7);
    v11 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
    v12 = a1[10].m128i_i64[0];
    v205[0] = _mm_loadu_si128(a1 + 6);
    v206 = v11;
    v209 = v12;
    v207 = v2;
    v205[1] = v10;
    v208 = v9;
    sub_9AC330((__int64)&v199, (__int64)v191, 0, v205);
    v184 = v200;
    if ( v200 > 0x40 )
    {
      v13 = sub_C44500((__int64)&v199);
    }
    else if ( v200 )
    {
      v13 = 64;
      if ( v199 << (64 - (unsigned __int8)v200) != -1 )
      {
        _BitScanReverse64(&v14, ~(v199 << (64 - (unsigned __int8)v200)));
        v13 = v14 ^ 0x3F;
      }
    }
    else
    {
      v13 = 0;
    }
    v189 = v202;
    if ( v202 > 0x40 )
    {
      v54 = sub_C44500((__int64)&v201);
    }
    else if ( v202 )
    {
      v54 = 64;
      if ( v201 << (64 - (unsigned __int8)v202) != -1 )
      {
        _BitScanReverse64(&v55, ~(v201 << (64 - (unsigned __int8)v202)));
        v54 = v55 ^ 0x3F;
      }
    }
    else
    {
      v54 = 0;
    }
    v56 = (*(_DWORD *)(v2 + 4) & 0x7FFFFFFu) >> 1;
    v57 = v56 - 1;
    if ( v56 != 1 )
    {
      v180 = v2;
      v58 = v56 - 1;
      v178 = v2;
      v59 = 1;
      v60 = *(_QWORD *)(v2 - 8);
      v61 = v54;
      while ( 1 )
      {
        v66 = *(_QWORD *)(v60 + 32LL * (unsigned int)(2 * v59));
        v67 = *(_DWORD *)(v66 + 32);
        if ( v67 <= 0x40 )
          break;
        v68 = sub_C444A0(v66 + 24);
        v69 = v66 + 24;
        if ( v13 > v68 )
          v13 = v68;
        v70 = sub_C44500(v69);
        if ( v61 > v70 )
          v61 = v70;
        v65 = v59 + 1;
        if ( v58 == v59 )
        {
LABEL_115:
          v54 = v61;
          v57 = v58;
          v2 = v178;
          v8 = v180;
          goto LABEL_116;
        }
LABEL_108:
        v59 = v65;
      }
      v62 = *(_QWORD *)(v66 + 24);
      if ( v62 )
      {
        _BitScanReverse64((unsigned __int64 *)&v63, v62);
        if ( v13 > ((unsigned int)v63 ^ 0x3F) + v67 - 64 )
          v13 = (v63 ^ 0x3F) + v67 - 64;
        if ( v67 )
        {
          v64 = ~(v62 << (64 - (unsigned __int8)v67));
          if ( !v64 )
          {
            if ( v61 > 0x40 )
              v61 = 64;
LABEL_107:
            v65 = v59 + 1;
            if ( v58 == v59 )
              goto LABEL_115;
            goto LABEL_108;
          }
LABEL_105:
          _BitScanReverse64(&v64, v64);
          LODWORD(v64) = v64 ^ 0x3F;
          if ( v61 > (unsigned int)v64 )
            v61 = v64;
          goto LABEL_107;
        }
      }
      else
      {
        if ( v13 > v67 )
          v13 = *(_DWORD *)(v66 + 32);
        if ( v67 )
        {
          v64 = -1;
          goto LABEL_105;
        }
      }
      v61 = 0;
      goto LABEL_107;
    }
LABEL_116:
    if ( v54 < v13 )
      v54 = v13;
    v71 = v184 - v54;
    if ( v184 != v54 && v71 < v184 && (unsigned __int8)sub_F0C790((__int64)a1, v184, v71) )
    {
      v75 = (_QWORD *)sub_BD5C60(v2);
      v76 = sub_BCCE00(v75, v184 - v54);
      sub_D5F1F0(a1[2].m128i_i64[0], v2);
      v77 = a1[2].m128i_i64[0];
      v78 = v184 - v54;
      *(_QWORD *)v203 = "trunc";
      v204 = 259;
      if ( v76 == *((_QWORD *)v191 + 1) )
      {
        v190 = (__int64)v191;
      }
      else
      {
        v185 = v184 - v54;
        v195 = v77;
        v79 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64))(**(_QWORD **)(v77 + 80) + 120LL))(
                *(_QWORD *)(v77 + 80),
                38,
                v191,
                v76);
        v78 = v185;
        v190 = v79;
        if ( !v79 )
        {
          LOWORD(v206) = 257;
          v190 = sub_B51D30(38, (__int64)v191, v76, (__int64)v205, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(v195 + 88) + 16LL))(
            *(_QWORD *)(v195 + 88),
            v190,
            v203,
            *(_QWORD *)(v195 + 56),
            *(_QWORD *)(v195 + 64));
          v78 = v185;
          v147 = *(_QWORD *)v195 + 16LL * *(unsigned int *)(v195 + 8);
          if ( *(_QWORD *)v195 != v147 )
          {
            v148 = *(_QWORD *)v195;
            do
            {
              v149 = *(_QWORD *)(v148 + 8);
              v150 = *(_DWORD *)v148;
              v148 += 16;
              sub_B99FD0(v190, v150, v149);
            }
            while ( v147 != v148 );
            v78 = v185;
          }
        }
      }
      v196 = ((*(_DWORD *)(v2 + 4) & 0x7FFFFFFu) >> 1) - 1;
      if ( (*(_DWORD *)(v2 + 4) & 0x7FFFFFFu) >> 1 != 1 )
      {
        v192 = v8;
        v80 = v78;
        v81 = 0;
        v82 = v2;
        do
        {
          v83 = 32LL * (unsigned int)(2 * ++v81);
          sub_C44740((__int64)v205, (char **)(*(_QWORD *)(*(_QWORD *)(v82 - 8) + v83) + 24LL), v80);
          v84 = (__int64 *)sub_BD5C60(v82);
          v85 = sub_ACCFD0(v84, (__int64)v205);
          v86 = v83 + *(_QWORD *)(v82 - 8);
          if ( *(_QWORD *)v86 )
          {
            v87 = *(_QWORD *)(v86 + 8);
            **(_QWORD **)(v86 + 16) = v87;
            if ( v87 )
              *(_QWORD *)(v87 + 16) = *(_QWORD *)(v86 + 16);
          }
          *(_QWORD *)v86 = v85;
          if ( v85 )
          {
            v88 = *(_QWORD *)(v85 + 16);
            *(_QWORD *)(v86 + 8) = v88;
            if ( v88 )
              *(_QWORD *)(v88 + 16) = v86 + 8;
            *(_QWORD *)(v86 + 16) = v85 + 16;
            *(_QWORD *)(v85 + 16) = v86;
          }
          if ( v205[0].m128i_i32[2] > 0x40u && v205[0].m128i_i64[0] )
            j_j___libc_free_0_0(v205[0].m128i_i64[0]);
        }
        while ( v196 != v81 );
        v8 = v192;
        v2 = v82;
      }
      if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
        v89 = *(_QWORD *)(v2 - 8);
      else
        v89 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
      v90 = *(_QWORD *)v89;
      if ( *(_QWORD *)v89 )
      {
        v91 = *(_QWORD *)(v89 + 8);
        **(_QWORD **)(v89 + 16) = v91;
        if ( v91 )
          *(_QWORD *)(v91 + 16) = *(_QWORD *)(v89 + 16);
      }
      *(_QWORD *)v89 = v190;
      if ( v190 )
      {
        v92 = *(_QWORD *)(v190 + 16);
        *(_QWORD *)(v89 + 8) = v92;
        if ( v92 )
          *(_QWORD *)(v92 + 16) = v89 + 8;
        *(_QWORD *)(v89 + 16) = v190 + 16;
        *(_QWORD *)(v190 + 16) = v89;
      }
      if ( *(_BYTE *)v90 > 0x1Cu )
      {
        v205[0].m128i_i64[0] = v90;
        v93 = a1[2].m128i_i64[1] + 2096;
        sub_F200C0(v93, v205[0].m128i_i64);
        v94 = *(_QWORD *)(v90 + 16);
        if ( v94 )
        {
          if ( !*(_QWORD *)(v94 + 8) )
          {
            v205[0].m128i_i64[0] = *(_QWORD *)(v94 + 24);
            sub_F200C0(v93, v205[0].m128i_i64);
          }
        }
      }
      v189 = v202;
      goto LABEL_123;
    }
    v72 = *v191;
    if ( (unsigned int)(v72 - 12) > 1 )
    {
      v8 = 0;
      if ( (_BYTE)v72 != 17 )
      {
LABEL_123:
        if ( v189 > 0x40 && v201 )
          j_j___libc_free_0_0(v201);
        if ( v200 > 0x40 )
        {
          if ( v199 )
            j_j___libc_free_0_0(v199);
        }
        return v8;
      }
      sub_F08550(v2, 0, v2, v57, (__int64)v191);
      v96 = 32;
      if ( v57 != v95 && (_DWORD)v95 != -2 )
        v96 = 32LL * (unsigned int)(2 * v95 + 3);
      v73 = *(_QWORD *)(v2 + 40);
      v74 = *(_QWORD *)(*(_QWORD *)(v2 - 8) + v96);
    }
    else
    {
      v73 = *(_QWORD *)(v2 + 40);
      v74 = 0;
    }
    v8 = 0;
    sub_F26260((__int64)a1, v73, v74);
    v189 = v202;
    goto LABEL_123;
  }
  v98 = sub_F0A410(v2, (__int64)v191, 1);
  if ( v98 )
  {
    if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
      v99 = *(_QWORD *)(v2 - 8);
    else
      v99 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    v49 = *(_QWORD *)v99;
    if ( *(_QWORD *)v99 )
    {
      v100 = *(_QWORD *)(v99 + 8);
      **(_QWORD **)(v99 + 16) = v100;
      if ( v100 )
        *(_QWORD *)(v100 + 16) = *(_QWORD *)(v99 + 16);
    }
    *(_QWORD *)v99 = v98;
    v101 = *(_QWORD *)(v98 + 16);
    *(_QWORD *)(v99 + 8) = v101;
    if ( v101 )
      *(_QWORD *)(v101 + 16) = v99 + 8;
    *(_QWORD *)(v99 + 16) = v98 + 16;
    *(_QWORD *)(v98 + 16) = v99;
    if ( *(_BYTE *)v49 <= 0x1Cu )
      return v8;
    goto LABEL_89;
  }
  v122 = sub_F0A410(v2, (__int64)v191, 0);
  if ( !v122 )
    goto LABEL_10;
  return sub_F20660((__int64)a1, v2, 0, v122);
}
