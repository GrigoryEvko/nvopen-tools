// Function: sub_1F84730
// Address: 0x1f84730
//
__int64 __fastcall sub_1F84730(_QWORD *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int16 v8; // dx
  char v9; // r13
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  __m128i v13; // xmm0
  int v14; // eax
  __int16 v15; // ax
  unsigned int v16; // r12d
  unsigned int v17; // r13d
  unsigned int v18; // r15d
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rdi
  unsigned int v25; // r15d
  __int64 v26; // r12
  char v29; // al
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rax
  _QWORD *v35; // r15
  char v36; // di
  __int64 v37; // rax
  unsigned int v38; // esi
  _QWORD *v39; // r8
  unsigned int v40; // esi
  char v41; // al
  __int64 *v42; // r15
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // rax
  _QWORD *v46; // r8
  int v47; // eax
  int v48; // edx
  __int64 v49; // r9
  char v50; // di
  __int64 v51; // rax
  unsigned int v52; // eax
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  int v57; // edx
  __int64 v58; // rax
  unsigned int v59; // ebx
  __int64 v60; // rax
  __int32 v63; // eax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  unsigned int v68; // ebx
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  unsigned int v73; // eax
  __int64 v74; // rax
  unsigned __int8 *v75; // r14
  __int64 v76; // rdi
  __int64 (*v77)(); // rax
  char v78; // di
  __int64 v79; // rax
  int v80; // eax
  int v81; // eax
  __int64 v82; // r13
  unsigned __int8 *v83; // rax
  unsigned int v84; // r14d
  unsigned int v85; // eax
  __int64 v86; // rsi
  __int128 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rax
  __int64 *v90; // r12
  unsigned __int16 v91; // r14
  unsigned __int64 v92; // rcx
  int v93; // eax
  __int64 v94; // rsi
  __int64 *v95; // rcx
  __int128 *v96; // r13
  __int64 v97; // rax
  __int64 v98; // rsi
  unsigned int v99; // edx
  __int64 v100; // r14
  __int64 v101; // r12
  __int64 v102; // rcx
  __int64 v103; // rdi
  unsigned __int64 v104; // r13
  __int8 v105; // r8
  __int64 v106; // rdx
  unsigned int v107; // eax
  __int64 *v108; // r14
  __int64 v109; // rax
  unsigned __int16 v110; // cx
  unsigned __int64 v111; // rsi
  int v112; // eax
  __int64 v113; // rsi
  __int128 *v114; // r8
  __int64 v115; // rax
  unsigned int v116; // edx
  unsigned __int8 *v117; // rdx
  __int64 v118; // r14
  __int64 v119; // rax
  char v120; // al
  __int64 v121; // rdx
  unsigned int v122; // eax
  __int64 v123; // rsi
  unsigned int v124; // eax
  __int64 *v125; // rdi
  __int64 v126; // rax
  __int64 v127; // r9
  __int64 v128; // rax
  char v129; // di
  __int64 v130; // rax
  int v131; // eax
  __int64 v132; // r13
  char v133; // si
  __int64 v134; // rax
  __int64 v135; // r13
  char v136; // di
  __int64 v137; // rax
  __int128 v138; // rax
  __int64 *v139; // rdx
  __int64 v140; // rcx
  int v141; // eax
  __int64 v142; // rcx
  _QWORD *v143; // rax
  int v144; // eax
  int v145; // eax
  int v146; // [rsp+0h] [rbp-120h]
  int v147; // [rsp+0h] [rbp-120h]
  __int128 *v148; // [rsp+0h] [rbp-120h]
  __int64 v149; // [rsp+8h] [rbp-118h]
  __int64 v150; // [rsp+8h] [rbp-118h]
  __int64 *v151; // [rsp+8h] [rbp-118h]
  __int64 *v152; // [rsp+8h] [rbp-118h]
  unsigned __int16 v153; // [rsp+8h] [rbp-118h]
  int v154; // [rsp+10h] [rbp-110h]
  int v155; // [rsp+10h] [rbp-110h]
  const void **v156; // [rsp+10h] [rbp-110h]
  __int64 *v157; // [rsp+10h] [rbp-110h]
  unsigned int v158; // [rsp+10h] [rbp-110h]
  __int64 v159; // [rsp+10h] [rbp-110h]
  __int64 v160; // [rsp+18h] [rbp-108h]
  __int64 v161; // [rsp+18h] [rbp-108h]
  __int64 v162; // [rsp+18h] [rbp-108h]
  __int64 v163; // [rsp+20h] [rbp-100h]
  __int64 v164; // [rsp+20h] [rbp-100h]
  char v165; // [rsp+28h] [rbp-F8h]
  __int64 v166; // [rsp+28h] [rbp-F8h]
  __int64 v167; // [rsp+28h] [rbp-F8h]
  unsigned __int16 v168; // [rsp+30h] [rbp-F0h]
  __int64 v169; // [rsp+30h] [rbp-F0h]
  unsigned int v170; // [rsp+30h] [rbp-F0h]
  _QWORD *v171; // [rsp+30h] [rbp-F0h]
  int v172; // [rsp+30h] [rbp-F0h]
  __int64 v173; // [rsp+30h] [rbp-F0h]
  __int64 v174; // [rsp+30h] [rbp-F0h]
  __m128i v176; // [rsp+40h] [rbp-E0h] BYREF
  __m128i v177; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v178; // [rsp+60h] [rbp-C0h] BYREF
  int v179; // [rsp+68h] [rbp-B8h]
  __m128i v180; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v181; // [rsp+80h] [rbp-A0h] BYREF
  int v182; // [rsp+88h] [rbp-98h]
  __int128 v183; // [rsp+90h] [rbp-90h] BYREF
  __int64 v184; // [rsp+A0h] [rbp-80h]
  __int128 v185; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v186; // [rsp+C0h] [rbp-60h]
  __m128i v187; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v188; // [rsp+E0h] [rbp-40h]
  _QWORD *v189; // [rsp+E8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_WORD *)(a2 + 24);
  v9 = *(_BYTE *)v6;
  v10 = *(_QWORD *)(v6 + 8);
  v11 = *(_QWORD *)v7;
  v12 = *(unsigned int *)(v7 + 8);
  v176.m128i_i8[0] = v9;
  v176.m128i_i64[1] = v10;
  v13 = _mm_loadu_si128(&v176);
  v177 = v13;
  if ( !v9 )
  {
    v168 = v8;
    if ( sub_1F58D20((__int64)&v176) )
      return 0;
    v14 = v168;
    if ( v168 != 148 )
      goto LABEL_4;
LABEL_15:
    v20 = *(_QWORD *)(v7 + 40);
    v16 = 2;
    v21 = *(_QWORD *)(v20 + 96);
    LOBYTE(v20) = *(_BYTE *)(v20 + 88);
    v177.m128i_i64[1] = v21;
    v177.m128i_i8[0] = v20;
    v15 = *(_WORD *)(v11 + 24);
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v9 - 14) <= 0x5Fu )
    return 0;
  v14 = v8;
  if ( v8 == 148 )
    goto LABEL_15;
LABEL_4:
  if ( v14 != 124 )
  {
    if ( v14 != 118 )
    {
      v15 = *(_WORD *)(v11 + 24);
      v16 = 0;
      goto LABEL_7;
    }
    v22 = *(_QWORD *)(v7 + 40);
    v23 = *(unsigned __int16 *)(v22 + 24);
    if ( v23 != 32 && v23 != 10 )
      return 0;
    v24 = *(_QWORD *)(v22 + 88);
    v25 = *(_DWORD *)(v24 + 32);
    if ( v25 > 0x40 )
    {
      LODWORD(_R12) = sub_16A58F0(v24 + 24);
      if ( !(_DWORD)_R12 || v25 != (_DWORD)_R12 + (unsigned int)sub_16A57B0(v24 + 24) )
        return 0;
    }
    else
    {
      v26 = *(_QWORD *)(v24 + 24);
      if ( !v26 || (v26 & (v26 + 1)) != 0 )
        return 0;
      if ( !~v26 )
        goto LABEL_151;
      __asm { tzcnt   r12, r12 }
    }
    if ( (_DWORD)_R12 == 32 )
    {
      v29 = 5;
      goto LABEL_27;
    }
    if ( (unsigned int)_R12 <= 0x20 )
    {
      if ( (_DWORD)_R12 == 8 )
      {
        v29 = 3;
      }
      else
      {
        v29 = 4;
        if ( (_DWORD)_R12 != 16 )
        {
          v29 = 2;
          if ( (_DWORD)_R12 != 1 )
            goto LABEL_101;
        }
      }
      goto LABEL_27;
    }
    if ( (_DWORD)_R12 != 64 )
    {
      if ( (_DWORD)_R12 != 128 )
      {
LABEL_101:
        v29 = sub_1F58CC0(*(_QWORD **)(*a1 + 48LL), _R12);
        goto LABEL_28;
      }
      v29 = 7;
LABEL_27:
      v30 = 0;
LABEL_28:
      v177.m128i_i8[0] = v29;
      v16 = 3;
      v15 = *(_WORD *)(v11 + 24);
      v177.m128i_i64[1] = v30;
      goto LABEL_7;
    }
LABEL_151:
    v29 = 6;
    goto LABEL_27;
  }
  v31 = *(_QWORD *)v7;
  v32 = *(_QWORD *)(v7 + 40);
  if ( *(_WORD *)(*(_QWORD *)v7 + 24LL) != 185 )
    return 0;
  v33 = *(unsigned __int16 *)(v32 + 24);
  if ( v33 != 10 && v33 != 32 )
    return 0;
  v34 = *(_QWORD *)(v32 + 88);
  v35 = *(_QWORD **)(v34 + 24);
  if ( *(_DWORD *)(v34 + 32) > 0x40u )
    v35 = (_QWORD *)*v35;
  v36 = *(_BYTE *)(v31 + 88);
  v37 = *(_QWORD *)(v31 + 96);
  v187.m128i_i8[0] = v36;
  v187.m128i_i64[1] = v37;
  if ( v36 )
    v38 = sub_1F6C8D0(v36);
  else
    v38 = sub_1F58D40((__int64)&v187);
  v39 = *(_QWORD **)(*a1 + 48LL);
  if ( ((*(_BYTE *)(v31 + 27) >> 2) & 3) == 2 || v38 <= (unsigned __int64)v35 )
  {
    if ( v9 )
    {
      v38 = sub_1F6C8D0(v9);
    }
    else
    {
      v171 = *(_QWORD **)(*a1 + 48LL);
      v107 = sub_1F58D40((__int64)&v176);
      v39 = v171;
      v38 = v107;
    }
  }
  v40 = v38 - (_DWORD)v35;
  if ( v40 == 32 )
  {
    v41 = 5;
    goto LABEL_44;
  }
  if ( v40 <= 0x20 )
  {
    if ( v40 == 8 )
    {
      v41 = 3;
    }
    else
    {
      v41 = 4;
      if ( v40 != 16 )
      {
        v41 = 2;
        if ( v40 != 1 )
          goto LABEL_103;
      }
    }
LABEL_44:
    v177.m128i_i8[0] = v41;
    v11 = a2;
    v12 = 0;
    v16 = 3;
    v177.m128i_i64[1] = 0;
    goto LABEL_45;
  }
  if ( v40 == 64 )
  {
    v41 = 6;
    goto LABEL_44;
  }
  if ( v40 == 128 )
  {
    v41 = 7;
    goto LABEL_44;
  }
LABEL_103:
  v11 = a2;
  v12 = 0;
  v105 = sub_1F58CC0(v39, v40);
  v177.m128i_i64[1] = v106;
  v15 = *(_WORD *)(a2 + 24);
  v16 = 3;
  v177.m128i_i8[0] = v105;
LABEL_7:
  if ( v15 != 124 )
    goto LABEL_8;
LABEL_45:
  if ( !sub_1D18C00(v11, 1, v12)
    || (v42 = *(__int64 **)(v11 + 32), v43 = v42[5], v44 = *(unsigned __int16 *)(v43 + 24), v44 != 32) && v44 != 10 )
  {
    v15 = *(_WORD *)(v11 + 24);
LABEL_8:
    v17 = 0;
    v18 = 0;
    if ( v15 == 122 )
    {
      if ( sub_1D18C00(v11, 1, v12)
        && v177.m128i_i8[0] == v176.m128i_i8[0]
        && (v177.m128i_i8[0] || v176.m128i_i64[1] == v177.m128i_i64[1]) )
      {
        v75 = (unsigned __int8 *)(*(_QWORD *)(v11 + 40) + 16 * v12);
        v76 = a1[1];
        v77 = *(__int64 (**)())(*(_QWORD *)v76 + 928LL);
        if ( v77 != sub_1F3CC00
          && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, __int64))v77)(
               v76,
               *v75,
               *((_QWORD *)v75 + 1),
               v176.m128i_u32[0],
               v176.m128i_i64[1])
          && ((v139 = *(__int64 **)(v11 + 32), v140 = v139[5], v141 = *(unsigned __int16 *)(v140 + 24), v141 == 10)
           || v141 == 32) )
        {
          v142 = *(_QWORD *)(v140 + 88);
          v143 = *(_QWORD **)(v142 + 24);
          if ( *(_DWORD *)(v142 + 32) > 0x40u )
            v143 = (_QWORD *)*v143;
          v11 = *v139;
          v18 = (unsigned int)v143;
          v17 = 0;
          v15 = *(_WORD *)(*v139 + 24);
        }
        else
        {
          v15 = *(_WORD *)(v11 + 24);
          v17 = 0;
          v18 = 0;
        }
      }
      else
      {
        v15 = *(_WORD *)(v11 + 24);
      }
    }
LABEL_9:
    if ( v15 == 185 )
      goto LABEL_10;
    return 0;
  }
  v45 = *(_QWORD *)(v43 + 88);
  v46 = *(_QWORD **)(v45 + 24);
  if ( *(_DWORD *)(v45 + 32) > 0x40u )
    v46 = (_QWORD *)*v46;
  v17 = (unsigned int)v46;
  if ( v177.m128i_i8[0] )
    v47 = sub_1F6C8D0(v177.m128i_i8[0]);
  else
    v47 = sub_1F58D40((__int64)&v177);
  v48 = v47 - 1;
  v49 = v11;
  if ( (v17 & (v47 - 1)) == 0 )
  {
    v127 = *v42;
    v12 = *((unsigned int *)v42 + 2);
    v128 = *(_QWORD *)(*v42 + 40) + 16 * v12;
    v129 = *(_BYTE *)v128;
    v130 = *(_QWORD *)(v128 + 8);
    v187.m128i_i8[0] = v129;
    v187.m128i_i64[1] = v130;
    if ( v129 )
    {
      v172 = v48;
      v131 = sub_1F6C8D0(v129);
    }
    else
    {
      v166 = v127;
      v172 = v48;
      v131 = sub_1F58D40((__int64)&v187);
      v49 = v166;
    }
    if ( (v131 & v172) != 0 )
      return 0;
  }
  if ( *(_WORD *)(v49 + 24) != 185 || ((*(_BYTE *)(v49 + 27) >> 2) & 3) == 2 )
    return 0;
  v50 = *(_BYTE *)(v49 + 88);
  v51 = *(_QWORD *)(v49 + 96);
  v187.m128i_i8[0] = v50;
  v187.m128i_i64[1] = v51;
  if ( v50 )
  {
    v52 = sub_1F6C8D0(v50);
  }
  else
  {
    v173 = v49;
    v52 = sub_1F58D40((__int64)&v187);
    v53 = v173;
  }
  if ( v17 >= v52 )
    return 0;
  v54 = *(_QWORD *)(v11 + 48);
  v18 = 0;
  v11 = v53;
  v55 = *(_QWORD *)(v54 + 16);
  if ( *(_WORD *)(v55 + 24) == 118 )
  {
    v56 = *(_QWORD *)(*(_QWORD *)(v55 + 32) + 40LL);
    v57 = *(unsigned __int16 *)(v56 + 24);
    if ( v57 == 10 || v57 == 32 )
    {
      v58 = *(_QWORD *)(v56 + 88);
      v59 = *(_DWORD *)(v58 + 32);
      if ( v59 > 0x40 )
      {
        v174 = v53;
        v167 = v58 + 24;
        v144 = sub_16A58F0(v58 + 24);
        v53 = v174;
        LODWORD(_R15) = v144;
        if ( !v144 )
          goto LABEL_71;
        v145 = sub_16A57B0(v167);
        v53 = v174;
        if ( v59 != (_DWORD)_R15 + v145 )
          goto LABEL_71;
        goto LABEL_66;
      }
      v60 = *(_QWORD *)(v58 + 24);
      if ( v60 )
      {
        if ( (v60 & (v60 + 1)) != 0 )
          goto LABEL_71;
        if ( ~v60 )
          __asm { tzcnt   r15, rax }
        else
          LODWORD(_R15) = 64;
LABEL_66:
        v169 = v53;
        v63 = sub_1F7DE30(*(_QWORD **)(*a1 + 48LL), _R15);
        v187.m128i_i64[1] = v64;
        v187.m128i_i32[0] = v63;
        v68 = sub_1D159A0(v177.m128i_i8, (unsigned int)_R15, v64, v65, v66, v67, v146, v149, v154, v160);
        v73 = sub_1D159A0(v187.m128i_i8, (__int64)&v187, v69, v70, v71, v72, v147, v150, v155, v161);
        v53 = v169;
        if ( v68 > v73 )
        {
          v74 = *(unsigned __int8 *)(*(_QWORD *)(v169 + 40) + 16LL * (unsigned int)v12);
          if ( v187.m128i_i8[0] )
          {
            if ( (_BYTE)v74
              && (((int)*(unsigned __int16 *)(a1[1] + 2 * (v187.m128i_u8[0] + 115 * v74 + 16104)) >> (4 * v16)) & 0xF) == 0 )
            {
              v177 = _mm_loadu_si128(&v187);
            }
          }
        }
LABEL_71:
        v15 = *(_WORD *)(v53 + 24);
        v11 = v53;
        v18 = 0;
        if ( v17 )
          goto LABEL_9;
        goto LABEL_8;
      }
      v11 = v53;
      v18 = 0;
    }
  }
LABEL_10:
  if ( !(unsigned __int8)sub_1F742D0((__int64)a1, v11, v16, (unsigned int *)&v177, v17) )
    return 0;
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL)) )
  {
    v78 = *(_BYTE *)(v11 + 88);
    v79 = *(_QWORD *)(v11 + 96);
    v187.m128i_i8[0] = v78;
    v187.m128i_i64[1] = v79;
    if ( v78 )
      v80 = sub_1F6C8D0(v78);
    else
      v80 = sub_1F58D40((__int64)&v187);
    v170 = (unsigned int)(v80 + 7) >> 3;
    if ( v177.m128i_i8[0] )
      v81 = sub_1F6C8D0(v177.m128i_i8[0]);
    else
      v81 = sub_1F58D40((__int64)&v177);
    v17 = 8 * (v170 - ((unsigned int)(v81 + 7) >> 3)) - v17;
  }
  v82 = v17 >> 3;
  v83 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(v11 + 32) + 48LL));
  v84 = *v83;
  v156 = (const void **)*((_QWORD *)v83 + 1);
  v85 = sub_1E34390(*(_QWORD *)(v11 + 104));
  v86 = *(_QWORD *)(v11 + 72);
  v178 = v86;
  v163 = (v82 | v85) & -(v82 | v85);
  if ( v86 )
    sub_1623A60((__int64)&v178, v86, 2);
  v179 = *(_DWORD *)(v11 + 64);
  v151 = (__int64 *)*a1;
  *(_QWORD *)&v87 = sub_1D38BB0(*a1, v82, (__int64)&v178, v84, v156, 0, v13, *(double *)a4.m128i_i64, a5, 0);
  v157 = sub_1D332F0(
           v151,
           52,
           (__int64)&v178,
           v84,
           v156,
           3u,
           *(double *)v13.m128i_i64,
           *(double *)a4.m128i_i64,
           a5,
           *(_QWORD *)(*(_QWORD *)(v11 + 32) + 40LL),
           *(_QWORD *)(*(_QWORD *)(v11 + 32) + 48LL),
           v87);
  v162 = v88;
  sub_1F81BC0((__int64)a1, (__int64)v157);
  if ( v16 )
  {
    v108 = (__int64 *)*a1;
    v109 = *(_QWORD *)(v11 + 104);
    a5 = _mm_loadu_si128((const __m128i *)(v109 + 40));
    v187 = a5;
    v188 = *(_QWORD *)(v109 + 56);
    v110 = *(_WORD *)(v109 + 32);
    v111 = *(_QWORD *)v109 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v111 )
    {
      v135 = *(_QWORD *)(v109 + 8) + v82;
      v136 = *(_BYTE *)(v109 + 16);
      if ( (*(_QWORD *)v109 & 4) != 0 )
      {
        *((_QWORD *)&v185 + 1) = v135;
        LOBYTE(v186) = v136;
        *(_QWORD *)&v185 = v111 | 4;
        HIDWORD(v186) = *(_DWORD *)(v111 + 12);
      }
      else
      {
        *(_QWORD *)&v185 = *(_QWORD *)v109 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v185 + 1) = v135;
        LOBYTE(v186) = v136;
        v137 = *(_QWORD *)v111;
        if ( *(_BYTE *)(*(_QWORD *)v111 + 8LL) == 16 )
          v137 = **(_QWORD **)(v137 + 16);
        HIDWORD(v186) = *(_DWORD *)(v137 + 8) >> 8;
      }
    }
    else
    {
      v112 = *(_DWORD *)(v109 + 20);
      LODWORD(v186) = 0;
      v185 = 0u;
      HIDWORD(v186) = v112;
    }
    v113 = *(_QWORD *)(v11 + 72);
    v114 = *(__int128 **)(v11 + 32);
    v96 = &v183;
    *(_QWORD *)&v183 = v113;
    if ( v113 )
    {
      v148 = v114;
      v153 = v110;
      sub_1623A60((__int64)&v183, v113, 2);
      v114 = v148;
      v110 = v153;
    }
    DWORD2(v183) = *(_DWORD *)(v11 + 64);
    v115 = sub_1D2B810(
             v108,
             v16,
             (__int64)&v183,
             v176.m128i_u32[0],
             v176.m128i_i64[1],
             v163,
             *v114,
             (__int64)v157,
             v162,
             v185,
             v186,
             v177.m128i_i64[0],
             v177.m128i_i64[1],
             v110,
             (__int64)&v187);
    v98 = v183;
    v158 = v116;
    v100 = v115;
    if ( !(_QWORD)v183 )
      goto LABEL_97;
  }
  else
  {
    v89 = *(_QWORD *)(v11 + 104);
    v90 = (__int64 *)*a1;
    a4 = _mm_loadu_si128((const __m128i *)(v89 + 40));
    v187 = a4;
    v188 = *(_QWORD *)(v89 + 56);
    v91 = *(_WORD *)(v89 + 32);
    v92 = *(_QWORD *)v89 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v92 )
    {
      v132 = *(_QWORD *)(v89 + 8) + v82;
      v133 = *(_BYTE *)(v89 + 16);
      if ( (*(_QWORD *)v89 & 4) != 0 )
      {
        *((_QWORD *)&v183 + 1) = v132;
        LOBYTE(v184) = v133;
        *(_QWORD *)&v183 = v92 | 4;
        HIDWORD(v184) = *(_DWORD *)(v92 + 12);
      }
      else
      {
        *(_QWORD *)&v183 = *(_QWORD *)v89 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v183 + 1) = v132;
        LOBYTE(v184) = v133;
        v134 = *(_QWORD *)v92;
        if ( *(_BYTE *)(*(_QWORD *)v92 + 8LL) == 16 )
          v134 = **(_QWORD **)(v134 + 16);
        HIDWORD(v184) = *(_DWORD *)(v134 + 8) >> 8;
      }
    }
    else
    {
      v93 = *(_DWORD *)(v89 + 20);
      LODWORD(v184) = 0;
      v183 = 0u;
      HIDWORD(v184) = v93;
    }
    v94 = *(_QWORD *)(v11 + 72);
    v95 = *(__int64 **)(v11 + 32);
    v96 = &v185;
    *(_QWORD *)&v185 = v94;
    if ( v94 )
    {
      v152 = v95;
      sub_1623A60((__int64)&v185, v94, 2);
      v95 = v152;
    }
    DWORD2(v185) = *(_DWORD *)(v11 + 64);
    v97 = sub_1D2B730(
            v90,
            v176.m128i_u32[0],
            v176.m128i_i64[1],
            (__int64)&v185,
            *v95,
            v95[1],
            (__int64)v157,
            v162,
            v183,
            v184,
            v163,
            v91,
            (__int64)&v187,
            0);
    v98 = v185;
    v158 = v99;
    v100 = v97;
    if ( !(_QWORD)v185 )
      goto LABEL_97;
  }
  sub_161E7C0((__int64)v96, v98);
LABEL_97:
  v101 = v100;
  v102 = *(_QWORD *)(*a1 + 664LL);
  v188 = *a1;
  v187.m128i_i64[1] = v102;
  *(_QWORD *)(v188 + 664) = &v187;
  v103 = *a1;
  v187.m128i_i64[0] = (__int64)off_49FFF30;
  v189 = a1;
  sub_1D44C70(v103, v11, 1, v100, 1u);
  v104 = v158;
  if ( v18 )
  {
    v117 = (unsigned __int8 *)(*(_QWORD *)(v100 + 40) + 16LL * v158);
    v118 = a1[1];
    v164 = *((_QWORD *)v117 + 1);
    v159 = *v117;
    v165 = *((_BYTE *)a1 + 25);
    v119 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
    v120 = sub_1F40B60(v118, v159, v164, v119, v165);
    v180.m128i_i8[0] = v120;
    v180.m128i_i64[1] = v121;
    if ( v120 )
      v122 = sub_1F6C8D0(v120);
    else
      v122 = sub_1F58D40((__int64)&v180);
    if ( v122 <= 0x3F && v18 > 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v122) )
      v180 = _mm_loadu_si128(&v176);
    v123 = *(_QWORD *)(v11 + 72);
    v181 = v123;
    if ( v123 )
      sub_1623A60((__int64)&v181, v123, 2);
    v182 = *(_DWORD *)(v11 + 64);
    if ( v176.m128i_i8[0] )
      v124 = sub_1F6C8D0(v176.m128i_i8[0]);
    else
      v124 = sub_1F58D40((__int64)&v176);
    v125 = (__int64 *)*a1;
    if ( v124 > v18 )
    {
      *(_QWORD *)&v138 = sub_1D38BB0(
                           (__int64)v125,
                           v18,
                           (__int64)&v181,
                           v180.m128i_u32[0],
                           (const void **)v180.m128i_i64[1],
                           0,
                           v13,
                           *(double *)a4.m128i_i64,
                           a5,
                           0);
      v126 = (__int64)sub_1D332F0(
                        v125,
                        122,
                        (__int64)&v181,
                        v176.m128i_u32[0],
                        (const void **)v176.m128i_i64[1],
                        0,
                        *(double *)v13.m128i_i64,
                        *(double *)a4.m128i_i64,
                        a5,
                        v101,
                        v104,
                        v138);
    }
    else
    {
      v126 = sub_1D38BB0(
               (__int64)v125,
               0,
               (__int64)&v181,
               v176.m128i_u32[0],
               (const void **)v176.m128i_i64[1],
               0,
               v13,
               *(double *)a4.m128i_i64,
               a5,
               0);
    }
    v100 = v126;
    if ( v181 )
      sub_161E7C0((__int64)&v181, v181);
  }
  result = v100;
  *(_QWORD *)(v188 + 664) = v187.m128i_i64[1];
  if ( v178 )
  {
    sub_161E7C0((__int64)&v178, v178);
    return v100;
  }
  return result;
}
