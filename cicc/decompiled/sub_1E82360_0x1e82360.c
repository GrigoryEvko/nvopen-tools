// Function: sub_1E82360
// Address: 0x1e82360
//
void __fastcall sub_1E82360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  _WORD *v9; // rax
  __int64 v10; // rdx
  int v11; // ecx
  unsigned int *v12; // rbx
  unsigned int v13; // r14d
  __int64 v14; // rcx
  unsigned int v15; // r8d
  unsigned int *v16; // r10
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned int v20; // edi
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // r14d
  unsigned int v26; // r12d
  unsigned int v27; // r12d
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int16 v30; // ax
  int v31; // eax
  int v32; // r8d
  int v33; // r8d
  __int64 v34; // r9
  unsigned int v35; // ecx
  int v36; // edx
  __int64 v37; // r11
  int v38; // edi
  __int64 *v39; // rsi
  int v40; // eax
  unsigned __int64 v41; // r9
  __int64 v42; // rax
  unsigned __int16 *v43; // r8
  __int64 v44; // r15
  unsigned __int16 *v45; // r13
  int v46; // r14d
  unsigned __int8 v47; // dl
  __int64 v48; // rdx
  unsigned int *v49; // r11
  unsigned int *v50; // r14
  __int64 v51; // rcx
  unsigned int v52; // edx
  __int16 v53; // ax
  _WORD *v54; // rsi
  __int16 *v55; // rdx
  unsigned __int16 v56; // r9
  __int16 *v57; // r8
  unsigned int v58; // ecx
  unsigned int v59; // eax
  __int64 v60; // rsi
  __m128i *v61; // rdx
  __int64 v62; // rax
  const __m128i *v63; // rax
  __int16 v64; // ax
  unsigned int *v65; // rdi
  unsigned int *v66; // r9
  __int64 v67; // r8
  __int64 v68; // r13
  unsigned int *v69; // r15
  __int64 v70; // rdi
  __int64 v71; // r12
  __int64 v72; // rcx
  unsigned int v73; // edx
  __int16 v74; // ax
  _WORD *v75; // rsi
  _WORD *v76; // rdx
  unsigned __int16 v77; // bx
  _WORD *v78; // r14
  unsigned int v79; // ecx
  _BYTE *v80; // r11
  unsigned int v81; // eax
  __int64 v82; // r10
  __int64 v83; // rdx
  __int16 v84; // ax
  unsigned int v85; // ecx
  __int16 v86; // dx
  int v87; // eax
  _WORD *v88; // rcx
  unsigned __int16 *v89; // rsi
  __int64 v90; // rcx
  unsigned int v91; // eax
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rax
  unsigned __int64 v95; // r14
  unsigned __int64 *v96; // rax
  __int64 v97; // rax
  __m128i *v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rax
  __int64 v101; // rdx
  int v102; // r15d
  __int64 *v103; // r10
  int v104; // esi
  int v105; // eax
  int v106; // r8d
  int v107; // r8d
  __int64 v108; // r9
  __int64 *v109; // rcx
  unsigned int v110; // ebx
  int v111; // esi
  __int64 v112; // rdi
  unsigned int *v113; // [rsp+8h] [rbp-188h]
  __int64 v114; // [rsp+10h] [rbp-180h]
  unsigned __int64 v115; // [rsp+10h] [rbp-180h]
  unsigned __int64 v116; // [rsp+10h] [rbp-180h]
  unsigned int *v118; // [rsp+38h] [rbp-158h]
  int v119; // [rsp+38h] [rbp-158h]
  __int64 v120; // [rsp+40h] [rbp-150h]
  unsigned __int16 *v121; // [rsp+40h] [rbp-150h]
  const void *v122; // [rsp+40h] [rbp-150h]
  int v124; // [rsp+50h] [rbp-140h] BYREF
  __int64 v125; // [rsp+54h] [rbp-13Ch]
  _BYTE v126[12]; // [rsp+5Ch] [rbp-134h]
  unsigned int *v127; // [rsp+70h] [rbp-120h] BYREF
  __int64 v128; // [rsp+78h] [rbp-118h]
  _BYTE v129[32]; // [rsp+80h] [rbp-110h] BYREF
  unsigned int *v130; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v131; // [rsp+A8h] [rbp-E8h]
  _BYTE v132[32]; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int *v133; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v134; // [rsp+D8h] [rbp-B8h]
  _BYTE v135[176]; // [rsp+E0h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = a3;
  v133 = (unsigned int *)v135;
  v9 = *(_WORD **)(a3 + 16);
  v134 = 0x800000000LL;
  v10 = *(_QWORD *)(a1 + 440);
  v11 = (unsigned __int16)*v9;
  if ( !*v9 || v11 == 45 )
  {
    if ( *(_QWORD *)a2 )
    {
      sub_1E7F930(*(_QWORD *)(v7 + 32), *(_DWORD *)(v7 + 40), (__int64)&v133, *(_QWORD *)a2, *(_QWORD *)(v10 + 256), a6);
      goto LABEL_5;
    }
LABEL_33:
    v15 = *(_DWORD *)(a1 + 400);
    v14 = *(_QWORD *)(a1 + 384);
    v13 = 0;
    v21 = a1 + 376;
    if ( v15 )
      goto LABEL_15;
    goto LABEL_34;
  }
  if ( (unsigned __int16)(v11 - 12) <= 1u )
    goto LABEL_33;
  if ( !(unsigned __int8)sub_1E7F7A0(v7, (__int64)&v133, *(_QWORD *)(v10 + 256)) )
    goto LABEL_5;
  v42 = *(_QWORD *)(a1 + 440);
  v43 = *(unsigned __int16 **)(v7 + 32);
  v128 = 0x800000000LL;
  v131 = 0x800000000LL;
  v44 = *(_QWORD *)(v42 + 248);
  v127 = (unsigned int *)v129;
  v130 = (unsigned int *)v132;
  v121 = &v43[20 * *(unsigned int *)(v7 + 40)];
  if ( v43 == v121 )
    goto LABEL_91;
  v45 = v43;
  do
  {
    if ( *(_BYTE *)v45 )
      goto LABEL_59;
    v46 = *((_DWORD *)v45 + 2);
    if ( v46 <= 0 )
      goto LABEL_59;
    v47 = *((_BYTE *)v45 + 3);
    if ( (v47 & 0x10) != 0 )
    {
      if ( (((*((_BYTE *)v45 + 3) & 0x40) != 0) & (v47 >> 4)) == 0 )
      {
        v48 = (unsigned int)v131;
        v41 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v45 - *(_QWORD *)(v7 + 32)) >> 3);
        if ( (unsigned int)v131 >= HIDWORD(v131) )
        {
          v116 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v45 - *(_QWORD *)(v7 + 32)) >> 3);
          sub_16CD150((__int64)&v130, v132, 0, 4, (int)v43, v41);
          v48 = (unsigned int)v131;
          LODWORD(v41) = v116;
        }
        v130[v48] = v41;
        LODWORD(v131) = v131 + 1;
        goto LABEL_55;
      }
    }
    else if ( (*((_BYTE *)v45 + 3) & 0x40) == 0 )
    {
      goto LABEL_55;
    }
    v101 = (unsigned int)v128;
    if ( (unsigned int)v128 >= HIDWORD(v128) )
    {
      sub_16CD150((__int64)&v127, v129, 0, 4, (int)v43, v41);
      v101 = (unsigned int)v128;
    }
    v127[v101] = v46;
    LODWORD(v128) = v128 + 1;
LABEL_55:
    if ( (v45[2] & 1) == 0
      && (v45[2] & 2) == 0
      && ((*((_BYTE *)v45 + 3) & 0x10) == 0 || (*(_DWORD *)v45 & 0xFFF00) != 0) )
    {
      if ( !v44 )
        BUG();
      v85 = *(_DWORD *)(*(_QWORD *)(v44 + 8) + 24LL * (unsigned int)v46 + 16);
      v86 = v85 & 0xF;
      v87 = v46 * (v85 & 0xF);
      v88 = (_WORD *)(*(_QWORD *)(v44 + 56) + 2LL * (v85 >> 4));
      LOWORD(v87) = *v88 + v46 * v86;
      v89 = v88 + 1;
      LODWORD(v41) = v87;
LABEL_95:
      v43 = v89;
      while ( v43 )
      {
        v90 = *(unsigned int *)(a4 + 8);
        v91 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 208) + (unsigned __int16)v41);
        if ( v91 < (unsigned int)v90 )
        {
          v92 = *(_QWORD *)a4;
          while ( 1 )
          {
            v93 = v92 + 24LL * v91;
            if ( (unsigned __int16)v41 == *(_DWORD *)v93 )
              break;
            v91 += 256;
            if ( (unsigned int)v90 <= v91 )
              goto LABEL_121;
          }
          if ( v93 != v92 + 24 * v90 )
          {
            v41 = *(_QWORD *)(v93 + 8);
            v94 = (unsigned int)v134;
            v95 = ((unsigned __int64)(-858993459 * (unsigned int)(((__int64)v45 - *(_QWORD *)(v7 + 32)) >> 3)) << 32)
                | *(unsigned int *)(v93 + 16);
            if ( (unsigned int)v134 >= HIDWORD(v134) )
            {
              v115 = *(_QWORD *)(v93 + 8);
              sub_16CD150((__int64)&v133, v135, 0, 16, (int)v43, v41);
              v94 = (unsigned int)v134;
              v41 = v115;
            }
            v96 = (unsigned __int64 *)&v133[4 * v94];
            *v96 = v41;
            v96[1] = v95;
            LODWORD(v134) = v134 + 1;
            break;
          }
        }
LABEL_121:
        v105 = *v43;
        v89 = 0;
        ++v43;
        LODWORD(v41) = v105 + v41;
        if ( !(_WORD)v105 )
          goto LABEL_95;
      }
    }
LABEL_59:
    v45 += 20;
  }
  while ( v121 != v45 );
  v49 = v127;
  v6 = a1;
  v50 = &v127[(unsigned int)v128];
  if ( v127 != v50 )
  {
    do
    {
      if ( !v44 )
        BUG();
      v51 = *v49;
      v52 = *(_DWORD *)(*(_QWORD *)(v44 + 8) + 24 * v51 + 16);
      v53 = v52 & 0xF;
      v54 = (_WORD *)(*(_QWORD *)(v44 + 56) + 2LL * (v52 >> 4));
      v55 = v54 + 1;
      v56 = *v54 + v51 * v53;
      while ( 1 )
      {
        v57 = v55;
        if ( !v55 )
          break;
        while ( 1 )
        {
          v58 = *(_DWORD *)(a4 + 8);
          v59 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 208) + v56);
          if ( v59 < v58 )
          {
            v60 = *(_QWORD *)a4;
            while ( 1 )
            {
              v61 = (__m128i *)(v60 + 24LL * v59);
              if ( v56 == v61->m128i_i32[0] )
                break;
              v59 += 256;
              if ( v58 <= v59 )
                goto LABEL_72;
            }
            v62 = 24LL * v58;
            if ( v61 != (__m128i *)(v60 + v62) )
            {
              v63 = (const __m128i *)(v60 + v62 - 24);
              if ( v61 != v63 )
              {
                *v61 = _mm_loadu_si128(v63);
                v61[1].m128i_i32[0] = v63[1].m128i_i32[0];
                *(_BYTE *)(*(_QWORD *)(a4 + 208)
                         + *(unsigned int *)(*(_QWORD *)a4 + 24LL * *(unsigned int *)(a4 + 8) - 24)) = -85 * (((__int64)v61->m128i_i64 - *(_QWORD *)a4) >> 3);
                v58 = *(_DWORD *)(a4 + 8);
              }
              *(_DWORD *)(a4 + 8) = v58 - 1;
            }
          }
LABEL_72:
          v64 = *v57;
          v55 = 0;
          ++v57;
          v56 += v64;
          if ( !v64 )
            break;
          if ( !v57 )
            goto LABEL_74;
        }
      }
LABEL_74:
      ++v49;
    }
    while ( v50 != v49 );
  }
  v65 = &v130[(unsigned int)v131];
  v66 = v65;
  if ( v130 != v65 )
  {
    v67 = v44;
    v68 = v7;
    v122 = (const void *)(a4 + 16);
    v69 = v130;
    v70 = a4;
    do
    {
      if ( !v67 )
        BUG();
      v71 = *v69;
      v72 = *(unsigned int *)(*(_QWORD *)(v68 + 32) + 40 * v71 + 8);
      v73 = *(_DWORD *)(*(_QWORD *)(v67 + 8) + 24 * v72 + 16);
      v74 = v73 & 0xF;
      v75 = (_WORD *)(*(_QWORD *)(v67 + 56) + 2LL * (v73 >> 4));
      v76 = v75 + 1;
      v77 = *v75 + v72 * v74;
      while ( 1 )
      {
        v78 = v76;
        if ( !v76 )
          break;
        while ( 1 )
        {
          v79 = *(_DWORD *)(v70 + 8);
          v80 = (_BYTE *)(*(_QWORD *)(v70 + 208) + v77);
          v81 = (unsigned __int8)*v80;
          v124 = v77;
          v125 = 0;
          *(_QWORD *)v126 = 0;
          if ( v81 >= v79 )
            goto LABEL_105;
          v82 = *(_QWORD *)v70;
          while ( 1 )
          {
            v83 = v82 + 24LL * v81;
            if ( v77 == *(_DWORD *)v83 )
              break;
            v81 += 256;
            if ( v79 <= v81 )
              goto LABEL_105;
          }
          if ( v83 == v82 + 24LL * v79 )
          {
LABEL_105:
            *v80 = v79;
            v97 = *(unsigned int *)(v70 + 8);
            if ( (unsigned int)v97 >= *(_DWORD *)(v70 + 12) )
            {
              v113 = v66;
              v114 = v67;
              sub_16CD150(v70, v122, 0, 24, v67, (int)v66);
              v66 = v113;
              v67 = v114;
              v97 = *(unsigned int *)(v70 + 8);
            }
            v98 = (__m128i *)(*(_QWORD *)v70 + 24 * v97);
            v99 = *(_QWORD *)&v126[4];
            *v98 = _mm_load_si128((const __m128i *)&v124);
            v98[1].m128i_i64[0] = v99;
            v100 = (unsigned int)(*(_DWORD *)(v70 + 8) + 1);
            *(_DWORD *)(v70 + 8) = v100;
            v83 = *(_QWORD *)v70 + 24 * v100 - 24;
          }
          *(_QWORD *)(v83 + 8) = v68;
          ++v78;
          *(_DWORD *)(v83 + 16) = v71;
          v84 = *(v78 - 1);
          v76 = 0;
          v77 += v84;
          if ( !v84 )
            break;
          if ( !v78 )
            goto LABEL_87;
        }
      }
LABEL_87:
      ++v69;
    }
    while ( v66 != v69 );
    v7 = v68;
    v6 = a1;
    v65 = v130;
  }
  if ( v65 != (unsigned int *)v132 )
    _libc_free((unsigned __int64)v65);
LABEL_91:
  if ( v127 != (unsigned int *)v129 )
    _libc_free((unsigned __int64)v127);
LABEL_5:
  v12 = v133;
  v13 = 0;
  v14 = *(_QWORD *)(v6 + 384);
  v15 = *(_DWORD *)(v6 + 400);
  if ( v133 != &v133[4 * (unsigned int)v134] )
  {
    v120 = v7;
    v16 = &v133[4 * (unsigned int)v134];
    do
    {
      v17 = *(_QWORD *)v12;
      v18 = *(_QWORD *)(v6 + 8) + 88LL * *(int *)(*(_QWORD *)(*(_QWORD *)v12 + 24LL) + 48LL);
      v19 = *(_DWORD *)(v18 + 24);
      if ( v19 != -1 )
      {
        v20 = *(_DWORD *)(a2 + 24);
        if ( v20 != -1 && *(_DWORD *)(v18 + 16) == *(_DWORD *)(a2 + 16) && v19 <= v20 && *(_BYTE *)(v18 + 32) )
        {
          v26 = 0;
          if ( v15 )
          {
            v27 = (v15 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v28 = (__int64 *)(v14 + 16LL * v27);
            v29 = *v28;
            if ( v17 == *v28 )
            {
LABEL_25:
              v26 = *((_DWORD *)v28 + 2);
            }
            else
            {
              v40 = 1;
              while ( v29 != -8 )
              {
                v27 = (v15 - 1) & (v40 + v27);
                v119 = v40 + 1;
                v28 = (__int64 *)(v14 + 16LL * v27);
                v29 = *v28;
                if ( v17 == *v28 )
                  goto LABEL_25;
                v40 = v119;
              }
              v26 = 0;
            }
          }
          v30 = **(_WORD **)(v17 + 16);
          switch ( v30 )
          {
            case 0:
            case 8:
            case 10:
            case 14:
            case 15:
            case 45:
              break;
            default:
              switch ( v30 )
              {
                case 2:
                case 3:
                case 4:
                case 6:
                case 9:
                case 12:
                case 13:
                case 17:
                case 18:
                  goto LABEL_29;
                default:
                  v118 = v16;
                  v31 = sub_1F4BB70(*(_QWORD *)(v6 + 440) + 272LL, v17, v12[2], v120, v12[3]);
                  v14 = *(_QWORD *)(v6 + 384);
                  v15 = *(_DWORD *)(v6 + 400);
                  v16 = v118;
                  v26 += v31;
                  break;
              }
              break;
          }
LABEL_29:
          if ( v13 < v26 )
            v13 = v26;
        }
      }
      v12 += 4;
    }
    while ( v16 != v12 );
    v7 = v120;
  }
  v21 = v6 + 376;
  if ( !v15 )
  {
LABEL_34:
    ++*(_QWORD *)(v6 + 376);
    goto LABEL_35;
  }
LABEL_15:
  v22 = (v15 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v23 = (__int64 *)(v14 + 16LL * v22);
  v24 = *v23;
  if ( v7 == *v23 )
    goto LABEL_16;
  v102 = 1;
  v103 = 0;
  while ( v24 != -8 )
  {
    if ( v24 == -16 && !v103 )
      v103 = v23;
    v22 = (v15 - 1) & (v102 + v22);
    v23 = (__int64 *)(v14 + 16LL * v22);
    v24 = *v23;
    if ( v7 == *v23 )
      goto LABEL_16;
    ++v102;
  }
  v104 = *(_DWORD *)(v6 + 392);
  if ( v103 )
    v23 = v103;
  ++*(_QWORD *)(v6 + 376);
  v36 = v104 + 1;
  if ( 4 * (v104 + 1) >= 3 * v15 )
  {
LABEL_35:
    sub_1E821A0(v21, 2 * v15);
    v32 = *(_DWORD *)(v6 + 400);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(v6 + 384);
      v35 = v33 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v36 = *(_DWORD *)(v6 + 392) + 1;
      v23 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v23;
      if ( v7 != *v23 )
      {
        v38 = 1;
        v39 = 0;
        while ( v37 != -8 )
        {
          if ( v37 == -16 && !v39 )
            v39 = v23;
          v35 = v33 & (v38 + v35);
          v23 = (__int64 *)(v34 + 16LL * v35);
          v37 = *v23;
          if ( v7 == *v23 )
            goto LABEL_118;
          ++v38;
        }
        if ( v39 )
          v23 = v39;
      }
      goto LABEL_118;
    }
    goto LABEL_146;
  }
  if ( v15 - (v36 + *(_DWORD *)(v6 + 396)) <= v15 >> 3 )
  {
    sub_1E821A0(v21, v15);
    v106 = *(_DWORD *)(v6 + 400);
    if ( v106 )
    {
      v107 = v106 - 1;
      v108 = *(_QWORD *)(v6 + 384);
      v109 = 0;
      v110 = v107 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v36 = *(_DWORD *)(v6 + 392) + 1;
      v111 = 1;
      v23 = (__int64 *)(v108 + 16LL * v110);
      v112 = *v23;
      if ( v7 != *v23 )
      {
        while ( v112 != -8 )
        {
          if ( !v109 && v112 == -16 )
            v109 = v23;
          v110 = v107 & (v111 + v110);
          v23 = (__int64 *)(v108 + 16LL * v110);
          v112 = *v23;
          if ( v7 == *v23 )
            goto LABEL_118;
          ++v111;
        }
        if ( v109 )
          v23 = v109;
      }
      goto LABEL_118;
    }
LABEL_146:
    ++*(_DWORD *)(v6 + 392);
    BUG();
  }
LABEL_118:
  *(_DWORD *)(v6 + 392) = v36;
  if ( *v23 != -8 )
    --*(_DWORD *)(v6 + 396);
  *v23 = v7;
  v23[1] = 0;
LABEL_16:
  *((_DWORD *)v23 + 2) = v13;
  if ( *(_BYTE *)(a2 + 33) )
  {
    v25 = *((_DWORD *)v23 + 3) + v13;
    if ( v25 < *(_DWORD *)(a2 + 36) )
      v25 = *(_DWORD *)(a2 + 36);
    *(_DWORD *)(a2 + 36) = v25;
  }
  if ( v133 != (unsigned int *)v135 )
    _libc_free((unsigned __int64)v133);
}
