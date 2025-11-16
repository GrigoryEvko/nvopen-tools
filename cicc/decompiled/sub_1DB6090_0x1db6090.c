// Function: sub_1DB6090
// Address: 0x1db6090
//
void __fastcall sub_1DB6090(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 j,
        __int64 a4,
        unsigned int *a5,
        unsigned __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r8
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // r15
  unsigned __int64 v19; // r15
  __int64 v20; // rsi
  unsigned int v21; // edi
  unsigned int *v22; // rax
  int v23; // eax
  __int64 v24; // r14
  int v25; // edi
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r15
  __int64 v32; // r12
  int v33; // r13d
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  _BYTE *v40; // r10
  _BYTE *v41; // rdi
  unsigned int **v42; // r12
  unsigned int **v43; // r15
  int v44; // eax
  unsigned int **v45; // r13
  __m128i v46; // xmm0
  __int64 v47; // rbx
  __int64 v48; // rax
  __m128i v49; // xmm1
  __int64 v50; // rax
  unsigned int **v51; // r13
  unsigned int v52; // r11d
  unsigned __int64 v53; // r12
  unsigned int v54; // r13d
  int v55; // eax
  __int64 v56; // rdx
  unsigned int v57; // ebx
  __int64 v58; // rdx
  unsigned int *v59; // r15
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  bool v63; // cf
  __int64 v64; // r15
  __int64 v65; // r11
  __int64 i; // rax
  __int64 v67; // rdx
  __int64 v68; // rdi
  unsigned int v69; // esi
  __int64 *v70; // rcx
  __int64 v71; // r9
  int v72; // eax
  _QWORD *v73; // rax
  char v74; // al
  int v75; // ecx
  __int64 v76; // rax
  _BYTE *v77; // rdi
  _BYTE **v78; // r8
  __int64 v79; // rdx
  __int64 v80; // rdx
  unsigned int **v81; // r12
  unsigned int **v82; // r14
  int v83; // eax
  __int64 v84; // r8
  unsigned int **v85; // r13
  __m128i v86; // xmm2
  __int64 v87; // r15
  __int64 v88; // rax
  __m128i v89; // xmm3
  __int64 v90; // rax
  __int64 v91; // rax
  unsigned int **v92; // r13
  unsigned int v93; // r15d
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // r12
  unsigned int v96; // r13d
  int v97; // eax
  __int64 v98; // rdx
  __int64 v99; // rdx
  unsigned int *v100; // r14
  unsigned __int64 v101; // rax
  __int64 v102; // r15
  __int64 v103; // rax
  bool v104; // cf
  __int64 v105; // rcx
  _QWORD *v106; // rax
  _QWORD *k; // rdx
  int v108; // r9d
  __int64 v109; // rax
  __int64 *v110; // [rsp+8h] [rbp-148h]
  int v111; // [rsp+14h] [rbp-13Ch]
  _BYTE *v112; // [rsp+18h] [rbp-138h]
  __int64 *v113; // [rsp+20h] [rbp-130h]
  _BYTE *v114; // [rsp+20h] [rbp-130h]
  unsigned int *v115; // [rsp+20h] [rbp-130h]
  __int64 v116; // [rsp+28h] [rbp-128h]
  unsigned __int64 v117; // [rsp+38h] [rbp-118h]
  __int64 v118; // [rsp+40h] [rbp-110h]
  _BYTE *v119; // [rsp+40h] [rbp-110h]
  unsigned int *v120; // [rsp+40h] [rbp-110h]
  unsigned __int64 v121; // [rsp+40h] [rbp-110h]
  unsigned __int64 v122; // [rsp+48h] [rbp-108h]
  unsigned __int64 v124; // [rsp+60h] [rbp-F0h]
  unsigned __int64 v125; // [rsp+68h] [rbp-E8h]
  int v126; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v127; // [rsp+68h] [rbp-E8h]
  int v128; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v129; // [rsp+68h] [rbp-E8h]
  __int64 v130; // [rsp+68h] [rbp-E8h]
  _BYTE *v131; // [rsp+68h] [rbp-E8h]
  __int64 v132; // [rsp+68h] [rbp-E8h]
  int v133; // [rsp+68h] [rbp-E8h]
  _BYTE *v134; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v135; // [rsp+78h] [rbp-D8h]
  unsigned int v136; // [rsp+7Ch] [rbp-D4h]
  _BYTE v137[32]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD v138[2]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE v139[32]; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE *v140; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v141; // [rsp+D8h] [rbp-78h]
  _BYTE v142[32]; // [rsp+E0h] [rbp-70h] BYREF
  int v143; // [rsp+100h] [rbp-50h]

  v6 = a2;
  v7 = *(unsigned int *)(a2 + 112);
  v117 = j;
  if ( (int)v7 < 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a4 + 24) + 16 * (v7 & 0x7FFFFFFF) + 8);
  }
  else
  {
    j = *(_QWORD *)(a4 + 272);
    v8 = *(_QWORD *)(j + 8 * v7);
  }
  if ( !v8 )
    goto LABEL_24;
  do
  {
    v9 = *(_QWORD *)(v8 + 16);
    v10 = v8;
    v8 = *(_QWORD *)(v8 + 32);
    v11 = *(_QWORD *)(*a1 + 272);
    if ( **(_WORD **)(v9 + 16) != 12 )
    {
      while ( 1 )
      {
        v12 = v9;
        if ( (*(_BYTE *)(v9 + 46) & 4) == 0 )
          break;
        v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v13 = *(unsigned int *)(v11 + 384);
      v14 = *(_QWORD *)(v11 + 368);
      if ( (_DWORD)v13 )
      {
        v15 = (v13 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v12 == *v16 )
        {
LABEL_10:
          v18 = v16[1];
          goto LABEL_11;
        }
        v72 = 1;
        while ( v17 != -8 )
        {
          v108 = v72 + 1;
          v109 = ((_DWORD)v13 - 1) & (v15 + v72);
          v15 = v109;
          v16 = (__int64 *)(v14 + 16 * v109);
          v17 = *v16;
          if ( *v16 == v12 )
            goto LABEL_10;
          v72 = v108;
        }
      }
      v16 = (__int64 *)(v14 + 16 * v13);
      goto LABEL_10;
    }
    v64 = *(_QWORD *)(v9 + 24);
    v65 = *(_QWORD *)(v64 + 32);
    if ( v9 == v65 )
    {
LABEL_97:
      v18 = *(_QWORD *)(*(_QWORD *)(v11 + 392) + 16LL * *(unsigned int *)(v64 + 48));
      goto LABEL_11;
    }
    while ( 1 )
    {
      v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v9 )
        BUG();
      if ( (*(_QWORD *)v9 & 4) == 0 && (*(_BYTE *)(v9 + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v9; ; i = *(_QWORD *)v9 )
        {
          v9 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v9 + 46) & 4) == 0 )
            break;
        }
      }
      v67 = *(unsigned int *)(v11 + 384);
      if ( (_DWORD)v67 )
      {
        v68 = *(_QWORD *)(v11 + 368);
        v69 = (v67 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v70 = (__int64 *)(v68 + 16LL * v69);
        v71 = *v70;
        if ( v9 != *v70 )
        {
          v75 = 1;
          while ( v71 != -8 )
          {
            v69 = (v67 - 1) & (v75 + v69);
            v128 = v75 + 1;
            v70 = (__int64 *)(v68 + 16LL * v69);
            v71 = *v70;
            if ( v9 == *v70 )
              goto LABEL_95;
            v75 = v128;
          }
          goto LABEL_96;
        }
LABEL_95:
        if ( v70 != (__int64 *)(v68 + 16 * v67) )
          break;
      }
LABEL_96:
      if ( v65 == v9 )
        goto LABEL_97;
    }
    v18 = v70[1];
LABEL_11:
    v19 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    j = sub_1DB3C70((__int64 *)v6, v19);
    a4 = 3LL * *(unsigned int *)(v6 + 8);
    v20 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
    if ( j == v20 )
    {
      v74 = *(_BYTE *)(v10 + 4);
      if ( (v74 & 1) == 0 && (v74 & 2) == 0 && (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
      {
        v22 = 0;
        a5 = 0;
        goto LABEL_115;
      }
    }
    else
    {
      v21 = *(_DWORD *)(v19 + 24);
      a4 = *(unsigned int *)((*(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( (unsigned __int64)((unsigned int)a4 | (*(__int64 *)j >> 1) & 3) > v21 )
      {
        a5 = 0;
        goto LABEL_16;
      }
      a5 = *(unsigned int **)(j + 16);
      if ( v19 == (*(_QWORD *)(j + 8) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        LODWORD(a6) = j + 24;
        v22 = 0;
        if ( v20 != j + 24 )
        {
          v76 = *(_QWORD *)(j + 24);
          j += 24LL;
          a4 = *(unsigned int *)((v76 & 0xFFFFFFFFFFFFFFF8LL) + 24);
          goto LABEL_14;
        }
      }
      else
      {
LABEL_14:
        if ( v19 == *((_QWORD *)a5 + 1) )
          a5 = 0;
LABEL_16:
        v22 = 0;
        if ( v21 >= (unsigned int)a4 )
          v22 = *(unsigned int **)(j + 16);
      }
      j = *(unsigned __int8 *)(v10 + 4);
      if ( (j & 1) != 0 )
        goto LABEL_19;
      j &= 2u;
      if ( (_DWORD)j )
        goto LABEL_19;
      if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
      {
LABEL_116:
        v22 = a5;
        goto LABEL_20;
      }
LABEL_115:
      if ( (*(_DWORD *)v10 & 0xFFF00) != 0 )
        goto LABEL_116;
LABEL_19:
      if ( a5 != v22 )
      {
LABEL_20:
        if ( v22 )
        {
          j = *v22;
          v23 = *(_DWORD *)(a1[1] + 4 * j);
          if ( v23 )
            sub_1E310D0(v10, *(unsigned int *)(*(_QWORD *)(v117 + 8LL * (unsigned int)(v23 - 1)) + 112LL));
        }
      }
    }
  }
  while ( v8 );
LABEL_24:
  v24 = *(_QWORD *)(v6 + 104);
  if ( !v24 )
    goto LABEL_131;
  v136 = 8;
  v25 = *((_DWORD *)a1 + 14);
  v26 = *a1;
  v134 = v137;
  v140 = v142;
  v110 = (__int64 *)(v26 + 296);
  v141 = 0x800000000LL;
  v111 = v25 - 1;
  v122 = (unsigned int)(v25 - 1);
  v116 = 8 * v122;
  v27 = 8;
  while ( 2 )
  {
    v135 = 0;
    v28 = *(unsigned int *)(v24 + 72);
    if ( (unsigned int)v28 > v27 )
      sub_16CD150((__int64)&v134, v137, (unsigned int)v28, 4, (int)a5, a6);
    LODWORD(v141) = 0;
    if ( v122 )
    {
      j = 0;
      if ( v122 > HIDWORD(v141) )
      {
        sub_16CD150((__int64)&v140, v142, v122, 8, (int)a5, a6);
        j = 8LL * (unsigned int)v141;
      }
      if ( &v140[j] != &v140[v116] )
        memset(&v140[j], 0, v116 - j);
      LODWORD(v141) = v111;
    }
    if ( !(_DWORD)v28 )
    {
      v35 = v135;
      goto LABEL_49;
    }
    v29 = v6;
    v30 = 8 * v28;
    v31 = 0;
    v32 = v29;
    while ( 2 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)(v24 + 64) + v31);
      v37 = *(_QWORD *)(v36 + 8);
      if ( (v37 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_36;
      v125 = *(_QWORD *)(v36 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      a4 = sub_1DB3C70((__int64 *)v32, *(_QWORD *)(v36 + 8));
      if ( a4 == *(_QWORD *)v32 + 24LL * *(unsigned int *)(v32 + 8)
        || (*(_DWORD *)((*(_QWORD *)a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)a4 >> 1) & 3) > (*(_DWORD *)(v125 + 24) | (unsigned int)(v37 >> 1) & 3) )
      {
        BUG();
      }
      v33 = *(_DWORD *)(a1[1] + 4LL * **(unsigned int **)(a4 + 16));
      if ( !v33 )
      {
LABEL_36:
        v33 = 0;
        goto LABEL_37;
      }
      v38 = (unsigned int)(v33 - 1);
      if ( *(_QWORD *)&v140[8 * v38] )
      {
LABEL_37:
        v34 = v135;
        if ( v135 >= v136 )
          goto LABEL_47;
        goto LABEL_38;
      }
      v113 = (__int64 *)&v140[8 * v38];
      v126 = *(_DWORD *)(v24 + 112);
      v118 = *(_QWORD *)(v117 + 8 * v38);
      v39 = sub_145CDC0(0x78u, v110);
      a4 = v118;
      if ( v39 )
      {
        LODWORD(a6) = v126;
        *(_QWORD *)(v39 + 96) = 0;
        *(_QWORD *)v39 = v39 + 16;
        *(_QWORD *)(v39 + 8) = 0x200000000LL;
        *(_QWORD *)(v39 + 64) = v39 + 80;
        *(_QWORD *)(v39 + 72) = 0x200000000LL;
        *(_DWORD *)(v39 + 112) = v126;
      }
      *(_QWORD *)(v39 + 104) = *(_QWORD *)(v118 + 104);
      *(_QWORD *)(v118 + 104) = v39;
      *v113 = v39;
      v34 = v135;
      if ( v135 >= v136 )
      {
LABEL_47:
        sub_16CD150((__int64)&v134, v137, 0, 4, (int)a5, a6);
        v34 = v135;
      }
LABEL_38:
      j = (unsigned __int64)v134;
      v31 += 8;
      *(_DWORD *)&v134[4 * v34] = v33;
      v35 = ++v135;
      if ( v31 != v30 )
        continue;
      break;
    }
    v6 = v32;
LABEL_49:
    v40 = v139;
    v138[0] = v139;
    v41 = v139;
    v138[1] = 0x800000000LL;
    if ( v35 )
    {
      sub_1DB34A0((__int64)v138, (__int64)&v134, j, a4, (int)a5, (int)v138);
      v41 = (_BYTE *)v138[0];
      v40 = v139;
    }
    j = *(_QWORD *)v24;
    a6 = (unsigned __int64)v140;
    v42 = *(unsigned int ***)v24;
    v43 = (unsigned int **)(*(_QWORD *)v24 + 24LL * *(unsigned int *)(v24 + 8));
    if ( *(unsigned int ***)v24 == v43 )
      goto LABEL_101;
    while ( 1 )
    {
      v44 = *(_DWORD *)&v41[4 * *v42[2]];
      if ( v44 )
        break;
      v42 += 3;
      if ( v43 == v42 )
        goto LABEL_101;
    }
    if ( v42 == v43 )
    {
LABEL_101:
      v51 = v42;
      goto LABEL_65;
    }
    v45 = v42;
    a5 = (unsigned int *)v6;
    while ( 2 )
    {
      if ( !v44 )
      {
        v46 = _mm_loadu_si128((const __m128i *)v45);
        v45 += 3;
        v42 += 3;
        *(__m128i *)(v42 - 3) = v46;
        *(v42 - 1) = *(v45 - 1);
        if ( v43 == v45 )
          break;
        goto LABEL_58;
      }
      v47 = *(_QWORD *)(a6 + 8LL * (unsigned int)(v44 - 1));
      v48 = *(unsigned int *)(v47 + 8);
      if ( (unsigned int)v48 >= *(_DWORD *)(v47 + 12) )
      {
        v114 = v40;
        v120 = a5;
        v129 = a6;
        sub_16CD150(v47, (const void *)(v47 + 16), 0, 24, (int)a5, a6);
        v48 = *(unsigned int *)(v47 + 8);
        v40 = v114;
        a5 = v120;
        a6 = v129;
      }
      v49 = _mm_loadu_si128((const __m128i *)v45);
      v45 += 3;
      v50 = *(_QWORD *)v47 + 24 * v48;
      *(__m128i *)v50 = v49;
      *(_QWORD *)(v50 + 16) = *(v45 - 1);
      ++*(_DWORD *)(v47 + 8);
      if ( v43 != v45 )
      {
LABEL_58:
        v44 = *(_DWORD *)(v138[0] + 4LL * *v45[2]);
        continue;
      }
      break;
    }
    j = *(_QWORD *)v24;
    v6 = (__int64)a5;
    v51 = (unsigned int **)((char *)v42 + *(_QWORD *)v24 + 24LL * *(unsigned int *)(v24 + 8) - (_QWORD)v43);
    if ( v43 == (unsigned int **)(*(_QWORD *)v24 + 24LL * *(unsigned int *)(v24 + 8)) )
    {
      v41 = (_BYTE *)v138[0];
    }
    else
    {
      v119 = v40;
      v127 = a6;
      memmove(v42, v43, *(_QWORD *)v24 + 24LL * *(unsigned int *)(v24 + 8) - (_QWORD)v43);
      v41 = (_BYTE *)v138[0];
      j = *(_QWORD *)v24;
      v40 = v119;
      a6 = v127;
    }
LABEL_65:
    v52 = *(_DWORD *)(v24 + 72);
    *(_DWORD *)(v24 + 8) = -1431655765 * ((__int64)((__int64)v51 - j) >> 3);
    if ( !v52 )
      goto LABEL_81;
    j = v52;
    v53 = 0;
    while ( 1 )
    {
      v55 = *(_DWORD *)&v41[4 * v53];
      v54 = v53;
      if ( v55 )
        break;
      v54 = ++v53;
      if ( v53 == v52 )
      {
        v53 = v54;
        v63 = v52 < (unsigned __int64)v54;
        if ( v52 > (unsigned __int64)v54 )
          goto LABEL_79;
        goto LABEL_103;
      }
    }
    if ( v52 == (_DWORD)v53 )
      goto LABEL_78;
    a5 = (unsigned int *)v6;
    v56 = (unsigned int)v53;
    v57 = v52;
    while ( 2 )
    {
      a4 = *(_QWORD *)(v24 + 64);
      v59 = *(unsigned int **)(a4 + 8 * v56);
      if ( !v55 )
      {
        *v59 = v54;
        v58 = v54;
        LODWORD(v53) = v53 + 1;
        ++v54;
        *(_QWORD *)(*(_QWORD *)(v24 + 64) + 8 * v58) = v59;
        if ( v57 == (_DWORD)v53 )
          break;
        goto LABEL_72;
      }
      v60 = a6 + 8LL * (unsigned int)(v55 - 1);
      *v59 = *(_DWORD *)(*(_QWORD *)v60 + 72LL);
      v61 = *(_QWORD *)v60;
      v62 = *(unsigned int *)(v61 + 72);
      if ( (unsigned int)v62 >= *(_DWORD *)(v61 + 76) )
      {
        v112 = v40;
        v115 = a5;
        v121 = a6;
        v130 = v61;
        sub_16CD150(v61 + 64, (const void *)(v61 + 80), 0, 8, (int)a5, a6);
        v61 = v130;
        v40 = v112;
        a5 = v115;
        a6 = v121;
        v62 = *(unsigned int *)(v130 + 72);
      }
      a4 = *(_QWORD *)(v61 + 64);
      LODWORD(v53) = v53 + 1;
      *(_QWORD *)(a4 + 8 * v62) = v59;
      ++*(_DWORD *)(v61 + 72);
      if ( v57 != (_DWORD)v53 )
      {
LABEL_72:
        v56 = (unsigned int)v53;
        v55 = *(_DWORD *)(v138[0] + 4LL * (unsigned int)v53);
        continue;
      }
      break;
    }
    j = *(unsigned int *)(v24 + 72);
    v6 = (__int64)a5;
    v53 = v54;
LABEL_78:
    v63 = j < v53;
    if ( j > v53 )
    {
LABEL_79:
      *(_DWORD *)(v24 + 72) = v54;
      goto LABEL_80;
    }
LABEL_103:
    if ( v63 )
    {
      if ( *(unsigned int *)(v24 + 76) < v53 )
      {
        v131 = v40;
        sub_16CD150(v24 + 64, (const void *)(v24 + 80), v53, 8, (int)a5, a6);
        j = *(unsigned int *)(v24 + 72);
        v40 = v131;
      }
      a4 = *(_QWORD *)(v24 + 64);
      v73 = (_QWORD *)(a4 + 8 * j);
      for ( j = a4 + 8 * v53; (_QWORD *)j != v73; ++v73 )
      {
        if ( v73 )
          *v73 = 0;
      }
      goto LABEL_79;
    }
LABEL_80:
    v41 = (_BYTE *)v138[0];
LABEL_81:
    if ( v41 != v40 )
      _libc_free((unsigned __int64)v41);
    v24 = *(_QWORD *)(v24 + 104);
    if ( v24 )
    {
      v27 = v136;
      continue;
    }
    break;
  }
  sub_1DB4C70(v6);
  if ( v140 != v142 )
    _libc_free((unsigned __int64)v140);
  if ( v134 != v137 )
    _libc_free((unsigned __int64)v134);
LABEL_131:
  v77 = v142;
  v78 = &v140;
  v141 = 0x800000000LL;
  v140 = v142;
  v79 = *((unsigned int *)a1 + 4);
  if ( (_DWORD)v79 )
  {
    sub_1DB34A0((__int64)&v140, (__int64)(a1 + 1), v79, a4, (int)&v140, a6);
    v77 = v140;
  }
  v80 = *(_QWORD *)v6;
  v81 = (unsigned int **)v80;
  v143 = *((_DWORD *)a1 + 14);
  v82 = (unsigned int **)(v80 + 24LL * *(unsigned int *)(v6 + 8));
  if ( (unsigned int **)v80 == v82 )
    goto LABEL_167;
  while ( 1 )
  {
    v83 = *(_DWORD *)&v77[4 * *v81[2]];
    if ( v83 )
      break;
    v81 += 3;
    if ( v82 == v81 )
      goto LABEL_167;
  }
  if ( v82 == v81 )
  {
LABEL_167:
    v92 = v81;
    goto LABEL_147;
  }
  v84 = v6;
  v85 = v81;
  while ( 2 )
  {
    if ( !v83 )
    {
      v86 = _mm_loadu_si128((const __m128i *)v85);
      v85 += 3;
      v81 += 3;
      *(__m128i *)(v81 - 3) = v86;
      *(v81 - 1) = *(v85 - 1);
      if ( v82 == v85 )
        break;
      goto LABEL_140;
    }
    v87 = *(_QWORD *)(v117 + 8LL * (unsigned int)(v83 - 1));
    v88 = *(unsigned int *)(v87 + 8);
    if ( (unsigned int)v88 >= *(_DWORD *)(v87 + 12) )
    {
      v132 = v84;
      sub_16CD150(v87, (const void *)(v87 + 16), 0, 24, v84, a6);
      v88 = *(unsigned int *)(v87 + 8);
      v84 = v132;
    }
    v89 = _mm_loadu_si128((const __m128i *)v85);
    v85 += 3;
    v90 = *(_QWORD *)v87 + 24 * v88;
    *(__m128i *)v90 = v89;
    *(_QWORD *)(v90 + 16) = *(v85 - 1);
    ++*(_DWORD *)(v87 + 8);
    if ( v82 != v85 )
    {
LABEL_140:
      v83 = *(_DWORD *)&v140[4 * *v85[2]];
      continue;
    }
    break;
  }
  v80 = *(_QWORD *)v84;
  v6 = v84;
  v91 = *(_QWORD *)v84 + 24LL * *(unsigned int *)(v84 + 8);
  v78 = (_BYTE **)(v91 - (_QWORD)v82);
  v92 = (unsigned int **)((char *)v81 + v91 - (_QWORD)v82);
  if ( v82 == (unsigned int **)v91 )
  {
    v77 = v140;
  }
  else
  {
    memmove(v81, v82, (size_t)v78);
    v77 = v140;
    v80 = *(_QWORD *)v6;
  }
LABEL_147:
  v93 = *(_DWORD *)(v6 + 72);
  *(_DWORD *)(v6 + 8) = -1431655765 * (((__int64)v92 - v80) >> 3);
  if ( !v93 )
    goto LABEL_163;
  v94 = v93;
  v95 = 0;
  while ( 1 )
  {
    v97 = *(_DWORD *)&v77[4 * v95];
    v96 = v95;
    if ( v97 )
      break;
    v96 = ++v95;
    if ( v93 == v95 )
    {
      v95 = v96;
      v104 = v93 < (unsigned __int64)v96;
      if ( v93 > (unsigned __int64)v96 )
        goto LABEL_161;
      goto LABEL_169;
    }
  }
  if ( v93 == (_DWORD)v95 )
    goto LABEL_160;
  a6 = v117;
  v98 = (unsigned int)v95;
  LODWORD(v78) = v93;
  while ( 2 )
  {
    v100 = *(unsigned int **)(*(_QWORD *)(v6 + 64) + 8 * v98);
    if ( !v97 )
    {
      *v100 = v96;
      v99 = v96;
      LODWORD(v95) = v95 + 1;
      ++v96;
      *(_QWORD *)(*(_QWORD *)(v6 + 64) + 8 * v99) = v100;
      if ( (_DWORD)v78 == (_DWORD)v95 )
        break;
      goto LABEL_154;
    }
    v101 = a6 + 8LL * (unsigned int)(v97 - 1);
    *v100 = *(_DWORD *)(*(_QWORD *)v101 + 72LL);
    v102 = *(_QWORD *)v101;
    v103 = *(unsigned int *)(*(_QWORD *)v101 + 72LL);
    if ( (unsigned int)v103 >= *(_DWORD *)(v102 + 76) )
    {
      v124 = a6;
      v133 = (int)v78;
      sub_16CD150(v102 + 64, (const void *)(v102 + 80), 0, 8, (int)v78, a6);
      v103 = *(unsigned int *)(v102 + 72);
      a6 = v124;
      LODWORD(v78) = v133;
    }
    LODWORD(v95) = v95 + 1;
    *(_QWORD *)(*(_QWORD *)(v102 + 64) + 8 * v103) = v100;
    ++*(_DWORD *)(v102 + 72);
    if ( (_DWORD)v78 != (_DWORD)v95 )
    {
LABEL_154:
      v98 = (unsigned int)v95;
      v97 = *(_DWORD *)&v140[4 * (unsigned int)v95];
      continue;
    }
    break;
  }
  v94 = *(unsigned int *)(v6 + 72);
  v95 = v96;
LABEL_160:
  v104 = v94 < v95;
  if ( v94 > v95 )
  {
LABEL_161:
    *(_DWORD *)(v6 + 72) = v96;
    goto LABEL_162;
  }
LABEL_169:
  if ( v104 )
  {
    if ( *(unsigned int *)(v6 + 76) < v95 )
    {
      sub_16CD150(v6 + 64, (const void *)(v6 + 80), v95, 8, (int)v78, a6);
      v94 = *(unsigned int *)(v6 + 72);
    }
    v105 = *(_QWORD *)(v6 + 64);
    v106 = (_QWORD *)(v105 + 8 * v94);
    for ( k = (_QWORD *)(v105 + 8 * v95); k != v106; ++v106 )
    {
      if ( v106 )
        *v106 = 0;
    }
    goto LABEL_161;
  }
LABEL_162:
  v77 = v140;
LABEL_163:
  if ( v77 != v142 )
    _libc_free((unsigned __int64)v77);
}
