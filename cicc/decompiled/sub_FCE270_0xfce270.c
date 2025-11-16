// Function: sub_FCE270
// Address: 0xfce270
//
unsigned __int64 __fastcall sub_FCE270(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _DWORD *v7; // rdx
  __int64 v8; // r13
  const char *v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  unsigned __int8 *v12; // rsi
  _BYTE *v13; // rax
  unsigned int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r8
  int *v19; // rbx
  _DWORD *v20; // rdx
  __int64 v21; // r14
  _DWORD *v22; // rdx
  int v23; // ecx
  int v24; // r13d
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdi
  _BYTE *v30; // rax
  void *v31; // rdx
  _DWORD *v32; // rdx
  int v33; // ecx
  int v34; // r13d
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rdi
  _BYTE *v40; // rax
  __m128i *v41; // rdx
  __m128i si128; // xmm0
  _DWORD *v43; // rdx
  int v44; // ecx
  int v45; // r13d
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // rdi
  _BYTE *v50; // rax
  int v51; // ecx
  int v52; // r13d
  int v53; // r14d
  int v54; // r9d
  _WORD *v55; // rax
  int v56; // ecx
  int v57; // r13d
  int v58; // r14d
  int v59; // r9d
  __int64 v60; // rdx
  __int64 v61; // rax
  __m128i v62; // xmm0
  int v63; // ecx
  __int64 v64; // r10
  unsigned int v65; // edi
  unsigned __int64 v66; // r9
  __int64 v67; // rdx
  __m128i *v69; // rdx
  unsigned __int64 result; // rax
  __m128i v71; // xmm0
  unsigned int i; // r14d
  _BYTE *v74; // rax
  int v75; // edx
  unsigned int v76; // eax
  unsigned int v77; // r9d
  unsigned int v78; // esi
  __int64 v79; // r10
  int v80; // ecx
  unsigned __int64 v81; // r8
  __int64 v82; // rdx
  unsigned __int64 v83; // r8
  _BYTE *v86; // rax
  __m128i *v87; // rdx
  __m128i v88; // xmm0
  int v89; // ecx
  __int64 v90; // r10
  unsigned int v91; // r9d
  unsigned __int64 v92; // rdi
  __int64 v93; // rdx
  __m128i *v95; // rdx
  __m128i v96; // xmm0
  _WORD *v97; // rax
  _WORD *v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  int v105; // eax
  unsigned int j; // r14d
  _BYTE *v108; // rax
  int v109; // edx
  unsigned int v110; // eax
  unsigned int v111; // r8d
  unsigned int v112; // esi
  __int64 v113; // r9
  int v114; // ecx
  unsigned __int64 v115; // rdi
  __int64 v116; // rdx
  unsigned __int64 v117; // rdi
  int v120; // r9d
  size_t v121; // [rsp+0h] [rbp-40h]
  int v122; // [rsp+0h] [rbp-40h]
  int v123; // [rsp+0h] [rbp-40h]
  int v124; // [rsp+0h] [rbp-40h]
  int v125; // [rsp+0h] [rbp-40h]
  int v126; // [rsp+0h] [rbp-40h]
  int v127; // [rsp+0h] [rbp-40h]
  int v128; // [rsp+0h] [rbp-40h]
  int v129; // [rsp+8h] [rbp-38h]
  int v130; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 3u )
  {
    v8 = sub_CB6200(a2, (unsigned __int8 *)"BB: ", 4u);
  }
  else
  {
    *v7 = 540688962;
    v8 = a2;
    *(_QWORD *)(a2 + 32) += 4LL;
  }
  v9 = sub_BD5D20(a3);
  v11 = *(_BYTE **)(v8 + 32);
  v12 = (unsigned __int8 *)v9;
  v13 = *(_BYTE **)(v8 + 24);
  if ( v13 - v11 < v10 )
  {
    v8 = sub_CB6200(v8, v12, v10);
    v13 = *(_BYTE **)(v8 + 24);
    v11 = *(_BYTE **)(v8 + 32);
  }
  else if ( v10 )
  {
    v121 = v10;
    memcpy(v11, v12, v10);
    v86 = *(_BYTE **)(v8 + 24);
    v11 = (_BYTE *)(*(_QWORD *)(v8 + 32) + v121);
    *(_QWORD *)(v8 + 32) = v11;
    if ( v11 != v86 )
      goto LABEL_6;
    goto LABEL_68;
  }
  if ( v11 != v13 )
  {
LABEL_6:
    *v11 = 10;
    ++*(_QWORD *)(v8 + 32);
    goto LABEL_7;
  }
LABEL_68:
  sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
LABEL_7:
  v14 = *(_DWORD *)(a1 + 136);
  v15 = *(_QWORD *)(a1 + 120);
  if ( v14 )
  {
    v16 = (v14 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v17 = (__int64 *)(v15 + 16LL * v16);
    v18 = *v17;
    if ( a3 == *v17 )
      goto LABEL_9;
    v105 = 1;
    while ( v18 != -4096 )
    {
      v120 = v105 + 1;
      v16 = (v14 - 1) & (v105 + v16);
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( a3 == *v17 )
        goto LABEL_9;
      v105 = v120;
    }
  }
  v17 = (__int64 *)(v15 + 16LL * v14);
LABEL_9:
  v19 = (int *)v17[1];
  v20 = *(_DWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v20 <= 3u )
  {
    v104 = sub_CB6200(a2, (unsigned __int8 *)"RP: ", 4u);
    v22 = *(_DWORD **)(v104 + 32);
    v21 = v104;
  }
  else
  {
    *v20 = 540692562;
    v21 = a2;
    v22 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 4LL);
    *(_QWORD *)(a2 + 32) = v22;
  }
  v23 = *v19;
  v24 = v19[1];
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 3u )
  {
    v124 = *v19;
    v103 = sub_CB6200(v21, "[R: ", 4u);
    v23 = v124;
    v25 = v103;
  }
  else
  {
    *v22 = 540693083;
    v25 = v21;
    *(_QWORD *)(v21 + 32) += 4LL;
  }
  v26 = sub_CB59F0(v25, v23);
  v27 = *(_QWORD *)(v26 + 32);
  v28 = v26;
  if ( (unsigned __int64)(*(_QWORD *)(v26 + 24) - v27) <= 4 )
  {
    v28 = sub_CB6200(v26, ", P: ", 5u);
  }
  else
  {
    *(_DWORD *)v27 = 978329644;
    *(_BYTE *)(v27 + 4) = 32;
    *(_QWORD *)(v26 + 32) += 5LL;
  }
  v29 = sub_CB59F0(v28, v24);
  v30 = *(_BYTE **)(v29 + 32);
  if ( *(_BYTE **)(v29 + 24) == v30 )
  {
    sub_CB6200(v29, (unsigned __int8 *)"]", 1u);
  }
  else
  {
    *v30 = 93;
    ++*(_QWORD *)(v29 + 32);
  }
  v31 = *(void **)(v21 + 32);
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v31 <= 0xCu )
  {
    v102 = sub_CB6200(v21, " Live-in RP: ", 0xDu);
    v32 = *(_DWORD **)(v102 + 32);
    v21 = v102;
  }
  else
  {
    qmemcpy(v31, " Live-in RP: ", 13);
    v32 = (_DWORD *)(*(_QWORD *)(v21 + 32) + 13LL);
    *(_QWORD *)(v21 + 32) = v32;
  }
  v33 = v19[2];
  v34 = v19[3];
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v32 <= 3u )
  {
    v123 = v19[2];
    v101 = sub_CB6200(v21, "[R: ", 4u);
    v33 = v123;
    v35 = v101;
  }
  else
  {
    *v32 = 540693083;
    v35 = v21;
    *(_QWORD *)(v21 + 32) += 4LL;
  }
  v36 = sub_CB59F0(v35, v33);
  v37 = *(_QWORD *)(v36 + 32);
  v38 = v36;
  if ( (unsigned __int64)(*(_QWORD *)(v36 + 24) - v37) <= 4 )
  {
    v38 = sub_CB6200(v36, ", P: ", 5u);
  }
  else
  {
    *(_DWORD *)v37 = 978329644;
    *(_BYTE *)(v37 + 4) = 32;
    *(_QWORD *)(v36 + 32) += 5LL;
  }
  v39 = sub_CB59F0(v38, v34);
  v40 = *(_BYTE **)(v39 + 32);
  if ( *(_BYTE **)(v39 + 24) == v40 )
  {
    sub_CB6200(v39, (unsigned __int8 *)"]", 1u);
  }
  else
  {
    *v40 = 93;
    ++*(_QWORD *)(v39 + 32);
  }
  v41 = *(__m128i **)(v21 + 32);
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v41 <= 0x11u )
  {
    v100 = sub_CB6200(v21, " Register Target: ", 0x12u);
    v43 = *(_DWORD **)(v100 + 32);
    v21 = v100;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C0B0);
    v41[1].m128i_i16[0] = 8250;
    *v41 = si128;
    v43 = (_DWORD *)(*(_QWORD *)(v21 + 32) + 18LL);
    *(_QWORD *)(v21 + 32) = v43;
  }
  v44 = v19[4];
  v45 = v19[5];
  if ( *(_QWORD *)(v21 + 24) - (_QWORD)v43 <= 3u )
  {
    v122 = v19[4];
    v99 = sub_CB6200(v21, "[R: ", 4u);
    v44 = v122;
    v21 = v99;
  }
  else
  {
    *v43 = 540693083;
    *(_QWORD *)(v21 + 32) += 4LL;
  }
  v46 = sub_CB59F0(v21, v44);
  v47 = *(_QWORD *)(v46 + 32);
  v48 = v46;
  if ( (unsigned __int64)(*(_QWORD *)(v46 + 24) - v47) <= 4 )
  {
    v48 = sub_CB6200(v46, ", P: ", 5u);
  }
  else
  {
    *(_DWORD *)v47 = 978329644;
    *(_BYTE *)(v47 + 4) = 32;
    *(_QWORD *)(v46 + 32) += 5LL;
  }
  v49 = sub_CB59F0(v48, v45);
  v50 = *(_BYTE **)(v49 + 32);
  if ( *(_BYTE **)(v49 + 24) == v50 )
  {
    sub_CB6200(v49, (unsigned __int8 *)"]", 1u);
  }
  else
  {
    *v50 = 93;
    ++*(_QWORD *)(v49 + 32);
  }
  v51 = *(_DWORD *)(a1 + 24);
  v52 = *v19;
  v53 = v19[1];
  v54 = *(_DWORD *)(a1 + 28);
  v55 = *(_WORD **)(a2 + 32);
  if ( v51 == *v19 || v54 == v53 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v55 <= 6u )
    {
      v129 = *(_DWORD *)(a1 + 28);
      v125 = *(_DWORD *)(a1 + 24);
      sub_CB6200(a2, " Max { ", 7u);
      v98 = *(_WORD **)(a2 + 32);
      v51 = v125;
      v54 = v129;
    }
    else
    {
      *(_DWORD *)v55 = 2019642656;
      v55[2] = 31520;
      *((_BYTE *)v55 + 6) = 32;
      v98 = (_WORD *)(*(_QWORD *)(a2 + 32) + 7LL);
      *(_QWORD *)(a2 + 32) = v98;
    }
    if ( v51 != v52
      || (*(_QWORD *)(a2 + 24) - (_QWORD)v98 <= 1u
        ? (v128 = v54, sub_CB6200(a2, "R ", 2u), v98 = *(_WORD **)(a2 + 32), v54 = v128)
        : (*v98 = 8274, v98 = (_WORD *)(*(_QWORD *)(a2 + 32) + 2LL), *(_QWORD *)(a2 + 32) = v98),
          v54 == v53) )
    {
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v98 <= 1u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"P ", 2u);
        v98 = *(_WORD **)(a2 + 32);
      }
      else
      {
        *v98 = 8272;
        v98 = (_WORD *)(*(_QWORD *)(a2 + 32) + 2LL);
        *(_QWORD *)(a2 + 32) = v98;
      }
    }
    if ( *(_WORD **)(a2 + 24) == v98 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"}", 1u);
      v55 = *(_WORD **)(a2 + 32);
    }
    else
    {
      *(_BYTE *)v98 = 125;
      v55 = (_WORD *)(*(_QWORD *)(a2 + 32) + 1LL);
      *(_QWORD *)(a2 + 32) = v55;
    }
  }
  v56 = *(_DWORD *)(a1 + 32);
  v57 = v19[2];
  v58 = v19[3];
  v59 = *(_DWORD *)(a1 + 36);
  if ( v56 == v57 || v59 == v58 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v55 <= 0xEu )
    {
      v130 = *(_DWORD *)(a1 + 36);
      v126 = *(_DWORD *)(a1 + 32);
      sub_CB6200(a2, " Max Live-in { ", 0xFu);
      v97 = *(_WORD **)(a2 + 32);
      v56 = v126;
      v59 = v130;
    }
    else
    {
      qmemcpy(v55, " Max Live-in { ", 15);
      v97 = (_WORD *)(*(_QWORD *)(a2 + 32) + 15LL);
      *(_QWORD *)(a2 + 32) = v97;
    }
    if ( v56 != v57
      || (*(_QWORD *)(a2 + 24) - (_QWORD)v97 <= 1u
        ? (v127 = v59, sub_CB6200(a2, "R ", 2u), v97 = *(_WORD **)(a2 + 32), v59 = v127)
        : (*v97 = 8274, v97 = (_WORD *)(*(_QWORD *)(a2 + 32) + 2LL), *(_QWORD *)(a2 + 32) = v97),
          v59 == v58) )
    {
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v97 <= 1u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"P ", 2u);
        v97 = *(_WORD **)(a2 + 32);
      }
      else
      {
        *v97 = 8272;
        v97 = (_WORD *)(*(_QWORD *)(a2 + 32) + 2LL);
        *(_QWORD *)(a2 + 32) = v97;
      }
    }
    if ( *(_WORD **)(a2 + 24) == v97 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"}", 1u);
      v55 = *(_WORD **)(a2 + 32);
    }
    else
    {
      *(_BYTE *)v97 = 125;
      v55 = (_WORD *)(*(_QWORD *)(a2 + 32) + 1LL);
      *(_QWORD *)(a2 + 32) = v55;
    }
  }
  if ( *(_WORD **)(a2 + 24) == v55 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    v60 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v60) > 0x14 )
      goto LABEL_39;
  }
  else
  {
    *(_BYTE *)v55 = 10;
    v60 = *(_QWORD *)(a2 + 32) + 1LL;
    v61 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 32) = v60;
    if ( (unsigned __int64)(v61 - v60) > 0x14 )
    {
LABEL_39:
      v62 = _mm_load_si128((const __m128i *)&xmmword_3F8C0C0);
      *(_DWORD *)(v60 + 16) = 1852401509;
      *(_BYTE *)(v60 + 20) = 10;
      *(__m128i *)v60 = v62;
      *(_QWORD *)(a2 + 32) += 21LL;
      goto LABEL_40;
    }
  }
  sub_CB6200(a2, "Live-in values begin\n", 0x15u);
LABEL_40:
  v63 = v19[22];
  if ( v63 )
  {
    v64 = *((_QWORD *)v19 + 3);
    v65 = (unsigned int)(v63 - 1) >> 6;
    v66 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v63;
    v67 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v64 + 8 * v67);
      if ( v65 == (_DWORD)v67 )
        _RCX = v66 & *(_QWORD *)(v64 + 8 * v67);
      if ( _RCX )
        break;
      if ( v65 + 1 == ++v67 )
        goto LABEL_46;
    }
    __asm { tzcnt   rcx, rcx }
    for ( i = ((_DWORD)v67 << 6) + _RCX; i != -1; i = ((_DWORD)v82 << 6) + _RAX )
    {
      sub_A69870(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * i), (_BYTE *)a2, 0);
      v74 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v74 )
      {
        sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v74 = 10;
        ++*(_QWORD *)(a2 + 32);
      }
      v75 = v19[22];
      v76 = i + 1;
      if ( v75 == i + 1 )
        break;
      v77 = v76 >> 6;
      v78 = (unsigned int)(v75 - 1) >> 6;
      if ( v76 >> 6 > v78 )
        break;
      v79 = *((_QWORD *)v19 + 3);
      v80 = 64 - (v76 & 0x3F);
      v81 = 0xFFFFFFFFFFFFFFFFLL >> v80;
      v82 = v77;
      if ( v80 == 64 )
        v81 = 0;
      v83 = ~v81;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v79 + 8 * v82);
        if ( v77 == (_DWORD)v82 )
          _RAX = v83 & *(_QWORD *)(v79 + 8 * v82);
        if ( v78 == (_DWORD)v82 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v19[22];
        if ( _RAX )
          break;
        if ( v78 < (unsigned int)++v82 )
          goto LABEL_46;
      }
      __asm { tzcnt   rax, rax }
    }
  }
LABEL_46:
  v69 = *(__m128i **)(a2 + 32);
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v69;
  if ( result <= 0x12 )
  {
    result = sub_CB6200(a2, "Live-in values end\n", 0x13u);
  }
  else
  {
    v71 = _mm_load_si128((const __m128i *)&xmmword_3F8C0D0);
    v69[1].m128i_i8[2] = 10;
    v69[1].m128i_i16[0] = 25710;
    *v69 = v71;
    *(_QWORD *)(a2 + 32) += 19LL;
  }
  if ( a4 )
  {
    v87 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v87 <= 0x15u )
    {
      sub_CB6200(a2, "Live-out values begin\n", 0x16u);
    }
    else
    {
      v88 = _mm_load_si128((const __m128i *)&xmmword_3F8C0E0);
      v87[1].m128i_i32[0] = 1768383842;
      v87[1].m128i_i16[2] = 2670;
      *v87 = v88;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v89 = v19[40];
    if ( v89 )
    {
      v90 = *((_QWORD *)v19 + 12);
      v91 = (unsigned int)(v89 - 1) >> 6;
      v92 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v89;
      v93 = 0;
      while ( 1 )
      {
        _RCX = *(_QWORD *)(v90 + 8 * v93);
        if ( v91 == (_DWORD)v93 )
          _RCX = v92 & *(_QWORD *)(v90 + 8 * v93);
        if ( _RCX )
          break;
        if ( v91 + 1 == ++v93 )
          goto LABEL_79;
      }
      __asm { tzcnt   rcx, rcx }
      for ( j = ((_DWORD)v93 << 6) + _RCX; j != -1; j = _RAX + ((_DWORD)v116 << 6) )
      {
        sub_A69870(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * j), (_BYTE *)a2, 0);
        v108 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v108 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v108 = 10;
          ++*(_QWORD *)(a2 + 32);
        }
        v109 = v19[40];
        v110 = j + 1;
        if ( v109 == j + 1 )
          break;
        v111 = v110 >> 6;
        v112 = (unsigned int)(v109 - 1) >> 6;
        if ( v110 >> 6 > v112 )
          break;
        v113 = *((_QWORD *)v19 + 12);
        v114 = 64 - (v110 & 0x3F);
        v115 = 0xFFFFFFFFFFFFFFFFLL >> v114;
        v116 = v111;
        if ( v114 == 64 )
          v115 = 0;
        v117 = ~v115;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v113 + 8 * v116);
          if ( v111 == (_DWORD)v116 )
            _RAX = v117 & *(_QWORD *)(v113 + 8 * v116);
          if ( v112 == (_DWORD)v116 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v19[40];
          if ( _RAX )
            break;
          if ( v112 < (unsigned int)++v116 )
            goto LABEL_79;
        }
        __asm { tzcnt   rax, rax }
      }
    }
LABEL_79:
    v95 = *(__m128i **)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - (_QWORD)v95;
    if ( result <= 0x13 )
    {
      return sub_CB6200(a2, "Live-out values end\n", 0x14u);
    }
    else
    {
      v96 = _mm_load_si128((const __m128i *)&xmmword_3F8C0E0);
      v95[1].m128i_i32[0] = 174354021;
      *v95 = v96;
      *(_QWORD *)(a2 + 32) += 20LL;
    }
  }
  return result;
}
