// Function: sub_2B3C700
// Address: 0x2b3c700
//
__int64 __fastcall sub_2B3C700(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // r13
  unsigned __int64 v4; // r15
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *i; // rdx
  bool v19; // zf
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 k; // r15
  __int64 v24; // r12
  int v25; // eax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *m; // rdx
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdx
  _QWORD *v32; // rax
  _QWORD *ii; // rdx
  int v34; // eax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *mm; // rdx
  int v38; // r15d
  __int64 result; // rax
  __int64 v40; // r12
  __int64 v41; // r14
  __int64 v42; // r13
  unsigned __int64 v43; // rdi
  unsigned int v44; // ecx
  unsigned int v45; // eax
  _QWORD *v46; // rdi
  __int64 v47; // rax
  _QWORD *v48; // rax
  unsigned int v49; // ecx
  unsigned int v50; // eax
  _QWORD *v51; // rdi
  int v52; // r12d
  _QWORD *v53; // rax
  unsigned int v54; // ecx
  unsigned int v55; // eax
  _QWORD *v56; // rdi
  __int64 v57; // r12
  _QWORD *v58; // rax
  unsigned int v59; // ecx
  unsigned int v60; // eax
  _QWORD *v61; // rdi
  __int64 v62; // r12
  _QWORD *v63; // rax
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rdx
  _QWORD *n; // rdx
  int v70; // edx
  __int64 v71; // r12
  unsigned int v72; // eax
  _QWORD *v73; // rdi
  __int64 v74; // rdx
  __int64 nn; // rdx
  unsigned int v76; // eax
  _QWORD *v77; // rax
  __int64 v78; // rdx
  _QWORD *jj; // rdx
  unsigned int v80; // eax
  _QWORD *v81; // rax
  __int64 v82; // rdx
  _QWORD *j; // rdx
  unsigned int v84; // eax
  _QWORD *v85; // rax
  __int64 v86; // rdx
  _QWORD *kk; // rdx
  __int64 v88; // [rsp+8h] [rbp-38h]
  int v89; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  v88 = *(_QWORD *)a1;
  while ( v88 != v3 )
  {
    v4 = *(_QWORD *)(v3 - 8);
    v3 -= 8;
    if ( v4 )
    {
      v5 = *(unsigned __int64 **)(v4 + 240);
      v6 = &v5[10 * *(unsigned int *)(v4 + 248)];
      if ( v5 != v6 )
      {
        do
        {
          v6 -= 10;
          if ( (unsigned __int64 *)*v6 != v6 + 2 )
            _libc_free(*v6);
        }
        while ( v5 != v6 );
        v6 = *(unsigned __int64 **)(v4 + 240);
      }
      if ( v6 != (unsigned __int64 *)(v4 + 256) )
        _libc_free((unsigned __int64)v6);
      v7 = *(_QWORD *)(v4 + 208);
      if ( v7 != v4 + 224 )
        _libc_free(v7);
      v8 = *(_QWORD *)(v4 + 144);
      if ( v8 != v4 + 160 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v4 + 112);
      if ( v9 != v4 + 128 )
        _libc_free(v9);
      v10 = *(_QWORD *)(v4 + 96);
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
        sub_BD60C0((_QWORD *)(v4 + 80));
      if ( *(_QWORD *)v4 != v4 + 16 )
        _libc_free(*(_QWORD *)v4);
      v2 = 440;
      j_j___libc_free_0(v4);
    }
  }
  *(_DWORD *)(a1 + 8) = 0;
  sub_2B39120(a1 + 80);
  sub_2B39120(a1 + 384);
  ++*(_QWORD *)(a1 + 768);
  if ( *(_BYTE *)(a1 + 796) )
    goto LABEL_28;
  v11 = 4 * (*(_DWORD *)(a1 + 788) - *(_DWORD *)(a1 + 792));
  v12 = *(unsigned int *)(a1 + 784);
  if ( v11 < 0x20 )
    v11 = 32;
  if ( (unsigned int)v12 <= v11 )
  {
    v2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 776), -1, 8 * v12);
LABEL_28:
    *(_QWORD *)(a1 + 788) = 0;
    goto LABEL_29;
  }
  sub_C8C990(a1 + 768, v2);
LABEL_29:
  ++*(_QWORD *)(a1 + 928);
  if ( *(_BYTE *)(a1 + 956) )
  {
LABEL_34:
    *(_QWORD *)(a1 + 948) = 0;
    goto LABEL_35;
  }
  v13 = 4 * (*(_DWORD *)(a1 + 948) - *(_DWORD *)(a1 + 952));
  v14 = *(unsigned int *)(a1 + 944);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    v2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 936), -1, 8 * v14);
    goto LABEL_34;
  }
  sub_C8C990(a1 + 928, v2);
LABEL_35:
  v15 = *(_DWORD *)(a1 + 1104);
  ++*(_QWORD *)(a1 + 1088);
  if ( !v15 )
  {
    if ( !*(_DWORD *)(a1 + 1108) )
      goto LABEL_41;
    v16 = *(unsigned int *)(a1 + 1112);
    if ( (unsigned int)v16 <= 0x40 )
      goto LABEL_38;
    v2 = 16LL * *(unsigned int *)(a1 + 1112);
    sub_C7D6A0(*(_QWORD *)(a1 + 1096), v2, 8);
    *(_DWORD *)(a1 + 1112) = 0;
LABEL_172:
    *(_QWORD *)(a1 + 1096) = 0;
LABEL_40:
    *(_QWORD *)(a1 + 1104) = 0;
    goto LABEL_41;
  }
  v49 = 4 * v15;
  v2 = 64;
  v16 = *(unsigned int *)(a1 + 1112);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v49 = 64;
  if ( (unsigned int)v16 <= v49 )
  {
LABEL_38:
    v17 = *(_QWORD **)(a1 + 1096);
    for ( i = &v17[2 * v16]; i != v17; v17 += 2 )
      *v17 = -4096;
    goto LABEL_40;
  }
  v50 = v15 - 1;
  if ( v50 )
  {
    _BitScanReverse(&v50, v50);
    v51 = *(_QWORD **)(a1 + 1096);
    v52 = 1 << (33 - (v50 ^ 0x1F));
    if ( v52 < 64 )
      v52 = 64;
    if ( (_DWORD)v16 == v52 )
    {
      *(_QWORD *)(a1 + 1104) = 0;
      v53 = &v51[2 * (unsigned int)v16];
      do
      {
        if ( v51 )
          *v51 = -4096;
        v51 += 2;
      }
      while ( v53 != v51 );
      goto LABEL_41;
    }
  }
  else
  {
    v51 = *(_QWORD **)(a1 + 1096);
    v52 = 64;
  }
  v2 = 16LL * *(unsigned int *)(a1 + 1112);
  sub_C7D6A0((__int64)v51, v2, 8);
  v80 = sub_2B149A0(v52);
  *(_DWORD *)(a1 + 1112) = v80;
  if ( !v80 )
    goto LABEL_172;
  v2 = 8;
  v81 = (_QWORD *)sub_C7D670(16LL * v80, 8);
  v82 = *(unsigned int *)(a1 + 1112);
  *(_QWORD *)(a1 + 1104) = 0;
  *(_QWORD *)(a1 + 1096) = v81;
  for ( j = &v81[2 * v82]; j != v81; v81 += 2 )
  {
    if ( v81 )
      *v81 = -4096;
  }
LABEL_41:
  sub_264E600(a1 + 1200);
  v19 = *(_BYTE *)(a1 + 1256) == 0;
  *(_DWORD *)(a1 + 1240) = 0;
  *(_BYTE *)(a1 + 1248) = 0;
  if ( !v19 )
    *(_BYTE *)(a1 + 1256) = 0;
  ++*(_QWORD *)(a1 + 2760);
  v19 = *(_BYTE *)(a1 + 2788) == 0;
  *(_DWORD *)(a1 + 2240) = 0;
  if ( !v19 )
    goto LABEL_48;
  v20 = 4 * (*(_DWORD *)(a1 + 2780) - *(_DWORD *)(a1 + 2784));
  v21 = *(unsigned int *)(a1 + 2776);
  if ( v20 < 0x20 )
    v20 = 32;
  if ( (unsigned int)v21 <= v20 )
  {
    memset(*(void **)(a1 + 2768), -1, 8 * v21);
LABEL_48:
    *(_QWORD *)(a1 + 2780) = 0;
    goto LABEL_49;
  }
  sub_C8C990(a1 + 2760, v2);
LABEL_49:
  v22 = *(_QWORD *)(a1 + 3256);
  for ( k = v22 + 16LL * *(unsigned int *)(a1 + 3264); v22 != k; *(_DWORD *)(v24 + 196) = 0 )
  {
    v24 = *(_QWORD *)(v22 + 8);
    v25 = *(_DWORD *)(v24 + 128);
    ++*(_QWORD *)(v24 + 112);
    if ( v25 )
    {
      v44 = 4 * v25;
      v26 = *(unsigned int *)(v24 + 136);
      if ( (unsigned int)(4 * v25) < 0x40 )
        v44 = 64;
      if ( v44 >= (unsigned int)v26 )
      {
LABEL_53:
        v27 = *(_QWORD **)(v24 + 120);
        for ( m = &v27[v26]; m != v27; ++v27 )
          *v27 = -4096;
        *(_QWORD *)(v24 + 128) = 0;
        goto LABEL_56;
      }
      v45 = v25 - 1;
      if ( !v45 )
      {
        v46 = *(_QWORD **)(v24 + 120);
        LODWORD(v47) = 64;
LABEL_146:
        v89 = v47;
        sub_C7D6A0((__int64)v46, 8 * v26, 8);
        v65 = (((((((4 * v89 / 3u + 1) | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 2)
                | (4 * v89 / 3u + 1)
                | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 4)
              | (((4 * v89 / 3u + 1) | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 2)
              | (4 * v89 / 3u + 1)
              | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 8)
            | (((((4 * v89 / 3u + 1) | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 2)
              | (4 * v89 / 3u + 1)
              | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 4)
            | (((4 * v89 / 3u + 1) | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1)) >> 2)
            | (4 * v89 / 3u + 1)
            | ((unsigned __int64)(4 * v89 / 3u + 1) >> 1);
        v66 = ((v65 >> 16) | v65) + 1;
        *(_DWORD *)(v24 + 136) = v66;
        v67 = (_QWORD *)sub_C7D670(8 * v66, 8);
        v68 = *(unsigned int *)(v24 + 136);
        *(_QWORD *)(v24 + 128) = 0;
        *(_QWORD *)(v24 + 120) = v67;
        for ( n = &v67[v68]; n != v67; ++v67 )
        {
          if ( v67 )
            *v67 = -4096;
        }
        goto LABEL_56;
      }
      _BitScanReverse(&v45, v45);
      v46 = *(_QWORD **)(v24 + 120);
      v47 = (unsigned int)(1 << (33 - (v45 ^ 0x1F)));
      if ( (int)v47 < 64 )
        v47 = 64;
      if ( (_DWORD)v47 != (_DWORD)v26 )
        goto LABEL_146;
      *(_QWORD *)(v24 + 128) = 0;
      v48 = &v46[v47];
      do
      {
        if ( v46 )
          *v46 = -4096;
        ++v46;
      }
      while ( v48 != v46 );
    }
    else if ( *(_DWORD *)(v24 + 132) )
    {
      v26 = *(unsigned int *)(v24 + 136);
      if ( (unsigned int)v26 <= 0x40 )
        goto LABEL_53;
      sub_C7D6A0(*(_QWORD *)(v24 + 120), 8 * v26, 8);
      *(_QWORD *)(v24 + 120) = 0;
      *(_QWORD *)(v24 + 128) = 0;
      *(_DWORD *)(v24 + 136) = 0;
    }
LABEL_56:
    v29 = *(_DWORD *)(v24 + 200) - *(_DWORD *)(v24 + 196);
    *(_DWORD *)(v24 + 152) = 0;
    *(_QWORD *)(v24 + 160) = 0;
    *(_QWORD *)(v24 + 168) = 0;
    if ( v29 < 16 )
      v29 = 16;
    v22 += 16;
    ++*(_DWORD *)(v24 + 204);
    *(_QWORD *)(v24 + 176) = 0;
    *(_QWORD *)(v24 + 184) = 0;
    *(_BYTE *)(v24 + 192) = 0;
    *(_DWORD *)(v24 + 200) = v29;
  }
  v30 = *(_DWORD *)(a1 + 3536);
  ++*(_QWORD *)(a1 + 3520);
  if ( !v30 )
  {
    if ( !*(_DWORD *)(a1 + 3540) )
      goto LABEL_65;
    v31 = *(unsigned int *)(a1 + 3544);
    if ( (unsigned int)v31 <= 0x40 )
      goto LABEL_62;
    sub_C7D6A0(*(_QWORD *)(a1 + 3528), 24 * v31, 8);
    *(_DWORD *)(a1 + 3544) = 0;
LABEL_170:
    *(_QWORD *)(a1 + 3528) = 0;
LABEL_64:
    *(_QWORD *)(a1 + 3536) = 0;
    goto LABEL_65;
  }
  v59 = 4 * v30;
  v31 = *(unsigned int *)(a1 + 3544);
  if ( (unsigned int)(4 * v30) < 0x40 )
    v59 = 64;
  if ( v59 >= (unsigned int)v31 )
  {
LABEL_62:
    v32 = *(_QWORD **)(a1 + 3528);
    for ( ii = &v32[3 * v31]; ii != v32; v32 += 3 )
      *v32 = -4096;
    goto LABEL_64;
  }
  v60 = v30 - 1;
  if ( v60 )
  {
    _BitScanReverse(&v60, v60);
    v61 = *(_QWORD **)(a1 + 3528);
    v62 = (unsigned int)(1 << (33 - (v60 ^ 0x1F)));
    if ( (int)v62 < 64 )
      v62 = 64;
    if ( (_DWORD)v62 == (_DWORD)v31 )
    {
      *(_QWORD *)(a1 + 3536) = 0;
      v63 = &v61[3 * v62];
      do
      {
        if ( v61 )
          *v61 = -4096;
        v61 += 3;
      }
      while ( v63 != v61 );
      goto LABEL_65;
    }
  }
  else
  {
    v61 = *(_QWORD **)(a1 + 3528);
    LODWORD(v62) = 64;
  }
  sub_C7D6A0((__int64)v61, 24 * v31, 8);
  v76 = sub_2B149A0(v62);
  *(_DWORD *)(a1 + 3544) = v76;
  if ( !v76 )
    goto LABEL_170;
  v77 = (_QWORD *)sub_C7D670(24LL * v76, 8);
  v78 = *(unsigned int *)(a1 + 3544);
  *(_QWORD *)(a1 + 3536) = 0;
  *(_QWORD *)(a1 + 3528) = v77;
  for ( jj = &v77[3 * v78]; jj != v77; v77 += 3 )
  {
    if ( v77 )
      *v77 = -4096;
  }
LABEL_65:
  v19 = *(_BYTE *)(a1 + 3568) == 0;
  *(_QWORD *)(a1 + 3552) = 0x100000000LL;
  if ( !v19 )
    *(_BYTE *)(a1 + 3568) = 0;
  sub_264E600(a1 + 3576);
  sub_2B395F0(a1 + 688);
  v34 = *(_DWORD *)(a1 + 1136);
  ++*(_QWORD *)(a1 + 1120);
  *(_QWORD *)(a1 + 3272) = 0;
  if ( v34 )
  {
    v54 = 4 * v34;
    v35 = *(unsigned int *)(a1 + 1144);
    if ( (unsigned int)(4 * v34) < 0x40 )
      v54 = 64;
    if ( v54 >= (unsigned int)v35 )
      goto LABEL_70;
    v55 = v34 - 1;
    if ( v55 )
    {
      _BitScanReverse(&v55, v55);
      v56 = *(_QWORD **)(a1 + 1128);
      v57 = (unsigned int)(1 << (33 - (v55 ^ 0x1F)));
      if ( (int)v57 < 64 )
        v57 = 64;
      if ( (_DWORD)v57 == (_DWORD)v35 )
      {
        *(_QWORD *)(a1 + 1136) = 0;
        v58 = &v56[v57];
        do
        {
          if ( v56 )
            *v56 = -4096;
          ++v56;
        }
        while ( v58 != v56 );
        goto LABEL_73;
      }
    }
    else
    {
      v56 = *(_QWORD **)(a1 + 1128);
      LODWORD(v57) = 64;
    }
    sub_C7D6A0((__int64)v56, 8 * v35, 8);
    v84 = sub_2B149A0(v57);
    *(_DWORD *)(a1 + 1144) = v84;
    if ( !v84 )
      goto LABEL_174;
    v85 = (_QWORD *)sub_C7D670(8LL * v84, 8);
    v86 = *(unsigned int *)(a1 + 1144);
    *(_QWORD *)(a1 + 1136) = 0;
    *(_QWORD *)(a1 + 1128) = v85;
    for ( kk = &v85[v86]; kk != v85; ++v85 )
    {
      if ( v85 )
        *v85 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 1140) )
  {
    v35 = *(unsigned int *)(a1 + 1144);
    if ( (unsigned int)v35 <= 0x40 )
    {
LABEL_70:
      v36 = *(_QWORD **)(a1 + 1128);
      for ( mm = &v36[v35]; mm != v36; ++v36 )
        *v36 = -4096;
      goto LABEL_72;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 1128), 8 * v35, 8);
    *(_DWORD *)(a1 + 1144) = 0;
LABEL_174:
    *(_QWORD *)(a1 + 1128) = 0;
LABEL_72:
    *(_QWORD *)(a1 + 1136) = 0;
  }
LABEL_73:
  v38 = *(_DWORD *)(a1 + 1184);
  ++*(_QWORD *)(a1 + 1168);
  *(_DWORD *)(a1 + 1160) = 0;
  if ( !v38 )
  {
    result = *(unsigned int *)(a1 + 1188);
    if ( !(_DWORD)result )
      return result;
  }
  v40 = *(_QWORD *)(a1 + 1176);
  result = (unsigned int)(4 * v38);
  v41 = 88LL * *(unsigned int *)(a1 + 1192);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v42 = v40 + v41;
  if ( *(_DWORD *)(a1 + 1192) <= (unsigned int)result )
  {
    while ( v40 != v42 )
    {
      result = *(_QWORD *)v40;
      if ( *(_QWORD *)v40 != -4096 )
      {
        if ( result != -8192 )
        {
          v43 = *(_QWORD *)(v40 + 40);
          if ( v43 != v40 + 56 )
            _libc_free(v43);
          result = sub_C7D6A0(*(_QWORD *)(v40 + 16), 8LL * *(unsigned int *)(v40 + 32), 8);
        }
        *(_QWORD *)v40 = -4096;
      }
      v40 += 88;
    }
    goto LABEL_87;
  }
  do
  {
    while ( 1 )
    {
      result = *(_QWORD *)v40;
      if ( *(_QWORD *)v40 != -4096 )
        break;
LABEL_140:
      v40 += 88;
      if ( v42 == v40 )
        goto LABEL_155;
    }
    if ( result != -8192 )
    {
      v64 = *(_QWORD *)(v40 + 40);
      if ( v64 != v40 + 56 )
        _libc_free(v64);
      result = sub_C7D6A0(*(_QWORD *)(v40 + 16), 8LL * *(unsigned int *)(v40 + 32), 8);
      goto LABEL_140;
    }
    v40 += 88;
  }
  while ( v42 != v40 );
LABEL_155:
  v70 = *(_DWORD *)(a1 + 1192);
  if ( !v38 )
  {
    if ( v70 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 1176), v41, 8);
      *(_DWORD *)(a1 + 1192) = 0;
      goto LABEL_168;
    }
LABEL_87:
    *(_QWORD *)(a1 + 1184) = 0;
    return result;
  }
  v71 = 64;
  if ( v38 != 1 )
  {
    _BitScanReverse(&v72, v38 - 1);
    v71 = (unsigned int)(1 << (33 - (v72 ^ 0x1F)));
    if ( (int)v71 < 64 )
      v71 = 64;
  }
  v73 = *(_QWORD **)(a1 + 1176);
  if ( (_DWORD)v71 != v70 )
  {
    sub_C7D6A0((__int64)v73, v41, 8);
    result = sub_2B149A0(v71);
    *(_DWORD *)(a1 + 1192) = result;
    if ( (_DWORD)result )
    {
      result = sub_C7D670(88LL * (unsigned int)result, 8);
      v74 = *(unsigned int *)(a1 + 1192);
      *(_QWORD *)(a1 + 1184) = 0;
      *(_QWORD *)(a1 + 1176) = result;
      for ( nn = result + 88 * v74; nn != result; result += 88 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
      return result;
    }
LABEL_168:
    *(_QWORD *)(a1 + 1176) = 0;
    *(_QWORD *)(a1 + 1184) = 0;
    return result;
  }
  *(_QWORD *)(a1 + 1184) = 0;
  result = (__int64)&v73[11 * v71];
  do
  {
    if ( v73 )
      *v73 = -4096;
    v73 += 11;
  }
  while ( (_QWORD *)result != v73 );
  return result;
}
