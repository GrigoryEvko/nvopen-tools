// Function: sub_18A8CF0
// Address: 0x18a8cf0
//
__int64 __fastcall sub_18A8CF0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *j; // rdx
  void *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *k; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // r13
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // rax
  _QWORD *v25; // r12
  _QWORD *v26; // r13
  __int64 v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 *v31; // r13
  __int64 v32; // r15
  __int64 *v33; // r12
  __int64 *v34; // r14
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rax
  void *v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rdx
  unsigned __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rdi
  __int64 v45; // rax
  unsigned __int64 *v46; // r12
  unsigned __int64 *v47; // r13
  unsigned __int64 v48; // rdi
  __int64 v49; // rax
  unsigned __int64 v50; // r13
  unsigned __int64 v51; // rdi
  __int64 v52; // rdi
  int v53; // r15d
  __int64 result; // rax
  __int64 *v55; // r12
  __int64 *v56; // r14
  unsigned int v57; // ecx
  unsigned int v58; // eax
  int v59; // r13d
  unsigned int v60; // eax
  unsigned int v61; // ecx
  unsigned int v62; // eax
  int v63; // r13d
  unsigned int v64; // eax
  unsigned int v65; // ecx
  unsigned int v66; // eax
  int v67; // eax
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rax
  int v70; // r13d
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // rdi
  unsigned __int64 *v74; // rdx
  unsigned __int64 v75; // rcx
  unsigned __int64 *v76; // r13
  unsigned __int64 *v77; // r12
  unsigned __int64 v78; // rdi
  unsigned __int64 *v79; // r12
  unsigned __int64 v80; // rdi
  int v81; // edx
  int v82; // r13d
  unsigned int v83; // eax
  unsigned __int64 v84; // rdx
  unsigned __int64 v85; // rax
  __int64 *v86; // [rsp+0h] [rbp-40h]
  __int64 v87; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 )
  {
    v65 = 4 * v2;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v65 = 64;
    if ( (unsigned int)v3 <= v65 )
      goto LABEL_4;
    v66 = v2 - 1;
    if ( v66 )
    {
      _BitScanReverse(&v66, v66);
      v67 = 1 << (33 - (v66 ^ 0x1F));
      if ( v67 < 64 )
        v67 = 64;
      if ( (_DWORD)v3 == v67 )
        goto LABEL_128;
      v68 = (4 * v67 / 3u + 1) | ((unsigned __int64)(4 * v67 / 3u + 1) >> 1);
      v69 = ((v68 | (v68 >> 2)) >> 4) | v68 | (v68 >> 2) | ((((v68 | (v68 >> 2)) >> 4) | v68 | (v68 >> 2)) >> 8);
      v70 = (v69 | (v69 >> 16)) + 1;
      v71 = 16 * ((v69 | (v69 >> 16)) + 1);
    }
    else
    {
      v71 = 2048;
      v70 = 128;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_DWORD *)(a1 + 24) = v70;
    *(_QWORD *)(a1 + 8) = sub_22077B0(v71);
LABEL_128:
    sub_18A8BF0(a1);
    goto LABEL_7;
  }
  if ( !*(_DWORD *)(a1 + 20) )
    goto LABEL_7;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v3 > 0x40 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
LABEL_4:
  v4 = *(_QWORD **)(a1 + 8);
  for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
    *v4 = -8;
  *(_QWORD *)(a1 + 16) = 0;
LABEL_7:
  v6 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 52) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v7 <= 0x40 )
      goto LABEL_10;
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    *(_DWORD *)(a1 + 56) = 0;
LABEL_158:
    *(_QWORD *)(a1 + 40) = 0;
LABEL_12:
    *(_QWORD *)(a1 + 48) = 0;
    goto LABEL_13;
  }
  v61 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v61 = 64;
  if ( (unsigned int)v7 <= v61 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 40);
    for ( j = &v8[3 * v7]; j != v8; *(v8 - 2) = -8 )
    {
      *v8 = -8;
      v8 += 3;
    }
    goto LABEL_12;
  }
  v62 = v6 - 1;
  if ( v62 )
  {
    _BitScanReverse(&v62, v62);
    v63 = 1 << (33 - (v62 ^ 0x1F));
    if ( v63 < 64 )
      v63 = 64;
    if ( (_DWORD)v7 == v63 )
      goto LABEL_118;
  }
  else
  {
    v63 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 40));
  v64 = sub_18A4140(v63);
  *(_DWORD *)(a1 + 56) = v64;
  if ( !v64 )
    goto LABEL_158;
  *(_QWORD *)(a1 + 40) = sub_22077B0(24LL * v64);
LABEL_118:
  sub_18A8C30(a1 + 32);
LABEL_13:
  ++*(_QWORD *)(a1 + 64);
  v10 = *(void **)(a1 + 80);
  if ( v10 == *(void **)(a1 + 72) )
  {
LABEL_18:
    *(_QWORD *)(a1 + 92) = 0;
    goto LABEL_19;
  }
  v11 = 4 * (*(_DWORD *)(a1 + 92) - *(_DWORD *)(a1 + 96));
  v12 = *(unsigned int *)(a1 + 88);
  if ( v11 < 0x20 )
    v11 = 32;
  if ( (unsigned int)v12 <= v11 )
  {
    memset(v10, -1, 8 * v12);
    goto LABEL_18;
  }
  sub_16CC920(a1 + 64);
LABEL_19:
  *(_DWORD *)(a1 + 368) = 0;
  sub_18A3DA0(*(_QWORD *)(a1 + 904));
  ++*(_QWORD *)(a1 + 936);
  *(_QWORD *)(a1 + 912) = a1 + 896;
  *(_QWORD *)(a1 + 920) = a1 + 896;
  v13 = *(_DWORD *)(a1 + 952);
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  if ( !v13 )
  {
    if ( !*(_DWORD *)(a1 + 956) )
      goto LABEL_25;
    v14 = *(unsigned int *)(a1 + 960);
    if ( (unsigned int)v14 <= 0x40 )
      goto LABEL_22;
    j___libc_free_0(*(_QWORD *)(a1 + 944));
    *(_DWORD *)(a1 + 960) = 0;
LABEL_156:
    *(_QWORD *)(a1 + 944) = 0;
LABEL_24:
    *(_QWORD *)(a1 + 952) = 0;
    goto LABEL_25;
  }
  v57 = 4 * v13;
  v14 = *(unsigned int *)(a1 + 960);
  if ( (unsigned int)(4 * v13) < 0x40 )
    v57 = 64;
  if ( (unsigned int)v14 <= v57 )
  {
LABEL_22:
    v15 = *(_QWORD **)(a1 + 944);
    for ( k = &v15[2 * v14]; k != v15; v15 += 2 )
      *v15 = -8;
    goto LABEL_24;
  }
  v58 = v13 - 1;
  if ( v58 )
  {
    _BitScanReverse(&v58, v58);
    v59 = 1 << (33 - (v58 ^ 0x1F));
    if ( v59 < 64 )
      v59 = 64;
    if ( (_DWORD)v14 == v59 )
      goto LABEL_108;
  }
  else
  {
    v59 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 944));
  v60 = sub_18A4140(v59);
  *(_DWORD *)(a1 + 960) = v60;
  if ( !v60 )
    goto LABEL_156;
  *(_QWORD *)(a1 + 944) = sub_22077B0(16LL * v60);
LABEL_108:
  sub_18A8C70(a1 + 936);
LABEL_25:
  v17 = *(_QWORD *)(a1 + 1000);
  *(_QWORD *)(a1 + 1000) = 0;
  if ( v17 )
  {
    v18 = *(unsigned int *)(v17 + 48);
    if ( (_DWORD)v18 )
    {
      v19 = *(_QWORD **)(v17 + 32);
      v20 = &v19[2 * v18];
      do
      {
        if ( *v19 != -8 && *v19 != -16 )
        {
          v21 = v19[1];
          if ( v21 )
          {
            v22 = *(_QWORD *)(v21 + 24);
            if ( v22 )
              j_j___libc_free_0(v22, *(_QWORD *)(v21 + 40) - v22);
            j_j___libc_free_0(v21, 56);
          }
        }
        v19 += 2;
      }
      while ( v20 != v19 );
    }
    j___libc_free_0(*(_QWORD *)(v17 + 32));
    if ( *(_QWORD *)v17 != v17 + 16 )
      _libc_free(*(_QWORD *)v17);
    j_j___libc_free_0(v17, 80);
  }
  v23 = *(_QWORD *)(a1 + 1008);
  *(_QWORD *)(a1 + 1008) = 0;
  if ( v23 )
  {
    v24 = *(unsigned int *)(v23 + 72);
    if ( (_DWORD)v24 )
    {
      v25 = *(_QWORD **)(v23 + 56);
      v26 = &v25[2 * v24];
      do
      {
        if ( *v25 != -16 && *v25 != -8 )
        {
          v27 = v25[1];
          if ( v27 )
          {
            v28 = *(_QWORD *)(v27 + 24);
            if ( v28 )
              j_j___libc_free_0(v28, *(_QWORD *)(v27 + 40) - v28);
            j_j___libc_free_0(v27, 56);
          }
        }
        v25 += 2;
      }
      while ( v26 != v25 );
    }
    j___libc_free_0(*(_QWORD *)(v23 + 56));
    if ( *(_QWORD *)v23 != v23 + 16 )
      _libc_free(*(_QWORD *)v23);
    j_j___libc_free_0(v23, 104);
  }
  v29 = *(_QWORD *)(a1 + 1016);
  *(_QWORD *)(a1 + 1016) = 0;
  v87 = v29;
  v30 = v29;
  if ( v29 )
  {
    sub_142D890(v29);
    v31 = *(__int64 **)(v30 + 32);
    v86 = *(__int64 **)(v30 + 40);
    if ( v31 != v86 )
    {
      do
      {
        v32 = *v31;
        v33 = *(__int64 **)(*v31 + 16);
        if ( *(__int64 **)(*v31 + 8) == v33 )
        {
          *(_BYTE *)(v32 + 160) = 1;
        }
        else
        {
          v34 = *(__int64 **)(*v31 + 8);
          do
          {
            v35 = *v34++;
            sub_13FACC0(v35);
          }
          while ( v33 != v34 );
          *(_BYTE *)(v32 + 160) = 1;
          v36 = *(_QWORD *)(v32 + 8);
          if ( *(_QWORD *)(v32 + 16) != v36 )
            *(_QWORD *)(v32 + 16) = v36;
        }
        v37 = *(_QWORD *)(v32 + 32);
        if ( v37 != *(_QWORD *)(v32 + 40) )
          *(_QWORD *)(v32 + 40) = v37;
        ++*(_QWORD *)(v32 + 56);
        v38 = *(void **)(v32 + 72);
        if ( v38 == *(void **)(v32 + 64) )
        {
          *(_QWORD *)v32 = 0;
        }
        else
        {
          v39 = 4 * (*(_DWORD *)(v32 + 84) - *(_DWORD *)(v32 + 88));
          v40 = *(unsigned int *)(v32 + 80);
          if ( v39 < 0x20 )
            v39 = 32;
          if ( (unsigned int)v40 > v39 )
            sub_16CC920(v32 + 56);
          else
            memset(v38, -1, 8 * v40);
          v41 = *(_QWORD *)(v32 + 72);
          v42 = *(_QWORD *)(v32 + 64);
          *(_QWORD *)v32 = 0;
          if ( v41 != v42 )
            _libc_free(v41);
        }
        v43 = *(_QWORD *)(v32 + 32);
        if ( v43 )
          j_j___libc_free_0(v43, *(_QWORD *)(v32 + 48) - v43);
        v44 = *(_QWORD *)(v32 + 8);
        if ( v44 )
          j_j___libc_free_0(v44, *(_QWORD *)(v32 + 24) - v44);
        ++v31;
      }
      while ( v86 != v31 );
      v45 = *(_QWORD *)(v87 + 32);
      if ( v45 != *(_QWORD *)(v87 + 40) )
        *(_QWORD *)(v87 + 40) = v45;
    }
    v46 = *(unsigned __int64 **)(v87 + 120);
    v47 = &v46[2 * *(unsigned int *)(v87 + 128)];
    while ( v46 != v47 )
    {
      v48 = *v46;
      v46 += 2;
      _libc_free(v48);
    }
    *(_DWORD *)(v87 + 128) = 0;
    v49 = *(unsigned int *)(v87 + 80);
    if ( (_DWORD)v49 )
    {
      *(_QWORD *)(v87 + 136) = 0;
      v74 = *(unsigned __int64 **)(v87 + 72);
      v75 = *v74;
      v76 = &v74[v49];
      v77 = v74 + 1;
      *(_QWORD *)(v87 + 56) = *v74;
      *(_QWORD *)(v87 + 64) = v75 + 4096;
      if ( v76 != v74 + 1 )
      {
        do
        {
          v78 = *v77++;
          _libc_free(v78);
        }
        while ( v76 != v77 );
        v74 = *(unsigned __int64 **)(v87 + 72);
      }
      *(_DWORD *)(v87 + 80) = 1;
      _libc_free(*v74);
      v79 = *(unsigned __int64 **)(v87 + 120);
      v50 = (unsigned __int64)&v79[2 * *(unsigned int *)(v87 + 128)];
      if ( v79 == (unsigned __int64 *)v50 )
        goto LABEL_78;
      do
      {
        v80 = *v79;
        v79 += 2;
        _libc_free(v80);
      }
      while ( (unsigned __int64 *)v50 != v79 );
    }
    v50 = *(_QWORD *)(v87 + 120);
LABEL_78:
    if ( v50 != v87 + 136 )
      _libc_free(v50);
    v51 = *(_QWORD *)(v87 + 72);
    if ( v51 != v87 + 88 )
      _libc_free(v51);
    v52 = *(_QWORD *)(v87 + 32);
    if ( v52 )
      j_j___libc_free_0(v52, *(_QWORD *)(v87 + 48) - v52);
    j___libc_free_0(*(_QWORD *)(v87 + 8));
    j_j___libc_free_0(v87, 160);
  }
  sub_18A7BC0(a1 + 1088);
  sub_18A7BC0(a1 + 1120);
  v53 = *(_DWORD *)(a1 + 1168);
  ++*(_QWORD *)(a1 + 1152);
  if ( !v53 )
  {
    result = *(unsigned int *)(a1 + 1172);
    if ( !(_DWORD)result )
      goto LABEL_97;
  }
  v55 = *(__int64 **)(a1 + 1160);
  v56 = &v55[7 * *(unsigned int *)(a1 + 1176)];
  result = (unsigned int)(4 * v53);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( *(_DWORD *)(a1 + 1176) <= (unsigned int)result )
  {
    for ( ; v55 != v56; v55 += 7 )
    {
      result = *v55;
      if ( *v55 != -8 )
      {
        if ( result != -16 )
          result = sub_18A3F70(v55[3]);
        *v55 = -8;
      }
    }
    *(_QWORD *)(a1 + 1168) = 0;
    goto LABEL_97;
  }
  do
  {
    while ( *v55 == -16 )
    {
LABEL_131:
      v55 += 7;
      if ( v55 == v56 )
        goto LABEL_144;
    }
    if ( *v55 != -8 )
    {
      v72 = v55[3];
      while ( v72 )
      {
        sub_18A3F70(*(_QWORD *)(v72 + 24));
        v73 = v72;
        v72 = *(_QWORD *)(v72 + 16);
        j_j___libc_free_0(v73, 48);
      }
      goto LABEL_131;
    }
    v55 += 7;
  }
  while ( v55 != v56 );
LABEL_144:
  v81 = *(_DWORD *)(a1 + 1176);
  if ( v53 )
  {
    v82 = 64;
    if ( v53 != 1 )
    {
      _BitScanReverse(&v83, v53 - 1);
      v82 = 1 << (33 - (v83 ^ 0x1F));
      if ( v82 < 64 )
        v82 = 64;
    }
    if ( v81 != v82 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 1160));
      v84 = ((((((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
               | (4 * v82 / 3u + 1)
               | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
             | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
             | (4 * v82 / 3u + 1)
             | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
             | (4 * v82 / 3u + 1)
             | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
           | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
           | (4 * v82 / 3u + 1)
           | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 16;
      v85 = (v84
           | (((((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
               | (4 * v82 / 3u + 1)
               | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
             | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
             | (4 * v82 / 3u + 1)
             | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
             | (4 * v82 / 3u + 1)
             | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
           | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
           | (4 * v82 / 3u + 1)
           | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 1176) = v85;
      *(_QWORD *)(a1 + 1160) = sub_22077B0(56 * v85);
    }
LABEL_150:
    result = (__int64)sub_18A8CB0(a1 + 1152);
  }
  else
  {
    if ( !v81 )
      goto LABEL_150;
    result = j___libc_free_0(*(_QWORD *)(a1 + 1160));
    *(_QWORD *)(a1 + 1160) = 0;
    *(_QWORD *)(a1 + 1168) = 0;
    *(_DWORD *)(a1 + 1176) = 0;
  }
LABEL_97:
  *(_QWORD *)(a1 + 1184) = 0;
  return result;
}
