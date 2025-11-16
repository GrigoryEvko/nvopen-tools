// Function: sub_26C8090
// Address: 0x26c8090
//
void __fastcall sub_26C8090(__int64 a1, char a2)
{
  int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  int v9; // eax
  unsigned int v10; // edi
  _QWORD *v11; // rax
  _QWORD *j; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  _QWORD *k; // rdx
  unsigned __int64 v19; // r15
  __int64 v20; // r14
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r15
  __int64 v25; // r14
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // r14
  __int64 v30; // r15
  __int64 *v31; // r12
  __int64 *v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // rdx
  char v38; // al
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  __int64 *v42; // r12
  __int64 *v43; // r13
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rdx
  unsigned __int64 v47; // r13
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  int v50; // r8d
  unsigned int v51; // eax
  _QWORD *v52; // r12
  __int64 v53; // r14
  _QWORD *v54; // r13
  unsigned int v55; // ecx
  unsigned int v56; // eax
  int v57; // r14d
  unsigned int v58; // eax
  unsigned int v59; // edx
  unsigned int v60; // eax
  int v61; // r14d
  unsigned int v62; // eax
  unsigned int v63; // ecx
  unsigned int v64; // eax
  int v65; // r14d
  unsigned int v66; // eax
  unsigned __int64 m; // r15
  unsigned __int64 v68; // rdi
  __int64 *v69; // rax
  __int64 v70; // rcx
  __int64 *v71; // r12
  __int64 *v72; // r15
  __int64 v73; // rdi
  unsigned int v74; // ecx
  __int64 v75; // rsi
  __int64 *v76; // r12
  __int64 v77; // rsi
  __int64 v78; // rdi
  __int64 v79; // rdx
  int v80; // r12d
  unsigned int v81; // eax
  _QWORD *v82; // rdi
  unsigned int v83; // eax
  _QWORD *v84; // rax
  __int64 v85; // rcx
  _QWORD *n; // rdx
  _QWORD *v87; // rax
  __int64 *v88; // [rsp+0h] [rbp-40h]
  __int64 *v89; // [rsp+8h] [rbp-38h]
  int v90; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v4 )
  {
    v5 = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)v5 )
      goto LABEL_7;
    v6 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v6 <= 0x40 )
      goto LABEL_4;
    v5 = 16LL * *(unsigned int *)(a1 + 64);
    sub_C7D6A0(*(_QWORD *)(a1 + 48), v5, 8);
    *(_DWORD *)(a1 + 64) = 0;
LABEL_168:
    *(_QWORD *)(a1 + 48) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_7;
  }
  v63 = 4 * v4;
  v5 = 64;
  v6 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v63 = 64;
  if ( v63 >= (unsigned int)v6 )
  {
LABEL_4:
    v7 = *(_QWORD **)(a1 + 48);
    for ( i = &v7[2 * v6]; i != v7; v7 += 2 )
      *v7 = -4096;
    goto LABEL_6;
  }
  v64 = v4 - 1;
  if ( v64 )
  {
    _BitScanReverse(&v64, v64);
    v65 = 1 << (33 - (v64 ^ 0x1F));
    if ( v65 < 64 )
      v65 = 64;
    if ( v65 == (_DWORD)v6 )
      goto LABEL_130;
  }
  else
  {
    v65 = 64;
  }
  v5 = 16LL * (unsigned int)v6;
  sub_C7D6A0(*(_QWORD *)(a1 + 48), v5, 8);
  v66 = sub_26BC060(v65);
  *(_DWORD *)(a1 + 64) = v66;
  if ( !v66 )
    goto LABEL_168;
  v5 = 8;
  *(_QWORD *)(a1 + 48) = sub_C7D670(16LL * v66, 8);
LABEL_130:
  sub_26C5740(a1 + 40);
LABEL_7:
  v9 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( !v9 )
  {
    if ( !*(_DWORD *)(a1 + 92) )
      goto LABEL_13;
    v10 = *(_DWORD *)(a1 + 96);
    if ( v10 <= 0x40 )
      goto LABEL_10;
    v5 = 24LL * v10;
    sub_C7D6A0(*(_QWORD *)(a1 + 80), v5, 8);
    *(_DWORD *)(a1 + 96) = 0;
LABEL_166:
    *(_QWORD *)(a1 + 80) = 0;
LABEL_12:
    *(_QWORD *)(a1 + 88) = 0;
    goto LABEL_13;
  }
  v59 = 4 * v9;
  v5 = 64;
  v10 = *(_DWORD *)(a1 + 96);
  if ( (unsigned int)(4 * v9) < 0x40 )
    v59 = 64;
  if ( v10 <= v59 )
  {
LABEL_10:
    v11 = *(_QWORD **)(a1 + 80);
    for ( j = &v11[3 * v10]; j != v11; *(v11 - 2) = -4096 )
    {
      *v11 = -4096;
      v11 += 3;
    }
    goto LABEL_12;
  }
  v60 = v9 - 1;
  if ( v60 )
  {
    _BitScanReverse(&v60, v60);
    v61 = 1 << (33 - (v60 ^ 0x1F));
    if ( v61 < 64 )
      v61 = 64;
    if ( v61 == v10 )
      goto LABEL_120;
  }
  else
  {
    v61 = 64;
  }
  v5 = 24LL * v10;
  sub_C7D6A0(*(_QWORD *)(a1 + 80), v5, 8);
  v62 = sub_26BC060(v61);
  *(_DWORD *)(a1 + 96) = v62;
  if ( !v62 )
    goto LABEL_166;
  v5 = 8;
  *(_QWORD *)(a1 + 80) = sub_C7D670(24LL * v62, 8);
LABEL_120:
  sub_26C8010(a1 + 72);
LABEL_13:
  ++*(_QWORD *)(a1 + 104);
  if ( *(_BYTE *)(a1 + 132) )
  {
LABEL_18:
    *(_QWORD *)(a1 + 124) = 0;
    goto LABEL_19;
  }
  v13 = 4 * (*(_DWORD *)(a1 + 124) - *(_DWORD *)(a1 + 128));
  v14 = *(unsigned int *)(a1 + 120);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    v5 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 112), -1, 8 * v14);
    goto LABEL_18;
  }
  sub_C8C990(a1 + 104, v5);
LABEL_19:
  *(_DWORD *)(a1 + 400) = 0;
  sub_26BBBD0(*(_QWORD *)(a1 + 936));
  ++*(_QWORD *)(a1 + 968);
  *(_QWORD *)(a1 + 944) = a1 + 928;
  *(_QWORD *)(a1 + 952) = a1 + 928;
  v15 = *(_DWORD *)(a1 + 984);
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  if ( v15 )
  {
    v55 = 4 * v15;
    v5 = 64;
    v16 = *(unsigned int *)(a1 + 992);
    if ( (unsigned int)(4 * v15) < 0x40 )
      v55 = 64;
    if ( (unsigned int)v16 <= v55 )
      goto LABEL_22;
    v56 = v15 - 1;
    if ( v56 )
    {
      _BitScanReverse(&v56, v56);
      v57 = 1 << (33 - (v56 ^ 0x1F));
      if ( v57 < 64 )
        v57 = 64;
      if ( v57 == (_DWORD)v16 )
        goto LABEL_110;
    }
    else
    {
      v57 = 64;
    }
    v5 = 16LL * (unsigned int)v16;
    sub_C7D6A0(*(_QWORD *)(a1 + 976), v5, 8);
    v58 = sub_26BC060(v57);
    *(_DWORD *)(a1 + 992) = v58;
    if ( !v58 )
      goto LABEL_164;
    v5 = 8;
    *(_QWORD *)(a1 + 976) = sub_C7D670(16LL * v58, 8);
LABEL_110:
    sub_26C8050(a1 + 968);
    goto LABEL_25;
  }
  if ( *(_DWORD *)(a1 + 988) )
  {
    v16 = *(unsigned int *)(a1 + 992);
    if ( (unsigned int)v16 <= 0x40 )
    {
LABEL_22:
      v17 = *(_QWORD **)(a1 + 976);
      for ( k = &v17[2 * v16]; k != v17; v17 += 2 )
        *v17 = -4096;
      goto LABEL_24;
    }
    v5 = 16LL * *(unsigned int *)(a1 + 992);
    sub_C7D6A0(*(_QWORD *)(a1 + 976), v5, 8);
    *(_DWORD *)(a1 + 992) = 0;
LABEL_164:
    *(_QWORD *)(a1 + 976) = 0;
LABEL_24:
    *(_QWORD *)(a1 + 984) = 0;
  }
LABEL_25:
  if ( a2 )
  {
    v19 = *(_QWORD *)(a1 + 1000);
    *(_QWORD *)(a1 + 1000) = 0;
    if ( v19 )
    {
      v20 = *(_QWORD *)(v19 + 24);
      v21 = v20 + 8LL * *(unsigned int *)(v19 + 32);
      if ( v20 != v21 )
      {
        do
        {
          v22 = *(_QWORD *)(v21 - 8);
          v21 -= 8LL;
          if ( v22 )
          {
            v23 = *(_QWORD *)(v22 + 24);
            if ( v23 != v22 + 40 )
              _libc_free(v23);
            j_j___libc_free_0(v22);
          }
        }
        while ( v20 != v21 );
        v21 = *(_QWORD *)(v19 + 24);
      }
      if ( v21 != v19 + 40 )
        _libc_free(v21);
      if ( *(_QWORD *)v19 != v19 + 16 )
        _libc_free(*(_QWORD *)v19);
      v5 = 128;
      j_j___libc_free_0(v19);
    }
    v24 = *(_QWORD *)(a1 + 1008);
    *(_QWORD *)(a1 + 1008) = 0;
    if ( v24 )
    {
      v25 = *(_QWORD *)(v24 + 48);
      v26 = v25 + 8LL * *(unsigned int *)(v24 + 56);
      if ( v25 != v26 )
      {
        do
        {
          v27 = *(_QWORD *)(v26 - 8);
          v26 -= 8LL;
          if ( v27 )
          {
            v28 = *(_QWORD *)(v27 + 24);
            if ( v28 != v27 + 40 )
              _libc_free(v28);
            j_j___libc_free_0(v27);
          }
        }
        while ( v25 != v26 );
        v26 = *(_QWORD *)(v24 + 48);
      }
      if ( v26 != v24 + 64 )
        _libc_free(v26);
      if ( *(_QWORD *)v24 != v24 + 16 )
        _libc_free(*(_QWORD *)v24);
      v5 = 152;
      j_j___libc_free_0(v24);
    }
    v29 = *(_QWORD *)(a1 + 1016);
    *(_QWORD *)(a1 + 1016) = 0;
    if ( v29 )
    {
      sub_D786F0(v29);
      v88 = *(__int64 **)(v29 + 40);
      if ( *(__int64 **)(v29 + 32) != v88 )
      {
        v89 = *(__int64 **)(v29 + 32);
        do
        {
          v30 = *v89;
          v31 = *(__int64 **)(*v89 + 16);
          if ( *(__int64 **)(*v89 + 8) == v31 )
          {
            *(_BYTE *)(v30 + 152) = 1;
          }
          else
          {
            v32 = *(__int64 **)(*v89 + 8);
            do
            {
              v33 = *v32++;
              sub_D47BB0(v33, v5);
            }
            while ( v31 != v32 );
            *(_BYTE *)(v30 + 152) = 1;
            v34 = *(_QWORD *)(v30 + 8);
            if ( *(_QWORD *)(v30 + 16) != v34 )
              *(_QWORD *)(v30 + 16) = v34;
          }
          v35 = *(_QWORD *)(v30 + 32);
          if ( v35 != *(_QWORD *)(v30 + 40) )
            *(_QWORD *)(v30 + 40) = v35;
          ++*(_QWORD *)(v30 + 56);
          if ( *(_BYTE *)(v30 + 84) )
          {
            *(_QWORD *)v30 = 0;
          }
          else
          {
            v36 = 4 * (*(_DWORD *)(v30 + 76) - *(_DWORD *)(v30 + 80));
            v37 = *(unsigned int *)(v30 + 72);
            if ( v36 < 0x20 )
              v36 = 32;
            if ( (unsigned int)v37 > v36 )
            {
              sub_C8C990(v30 + 56, v5);
            }
            else
            {
              v5 = 0xFFFFFFFFLL;
              memset(*(void **)(v30 + 64), -1, 8 * v37);
            }
            v38 = *(_BYTE *)(v30 + 84);
            *(_QWORD *)v30 = 0;
            if ( !v38 )
              _libc_free(*(_QWORD *)(v30 + 64));
          }
          v39 = *(_QWORD *)(v30 + 32);
          if ( v39 )
          {
            v5 = *(_QWORD *)(v30 + 48) - v39;
            j_j___libc_free_0(v39);
          }
          v40 = *(_QWORD *)(v30 + 8);
          if ( v40 )
          {
            v5 = *(_QWORD *)(v30 + 24) - v40;
            j_j___libc_free_0(v40);
          }
          ++v89;
        }
        while ( v88 != v89 );
        v41 = *(_QWORD *)(v29 + 32);
        if ( v41 != *(_QWORD *)(v29 + 40) )
          *(_QWORD *)(v29 + 40) = v41;
      }
      v42 = *(__int64 **)(v29 + 120);
      v43 = &v42[2 * *(unsigned int *)(v29 + 128)];
      while ( v43 != v42 )
      {
        v44 = v42[1];
        v45 = *v42;
        v42 += 2;
        sub_C7D6A0(v45, v44, 16);
      }
      *(_DWORD *)(v29 + 128) = 0;
      v46 = *(unsigned int *)(v29 + 80);
      if ( (_DWORD)v46 )
      {
        *(_QWORD *)(v29 + 136) = 0;
        v69 = *(__int64 **)(v29 + 72);
        v70 = *v69;
        v71 = &v69[v46];
        v72 = v69 + 1;
        *(_QWORD *)(v29 + 56) = *v69;
        for ( *(_QWORD *)(v29 + 64) = v70 + 4096; v71 != v72; v69 = *(__int64 **)(v29 + 72) )
        {
          v73 = *v72;
          v74 = (unsigned int)(v72 - v69) >> 7;
          v75 = 4096LL << v74;
          if ( v74 >= 0x1E )
            v75 = 0x40000000000LL;
          ++v72;
          sub_C7D6A0(v73, v75, 16);
        }
        *(_DWORD *)(v29 + 80) = 1;
        sub_C7D6A0(*v69, 4096, 16);
        v76 = *(__int64 **)(v29 + 120);
        v47 = (unsigned __int64)&v76[2 * *(unsigned int *)(v29 + 128)];
        if ( v76 == (__int64 *)v47 )
          goto LABEL_80;
        do
        {
          v77 = v76[1];
          v78 = *v76;
          v76 += 2;
          sub_C7D6A0(v78, v77, 16);
        }
        while ( (__int64 *)v47 != v76 );
      }
      v47 = *(_QWORD *)(v29 + 120);
LABEL_80:
      if ( v47 != v29 + 136 )
        _libc_free(v47);
      v48 = *(_QWORD *)(v29 + 72);
      if ( v48 != v29 + 88 )
        _libc_free(v48);
      v49 = *(_QWORD *)(v29 + 32);
      if ( v49 )
        j_j___libc_free_0(v49);
      sub_C7D6A0(*(_QWORD *)(v29 + 8), 16LL * *(unsigned int *)(v29 + 24), 8);
      j_j___libc_free_0(v29);
    }
  }
  sub_26C3830(a1 + 1024);
  sub_26C3830(a1 + 1056);
  v50 = *(_DWORD *)(a1 + 1104);
  ++*(_QWORD *)(a1 + 1088);
  if ( !v50 && !*(_DWORD *)(a1 + 1108) )
    goto LABEL_100;
  v51 = 4 * v50;
  v52 = *(_QWORD **)(a1 + 1096);
  v53 = 56LL * *(unsigned int *)(a1 + 1112);
  if ( (unsigned int)(4 * v50) < 0x40 )
    v51 = 64;
  v54 = &v52[(unsigned __int64)v53 / 8];
  if ( *(_DWORD *)(a1 + 1112) <= v51 )
  {
    while ( v52 != v54 )
    {
      if ( *v52 != -4096 )
      {
        if ( *v52 != -8192 )
          sub_26BBA00(v52[3]);
        *v52 = -4096;
      }
      v52 += 7;
    }
    goto LABEL_99;
  }
  do
  {
    while ( *v52 == -8192 )
    {
LABEL_133:
      v52 += 7;
      if ( v52 == v54 )
        goto LABEL_148;
    }
    if ( *v52 != -4096 )
    {
      for ( m = v52[3]; m; v50 = v90 )
      {
        v90 = v50;
        sub_26BBA00(*(_QWORD *)(m + 24));
        v68 = m;
        m = *(_QWORD *)(m + 16);
        j_j___libc_free_0(v68);
      }
      goto LABEL_133;
    }
    v52 += 7;
  }
  while ( v52 != v54 );
LABEL_148:
  v79 = *(unsigned int *)(a1 + 1112);
  if ( !v50 )
  {
    if ( (_DWORD)v79 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1096), v53, 8);
      *(_DWORD *)(a1 + 1112) = 0;
      goto LABEL_162;
    }
LABEL_99:
    *(_QWORD *)(a1 + 1104) = 0;
    goto LABEL_100;
  }
  v80 = 64;
  if ( v50 != 1 )
  {
    _BitScanReverse(&v81, v50 - 1);
    v80 = 1 << (33 - (v81 ^ 0x1F));
    if ( v80 < 64 )
      v80 = 64;
  }
  v82 = *(_QWORD **)(a1 + 1096);
  if ( (_DWORD)v79 != v80 )
  {
    sub_C7D6A0((__int64)v82, v53, 8);
    v83 = sub_26BC060(v80);
    *(_DWORD *)(a1 + 1112) = v83;
    if ( v83 )
    {
      v84 = (_QWORD *)sub_C7D670(56LL * v83, 8);
      v85 = *(unsigned int *)(a1 + 1112);
      *(_QWORD *)(a1 + 1104) = 0;
      *(_QWORD *)(a1 + 1096) = v84;
      for ( n = &v84[7 * v85]; n != v84; v84 += 7 )
      {
        if ( v84 )
          *v84 = -4096;
      }
      goto LABEL_100;
    }
LABEL_162:
    *(_QWORD *)(a1 + 1096) = 0;
    *(_QWORD *)(a1 + 1104) = 0;
    goto LABEL_100;
  }
  *(_QWORD *)(a1 + 1104) = 0;
  v87 = &v82[7 * v79];
  do
  {
    if ( v82 )
      *v82 = -4096;
    v82 += 7;
  }
  while ( v87 != v82 );
LABEL_100:
  *(_QWORD *)(a1 + 1120) = 0;
}
