// Function: sub_33CCE30
// Address: 0x33cce30
//
void __fastcall sub_33CCE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 *v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  int v13; // esi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r14
  _QWORD **v17; // r13
  _QWORD *v18; // rdi
  int v19; // eax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *i; // rdx
  int v23; // r15d
  _QWORD *v24; // r12
  unsigned int v25; // eax
  __int64 v26; // r14
  _QWORD *v27; // r13
  unsigned __int64 v28; // rdi
  _BYTE *v29; // rdx
  _BYTE *v30; // rdi
  _BYTE *v31; // rdx
  _BYTE *v32; // rdi
  __int64 v33; // r12
  int v34; // r15d
  _QWORD *v35; // rbx
  unsigned int v36; // eax
  __int64 v37; // r14
  _QWORD *v38; // r13
  unsigned __int64 v39; // rdi
  __int64 v40; // r13
  __int64 *v41; // rbx
  __int64 *v42; // r13
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdx
  unsigned int v46; // ecx
  unsigned int v47; // eax
  _QWORD *v48; // rdi
  int v49; // r12d
  _QWORD *v50; // rax
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  __int64 *v53; // rax
  __int64 v54; // rcx
  __int64 *v55; // rbx
  __int64 *v56; // r14
  __int64 v57; // rdi
  unsigned int v58; // ecx
  __int64 v59; // rsi
  __int64 *v60; // rax
  __int64 v61; // rcx
  __int64 *v62; // r12
  __int64 *v63; // r14
  __int64 v64; // rdi
  unsigned int v65; // ecx
  __int64 v66; // rsi
  int v67; // edx
  __int64 v68; // r12
  unsigned int v69; // r15d
  unsigned int v70; // eax
  _QWORD *v71; // rdi
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rdx
  _QWORD *k; // rdx
  int v77; // edx
  __int64 v78; // rbx
  unsigned int v79; // r15d
  unsigned int v80; // eax
  _QWORD *v81; // rdi
  unsigned __int64 v82; // rdx
  unsigned __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rdx
  _QWORD *m; // rdx
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rdi
  _QWORD *v89; // rax
  __int64 v90; // rdx
  _QWORD *j; // rdx
  _QWORD *v92; // rax
  _QWORD *v93; // rax

  sub_33CC630(a1, a2, a3, a4, a5, a6);
  v7 = *(unsigned int *)(a1 + 616);
  v8 = *(__int64 **)(a1 + 608);
  *(_DWORD *)(a1 + 648) = 0;
  v9 = &v8[2 * v7];
  while ( v9 != v8 )
  {
    v10 = v8[1];
    v11 = *v8;
    v8 += 2;
    sub_C7D6A0(v11, v10, 16);
  }
  *(_DWORD *)(a1 + 616) = 0;
  v12 = *(unsigned int *)(a1 + 568);
  if ( (_DWORD)v12 )
  {
    v60 = *(__int64 **)(a1 + 560);
    *(_QWORD *)(a1 + 624) = 0;
    v61 = *v60;
    v62 = &v60[v12];
    v63 = v60 + 1;
    *(_QWORD *)(a1 + 544) = *v60;
    *(_QWORD *)(a1 + 552) = v61 + 4096;
    if ( v62 != v60 + 1 )
    {
      while ( 1 )
      {
        v64 = *v63;
        v65 = (unsigned int)(v63 - v60) >> 7;
        v66 = 4096LL << v65;
        if ( v65 >= 0x1E )
          v66 = 0x40000000000LL;
        ++v63;
        sub_C7D6A0(v64, v66, 16);
        if ( v62 == v63 )
          break;
        v60 = *(__int64 **)(a1 + 560);
      }
    }
    *(_DWORD *)(a1 + 568) = 1;
  }
  sub_C65780(a1 + 520);
  sub_33C8980(*(_QWORD *)(a1 + 888));
  v13 = *(_DWORD *)(a1 + 932);
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = a1 + 880;
  *(_QWORD *)(a1 + 904) = a1 + 880;
  *(_QWORD *)(a1 + 912) = 0;
  if ( v13 )
  {
    v14 = 0;
    v15 = *(unsigned int *)(a1 + 928);
    v16 = 8 * v15;
    if ( (_DWORD)v15 )
    {
      do
      {
        v17 = (_QWORD **)(v14 + *(_QWORD *)(a1 + 920));
        v18 = *v17;
        if ( *v17 != (_QWORD *)-8LL && v18 )
          sub_C7D6A0((__int64)v18, *v18 + 17LL, 8);
        v14 += 8;
        *v17 = 0;
      }
      while ( v14 != v16 );
    }
    *(_QWORD *)(a1 + 932) = 0;
  }
  sub_33C9130(*(_QWORD **)(a1 + 960));
  ++*(_QWORD *)(a1 + 992);
  *(_QWORD *)(a1 + 968) = a1 + 952;
  *(_QWORD *)(a1 + 976) = a1 + 952;
  v19 = *(_DWORD *)(a1 + 1008);
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  if ( !v19 )
  {
    if ( !*(_DWORD *)(a1 + 1012) )
      goto LABEL_17;
    v20 = *(unsigned int *)(a1 + 1016);
    if ( (unsigned int)v20 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 1000), 16LL * (unsigned int)v20, 8);
      *(_QWORD *)(a1 + 1000) = 0;
      *(_QWORD *)(a1 + 1008) = 0;
      *(_DWORD *)(a1 + 1016) = 0;
      goto LABEL_17;
    }
    goto LABEL_14;
  }
  v46 = 4 * v19;
  v20 = *(unsigned int *)(a1 + 1016);
  if ( (unsigned int)(4 * v19) < 0x40 )
    v46 = 64;
  if ( (unsigned int)v20 <= v46 )
  {
LABEL_14:
    v21 = *(_QWORD **)(a1 + 1000);
    for ( i = &v21[2 * v20]; i != v21; v21 += 2 )
      *v21 = -4096;
    *(_QWORD *)(a1 + 1008) = 0;
    goto LABEL_17;
  }
  v47 = v19 - 1;
  if ( v47 )
  {
    _BitScanReverse(&v47, v47);
    v48 = *(_QWORD **)(a1 + 1000);
    v49 = 1 << (33 - (v47 ^ 0x1F));
    if ( v49 < 64 )
      v49 = 64;
    if ( (_DWORD)v20 == v49 )
    {
      *(_QWORD *)(a1 + 1008) = 0;
      v50 = &v48[2 * (unsigned int)v20];
      do
      {
        if ( v48 )
          *v48 = -4096;
        v48 += 2;
      }
      while ( v50 != v48 );
      goto LABEL_17;
    }
  }
  else
  {
    v48 = *(_QWORD **)(a1 + 1000);
    v49 = 64;
  }
  sub_C7D6A0((__int64)v48, 16LL * (unsigned int)v20, 8);
  v87 = ((((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
       | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
       | (4 * v49 / 3u + 1)
       | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 16;
  v88 = (v87
       | (((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
       | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
       | (4 * v49 / 3u + 1)
       | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 1016) = v88;
  v89 = (_QWORD *)sub_C7D670(16 * v88, 8);
  v90 = *(unsigned int *)(a1 + 1016);
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1000) = v89;
  for ( j = &v89[2 * v90]; j != v89; v89 += 2 )
  {
    if ( v89 )
      *v89 = -4096;
  }
LABEL_17:
  v23 = *(_DWORD *)(a1 + 744);
  ++*(_QWORD *)(a1 + 728);
  if ( v23 || *(_DWORD *)(a1 + 748) )
  {
    v24 = *(_QWORD **)(a1 + 736);
    v25 = 4 * v23;
    v26 = 80LL * *(unsigned int *)(a1 + 752);
    if ( (unsigned int)(4 * v23) < 0x40 )
      v25 = 64;
    v27 = &v24[(unsigned __int64)v26 / 8];
    if ( *(_DWORD *)(a1 + 752) <= v25 )
    {
      for ( ; v24 != v27; v24 += 10 )
      {
        if ( *v24 != -4096 )
        {
          if ( *v24 != -8192 )
          {
            v28 = v24[1];
            if ( (_QWORD *)v28 != v24 + 3 )
              _libc_free(v28);
          }
          *v24 = -4096;
        }
      }
      goto LABEL_29;
    }
    do
    {
      if ( *v24 != -8192 && *v24 != -4096 )
      {
        v51 = v24[1];
        if ( (_QWORD *)v51 != v24 + 3 )
          _libc_free(v51);
      }
      v24 += 10;
    }
    while ( v27 != v24 );
    v67 = *(_DWORD *)(a1 + 752);
    if ( v23 )
    {
      v68 = 64;
      v69 = v23 - 1;
      if ( v69 )
      {
        _BitScanReverse(&v70, v69);
        v68 = (unsigned int)(1 << (33 - (v70 ^ 0x1F)));
        if ( (int)v68 < 64 )
          v68 = 64;
      }
      v71 = *(_QWORD **)(a1 + 736);
      if ( (_DWORD)v68 == v67 )
      {
        *(_QWORD *)(a1 + 744) = 0;
        v92 = &v71[10 * v68];
        do
        {
          if ( v71 )
            *v71 = -4096;
          v71 += 10;
        }
        while ( v92 != v71 );
      }
      else
      {
        sub_C7D6A0((__int64)v71, v26, 8);
        v72 = ((((((((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v68 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v68 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v68 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v68 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 16;
        v73 = (v72
             | (((((((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v68 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v68 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v68 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v68 / 3u + 1) | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v68 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v68 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 752) = v73;
        v74 = (_QWORD *)sub_C7D670(80 * v73, 8);
        v75 = *(unsigned int *)(a1 + 752);
        *(_QWORD *)(a1 + 744) = 0;
        *(_QWORD *)(a1 + 736) = v74;
        for ( k = &v74[10 * v75]; k != v74; v74 += 10 )
        {
          if ( v74 )
            *v74 = -4096;
        }
      }
    }
    else
    {
      if ( !v67 )
      {
LABEL_29:
        *(_QWORD *)(a1 + 744) = 0;
        goto LABEL_30;
      }
      sub_C7D6A0(*(_QWORD *)(a1 + 736), v26, 8);
      *(_QWORD *)(a1 + 736) = 0;
      *(_QWORD *)(a1 + 744) = 0;
      *(_DWORD *)(a1 + 752) = 0;
    }
  }
LABEL_30:
  v29 = *(_BYTE **)(a1 + 832);
  v30 = *(_BYTE **)(a1 + 824);
  if ( v30 != v29 )
    memset(v30, 0, v29 - v30);
  v31 = *(_BYTE **)(a1 + 856);
  v32 = *(_BYTE **)(a1 + 848);
  if ( v32 != v31 )
    memset(v32, 0, v31 - v32);
  *(_QWORD *)(a1 + 344) = 0;
  sub_33CC420(a1, a1 + 288);
  *(_QWORD *)(a1 + 384) = a1 + 288;
  v33 = *(_QWORD *)(a1 + 720);
  *(_DWORD *)(a1 + 392) = 0;
  v34 = *(_DWORD *)(v33 + 704);
  ++*(_QWORD *)(v33 + 688);
  if ( v34 || *(_DWORD *)(v33 + 708) )
  {
    v35 = *(_QWORD **)(v33 + 696);
    v36 = 4 * v34;
    v37 = 40LL * *(unsigned int *)(v33 + 712);
    if ( (unsigned int)(4 * v34) < 0x40 )
      v36 = 64;
    v38 = &v35[(unsigned __int64)v37 / 8];
    if ( *(_DWORD *)(v33 + 712) <= v36 )
    {
      while ( v35 != v38 )
      {
        if ( *v35 != -4096 )
        {
          if ( *v35 != -8192 )
          {
            v39 = v35[1];
            if ( (_QWORD *)v39 != v35 + 3 )
              _libc_free(v39);
          }
          *v35 = -4096;
        }
        v35 += 5;
      }
    }
    else
    {
      do
      {
        if ( *v35 != -8192 && *v35 != -4096 )
        {
          v52 = v35[1];
          if ( (_QWORD *)v52 != v35 + 3 )
            _libc_free(v52);
        }
        v35 += 5;
      }
      while ( v35 != v38 );
      v77 = *(_DWORD *)(v33 + 712);
      if ( v34 )
      {
        v78 = 64;
        v79 = v34 - 1;
        if ( v79 )
        {
          _BitScanReverse(&v80, v79);
          v78 = (unsigned int)(1 << (33 - (v80 ^ 0x1F)));
          if ( (int)v78 < 64 )
            v78 = 64;
        }
        v81 = *(_QWORD **)(v33 + 696);
        if ( (_DWORD)v78 == v77 )
        {
          *(_QWORD *)(v33 + 704) = 0;
          v93 = &v81[5 * v78];
          do
          {
            if ( v81 )
              *v81 = -4096;
            v81 += 5;
          }
          while ( v93 != v81 );
        }
        else
        {
          sub_C7D6A0((__int64)v81, v37, 8);
          v82 = ((((((((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v78 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v78 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v78 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v78 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 16;
          v83 = (v82
               | (((((((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v78 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v78 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v78 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v78 / 3u + 1) | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v78 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v78 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(v33 + 712) = v83;
          v84 = (_QWORD *)sub_C7D670(40 * v83, 8);
          v85 = *(unsigned int *)(v33 + 712);
          *(_QWORD *)(v33 + 704) = 0;
          *(_QWORD *)(v33 + 696) = v84;
          for ( m = &v84[5 * v85]; m != v84; v84 += 5 )
          {
            if ( v84 )
              *v84 = -4096;
          }
        }
        goto LABEL_48;
      }
      if ( v77 )
      {
        sub_C7D6A0(*(_QWORD *)(v33 + 696), v37, 8);
        *(_QWORD *)(v33 + 696) = 0;
        *(_QWORD *)(v33 + 704) = 0;
        *(_DWORD *)(v33 + 712) = 0;
        goto LABEL_48;
      }
    }
    *(_QWORD *)(v33 + 704) = 0;
  }
LABEL_48:
  v40 = *(unsigned int *)(v33 + 72);
  v41 = *(__int64 **)(v33 + 64);
  *(_DWORD *)(v33 + 104) = 0;
  *(_DWORD *)(v33 + 376) = 0;
  *(_DWORD *)(v33 + 648) = 0;
  v42 = &v41[2 * v40];
  while ( v42 != v41 )
  {
    v43 = v41[1];
    v44 = *v41;
    v41 += 2;
    sub_C7D6A0(v44, v43, 16);
  }
  *(_DWORD *)(v33 + 72) = 0;
  v45 = *(unsigned int *)(v33 + 24);
  if ( (_DWORD)v45 )
  {
    *(_QWORD *)(v33 + 80) = 0;
    v53 = *(__int64 **)(v33 + 16);
    v54 = *v53;
    v55 = &v53[v45];
    v56 = v53 + 1;
    *(_QWORD *)v33 = *v53;
    *(_QWORD *)(v33 + 8) = v54 + 4096;
    if ( v55 != v53 + 1 )
    {
      while ( 1 )
      {
        v57 = *v56;
        v58 = (unsigned int)(v56 - v53) >> 7;
        v59 = 4096LL << v58;
        if ( v58 >= 0x1E )
          v59 = 0x40000000000LL;
        ++v56;
        sub_C7D6A0(v57, v59, 16);
        if ( v55 == v56 )
          break;
        v53 = *(__int64 **)(v33 + 16);
      }
    }
    *(_DWORD *)(v33 + 24) = 1;
  }
}
