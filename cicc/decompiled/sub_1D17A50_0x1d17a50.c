// Function: sub_1D17A50
// Address: 0x1d17a50
//
void __fastcall sub_1D17A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // r13
  unsigned __int64 *v8; // r12
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // r14
  unsigned __int64 *v16; // r13
  unsigned __int64 v17; // rdi
  int v18; // eax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *i; // rdx
  _BYTE *v22; // rdx
  _BYTE *v23; // rdi
  _BYTE *v24; // rdx
  _BYTE *v25; // rdi
  __int64 v26; // r12
  int v27; // r14d
  _QWORD *v28; // rbx
  __int64 v29; // rdx
  _QWORD *v30; // r13
  unsigned int v31; // eax
  unsigned __int64 v32; // rdi
  __int64 v33; // r13
  unsigned __int64 *v34; // rbx
  unsigned __int64 *v35; // r13
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // ecx
  _QWORD *v39; // rdi
  unsigned int v40; // eax
  int v41; // eax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  int v44; // r13d
  __int64 v45; // r12
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v49; // rdi
  _QWORD *v50; // r12
  __int64 v51; // rdx
  unsigned __int64 *v52; // r13
  unsigned __int64 *v53; // r12
  unsigned __int64 v54; // rdi
  _QWORD *v55; // rbx
  __int64 v56; // rdx
  unsigned __int64 *v57; // r13
  unsigned __int64 *v58; // rbx
  unsigned __int64 v59; // rdi
  __int64 v60; // rdx
  int v61; // ebx
  unsigned int v62; // r14d
  unsigned int v63; // eax
  _QWORD *v64; // rdi
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rdx
  _QWORD *k; // rdx
  _QWORD *v70; // rax
  _QWORD *v71; // rax

  sub_1D173F0(a1, a2, a3, a4, a5, a6);
  v7 = *(unsigned int *)(a1 + 432);
  v8 = *(unsigned __int64 **)(a1 + 424);
  *(_DWORD *)(a1 + 472) = 0;
  v9 = &v8[2 * v7];
  while ( v9 != v8 )
  {
    v10 = *v8;
    v8 += 2;
    _libc_free(v10);
  }
  *(_DWORD *)(a1 + 432) = 0;
  v11 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v11 )
  {
    v50 = *(_QWORD **)(a1 + 376);
    *(_QWORD *)(a1 + 440) = 0;
    v51 = *v50;
    v52 = &v50[v11];
    v53 = v50 + 1;
    *(_QWORD *)(a1 + 360) = v51;
    *(_QWORD *)(a1 + 368) = v51 + 4096;
    while ( v52 != v53 )
    {
      v54 = *v53++;
      _libc_free(v54);
    }
    *(_DWORD *)(a1 + 384) = 1;
  }
  sub_16BD9E0(a1 + 320);
  sub_1D13CC0(*(_QWORD *)(a1 + 760));
  v12 = *(_DWORD *)(a1 + 804);
  *(_QWORD *)(a1 + 760) = 0;
  *(_QWORD *)(a1 + 768) = a1 + 752;
  *(_QWORD *)(a1 + 776) = a1 + 752;
  *(_QWORD *)(a1 + 784) = 0;
  if ( v12 )
  {
    v13 = 0;
    v14 = *(unsigned int *)(a1 + 800);
    v15 = 8 * v14;
    if ( (_DWORD)v14 )
    {
      do
      {
        v16 = (unsigned __int64 *)(v13 + *(_QWORD *)(a1 + 792));
        v17 = *v16;
        if ( *v16 != -8 && v17 )
          _libc_free(v17);
        v13 += 8;
        *v16 = 0;
      }
      while ( v15 != v13 );
    }
    *(_QWORD *)(a1 + 804) = 0;
  }
  sub_1D13F90(*(_QWORD **)(a1 + 840));
  ++*(_QWORD *)(a1 + 872);
  *(_QWORD *)(a1 + 848) = a1 + 832;
  *(_QWORD *)(a1 + 856) = a1 + 832;
  v18 = *(_DWORD *)(a1 + 888);
  *(_QWORD *)(a1 + 840) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  if ( !v18 )
  {
    if ( !*(_DWORD *)(a1 + 892) )
      goto LABEL_17;
    v19 = *(unsigned int *)(a1 + 896);
    if ( (unsigned int)v19 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 880));
      *(_QWORD *)(a1 + 880) = 0;
      *(_QWORD *)(a1 + 888) = 0;
      *(_DWORD *)(a1 + 896) = 0;
      goto LABEL_17;
    }
    goto LABEL_14;
  }
  v38 = 4 * v18;
  v19 = *(unsigned int *)(a1 + 896);
  if ( (unsigned int)(4 * v18) < 0x40 )
    v38 = 64;
  if ( (unsigned int)v19 <= v38 )
  {
LABEL_14:
    v20 = *(_QWORD **)(a1 + 880);
    for ( i = &v20[2 * v19]; i != v20; v20 += 2 )
      *v20 = -8;
    *(_QWORD *)(a1 + 888) = 0;
    goto LABEL_17;
  }
  v39 = *(_QWORD **)(a1 + 880);
  v40 = v18 - 1;
  if ( !v40 )
  {
    v45 = 2048;
    v44 = 128;
LABEL_47:
    j___libc_free_0(v39);
    *(_DWORD *)(a1 + 896) = v44;
    v46 = (_QWORD *)sub_22077B0(v45);
    v47 = *(unsigned int *)(a1 + 896);
    *(_QWORD *)(a1 + 888) = 0;
    *(_QWORD *)(a1 + 880) = v46;
    for ( j = &v46[2 * v47]; j != v46; v46 += 2 )
    {
      if ( v46 )
        *v46 = -8;
    }
    goto LABEL_17;
  }
  _BitScanReverse(&v40, v40);
  v41 = 1 << (33 - (v40 ^ 0x1F));
  if ( v41 < 64 )
    v41 = 64;
  if ( (_DWORD)v19 != v41 )
  {
    v42 = (((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
        | (4 * v41 / 3u + 1)
        | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)
        | (((((4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 2)
          | (4 * v41 / 3u + 1)
          | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1)) >> 4);
    v43 = (v42 >> 8) | v42;
    v44 = (v43 | (v43 >> 16)) + 1;
    v45 = 16 * ((v43 | (v43 >> 16)) + 1);
    goto LABEL_47;
  }
  *(_QWORD *)(a1 + 888) = 0;
  v70 = &v39[2 * (unsigned int)v19];
  do
  {
    if ( v39 )
      *v39 = -8;
    v39 += 2;
  }
  while ( v70 != v39 );
LABEL_17:
  v22 = *(_BYTE **)(a1 + 704);
  v23 = *(_BYTE **)(a1 + 696);
  if ( v23 != v22 )
    memset(v23, 0, v22 - v23);
  v24 = *(_BYTE **)(a1 + 728);
  v25 = *(_BYTE **)(a1 + 720);
  if ( v25 != v24 )
    memset(v25, 0, v24 - v25);
  *(_QWORD *)(a1 + 136) = 0;
  sub_1D172A0(a1, a1 + 88);
  *(_QWORD *)(a1 + 176) = a1 + 88;
  v26 = *(_QWORD *)(a1 + 648);
  *(_DWORD *)(a1 + 184) = 0;
  v27 = *(_DWORD *)(v26 + 712);
  ++*(_QWORD *)(v26 + 696);
  if ( v27 || *(_DWORD *)(v26 + 716) )
  {
    v28 = *(_QWORD **)(v26 + 704);
    v29 = *(unsigned int *)(v26 + 720);
    v30 = &v28[5 * v29];
    v31 = 4 * v27;
    if ( (unsigned int)(4 * v27) < 0x40 )
      v31 = 64;
    if ( (unsigned int)v29 <= v31 )
    {
      while ( v28 != v30 )
      {
        if ( *v28 != -8 )
        {
          if ( *v28 != -16 )
          {
            v32 = v28[1];
            if ( (_QWORD *)v32 != v28 + 3 )
              _libc_free(v32);
          }
          *v28 = -8;
        }
        v28 += 5;
      }
      goto LABEL_34;
    }
    do
    {
      if ( *v28 != -16 && *v28 != -8 )
      {
        v49 = v28[1];
        if ( (_QWORD *)v49 != v28 + 3 )
          _libc_free(v49);
      }
      v28 += 5;
    }
    while ( v28 != v30 );
    v60 = *(unsigned int *)(v26 + 720);
    if ( !v27 )
    {
      if ( (_DWORD)v60 )
      {
        j___libc_free_0(*(_QWORD *)(v26 + 704));
        *(_QWORD *)(v26 + 704) = 0;
        *(_QWORD *)(v26 + 712) = 0;
        *(_DWORD *)(v26 + 720) = 0;
        goto LABEL_35;
      }
LABEL_34:
      *(_QWORD *)(v26 + 712) = 0;
      goto LABEL_35;
    }
    v61 = 64;
    v62 = v27 - 1;
    if ( v62 )
    {
      _BitScanReverse(&v63, v62);
      v61 = 1 << (33 - (v63 ^ 0x1F));
      if ( v61 < 64 )
        v61 = 64;
    }
    v64 = *(_QWORD **)(v26 + 704);
    if ( (_DWORD)v60 == v61 )
    {
      *(_QWORD *)(v26 + 712) = 0;
      v71 = &v64[5 * v60];
      do
      {
        if ( v64 )
          *v64 = -8;
        v64 += 5;
      }
      while ( v71 != v64 );
    }
    else
    {
      j___libc_free_0(v64);
      v65 = ((((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
               | (4 * v61 / 3u + 1)
               | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
             | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
           | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 16;
      v66 = (v65
           | (((((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
               | (4 * v61 / 3u + 1)
               | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
             | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
             | (4 * v61 / 3u + 1)
             | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 4)
           | (((4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)) >> 2)
           | (4 * v61 / 3u + 1)
           | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v26 + 720) = v66;
      v67 = (_QWORD *)sub_22077B0(40 * v66);
      v68 = *(unsigned int *)(v26 + 720);
      *(_QWORD *)(v26 + 712) = 0;
      *(_QWORD *)(v26 + 704) = v67;
      for ( k = &v67[5 * v68]; k != v67; v67 += 5 )
      {
        if ( v67 )
          *v67 = -8;
      }
    }
  }
LABEL_35:
  v33 = *(unsigned int *)(v26 + 72);
  v34 = *(unsigned __int64 **)(v26 + 64);
  *(_DWORD *)(v26 + 112) = 0;
  *(_DWORD *)(v26 + 384) = 0;
  *(_DWORD *)(v26 + 656) = 0;
  v35 = &v34[2 * v33];
  while ( v35 != v34 )
  {
    v36 = *v34;
    v34 += 2;
    _libc_free(v36);
  }
  *(_DWORD *)(v26 + 72) = 0;
  v37 = *(unsigned int *)(v26 + 24);
  if ( (_DWORD)v37 )
  {
    *(_QWORD *)(v26 + 80) = 0;
    v55 = *(_QWORD **)(v26 + 16);
    v56 = *v55;
    v57 = &v55[v37];
    v58 = v55 + 1;
    *(_QWORD *)v26 = v56;
    *(_QWORD *)(v26 + 8) = v56 + 4096;
    while ( v57 != v58 )
    {
      v59 = *v58++;
      _libc_free(v59);
    }
    *(_DWORD *)(v26 + 24) = 1;
  }
}
