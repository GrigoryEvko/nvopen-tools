// Function: sub_2FDA680
// Address: 0x2fda680
//
__int64 __fastcall sub_2FDA680(
        __int64 a1,
        char a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r15
  __int64 *v9; // r14
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 *v12; // rsi
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  int v18; // ebx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r14
  unsigned int v21; // edx
  __int64 v22; // rcx
  unsigned int v23; // esi
  unsigned int *v24; // rax
  __int64 *v25; // r13
  __int64 *i; // r15
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 j; // rbx
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  int v36; // r13d
  unsigned int v37; // eax
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // r14
  __int64 v41; // r13
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rbx
  _BYTE *v44; // r13
  __int64 v45; // r12
  __int64 v46; // r14
  __int64 v47; // rax
  unsigned int v48; // r15d
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rax
  int v52; // ebx
  _BYTE *v53; // rcx
  size_t v54; // r13
  _BYTE *v55; // rsi
  __int64 v56; // rdx
  __int64 *v57; // rbx
  __int64 *v58; // r13
  __int64 v59; // r14
  int v60; // eax
  int v61; // eax
  __int64 v62; // rax
  __int64 v63; // r12
  unsigned __int64 v64; // rdi
  int v65; // edx
  int v66; // ebx
  unsigned int v67; // eax
  _DWORD *v68; // rdi
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rax
  _DWORD *v71; // rax
  __int64 v72; // rdx
  _DWORD *k; // rdx
  _DWORD *v74; // rax
  unsigned __int8 v75; // [rsp+2Fh] [rbp-271h]
  int *v76; // [rsp+30h] [rbp-270h]
  int *v82; // [rsp+58h] [rbp-248h]
  int v83; // [rsp+58h] [rbp-248h]
  _BYTE v84[48]; // [rsp+60h] [rbp-240h] BYREF
  __int64 *v85; // [rsp+90h] [rbp-210h] BYREF
  __int64 v86; // [rsp+98h] [rbp-208h]
  _BYTE v87[48]; // [rsp+A0h] [rbp-200h] BYREF
  _BYTE *v88; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v89; // [rsp+D8h] [rbp-1C8h]
  _BYTE v90[64]; // [rsp+E0h] [rbp-1C0h] BYREF
  unsigned __int64 v91[2]; // [rsp+120h] [rbp-180h] BYREF
  _BYTE v92[64]; // [rsp+130h] [rbp-170h] BYREF
  __int64 v93; // [rsp+170h] [rbp-130h] BYREF
  __int64 v94; // [rsp+178h] [rbp-128h]
  __int64 v95; // [rsp+180h] [rbp-120h]
  __int64 v96; // [rsp+188h] [rbp-118h]
  _BYTE *v97; // [rsp+190h] [rbp-110h]
  __int64 v98; // [rsp+198h] [rbp-108h]
  _BYTE v99[64]; // [rsp+1A0h] [rbp-100h] BYREF
  _BYTE *v100; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v101; // [rsp+1E8h] [rbp-B8h]
  _BYTE v102[176]; // [rsp+1F0h] [rbp-B0h] BYREF

  v7 = a1;
  v9 = *(__int64 **)(a3 + 112);
  v10 = *(unsigned int *)(a3 + 120);
  v97 = v99;
  v11 = &v9[v10];
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v98 = 0x800000000LL;
  while ( v11 != v9 )
  {
    v12 = v9++;
    sub_2FD92E0((__int64)&v93, v12);
  }
  v88 = v90;
  v100 = v102;
  v89 = 0x800000000LL;
  v101 = 0x1000000000LL;
  v75 = sub_2FD9800((__int64 *)a1, a2, (__int64 *)a3, a4, (__int64)&v88, (__int64)&v100, a7);
  if ( !v75 )
    goto LABEL_4;
  v14 = *(_QWORD *)(a1 + 32);
  v91[1] = 0x800000000LL;
  v91[0] = (unsigned __int64)v92;
  sub_356E200(v84, v14, v91);
  if ( *(_DWORD *)(a3 + 72) || *(_BYTE *)(a3 + 217) || *(_QWORD *)(a3 + 224) )
  {
    if ( *(_BYTE *)(a1 + 56) )
      sub_2FD5E60(a1, a3, 0, (__int64)&v88, (__int64)&v93);
  }
  else
  {
    if ( *(_BYTE *)(a1 + 56) )
      sub_2FD5E60(a1, a3, 1, (__int64)&v88, (__int64)&v93);
    sub_2FD76C0(a1, a3, a6);
  }
  v17 = *(unsigned int *)(a1 + 72);
  if ( !(_DWORD)v17 )
    goto LABEL_53;
  v82 = *(int **)(a1 + 64);
  v76 = &v82[v17];
  do
  {
    v18 = *v82;
    sub_356E2B0(v84, (unsigned int)*v82);
    v19 = sub_2EBEE10(*(_QWORD *)(a1 + 24), v18);
    v20 = v19;
    if ( v19 )
    {
      v20 = *(_QWORD *)(v19 + 24);
      sub_356E840(v84, v20, (unsigned int)v18);
    }
    v21 = *(_DWORD *)(a1 + 168);
    v22 = *(_QWORD *)(a1 + 152);
    if ( v21 )
    {
      v23 = (v21 - 1) & (37 * v18);
      v24 = (unsigned int *)(v22 + 32LL * v23);
      v15 = *v24;
      if ( (_DWORD)v15 == v18 )
        goto LABEL_21;
      v61 = 1;
      while ( (_DWORD)v15 != -1 )
      {
        v16 = (unsigned int)(v61 + 1);
        v23 = (v21 - 1) & (v61 + v23);
        v24 = (unsigned int *)(v22 + 32LL * v23);
        v15 = *v24;
        if ( (_DWORD)v15 == v18 )
          goto LABEL_21;
        v61 = v16;
      }
    }
    v24 = (unsigned int *)(v22 + 32LL * v21);
LABEL_21:
    v25 = (__int64 *)*((_QWORD *)v24 + 2);
    for ( i = (__int64 *)*((_QWORD *)v24 + 1); v25 != i; i += 2 )
    {
      v27 = *i;
      v28 = *((unsigned int *)i + 2);
      sub_356E840(v84, v27, v28);
    }
    v85 = (__int64 *)v87;
    v86 = 0x600000000LL;
    v29 = *(_QWORD *)(a1 + 24);
    if ( v18 < 0 )
      v30 = *(_QWORD *)(*(_QWORD *)(v29 + 56) + 16LL * (v18 & 0x7FFFFFFF) + 8);
    else
      v30 = *(_QWORD *)(*(_QWORD *)(v29 + 304) + 8LL * (unsigned int)v18);
    if ( v30 )
    {
      if ( (*(_BYTE *)(v30 + 3) & 0x10) == 0 )
        goto LABEL_27;
      do
      {
        v30 = *(_QWORD *)(v30 + 32);
        if ( !v30 )
          goto LABEL_39;
      }
      while ( (*(_BYTE *)(v30 + 3) & 0x10) != 0 );
LABEL_27:
      while ( 2 )
      {
        for ( j = *(_QWORD *)(v30 + 32); j; j = *(_QWORD *)(j + 32) )
        {
          if ( (*(_BYTE *)(j + 3) & 0x10) == 0 )
            break;
        }
        v32 = *(_QWORD *)(v30 + 16);
        v33 = *(unsigned __int16 *)(v32 + 68);
        if ( (unsigned __int16)(v33 - 14) <= 1u )
        {
          v34 = (unsigned int)v86;
          v35 = (unsigned int)v86 + 1LL;
          if ( v35 > HIDWORD(v86) )
          {
            sub_C8D5F0((__int64)&v85, v87, v35, 8u, v15, v16);
            v34 = (unsigned int)v86;
          }
          v85[v34] = v30;
          LODWORD(v86) = v86 + 1;
          goto LABEL_35;
        }
        if ( v20 == *(_QWORD *)(v32 + 24) && *(_WORD *)(v32 + 68) && v33 != 68 )
        {
LABEL_35:
          if ( !j )
            goto LABEL_73;
        }
        else
        {
          sub_3572410(v84, v30);
          if ( !j )
          {
LABEL_73:
            v57 = v85;
            v58 = &v85[(unsigned int)v86];
            if ( v85 != v58 )
            {
              do
              {
                v59 = *v57++;
                v60 = sub_3571960(v84, *(_QWORD *)(*(_QWORD *)(v59 + 16) + 24LL), 1);
                sub_2EAB0C0(v59, v60);
              }
              while ( v58 != v57 );
              v58 = v85;
            }
            if ( v58 != (__int64 *)v87 )
              _libc_free((unsigned __int64)v58);
            break;
          }
        }
        v30 = j;
        continue;
      }
    }
LABEL_39:
    ++v82;
  }
  while ( v76 != v82 );
  v7 = a1;
  v36 = *(_DWORD *)(a1 + 160);
  ++*(_QWORD *)(a1 + 144);
  *(_DWORD *)(a1 + 72) = 0;
  if ( v36 || *(_DWORD *)(a1 + 164) )
  {
    v37 = 4 * v36;
    v38 = *(_QWORD *)(a1 + 152);
    v39 = *(unsigned int *)(a1 + 168);
    v40 = 32 * v39;
    if ( (unsigned int)(4 * v36) < 0x40 )
      v37 = 64;
    v15 = v38 + v40;
    if ( v37 < (unsigned int)v39 )
    {
      v63 = v38 + v40;
      do
      {
        if ( *(_DWORD *)v38 <= 0xFFFFFFFD )
        {
          v64 = *(_QWORD *)(v38 + 8);
          if ( v64 )
            j_j___libc_free_0(v64);
        }
        v38 += 32;
      }
      while ( v38 != v63 );
      v65 = *(_DWORD *)(a1 + 168);
      if ( v36 )
      {
        v66 = 64;
        if ( v36 != 1 )
        {
          _BitScanReverse(&v67, v36 - 1);
          v66 = 1 << (33 - (v67 ^ 0x1F));
          if ( v66 < 64 )
            v66 = 64;
        }
        v68 = *(_DWORD **)(a1 + 152);
        if ( v65 == v66 )
        {
          *(_QWORD *)(a1 + 160) = 0;
          v74 = &v68[8 * v65];
          do
          {
            if ( v68 )
              *v68 = -1;
            v68 += 8;
          }
          while ( v74 != v68 );
        }
        else
        {
          sub_C7D6A0((__int64)v68, v40, 8);
          v69 = ((((((((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                   | (4 * v66 / 3u + 1)
                   | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                 | (4 * v66 / 3u + 1)
                 | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                 | (4 * v66 / 3u + 1)
                 | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 4)
               | (((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
               | (4 * v66 / 3u + 1)
               | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 16;
          v70 = (v69
               | (((((((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                   | (4 * v66 / 3u + 1)
                   | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                 | (4 * v66 / 3u + 1)
                 | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
                 | (4 * v66 / 3u + 1)
                 | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 4)
               | (((4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1)) >> 2)
               | (4 * v66 / 3u + 1)
               | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 168) = v70;
          v71 = (_DWORD *)sub_C7D670(32 * v70, 8);
          v72 = *(unsigned int *)(a1 + 168);
          *(_QWORD *)(a1 + 160) = 0;
          *(_QWORD *)(a1 + 152) = v71;
          for ( k = &v71[8 * v72]; k != v71; v71 += 8 )
          {
            if ( v71 )
              *v71 = -1;
          }
        }
      }
      else
      {
        if ( !v65 )
          goto LABEL_52;
        sub_C7D6A0(*(_QWORD *)(a1 + 152), v40, 8);
        *(_QWORD *)(a1 + 152) = 0;
        *(_QWORD *)(a1 + 160) = 0;
        *(_DWORD *)(a1 + 168) = 0;
      }
    }
    else
    {
      v41 = v38 + v40;
      if ( v38 != v15 )
      {
        do
        {
          if ( *(_DWORD *)v38 != -1 )
          {
            if ( *(_DWORD *)v38 != -2 )
            {
              v42 = *(_QWORD *)(v38 + 8);
              if ( v42 )
                j_j___libc_free_0(v42);
            }
            *(_DWORD *)v38 = -1;
          }
          v38 += 32;
        }
        while ( v38 != v41 );
      }
LABEL_52:
      *(_QWORD *)(a1 + 160) = 0;
    }
  }
LABEL_53:
  v43 = (unsigned __int64)v100;
  v44 = &v100[8 * (unsigned int)v101];
  if ( v44 != v100 )
  {
    v45 = v7;
    do
    {
      while ( 1 )
      {
        v46 = *(_QWORD *)v43;
        if ( *(_WORD *)(*(_QWORD *)v43 + 68LL) == 20 )
        {
          v47 = *(_QWORD *)(v46 + 32);
          v48 = *(_DWORD *)(v47 + 48);
          v83 = *(_DWORD *)(v47 + 8);
          if ( (unsigned __int8)sub_2EBEF70(*(_QWORD *)(v45 + 24), v48) )
          {
            if ( sub_2EBE590(
                   *(_QWORD *)(v45 + 24),
                   v48,
                   *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v45 + 24) + 56LL) + 16LL * (v83 & 0x7FFFFFFF))
                 & 0xFFFFFFFFFFFFFFF8LL,
                   0) )
            {
              break;
            }
          }
        }
        v43 += 8LL;
        if ( v44 == (_BYTE *)v43 )
          goto LABEL_60;
      }
      v43 += 8LL;
      sub_2EBECB0(*(_QWORD **)(v45 + 24), v83, v48);
      sub_2E88E20(v46);
    }
    while ( v44 != (_BYTE *)v43 );
  }
LABEL_60:
  if ( a5 )
  {
    v49 = (unsigned __int64)v88;
    if ( v88 == v90 )
    {
      v50 = (unsigned int)v89;
      v51 = *(unsigned int *)(a5 + 8);
      v52 = v89;
      if ( (unsigned int)v89 <= v51 )
      {
        if ( (_DWORD)v89 )
          memmove(*(void **)a5, v90, 8LL * (unsigned int)v89);
      }
      else
      {
        if ( (unsigned int)v89 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          *(_DWORD *)(a5 + 8) = 0;
          sub_C8D5F0(a5, (const void *)(a5 + 16), v50, 8u, v15, v16);
          v53 = v88;
          v50 = (unsigned int)v89;
          v51 = 0;
          v55 = v88;
        }
        else
        {
          v53 = v90;
          v54 = 8 * v51;
          v55 = v90;
          if ( *(_DWORD *)(a5 + 8) )
          {
            memmove(*(void **)a5, v90, v54);
            v53 = v88;
            v50 = (unsigned int)v89;
            v51 = v54;
            v55 = &v88[v54];
          }
        }
        v56 = 8 * v50;
        if ( v55 != &v53[v56] )
          memcpy((void *)(v51 + *(_QWORD *)a5), v55, v56 - v51);
      }
      LODWORD(v89) = 0;
      *(_DWORD *)(a5 + 8) = v52;
    }
    else
    {
      if ( *(_QWORD *)a5 != a5 + 16 )
      {
        _libc_free(*(_QWORD *)a5);
        v49 = (unsigned __int64)v88;
      }
      *(_QWORD *)a5 = v49;
      v62 = v89;
      v89 = 0;
      *(_QWORD *)(a5 + 8) = v62;
      v88 = v90;
    }
  }
  sub_356E260(v84);
  if ( (_BYTE *)v91[0] != v92 )
    _libc_free(v91[0]);
LABEL_4:
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  if ( v88 != v90 )
    _libc_free((unsigned __int64)v88);
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  sub_C7D6A0(v94, 8LL * (unsigned int)v96, 8);
  return v75;
}
