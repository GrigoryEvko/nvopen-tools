// Function: sub_2FCFFF0
// Address: 0x2fcfff0
//
void __fastcall sub_2FCFFF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // r15
  unsigned __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  _QWORD *v11; // r14
  _QWORD *v12; // rax
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 *i; // rax
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rsi
  __int64 *v20; // rbx
  unsigned __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // rbx
  unsigned __int64 *v26; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r13
  unsigned __int64 v34; // rdi

  v7 = *(_QWORD *)(a1 + 1928);
  v8 = v7 + 232LL * *(unsigned int *)(a1 + 1936);
  if ( v7 != v8 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v8 - 224);
        v8 -= 232LL;
        if ( v9 )
        {
          if ( *(_DWORD *)(v9 + 200) )
            break;
        }
        if ( v7 == v8 )
          goto LABEL_9;
      }
      v10 = v9 + 8;
      v11 = (_QWORD *)(v9 + 136);
      sub_2E19AD0(v10, (char *)sub_2E199D0, 0, a4, a5, a6);
      v12 = (_QWORD *)v10;
      do
      {
        *v12 = 0;
        v12 += 2;
        *(v12 - 1) = 0;
      }
      while ( v11 != v12 );
    }
    while ( v7 != v8 );
LABEL_9:
    v8 = *(_QWORD *)(a1 + 1928);
  }
  if ( v8 != a1 + 1944 )
    _libc_free(v8);
  v13 = *(__int64 **)(a1 + 1848);
  v14 = *(unsigned int *)(a1 + 1856);
  *(_QWORD *)(a1 + 1824) = 0;
  v15 = &v13[v14];
  if ( v13 != v15 )
  {
    for ( i = v13; ; i = *(__int64 **)(a1 + 1848) )
    {
      v17 = *v13;
      v18 = (unsigned int)(v13 - i) >> 7;
      v19 = 4096LL << v18;
      if ( v18 >= 0x1E )
        v19 = 0x40000000000LL;
      ++v13;
      sub_C7D6A0(v17, v19, 16);
      if ( v15 == v13 )
        break;
    }
  }
  v20 = *(__int64 **)(a1 + 1896);
  v21 = (unsigned __int64)&v20[2 * *(unsigned int *)(a1 + 1904)];
  if ( v20 != (__int64 *)v21 )
  {
    do
    {
      v22 = v20[1];
      v23 = *v20;
      v20 += 2;
      sub_C7D6A0(v23, v22, 16);
    }
    while ( (__int64 *)v21 != v20 );
    v21 = *(_QWORD *)(a1 + 1896);
  }
  if ( v21 != a1 + 1912 )
    _libc_free(v21);
  v24 = *(_QWORD *)(a1 + 1848);
  if ( v24 != a1 + 1864 )
    _libc_free(v24);
  v25 = *(unsigned __int64 **)(a1 + 1664);
  v26 = &v25[9 * *(unsigned int *)(a1 + 1672)];
  if ( v25 != v26 )
  {
    do
    {
      v26 -= 9;
      if ( (unsigned __int64 *)*v26 != v26 + 2 )
        _libc_free(*v26);
    }
    while ( v25 != v26 );
    v26 = *(unsigned __int64 **)(a1 + 1664);
  }
  if ( v26 != (unsigned __int64 *)(a1 + 1680) )
    _libc_free((unsigned __int64)v26);
  v27 = *(_QWORD *)(a1 + 1640);
  if ( v27 != a1 + 1656 )
    _libc_free(v27);
  v28 = *(unsigned __int64 **)(a1 + 1480);
  v29 = &v28[9 * *(unsigned int *)(a1 + 1488)];
  if ( v28 != v29 )
  {
    do
    {
      v29 -= 9;
      if ( (unsigned __int64 *)*v29 != v29 + 2 )
        _libc_free(*v29);
    }
    while ( v28 != v29 );
    v29 = *(unsigned __int64 **)(a1 + 1480);
  }
  if ( v29 != (unsigned __int64 *)(a1 + 1496) )
    _libc_free((unsigned __int64)v29);
  v30 = *(_QWORD *)(a1 + 1400);
  if ( v30 != a1 + 1416 )
    _libc_free(v30);
  v31 = *(_QWORD *)(a1 + 1360);
  if ( v31 != a1 + 1384 )
    _libc_free(v31);
  v32 = *(unsigned __int64 **)(a1 + 64);
  v33 = &v32[10 * *(unsigned int *)(a1 + 72)];
  if ( v32 != v33 )
  {
    do
    {
      v33 -= 10;
      if ( (unsigned __int64 *)*v33 != v33 + 2 )
        _libc_free(*v33);
    }
    while ( v32 != v33 );
    v33 = *(unsigned __int64 **)(a1 + 64);
  }
  if ( v33 != (unsigned __int64 *)(a1 + 80) )
    _libc_free((unsigned __int64)v33);
  v34 = *(_QWORD *)(a1 + 40);
  if ( v34 )
    j_j___libc_free_0(v34);
}
