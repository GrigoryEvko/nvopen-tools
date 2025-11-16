// Function: sub_C206E0
// Address: 0xc206e0
//
__int64 __fastcall sub_C206E0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 *v6; // r15
  __int64 *v7; // rbx
  __int64 i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 j; // rax
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  _QWORD *v26; // rbx
  _QWORD *v27; // r13
  __int64 v28; // rdi

  v2 = 16LL * *(unsigned int *)(a1 + 512);
  *(_QWORD *)a1 = &unk_49DBC48;
  sub_C7D6A0(*(_QWORD *)(a1 + 496), v2, 8);
  v3 = *(_QWORD *)(a1 + 464);
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 480) - v3);
  v4 = 16LL * *(unsigned int *)(a1 + 456);
  sub_C7D6A0(*(_QWORD *)(a1 + 440), v4, 8);
  v5 = *(_QWORD *)(a1 + 424);
  if ( v5 )
  {
    v6 = *(__int64 **)(v5 + 56);
    v7 = &v6[*(unsigned int *)(v5 + 64)];
    if ( v6 != v7 )
    {
      for ( i = *(_QWORD *)(v5 + 56); ; i = *(_QWORD *)(v5 + 56) )
      {
        v9 = *v6;
        v10 = (unsigned int)(((__int64)v6 - i) >> 3) >> 7;
        v4 = 4096LL << v10;
        if ( v10 >= 0x1E )
          v4 = 0x40000000000LL;
        ++v6;
        sub_C7D6A0(v9, v4, 16);
        if ( v7 == v6 )
          break;
      }
    }
    v11 = *(__int64 **)(v5 + 104);
    v12 = &v11[2 * *(unsigned int *)(v5 + 112)];
    if ( v11 != v12 )
    {
      do
      {
        v4 = v11[1];
        v13 = *v11;
        v11 += 2;
        sub_C7D6A0(v13, v4, 16);
      }
      while ( v12 != v11 );
      v12 = *(__int64 **)(v5 + 104);
    }
    if ( v12 != (__int64 *)(v5 + 120) )
      _libc_free(v12, v4);
    v14 = *(_QWORD *)(v5 + 56);
    if ( v14 != v5 + 72 )
      _libc_free(v14, v4);
    sub_C7D6A0(*(_QWORD *)(v5 + 16), 16LL * *(unsigned int *)(v5 + 32), 8);
    v4 = 136;
    j_j___libc_free_0(v5, 136);
  }
  v15 = *(_QWORD *)(a1 + 400);
  if ( v15 )
  {
    v4 = *(_QWORD *)(a1 + 416) - v15;
    j_j___libc_free_0(v15, v4);
  }
  v16 = *(__int64 **)(a1 + 320);
  v17 = &v16[*(unsigned int *)(a1 + 328)];
  if ( v16 != v17 )
  {
    for ( j = *(_QWORD *)(a1 + 320); ; j = *(_QWORD *)(a1 + 320) )
    {
      v19 = *v16;
      v20 = (unsigned int)(((__int64)v16 - j) >> 3) >> 7;
      v4 = 4096LL << v20;
      if ( v20 >= 0x1E )
        v4 = 0x40000000000LL;
      ++v16;
      sub_C7D6A0(v19, v4, 16);
      if ( v17 == v16 )
        break;
    }
  }
  v21 = *(__int64 **)(a1 + 368);
  v22 = &v21[2 * *(unsigned int *)(a1 + 376)];
  if ( v21 != v22 )
  {
    do
    {
      v4 = v21[1];
      v23 = *v21;
      v21 += 2;
      sub_C7D6A0(v23, v4, 16);
    }
    while ( v22 != v21 );
    v22 = *(__int64 **)(a1 + 368);
  }
  if ( v22 != (__int64 *)(a1 + 384) )
    _libc_free(v22, v4);
  v24 = *(_QWORD *)(a1 + 320);
  if ( v24 != a1 + 336 )
    _libc_free(v24, v4);
  v25 = *(_QWORD *)(a1 + 272);
  *(_QWORD *)a1 = &unk_49DBCC8;
  if ( v25 )
  {
    v4 = *(_QWORD *)(a1 + 288) - v25;
    j_j___libc_free_0(v25, v4);
  }
  v26 = *(_QWORD **)(a1 + 256);
  v27 = *(_QWORD **)(a1 + 248);
  if ( v26 != v27 )
  {
    do
    {
      if ( (_QWORD *)*v27 != v27 + 2 )
        _libc_free(*v27, v4);
      v27 += 5;
    }
    while ( v26 != v27 );
    v27 = *(_QWORD **)(a1 + 248);
  }
  if ( v27 )
    j_j___libc_free_0(v27, *(_QWORD *)(a1 + 264) - (_QWORD)v27);
  v28 = *(_QWORD *)(a1 + 224);
  if ( v28 )
    j_j___libc_free_0(v28, *(_QWORD *)(a1 + 240) - v28);
  return sub_C201C0(a1);
}
