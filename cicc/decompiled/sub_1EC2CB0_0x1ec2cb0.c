// Function: sub_1EC2CB0
// Address: 0x1ec2cb0
//
void *__fastcall sub_1EC2CB0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 i; // rbx
  __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 j; // rbx
  __int64 v34; // rdi

  *(_QWORD *)a1 = off_49FDB68;
  *(_QWORD *)(a1 + 232) = &unk_49FDC70;
  *(_QWORD *)(a1 + 672) = &unk_49FDCC8;
  v2 = *(_QWORD *)(a1 + 27496);
  if ( v2 != a1 + 27512 )
    _libc_free(v2);
  if ( (*(_BYTE *)(a1 + 27424) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 27432));
  v3 = *(_QWORD *)(a1 + 27256);
  if ( v3 != a1 + 27272 )
    _libc_free(v3);
  v4 = *(_QWORD **)(a1 + 24168);
  v5 = &v4[12 * *(unsigned int *)(a1 + 24176)];
  if ( v4 != v5 )
  {
    do
    {
      v5 -= 12;
      v6 = v5[6];
      if ( (_QWORD *)v6 != v5 + 8 )
        _libc_free(v6);
      _libc_free(v5[3]);
      v7 = v5[1];
      v5[2] = 0;
      if ( v7 )
        --*(_DWORD *)(v7 + 8);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 24168);
  }
  if ( v5 != (_QWORD *)(a1 + 24184) )
    _libc_free((unsigned __int64)v5);
  v8 = *(_QWORD *)(a1 + 24088);
  if ( v8 != a1 + 24104 )
    _libc_free(v8);
  v9 = a1 + 23368;
  _libc_free(*(_QWORD *)(a1 + 1024));
  do
  {
    v10 = *(_QWORD *)(v9 + 512);
    if ( v10 != v9 + 528 )
      _libc_free(v10);
    v11 = *(_QWORD *)(v9 + 48);
    v12 = v11 + 112LL * *(unsigned int *)(v9 + 56);
    if ( v11 != v12 )
    {
      do
      {
        v12 -= 112LL;
        v13 = *(_QWORD *)(v12 + 8);
        if ( v13 != v12 + 24 )
          _libc_free(v13);
      }
      while ( v11 != v12 );
      v11 = *(_QWORD *)(v9 + 48);
    }
    if ( v11 != v9 + 64 )
      _libc_free(v11);
    v9 -= 720;
  }
  while ( a1 + 328 != v9 );
  v14 = *(_QWORD *)(a1 + 992);
  if ( v14 )
    sub_1EC2B00(v14);
  v15 = *(_QWORD **)(a1 + 984);
  if ( v15 )
  {
    _libc_free(v15[78]);
    v16 = v15[35];
    if ( (_QWORD *)v16 != v15 + 37 )
      _libc_free(v16);
    v17 = v15[25];
    if ( (_QWORD *)v17 != v15 + 27 )
      _libc_free(v17);
    v18 = v15[7];
    if ( (_QWORD *)v18 != v15 + 9 )
      _libc_free(v18);
    j_j___libc_free_0(v15, 656);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 960));
  v19 = *(_QWORD *)(a1 + 920);
  if ( v19 != a1 + 936 )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 880);
  if ( v20 )
    j_j___libc_free_0(v20, *(_QWORD *)(a1 + 896) - v20);
  v21 = *(_QWORD *)(a1 + 872);
  if ( v21 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 16LL))(v21);
  v22 = *(_QWORD *)(a1 + 792);
  if ( v22 )
    j_j___libc_free_0_0(v22);
  _libc_free(*(_QWORD *)(a1 + 768));
  v23 = *(_QWORD *)(a1 + 744);
  if ( v23 != a1 + 760 )
    _libc_free(v23);
  v24 = *(_QWORD *)(a1 + 704);
  if ( v24 )
  {
    v25 = 24LL * *(_QWORD *)(v24 - 8);
    for ( i = v24 + v25; v24 != i; i -= 24 )
    {
      v27 = *(_QWORD *)(i - 8);
      if ( v27 )
        j_j___libc_free_0_0(v27);
    }
    j_j_j___libc_free_0_0(v24 - 8);
  }
  v28 = *(_QWORD *)(a1 + 392);
  *(_QWORD *)(a1 + 232) = &unk_4A00E78;
  if ( v28 != *(_QWORD *)(a1 + 384) )
    _libc_free(v28);
  v29 = *(_QWORD *)(a1 + 368);
  if ( v29 )
    j_j___libc_free_0_0(v29);
  _libc_free(*(_QWORD *)(a1 + 344));
  v30 = *(_QWORD *)(a1 + 320);
  if ( v30 != a1 + 336 )
    _libc_free(v30);
  v31 = *(_QWORD *)(a1 + 280);
  if ( v31 )
  {
    v32 = 24LL * *(_QWORD *)(v31 - 8);
    for ( j = v31 + v32; v31 != j; j -= 24 )
    {
      v34 = *(_QWORD *)(j - 8);
      if ( v34 )
        j_j___libc_free_0_0(v34);
    }
    j_j_j___libc_free_0_0(v31 - 8);
  }
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
