// Function: sub_1610730
// Address: 0x1610730
//
void __fastcall sub_1610730(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *i; // r13
  _QWORD *v4; // rbx
  _QWORD *j; // r13
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi

  v2 = *(_QWORD **)(a1 + 32);
  *(_QWORD *)a1 = &unk_49EDA38;
  for ( i = &v2[*(unsigned int *)(a1 + 40)]; i != v2; ++v2 )
  {
    if ( *v2 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
  }
  v4 = *(_QWORD **)(a1 + 256);
  for ( j = &v4[*(unsigned int *)(a1 + 264)]; j != v4; ++v4 )
  {
    if ( *v4 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 712));
  j___libc_free_0(*(_QWORD *)(a1 + 680));
  sub_16104D0(a1 + 568);
  v6 = *(unsigned __int64 **)(a1 + 584);
  v7 = &v6[*(unsigned int *)(a1 + 592)];
  while ( v7 != v6 )
  {
    v8 = *v6++;
    _libc_free(v8);
  }
  v9 = *(unsigned __int64 **)(a1 + 632);
  v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(a1 + 640)];
  if ( v9 != (unsigned __int64 *)v10 )
  {
    do
    {
      v11 = *v9;
      v9 += 2;
      _libc_free(v11);
    }
    while ( v9 != (unsigned __int64 *)v10 );
    v10 = *(_QWORD *)(a1 + 632);
  }
  if ( v10 != a1 + 648 )
    _libc_free(v10);
  v12 = *(_QWORD *)(a1 + 584);
  if ( v12 != a1 + 600 )
    _libc_free(v12);
  *(_QWORD *)(a1 + 544) = &unk_49ED500;
  sub_16BD9D0(a1 + 544);
  if ( (*(_BYTE *)(a1 + 408) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 416));
  v13 = *(_QWORD *)(a1 + 256);
  if ( v13 != a1 + 272 )
    _libc_free(v13);
  v14 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 232);
    v16 = &v15[14 * v14];
    do
    {
      if ( *v15 != -16 && *v15 != -8 )
      {
        v17 = v15[3];
        if ( v17 != v15[2] )
          _libc_free(v17);
      }
      v15 += 14;
    }
    while ( v16 != v15 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 232));
  j___libc_free_0(*(_QWORD *)(a1 + 200));
  v18 = *(_QWORD *)(a1 + 112);
  if ( v18 != a1 + 128 )
    _libc_free(v18);
  v19 = *(_QWORD *)(a1 + 32);
  if ( v19 != a1 + 48 )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 8);
  if ( v20 )
    j_j___libc_free_0(v20, *(_QWORD *)(a1 + 24) - v20);
}
