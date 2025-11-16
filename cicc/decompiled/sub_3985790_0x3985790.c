// Function: sub_3985790
// Address: 0x3985790
//
void __fastcall sub_3985790(unsigned __int64 a1)
{
  __int64 v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 v17; // r8
  __int64 v18; // rbx
  __int64 v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rdi

  v2 = *(unsigned int *)(a1 + 920);
  *(_QWORD *)a1 = &unk_4A40688;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 904);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( v5 )
        {
          v6 = *(_QWORD *)(v5 + 40);
          if ( v6 != v5 + 56 )
            _libc_free(v6);
          j_j___libc_free_0(v5);
        }
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 904));
  j___libc_free_0(*(_QWORD *)(a1 + 872));
  v7 = *(_QWORD *)(a1 + 808);
  if ( v7 != a1 + 824 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 736);
  v9 = v8 + 56LL * *(unsigned int *)(a1 + 744);
  if ( v8 != v9 )
  {
    do
    {
      v9 -= 56LL;
      v10 = *(_QWORD *)(v9 + 8);
      if ( v10 != v9 + 24 )
        _libc_free(v10);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 736);
  }
  if ( v9 != a1 + 752 )
    _libc_free(v9);
  v11 = *(_QWORD *)(a1 + 704);
  if ( *(_DWORD *)(a1 + 716) )
  {
    v12 = *(unsigned int *)(a1 + 712);
    if ( (_DWORD)v12 )
    {
      v13 = 8 * v12;
      v14 = 0;
      do
      {
        v15 = *(_QWORD *)(v11 + v14);
        if ( v15 && v15 != -8 )
        {
          _libc_free(v15);
          v11 = *(_QWORD *)(a1 + 704);
        }
        v14 += 8;
      }
      while ( v13 != v14 );
    }
  }
  _libc_free(v11);
  if ( *(_DWORD *)(a1 + 684) )
  {
    v16 = *(unsigned int *)(a1 + 680);
    v17 = *(_QWORD *)(a1 + 672);
    if ( (_DWORD)v16 )
    {
      v18 = 8 * v16;
      v19 = 0;
      do
      {
        v20 = *(_QWORD *)(v17 + v19);
        if ( v20 && v20 != -8 )
        {
          _libc_free(v20);
          v17 = *(_QWORD *)(a1 + 672);
        }
        v19 += 8;
      }
      while ( v18 != v19 );
    }
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 672);
  }
  _libc_free(v17);
  v21 = *(unsigned int *)(a1 + 664);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 648);
    v23 = &v22[11 * v21];
    do
    {
      if ( *v22 != -8 && *v22 != -16 )
      {
        v24 = v22[1];
        if ( (_QWORD *)v24 != v22 + 3 )
          _libc_free(v24);
      }
      v22 += 11;
    }
    while ( v23 != v22 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 648));
  sub_39A20E0(a1);
  j_j___libc_free_0(a1);
}
