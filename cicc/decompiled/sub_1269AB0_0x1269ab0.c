// Function: sub_1269AB0
// Address: 0x1269ab0
//
__int64 __fastcall sub_1269AB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r14
  _QWORD *i; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rax

  v3 = *(_QWORD *)(a1 + 384);
  if ( v3 )
  {
    sub_129E600(*(_QWORD *)(a1 + 384));
    a2 = 816;
    j_j___libc_free_0(v3, 816);
  }
  v4 = *(_QWORD **)(a1 + 456);
  for ( i = *(_QWORD **)(a1 + 448); v4 != i; ++i )
  {
    v6 = (_QWORD *)*i;
    if ( *i )
    {
      v7 = v6[3];
      if ( v7 != v6[4] )
        v6[4] = v7;
      if ( v7 )
        j_j___libc_free_0(v7, v6[5] - v7);
      v8 = v6[2];
      if ( v8 != -8 && v8 != 0 && v8 != -16 )
        sub_1649B30(v6);
      a2 = 48;
      j_j___libc_free_0(v6, 48);
    }
  }
  sub_1268D60(*(_QWORD *)(a1 + 696));
  sub_1269100(*(_QWORD *)(a1 + 648));
  sub_12692D0(*(_QWORD *)(a1 + 600));
  sub_1268F30(*(_QWORD *)(a1 + 552));
  if ( *(_DWORD *)(a1 + 516) )
  {
    v9 = *(unsigned int *)(a1 + 512);
    v10 = *(_QWORD *)(a1 + 504);
    if ( (_DWORD)v9 )
    {
      v11 = 8 * v9;
      v12 = 0;
      do
      {
        v13 = *(_QWORD *)(v10 + v12);
        if ( v13 != -8 && v13 )
        {
          _libc_free(v13, a2);
          v10 = *(_QWORD *)(a1 + 504);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 504);
  }
  _libc_free(v10, a2);
  if ( *(_DWORD *)(a1 + 484) )
  {
    v14 = *(unsigned int *)(a1 + 480);
    v15 = *(_QWORD *)(a1 + 472);
    if ( (_DWORD)v14 )
    {
      v16 = 8 * v14;
      v17 = 0;
      do
      {
        v18 = *(_QWORD *)(v15 + v17);
        if ( v18 != -8 && v18 )
        {
          _libc_free(v18, a2);
          v15 = *(_QWORD *)(a1 + 472);
        }
        v17 += 8;
      }
      while ( v16 != v17 );
    }
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 472);
  }
  _libc_free(v15, a2);
  v19 = *(_QWORD *)(a1 + 448);
  if ( v19 )
    j_j___libc_free_0(v19, *(_QWORD *)(a1 + 464) - v19);
  v20 = *(_QWORD *)(a1 + 432);
  v21 = *(_QWORD *)(a1 + 424);
  if ( v20 != v21 )
  {
    do
    {
      v22 = *(_QWORD *)(v21 + 16);
      if ( v22 != -8 && v22 != 0 && v22 != -16 )
        sub_1649B30(v21);
      v21 += 24;
    }
    while ( v20 != v21 );
    v21 = *(_QWORD *)(a1 + 424);
  }
  if ( v21 )
    j_j___libc_free_0(v21, *(_QWORD *)(a1 + 440) - v21);
  j___libc_free_0(*(_QWORD *)(a1 + 400));
  return sub_1277A00(a1 + 8);
}
