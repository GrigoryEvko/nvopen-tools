// Function: sub_3988C60
// Address: 0x3988c60
//
void __fastcall sub_3988C60(__int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi

  j___libc_free_0(*(_QWORD *)(a1 + 368));
  v2 = *(unsigned int *)(a1 + 352);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 336);
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
  j___libc_free_0(*(_QWORD *)(a1 + 336));
  j___libc_free_0(*(_QWORD *)(a1 + 304));
  v7 = *(unsigned int *)(a1 + 288);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 272);
    v9 = &v8[17 * v7];
    do
    {
      if ( *v8 != -16 && *v8 != -8 )
      {
        v10 = v8[7];
        if ( (_QWORD *)v10 != v8 + 9 )
          _libc_free(v10);
        v11 = v8[3];
        while ( v11 )
        {
          sub_3985EB0(*(_QWORD *)(v11 + 24));
          v12 = v11;
          v11 = *(_QWORD *)(v11 + 16);
          j_j___libc_free_0(v12);
        }
      }
      v8 += 17;
    }
    while ( v9 != v8 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 272));
  _libc_free(*(_QWORD *)(a1 + 192));
  v13 = *(_QWORD *)(a1 + 168);
  v14 = v13 + 8LL * *(unsigned int *)(a1 + 176);
  if ( v13 != v14 )
  {
    do
    {
      v15 = *(_QWORD *)(v14 - 8);
      v14 -= 8LL;
      if ( v15 )
        sub_3985790(v15);
    }
    while ( v13 != v14 );
    v14 = *(_QWORD *)(a1 + 168);
  }
  if ( v14 != a1 + 184 )
    _libc_free(v14);
  sub_3981BC0((_QWORD *)(a1 + 112));
  v16 = *(unsigned __int64 **)(a1 + 24);
  v17 = &v16[*(unsigned int *)(a1 + 32)];
  while ( v17 != v16 )
  {
    v18 = *v16++;
    _libc_free(v18);
  }
  v19 = *(unsigned __int64 **)(a1 + 72);
  v20 = (unsigned __int64)&v19[2 * *(unsigned int *)(a1 + 80)];
  if ( v19 != (unsigned __int64 *)v20 )
  {
    do
    {
      v21 = *v19;
      v19 += 2;
      _libc_free(v21);
    }
    while ( v19 != (unsigned __int64 *)v20 );
    v20 = *(_QWORD *)(a1 + 72);
  }
  if ( v20 != a1 + 88 )
    _libc_free(v20);
  v22 = *(_QWORD *)(a1 + 24);
  if ( v22 != a1 + 40 )
    _libc_free(v22);
}
