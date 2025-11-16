// Function: sub_3988E70
// Address: 0x3988e70
//
void __fastcall sub_3988E70(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 *v10; // r12
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi

  v2 = *(unsigned __int64 **)(a1 + 184);
  v3 = *(unsigned __int64 **)(a1 + 176);
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        j_j___libc_free_0(*v3);
      v3 += 3;
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 176);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 152);
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = *(_QWORD *)(a1 + 104);
  if ( *(_DWORD *)(a1 + 116) )
  {
    v6 = *(unsigned int *)(a1 + 112);
    if ( (_DWORD)v6 )
    {
      v7 = 8 * v6;
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(v5 + v8);
        if ( v9 != -8 && v9 && *(_QWORD *)(v9 + 24) )
        {
          j_j___libc_free_0(*(_QWORD *)(v9 + 24));
          v5 = *(_QWORD *)(a1 + 104);
        }
        v8 += 8;
      }
      while ( v7 != v8 );
    }
  }
  _libc_free(v5);
  v10 = *(unsigned __int64 **)(a1 + 16);
  v11 = &v10[*(unsigned int *)(a1 + 24)];
  while ( v11 != v10 )
  {
    v12 = *v10++;
    _libc_free(v12);
  }
  v13 = *(unsigned __int64 **)(a1 + 64);
  v14 = (unsigned __int64)&v13[2 * *(unsigned int *)(a1 + 72)];
  if ( v13 != (unsigned __int64 *)v14 )
  {
    do
    {
      v15 = *v13;
      v13 += 2;
      _libc_free(v15);
    }
    while ( (unsigned __int64 *)v14 != v13 );
    v14 = *(_QWORD *)(a1 + 64);
  }
  if ( v14 != a1 + 80 )
    _libc_free(v14);
  v16 = *(_QWORD *)(a1 + 16);
  if ( v16 != a1 + 32 )
    _libc_free(v16);
}
