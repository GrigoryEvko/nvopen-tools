// Function: sub_1383070
// Address: 0x1383070
//
__int64 __fastcall sub_1383070(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rdi

  v2 = *(_QWORD **)(a1 + 48);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = v3[4];
    v3[1] = &unk_49EE2B0;
    if ( v4 != -8 && v4 != 0 && v4 != -16 )
      sub_1649B30(v3 + 2);
    j_j___libc_free_0(v3, 48);
  }
  v5 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 24);
    v7 = v6 + 432 * v5;
    do
    {
      while ( *(_QWORD *)v6 == -8 || *(_QWORD *)v6 == -16 || !*(_BYTE *)(v6 + 424) )
      {
        v6 += 432;
        if ( v7 == v6 )
          return j___libc_free_0(*(_QWORD *)(a1 + 24));
      }
      v8 = *(_QWORD *)(v6 + 280);
      if ( v8 != v6 + 296 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v6 + 72);
      if ( v9 != v6 + 88 )
        _libc_free(v9);
      j___libc_free_0(*(_QWORD *)(v6 + 48));
      v10 = *(unsigned int *)(v6 + 32);
      if ( (_DWORD)v10 )
      {
        v11 = *(_QWORD **)(v6 + 16);
        v12 = &v11[4 * v10];
        do
        {
          if ( *v11 != -8 && *v11 != -16 )
          {
            v13 = v11[1];
            if ( v13 )
              j_j___libc_free_0(v13, v11[3] - v13);
          }
          v11 += 4;
        }
        while ( v12 != v11 );
      }
      v14 = *(_QWORD *)(v6 + 16);
      v6 += 432;
      j___libc_free_0(v14);
    }
    while ( v7 != v6 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 24));
}
