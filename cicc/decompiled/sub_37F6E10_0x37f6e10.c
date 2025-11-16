// Function: sub_37F6E10
// Address: 0x37f6e10
//
void __fastcall sub_37F6E10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r13
  _QWORD *v9; // rax
  unsigned __int64 v10; // r12
  _QWORD *v11; // rdx
  _QWORD *v12; // r15
  unsigned __int64 v13; // rdi
  int v14; // r15d
  unsigned __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x18u, v15, a6);
  v9 = *(_QWORD **)a1;
  v10 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = (_QWORD *)v8;
    do
    {
      if ( v11 )
      {
        *v11 = *v9;
        v11[1] = v9[1];
        v11[2] = v9[2];
        v9[2] = 0;
        v9[1] = 0;
        *v9 = 0;
      }
      v9 += 3;
      v11 += 3;
    }
    while ( (_QWORD *)v10 != v9 );
    v12 = *(_QWORD **)a1;
    v10 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v13 = *(_QWORD *)(v10 - 24);
        v10 -= 24LL;
        if ( v13 )
          j_j___libc_free_0(v13);
      }
      while ( (_QWORD *)v10 != v12 );
      v10 = *(_QWORD *)a1;
    }
  }
  v14 = v15[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v14;
}
