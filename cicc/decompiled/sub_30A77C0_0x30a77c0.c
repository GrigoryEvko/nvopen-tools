// Function: sub_30A77C0
// Address: 0x30a77c0
//
void __fastcall sub_30A77C0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  _QWORD *v10; // [rsp+8h] [rbp-38h]

  v10 = a1;
  while ( v10 )
  {
    v1 = v10;
    sub_30A77C0(v10[3]);
    v2 = v10[33];
    v10 = (_QWORD *)v10[2];
    while ( v2 )
    {
      v3 = v2;
      sub_30A7420(*(_QWORD **)(v2 + 24));
      v4 = *(_QWORD *)(v2 + 56);
      v2 = *(_QWORD *)(v2 + 16);
      while ( v4 )
      {
        v5 = v4;
        sub_30A7670(*(_QWORD **)(v4 + 24));
        v4 = *(_QWORD *)(v4 + 16);
        sub_30A7730((_QWORD *)(v5 + 40));
        j_j___libc_free_0(v5);
      }
      j_j___libc_free_0(v3);
    }
    v6 = v1[13];
    if ( (_QWORD *)v6 != v1 + 15 )
      _libc_free(v6);
    v7 = (_QWORD *)v1[11];
    if ( v7 )
      *v7 = v1[10];
    v8 = v1[10];
    if ( v8 )
      *(_QWORD *)(v8 + 8) = v1[11];
    v9 = v1[6];
    if ( (_QWORD *)v9 != v1 + 8 )
      j_j___libc_free_0(v9);
    j_j___libc_free_0((unsigned __int64)v1);
  }
}
