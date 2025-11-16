// Function: sub_230BAD0
// Address: 0x230bad0
//
void __fastcall sub_230BAD0(_QWORD *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_230BAD0(v1[3]);
      v3 = v1[28];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_230B880(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 56);
        v3 = *(_QWORD *)(v3 + 16);
        sub_230BAD0(v5);
        j_j___libc_free_0(v4);
      }
      v6 = v2[8];
      if ( (_QWORD *)v6 != v2 + 10 )
        _libc_free(v6);
      v7 = (_QWORD *)v2[6];
      if ( v7 )
        *v7 = v2[5];
      v8 = v2[5];
      if ( v8 )
        *(_QWORD *)(v8 + 8) = v2[6];
      j_j___libc_free_0((unsigned __int64)v2);
    }
    while ( v1 );
  }
}
