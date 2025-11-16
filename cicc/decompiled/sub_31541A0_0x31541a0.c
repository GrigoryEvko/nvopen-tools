// Function: sub_31541A0
// Address: 0x31541a0
//
void __fastcall sub_31541A0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_31541A0(v1[3]);
      v3 = v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = (_QWORD *)v3;
        sub_3153D30(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 224);
        v3 = *(_QWORD *)(v3 + 16);
        sub_31541A0(v5);
        v6 = v4[8];
        if ( (_QWORD *)v6 != v4 + 10 )
          _libc_free(v6);
        v7 = (_QWORD *)v4[6];
        if ( v7 )
          *v7 = v4[5];
        v8 = v4[5];
        if ( v8 )
          *(_QWORD *)(v8 + 8) = v4[6];
        j_j___libc_free_0((unsigned __int64)v4);
      }
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
