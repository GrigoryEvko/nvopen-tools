// Function: sub_26BCBE0
// Address: 0x26bcbe0
//
void __fastcall sub_26BCBE0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_26BCBE0(v1[3]);
      v3 = v1[23];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_26BC990(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 56);
        v3 = *(_QWORD *)(v3 + 16);
        sub_26BCBE0(v5);
        j_j___libc_free_0(v4);
      }
      sub_26BB480(*(_QWORD **)(v2 + 136));
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
