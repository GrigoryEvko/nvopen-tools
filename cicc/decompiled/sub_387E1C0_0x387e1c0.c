// Function: sub_387E1C0
// Address: 0x387e1c0
//
void __fastcall sub_387E1C0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_387E1C0(v1[3]);
      v3 = v1[8];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_387E140(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 64);
        v3 = *(_QWORD *)(v3 + 16);
        if ( v5 != v4 + 80 )
          j_j___libc_free_0(v5);
        v6 = *(_QWORD *)(v4 + 32);
        if ( v6 != v4 + 48 )
          j_j___libc_free_0(v6);
        j_j___libc_free_0(v4);
      }
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
