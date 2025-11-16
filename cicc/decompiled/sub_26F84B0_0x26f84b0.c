// Function: sub_26F84B0
// Address: 0x26f84b0
//
void __fastcall sub_26F84B0(_QWORD *a1)
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
      sub_26F84B0(v1[3]);
      v3 = v1[12];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_26F81E0(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 32);
        v3 = *(_QWORD *)(v3 + 16);
        if ( v5 )
          j_j___libc_free_0(v5);
        j_j___libc_free_0(v4);
      }
      v6 = *(_QWORD *)(v2 + 48);
      if ( v6 != v2 + 64 )
        j_j___libc_free_0(v6);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
