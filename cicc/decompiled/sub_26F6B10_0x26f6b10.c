// Function: sub_26F6B10
// Address: 0x26f6b10
//
void __fastcall sub_26F6B10(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_26F6B10(v1[3]);
      v3 = v1[14];
      v1 = (_QWORD *)v1[2];
      if ( v3 )
        j_j___libc_free_0(v3);
      v4 = v2[11];
      if ( v4 )
        j_j___libc_free_0(v4);
      v5 = v2[7];
      if ( v5 )
        j_j___libc_free_0(v5);
      v6 = v2[4];
      if ( v6 )
        j_j___libc_free_0(v6);
      j_j___libc_free_0((unsigned __int64)v2);
    }
    while ( v1 );
  }
}
