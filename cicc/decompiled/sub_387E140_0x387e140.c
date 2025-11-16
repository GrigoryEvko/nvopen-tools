// Function: sub_387E140
// Address: 0x387e140
//
void __fastcall sub_387E140(_QWORD *a1)
{
  _QWORD *v1; // rbx
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_387E140(v1[3]);
      v3 = v1[8];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v2 + 80 )
        j_j___libc_free_0(v3);
      v4 = *(_QWORD *)(v2 + 32);
      if ( v4 != v2 + 48 )
        j_j___libc_free_0(v4);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
