// Function: sub_30854D0
// Address: 0x30854d0
//
void __fastcall sub_30854D0(_QWORD *a1)
{
  _QWORD *v1; // r12
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_30854D0(v1[3]);
      v3 = v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        sub_3085300(*(_QWORD *)(v3 + 24));
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 16);
        j_j___libc_free_0(v4);
      }
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
