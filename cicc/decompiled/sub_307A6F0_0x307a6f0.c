// Function: sub_307A6F0
// Address: 0x307a6f0
//
void __fastcall sub_307A6F0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r12
  _QWORD *v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_307A6F0(v1[3]);
      v3 = (_QWORD *)v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = (unsigned __int64)v3;
        sub_307A4A0(v3[3]);
        v5 = (_QWORD *)v3[7];
        v3 = (_QWORD *)v3[2];
        sub_307A420(v5);
        j_j___libc_free_0(v4);
      }
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
