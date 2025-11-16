// Function: sub_D86030
// Address: 0xd86030
//
__int64 __fastcall sub_D86030(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_D86030(v1[3]);
      v3 = (_QWORD *)v1[13];
      v1 = (_QWORD *)v1[2];
      sub_D85F30(v3);
      sub_D85E30((_QWORD *)v2[7]);
      result = j_j___libc_free_0(v2, 144);
    }
    while ( v1 );
  }
  return result;
}
