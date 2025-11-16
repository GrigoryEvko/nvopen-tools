// Function: sub_EF88E0
// Address: 0xef88e0
//
__int64 __fastcall sub_EF88E0(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_EF88E0(v1[3]);
      v3 = (_QWORD *)v1[8];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        v3 = (_QWORD *)*v3;
        j_j___libc_free_0(v4, 40);
      }
      memset((void *)v2[6], 0, 8LL * v2[7]);
      v5 = (_QWORD *)v2[6];
      v2[9] = 0;
      v2[8] = 0;
      if ( v5 != v2 + 12 )
        j_j___libc_free_0(v5, 8LL * v2[7]);
      result = j_j___libc_free_0(v2, 104);
    }
    while ( v1 );
  }
  return result;
}
