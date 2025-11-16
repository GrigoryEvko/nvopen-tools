// Function: sub_19947E0
// Address: 0x19947e0
//
__int64 __fastcall sub_19947E0(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_19947E0(v1[3]);
      v3 = v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        sub_1994610(*(_QWORD *)(v3 + 24));
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 16);
        j_j___libc_free_0(v4, 40);
      }
      result = j_j___libc_free_0(v2, 88);
    }
    while ( v1 );
  }
  return result;
}
