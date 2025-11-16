// Function: sub_1842330
// Address: 0x1842330
//
__int64 __fastcall sub_1842330(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1842330(v1[3]);
      v3 = (_QWORD *)v1[8];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v2 + 10 )
        j_j___libc_free_0(v3, v2[10] + 1LL);
      v4 = (_QWORD *)v2[4];
      if ( v4 != v2 + 6 )
        j_j___libc_free_0(v4, v2[6] + 1LL);
      result = j_j___libc_free_0(v2, 96);
    }
    while ( v1 );
  }
  return result;
}
