// Function: sub_9C9600
// Address: 0x9c9600
//
__int64 __fastcall sub_9C9600(_QWORD *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r12
  _QWORD *v4; // rbx
  __int64 v5; // rdi
  __int64 result; // rax

  v1 = (_QWORD *)*a1;
  v2 = (_QWORD *)a1[1];
  if ( (_QWORD *)*a1 != v2 )
  {
    v4 = (_QWORD *)*a1;
    do
    {
      v5 = v4[4];
      if ( v5 )
        j_j___libc_free_0(v5, v4[6] - v5);
      result = (__int64)(v4 + 2);
      if ( (_QWORD *)*v4 != v4 + 2 )
        result = j_j___libc_free_0(*v4, v4[2] + 1LL);
      v4 += 7;
    }
    while ( v2 != v4 );
    a1[1] = v1;
  }
  return result;
}
