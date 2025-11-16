// Function: sub_16BA4F0
// Address: 0x16ba4f0
//
__int64 __fastcall sub_16BA4F0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v1 = (_QWORD *)a1[1];
    v2 = (_QWORD *)*a1;
    if ( v1 != (_QWORD *)*a1 )
    {
      do
      {
        if ( (_QWORD *)*v2 != v2 + 2 )
          j_j___libc_free_0(*v2, v2[2] + 1LL);
        v2 += 4;
      }
      while ( v1 != v2 );
      v2 = (_QWORD *)*a1;
    }
    if ( v2 )
      j_j___libc_free_0(v2, a1[2] - (_QWORD)v2);
    return j_j___libc_free_0(a1, 24);
  }
  return result;
}
