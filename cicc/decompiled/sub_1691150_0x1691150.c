// Function: sub_1691150
// Address: 0x1691150
//
__int64 __fastcall sub_1691150(_QWORD *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  _QWORD *v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1691150(v1[3]);
      v3 = (_QWORD *)v1[9];
      v4 = (_QWORD *)v1[8];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v4 )
      {
        do
        {
          if ( (_QWORD *)*v4 != v4 + 2 )
            j_j___libc_free_0(*v4, v4[2] + 1LL);
          v4 += 6;
        }
        while ( v3 != v4 );
        v4 = (_QWORD *)v2[8];
      }
      if ( v4 )
        j_j___libc_free_0(v4, v2[10] - (_QWORD)v4);
      v5 = (_QWORD *)v2[4];
      if ( v5 != v2 + 6 )
        j_j___libc_free_0(v5, v2[6] + 1LL);
      result = j_j___libc_free_0(v2, 88);
    }
    while ( v1 );
  }
  return result;
}
