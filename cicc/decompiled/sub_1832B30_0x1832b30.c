// Function: sub_1832B30
// Address: 0x1832b30
//
__int64 __fastcall sub_1832B30(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1832B30(v1[3]);
      v3 = v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_1832860(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 40);
        v3 = *(_QWORD *)(v3 + 16);
        if ( v5 )
          j_j___libc_free_0(v5, *(_QWORD *)(v4 + 56) - v5);
        j_j___libc_free_0(v4, 64);
      }
      result = j_j___libc_free_0(v2, 88);
    }
    while ( v1 );
  }
  return result;
}
