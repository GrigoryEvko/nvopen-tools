// Function: sub_EF9170
// Address: 0xef9170
//
__int64 __fastcall sub_EF9170(_QWORD *a1)
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
      sub_EF9170(v1[3]);
      v3 = v1[23];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_EF8F20(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 56);
        v3 = *(_QWORD *)(v3 + 16);
        sub_EF9170(v5);
        j_j___libc_free_0(v4, 88);
      }
      sub_EF88E0((_QWORD *)v2[17]);
      result = j_j___libc_free_0(v2, 224);
    }
    while ( v1 );
  }
  return result;
}
