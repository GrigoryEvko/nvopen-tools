// Function: sub_1444390
// Address: 0x1444390
//
__int64 __fastcall sub_1444390(_QWORD *a1)
{
  __int64 i; // r12
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 result; // rax
  _QWORD *v6; // r14
  _QWORD *v7; // r12
  __int64 v8; // r13

  for ( i = a1[10]; i; result = j_j___libc_free_0(v3, 48) )
  {
    v3 = i;
    sub_1442C80(*(_QWORD **)(i + 24));
    v4 = *(_QWORD *)(i + 40);
    i = *(_QWORD *)(i + 16);
    if ( v4 )
      j_j___libc_free_0(v4, 16);
  }
  v6 = (_QWORD *)a1[6];
  v7 = (_QWORD *)a1[5];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *v7;
      if ( *v7 )
      {
        sub_1444060(*v7);
        result = j_j___libc_free_0(v8, 112);
      }
      ++v7;
    }
    while ( v6 != v7 );
    v7 = (_QWORD *)a1[5];
  }
  if ( v7 )
    return j_j___libc_free_0(v7, a1[7] - (_QWORD)v7);
  return result;
}
