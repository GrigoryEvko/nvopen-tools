// Function: sub_16366C0
// Address: 0x16366c0
//
void *__fastcall sub_16366C0(_QWORD *a1)
{
  _QWORD *v2; // r13
  void *result; // rax
  __int64 i; // rbx
  __int64 v5; // rdi
  __int64 j; // rbx
  __int64 v7; // rdi
  __int64 v8; // rdi

  v2 = (_QWORD *)a1[1];
  result = &unk_49EDE80;
  *a1 = &unk_49EDE80;
  if ( v2 )
  {
    if ( *v2 )
      j_j___libc_free_0(*v2, v2[2] - *v2);
    result = (void *)j_j___libc_free_0(v2, 32);
  }
  for ( i = a1[15]; i; result = (void *)j_j___libc_free_0(v5, 40) )
  {
    sub_1636080(*(_QWORD *)(i + 24));
    v5 = i;
    i = *(_QWORD *)(i + 16);
  }
  for ( j = a1[9]; j; result = (void *)j_j___libc_free_0(v7, 40) )
  {
    sub_1636250(*(_QWORD *)(j + 24));
    v7 = j;
    j = *(_QWORD *)(j + 16);
  }
  v8 = a1[4];
  if ( v8 )
    return (void *)j_j___libc_free_0(v8, a1[6] - v8);
  return result;
}
