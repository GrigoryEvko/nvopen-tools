// Function: sub_1B8EB10
// Address: 0x1b8eb10
//
void *__fastcall sub_1B8EB10(_QWORD *a1)
{
  __int64 v1; // r12
  void *result; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  v1 = a1[6];
  result = &unk_49F7038;
  *a1 = &unk_49F7038;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 40);
    if ( v3 != v1 + 56 )
      _libc_free(v3);
    v4 = *(_QWORD *)(v1 + 8);
    if ( v4 != v1 + 24 )
      _libc_free(v4);
    return (void *)j_j___libc_free_0(v1, 72);
  }
  return result;
}
