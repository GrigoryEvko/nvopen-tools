// Function: sub_CB5840
// Address: 0xcb5840
//
void *__fastcall sub_CB5840(__int64 a1)
{
  void *result; // rax
  bool v2; // zf
  __int64 v3; // rdi

  result = &unk_49DD118;
  v2 = *(_DWORD *)(a1 + 44) == 1;
  *(_QWORD *)a1 = &unk_49DD118;
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      return (void *)j_j___libc_free_0_0(v3);
  }
  return result;
}
