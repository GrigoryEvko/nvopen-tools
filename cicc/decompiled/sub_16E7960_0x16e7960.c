// Function: sub_16E7960
// Address: 0x16e7960
//
void *__fastcall sub_16E7960(__int64 a1)
{
  void *result; // rax
  bool v2; // zf
  __int64 v3; // rdi

  result = &unk_49EFB08;
  v2 = *(_DWORD *)(a1 + 32) == 1;
  *(_QWORD *)a1 = &unk_49EFB08;
  if ( v2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( v3 )
      return (void *)j_j___libc_free_0_0(v3);
  }
  return result;
}
