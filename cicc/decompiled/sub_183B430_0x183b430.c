// Function: sub_183B430
// Address: 0x183b430
//
void *__fastcall sub_183B430(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rdi
  void *result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdi

  v2 = a1[15];
  if ( v2 != a1[14] )
    _libc_free(v2);
  v3 = a1[10];
  result = &unk_49F0D08;
  *a1 = &unk_49F0D08;
  if ( v3 )
    result = (void *)j_j___libc_free_0(v3, a1[12] - v3);
  v5 = a1[6];
  if ( v5 )
    result = (void *)j_j___libc_free_0(v5, a1[8] - v5);
  v6 = a1[2];
  if ( v6 )
    return (void *)j_j___libc_free_0(v6, a1[4] - v6);
  return result;
}
