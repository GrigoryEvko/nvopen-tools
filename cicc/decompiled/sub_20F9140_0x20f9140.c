// Function: sub_20F9140
// Address: 0x20f9140
//
void *__fastcall sub_20F9140(_QWORD *a1)
{
  void *result; // rax
  __int64 v3; // rdi
  __int64 v4; // rdi

  result = &unk_4A00AB0;
  *a1 = &unk_4A00AB0;
  v3 = a1[6];
  if ( v3 )
    result = (void *)j_j___libc_free_0(v3, a1[8] - v3);
  v4 = a1[3];
  if ( v4 )
    return (void *)j_j___libc_free_0(v4, a1[5] - v4);
  return result;
}
