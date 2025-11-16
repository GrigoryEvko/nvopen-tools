// Function: sub_2041C10
// Address: 0x2041c10
//
void *__fastcall sub_2041C10(_QWORD *a1)
{
  void *result; // rax
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi

  result = &unk_49FFF60;
  *a1 = &unk_49FFF60;
  v3 = a1[21];
  if ( v3 )
    result = (void *)j_j___libc_free_0(v3, a1[23] - v3);
  v4 = a1[20];
  if ( v4 )
  {
    j___libc_free_0(*(_QWORD *)(v4 + 40));
    result = (void *)j_j___libc_free_0(v4, 64);
  }
  v5 = a1[12];
  if ( v5 )
    result = (void *)j_j___libc_free_0(v5, a1[14] - v5);
  v6 = a1[9];
  if ( v6 )
    result = (void *)j_j___libc_free_0(v6, a1[11] - v6);
  v7 = a1[6];
  if ( v7 )
    result = (void *)j_j___libc_free_0(v7, a1[8] - v7);
  v8 = a1[3];
  if ( v8 )
    return (void *)j_j___libc_free_0(v8, a1[5] - v8);
  return result;
}
