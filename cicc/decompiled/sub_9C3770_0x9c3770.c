// Function: sub_9C3770
// Address: 0x9c3770
//
void *__fastcall sub_9C3770(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  void *result; // rax
  _QWORD *v5; // rdi

  v2 = a1 + 7;
  v3 = (_QWORD *)a1[7];
  result = &unk_49D97D0;
  *a1 = &unk_49D97D0;
  if ( v3 )
  {
    if ( *v3 )
      j_j___libc_free_0(*v3, v3[2] - *v3);
    a2 = 24;
    result = (void *)j_j___libc_free_0(v3, 24);
  }
  v5 = (_QWORD *)a1[5];
  if ( v2 != v5 )
    return (void *)_libc_free(v5, a2);
  return result;
}
