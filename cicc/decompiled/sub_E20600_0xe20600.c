// Function: sub_E20600
// Address: 0xe20600
//
void *__fastcall sub_E20600(_QWORD *a1)
{
  void *result; // rax
  _QWORD *v2; // rbx
  __int64 v4; // rdi

  result = &unk_49E0E68;
  v2 = (_QWORD *)a1[2];
  for ( *a1 = &unk_49E0E68; v2; a1[2] = v2 )
  {
    if ( *v2 )
      j_j___libc_free_0_0(*v2);
    v4 = a1[2];
    v2 = *(_QWORD **)(v4 + 24);
    result = (void *)j_j___libc_free_0(v4, 32);
  }
  return result;
}
