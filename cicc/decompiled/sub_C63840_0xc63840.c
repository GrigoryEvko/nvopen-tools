// Function: sub_C63840
// Address: 0xc63840
//
void *__fastcall sub_C63840(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  void *result; // rax

  v1 = (_QWORD *)a1[2];
  v2 = (_QWORD *)a1[1];
  result = &unk_49DC7A0;
  *a1 = &unk_49DC7A0;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        result = (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[1];
  }
  if ( v2 )
    return (void *)j_j___libc_free_0(v2, a1[3] - (_QWORD)v2);
  return result;
}
