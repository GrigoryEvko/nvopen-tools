// Function: sub_C9AF20
// Address: 0xc9af20
//
__int64 *sub_C9AF20()
{
  __int64 *result; // rax
  __int64 v1; // rdi

  result = (__int64 *)sub_C94E20((__int64)&qword_4F84F00);
  v1 = qword_4F84F10;
  if ( result )
    v1 = *result;
  if ( v1 )
    return (__int64 *)sub_C9A3C0(v1, *(_QWORD *)(*(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8) - 8));
  return result;
}
