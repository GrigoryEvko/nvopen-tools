// Function: sub_FE9450
// Address: 0xfe9450
//
__int64 __fastcall sub_FE9450(_QWORD *a1, __int64 a2)
{
  _QWORD **v3; // rdi
  __int64 result; // rax
  __int64 v5; // rdi
  _QWORD *i; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // rdi

  v3 = (_QWORD **)(a1 + 11);
  *(v3 - 11) = &unk_49E5580;
  result = sub_FE92E0(v3, a2);
  v5 = a1[8];
  if ( v5 )
    result = j_j___libc_free_0(v5, a1[10] - v5);
  for ( i = (_QWORD *)a1[4]; a1 + 4 != i; result = j_j___libc_free_0(v7, 40) )
  {
    v7 = i;
    i = (_QWORD *)*i;
  }
  v8 = a1[1];
  if ( v8 )
    return j_j___libc_free_0(v8, a1[3] - v8);
  return result;
}
