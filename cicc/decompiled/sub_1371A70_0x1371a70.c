// Function: sub_1371A70
// Address: 0x1371a70
//
__int64 __fastcall sub_1371A70(_QWORD *a1)
{
  _QWORD **v2; // rdi
  __int64 result; // rax
  __int64 v4; // rdi
  _QWORD *i; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rdi

  v2 = (_QWORD **)(a1 + 11);
  *(v2 - 11) = &unk_49E8A50;
  result = sub_1371900(v2);
  v4 = a1[8];
  if ( v4 )
    result = j_j___libc_free_0(v4, a1[10] - v4);
  for ( i = (_QWORD *)a1[5]; a1 + 5 != i; result = j_j___libc_free_0(v6, 40) )
  {
    v6 = i;
    i = (_QWORD *)*i;
  }
  v7 = a1[1];
  if ( v7 )
    return j_j___libc_free_0(v7, a1[3] - v7);
  return result;
}
