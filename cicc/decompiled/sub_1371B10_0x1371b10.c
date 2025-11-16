// Function: sub_1371B10
// Address: 0x1371b10
//
__int64 __fastcall sub_1371B10(_QWORD *a1)
{
  _QWORD **v2; // rdi
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // rdi
  __int64 v6; // rdi

  v2 = (_QWORD **)(a1 + 11);
  *(v2 - 11) = &unk_49E8A50;
  sub_1371900(v2);
  v3 = a1[8];
  if ( v3 )
    j_j___libc_free_0(v3, a1[10] - v3);
  v4 = (_QWORD *)a1[5];
  while ( a1 + 5 != v4 )
  {
    v5 = v4;
    v4 = (_QWORD *)*v4;
    j_j___libc_free_0(v5, 40);
  }
  v6 = a1[1];
  if ( v6 )
    j_j___libc_free_0(v6, a1[3] - v6);
  return j_j___libc_free_0(a1, 112);
}
