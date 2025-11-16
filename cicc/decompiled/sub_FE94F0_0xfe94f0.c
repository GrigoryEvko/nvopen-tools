// Function: sub_FE94F0
// Address: 0xfe94f0
//
__int64 __fastcall sub_FE94F0(_QWORD *a1, __int64 a2)
{
  _QWORD **v3; // rdi
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rdi

  v3 = (_QWORD **)(a1 + 11);
  *(v3 - 11) = &unk_49E5580;
  sub_FE92E0(v3, a2);
  v4 = a1[8];
  if ( v4 )
    j_j___libc_free_0(v4, a1[10] - v4);
  v5 = (_QWORD *)a1[4];
  while ( a1 + 4 != v5 )
  {
    v6 = v5;
    v5 = (_QWORD *)*v5;
    j_j___libc_free_0(v6, 40);
  }
  v7 = a1[1];
  if ( v7 )
    j_j___libc_free_0(v7, a1[3] - v7);
  return j_j___libc_free_0(a1, 112);
}
