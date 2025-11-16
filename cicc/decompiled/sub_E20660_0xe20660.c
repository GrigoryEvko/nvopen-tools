// Function: sub_E20660
// Address: 0xe20660
//
__int64 __fastcall sub_E20660(_QWORD *a1)
{
  _QWORD *v2; // rbx
  __int64 v3; // rdi

  v2 = (_QWORD *)a1[2];
  for ( *a1 = &unk_49E0E68; v2; a1[2] = v2 )
  {
    if ( *v2 )
      j_j___libc_free_0_0(*v2);
    v3 = a1[2];
    v2 = *(_QWORD **)(v3 + 24);
    j_j___libc_free_0(v3, 32);
  }
  return j_j___libc_free_0(a1, 200);
}
