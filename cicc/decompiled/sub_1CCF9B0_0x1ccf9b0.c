// Function: sub_1CCF9B0
// Address: 0x1ccf9b0
//
__int64 __fastcall sub_1CCF9B0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 22;
  v3 = a1[20];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 208);
}
