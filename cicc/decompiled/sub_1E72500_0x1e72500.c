// Function: sub_1E72500
// Address: 0x1e72500
//
__int64 __fastcall sub_1E72500(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *a1 = &unk_49FCA80;
  v2 = a1[63];
  if ( (_QWORD *)v2 != a1 + 65 )
    _libc_free(v2);
  sub_1E72310(a1 + 17);
  v3 = a1[6];
  if ( (_QWORD *)v3 != a1 + 8 )
    _libc_free(v3);
  return j_j___libc_free_0(a1, 584);
}
