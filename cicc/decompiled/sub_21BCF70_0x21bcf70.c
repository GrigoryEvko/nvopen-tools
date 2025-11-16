// Function: sub_21BCF70
// Address: 0x21bcf70
//
__int64 __fastcall sub_21BCF70(_QWORD *a1)
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
