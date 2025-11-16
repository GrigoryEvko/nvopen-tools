// Function: sub_36D4A10
// Address: 0x36d4a10
//
__int64 __fastcall sub_36D4A10(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 24;
  v3 = a1[22];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
