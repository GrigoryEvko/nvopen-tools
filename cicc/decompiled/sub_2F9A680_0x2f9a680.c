// Function: sub_2F9A680
// Address: 0x2f9a680
//
__int64 __fastcall sub_2F9A680(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi

  v2 = a1 + 58;
  v3 = a1[56];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
