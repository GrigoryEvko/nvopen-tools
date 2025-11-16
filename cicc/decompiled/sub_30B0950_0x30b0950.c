// Function: sub_30B0950
// Address: 0x30b0950
//
__int64 __fastcall sub_30B0950(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A32430;
  v2 = a1[8];
  if ( (_QWORD *)v2 != a1 + 10 )
    _libc_free(v2);
  return sub_30B08A0((__int64)a1);
}
