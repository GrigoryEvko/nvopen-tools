// Function: sub_30B09C0
// Address: 0x30b09c0
//
__int64 __fastcall sub_30B09C0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A32450;
  v2 = a1[8];
  if ( (_QWORD *)v2 != a1 + 10 )
    _libc_free(v2);
  return sub_30B08A0((__int64)a1);
}
