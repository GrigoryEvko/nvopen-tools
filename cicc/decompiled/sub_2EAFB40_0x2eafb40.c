// Function: sub_2EAFB40
// Address: 0x2eafb40
//
__int64 __fastcall sub_2EAFB40(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = &unk_4A297C0;
  v2 = a1[25];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
