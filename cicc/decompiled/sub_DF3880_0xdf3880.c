// Function: sub_DF3880
// Address: 0xdf3880
//
__int64 __fastcall sub_DF3880(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49DEAA8;
  v2 = a1[22];
  if ( v2 )
    j_j___libc_free_0(v2, 8);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
