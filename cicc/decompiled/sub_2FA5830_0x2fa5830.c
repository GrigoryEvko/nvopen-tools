// Function: sub_2FA5830
// Address: 0x2fa5830
//
__int64 __fastcall sub_2FA5830(_QWORD *a1)
{
  unsigned __int64 v2; // rdi

  *a1 = off_4A2BEA8;
  v2 = a1[25];
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
