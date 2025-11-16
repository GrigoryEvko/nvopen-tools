// Function: sub_10282A0
// Address: 0x10282a0
//
__int64 __fastcall sub_10282A0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49E57F0;
  v2 = a1[22];
  if ( v2 )
    sub_1027BE0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 184);
}
