// Function: sub_D77820
// Address: 0xd77820
//
__int64 __fastcall sub_D77820(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 760) == 0;
  *(_QWORD *)a1 = &unk_49DE498;
  if ( !v1 )
  {
    *(_BYTE *)(a1 + 760) = 0;
    sub_9CD560(a1 + 176);
  }
  sub_BB9260(a1);
  return j_j___libc_free_0(a1, 768);
}
