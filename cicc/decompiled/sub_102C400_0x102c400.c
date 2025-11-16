// Function: sub_102C400
// Address: 0x102c400
//
__int64 __fastcall sub_102C400(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 1192) == 0;
  *(_QWORD *)a1 = &unk_49E5898;
  if ( !v1 )
  {
    *(_BYTE *)(a1 + 1192) = 0;
    sub_102BD40(a1 + 176);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
