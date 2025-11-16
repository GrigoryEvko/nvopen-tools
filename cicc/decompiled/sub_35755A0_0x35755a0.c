// Function: sub_35755A0
// Address: 0x35755a0
//
__int64 __fastcall sub_35755A0(_QWORD *a1)
{
  unsigned __int64 v1; // rsi

  v1 = a1[25];
  *a1 = &unk_4A394F0;
  if ( v1 )
    sub_3575560((__int64)(a1 + 25), v1);
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
