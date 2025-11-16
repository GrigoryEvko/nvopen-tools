// Function: sub_877F10
// Address: 0x877f10
//
void __fastcall sub_877F10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx

  v6 = *(_QWORD *)(a2 + 64);
  if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0 )
  {
    sub_877E20(0, a1, v6, a4, a5, a6);
  }
  else if ( v6 )
  {
    sub_877E90(0, a1, v6);
  }
}
