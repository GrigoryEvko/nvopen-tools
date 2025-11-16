// Function: sub_B8E070
// Address: 0xb8e070
//
void __fastcall sub_B8E070(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rsi

  v4 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    v4 = (_BYTE *)sub_B91C10(a2, 40);
  sub_B8DF90(a1, v4);
}
