// Function: sub_B31FB0
// Address: 0xb31fb0
//
void __fastcall sub_B31FB0(__int64 a1, __int64 a2)
{
  char v2; // bl
  __int16 v4; // ax
  char v5; // dl
  unsigned __int16 v6; // ax
  __int64 v7; // rdx
  __int64 v8; // rsi

  sub_B31710(a1, (_BYTE *)a2);
  v4 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
  if ( v4 )
  {
    v2 = v4 - 1;
    v5 = 1;
  }
  else
  {
    v5 = 0;
  }
  LOBYTE(v6) = v2;
  HIBYTE(v6) = v5;
  sub_B2F740(a1, v6);
  v7 = 0;
  v8 = 0;
  if ( (*(_BYTE *)(a2 + 35) & 4) != 0 )
    v8 = sub_B31D10(a2, 0, 0);
  sub_B31A00(a1, v8, v7);
}
