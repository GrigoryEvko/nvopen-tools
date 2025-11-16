// Function: sub_1AB1780
// Address: 0x1ab1780
//
bool __fastcall sub_1AB1780(__int64 *a1, __int64 a2, int a3, int a4, int a5)
{
  char v5; // al
  __int64 v6; // rsi

  v5 = *(_BYTE *)(a2 + 8);
  v6 = *a1;
  if ( v5 == 2 )
    return (((int)*(unsigned __int8 *)(v6 + a4 / 4) >> (2 * (a4 & 3))) & 3) != 0;
  if ( v5 == 3 )
    return (((int)*(unsigned __int8 *)(v6 + a3 / 4) >> (2 * (a3 & 3))) & 3) != 0;
  return (((int)*(unsigned __int8 *)(v6 + a5 / 4) >> (2 * (a5 & 3))) & 3) != 0;
}
