// Function: sub_B54680
// Address: 0xb54680
//
__int64 __fastcall sub_B54680(__int64 a1)
{
  __int64 v1; // rsi

  v1 = 2 * (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu);
  *(_DWORD *)(a1 + 72) = v1;
  return sub_BD2A80(a1, v1, 0);
}
