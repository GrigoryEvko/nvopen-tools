// Function: sub_B53E10
// Address: 0xb53e10
//
__int64 __fastcall sub_B53E10(__int64 a1)
{
  __int64 v1; // rsi

  v1 = 3 * (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu);
  *(_DWORD *)(a1 + 72) = v1;
  return sub_BD2A80(a1, v1, 0);
}
