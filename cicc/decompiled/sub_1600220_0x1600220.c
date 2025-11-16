// Function: sub_1600220
// Address: 0x1600220
//
__int64 __fastcall sub_1600220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi

  v6 = 2 * (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu);
  *(_DWORD *)(a1 + 56) = v6;
  return sub_16488D0(a1, v6, 0, a4, a5, a6);
}
