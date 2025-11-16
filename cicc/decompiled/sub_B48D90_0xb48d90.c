// Function: sub_B48D90
// Address: 0xb48d90
//
__int64 __fastcall sub_B48D90(__int64 a1)
{
  __int64 v1; // rsi

  v1 = (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) + ((*(_DWORD *)(a1 + 4) & 0x7FFFFFFu) >> 1);
  if ( (unsigned int)v1 < 2 )
    v1 = 2;
  *(_DWORD *)(a1 + 72) = v1;
  return sub_BD2A80(a1, v1, 1);
}
