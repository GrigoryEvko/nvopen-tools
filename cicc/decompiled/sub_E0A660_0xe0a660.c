// Function: sub_E0A660
// Address: 0xe0a660
//
__int64 __fastcall sub_E0A660(__int64 a1, __int64 a2)
{
  if ( a2 != 25 )
  {
    if ( a2 == 23
      && !(*(_QWORD *)a1 ^ 0x454C5050415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x494B5F4D554E455FLL)
      && *(_DWORD *)(a1 + 16) == 1331643470
      && *(_WORD *)(a1 + 20) == 25968
      && *(_BYTE *)(a1 + 22) == 110 )
    {
      return 1;
    }
    return 0xFFFFFFFFLL;
  }
  if ( *(_QWORD *)a1 ^ 0x454C5050415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x494B5F4D554E455FLL
    || *(_QWORD *)(a1 + 16) != 0x65736F6C435F444ELL
    || *(_BYTE *)(a1 + 24) != 100 )
  {
    return 0xFFFFFFFFLL;
  }
  return 0;
}
