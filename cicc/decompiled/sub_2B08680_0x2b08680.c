// Function: sub_2B08680
// Address: 0x2b08680
//
__int64 __fastcall sub_2B08680(__int64 a1, int a2)
{
  int v2; // eax

  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v2 == 17 )
  {
    a2 *= *(_DWORD *)(a1 + 32);
    return sub_BCDA70(**(__int64 ***)(a1 + 16), a2);
  }
  if ( (unsigned int)(v2 - 17) <= 1 )
    return sub_BCDA70(**(__int64 ***)(a1 + 16), a2);
  return sub_BCDA70((__int64 *)a1, a2);
}
