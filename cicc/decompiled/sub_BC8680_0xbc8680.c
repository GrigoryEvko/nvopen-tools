// Function: sub_BC8680
// Address: 0xbc8680
//
__int64 __fastcall sub_BC8680(__int64 a1)
{
  unsigned int v1; // eax

  if ( a1
    && ((*(_BYTE *)(a1 - 16) & 2) == 0 ? (v1 = (*(_WORD *)(a1 - 16) >> 6) & 0xF) : (v1 = *(_DWORD *)(a1 - 24)), v1 > 2) )
  {
    return sub_BC85F0(a1, "branch_weights");
  }
  else
  {
    return 0;
  }
}
