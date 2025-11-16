// Function: sub_2EC8BB0
// Address: 0x2ec8bb0
//
__int64 __fastcall sub_2EC8BB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v3; // edx
  unsigned int v4; // ecx

  result = 0;
  if ( (*(_BYTE *)(a2 + 249) & 0x40) != 0 )
  {
    v3 = *(_DWORD *)(a2 + 236);
    if ( *(_DWORD *)(a1 + 24) == 1 )
      v3 = *(_DWORD *)(a2 + 232);
    v4 = *(_DWORD *)(a1 + 164);
    if ( v4 < v3 )
      return v3 - v4;
  }
  return result;
}
