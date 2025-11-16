// Function: sub_8D19C0
// Address: 0x8d19c0
//
__int64 __fastcall sub_8D19C0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 8 && (*(_BYTE *)(a1 + 169) & 2) != 0 && (*(_BYTE *)(a1 + 169) & 0x10) == 0 )
  {
    *a2 = 1;
    return 1;
  }
  return result;
}
