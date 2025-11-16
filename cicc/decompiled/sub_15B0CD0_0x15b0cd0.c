// Function: sub_15B0CD0
// Address: 0x15b0cd0
//
__int64 __fastcall sub_15B0CD0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  v2 = *(_DWORD *)(a2 + 52);
  if ( v2 > 6 )
  {
    if ( v2 - 7 <= 1 )
    {
      *(_BYTE *)(a1 + 4) = 1;
      *(_DWORD *)a1 = 1;
      return a1;
    }
    goto LABEL_5;
  }
  if ( v2 <= 4 )
  {
LABEL_5:
    *(_BYTE *)(a1 + 4) = 0;
    return a1;
  }
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = 0;
  return a1;
}
