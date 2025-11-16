// Function: sub_34E26C0
// Address: 0x34e26c0
//
__int64 __fastcall sub_34E26C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 == 5 )
  {
    if ( *(_DWORD *)a1 == 1634166092 && *(_BYTE *)(a1 + 4) == 108 )
      return 0;
  }
  else if ( a2 == 7 )
  {
    if ( *(_DWORD *)a1 == 1668507972 && *(_WORD *)(a1 + 4) == 29281 && *(_BYTE *)(a1 + 6) == 100 )
    {
      return 1;
    }
    else if ( *(_DWORD *)a1 == 1986948931 && *(_WORD *)(a1 + 4) == 29285 && *(_BYTE *)(a1 + 6) == 116 )
    {
      return 2;
    }
  }
  return result;
}
