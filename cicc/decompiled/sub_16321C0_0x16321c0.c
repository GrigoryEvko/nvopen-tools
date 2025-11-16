// Function: sub_16321C0
// Address: 0x16321c0
//
__int64 __fastcall sub_16321C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax

  result = sub_1632000(a1, a2, a3);
  if ( result )
  {
    if ( *(_BYTE *)(result + 16) == 3 )
    {
      if ( !a4 && (*(_BYTE *)(result + 32) & 0xFu) - 7 <= 1 )
        return 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
