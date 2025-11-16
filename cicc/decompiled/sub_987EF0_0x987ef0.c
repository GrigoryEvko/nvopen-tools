// Function: sub_987EF0
// Address: 0x987ef0
//
__int64 __fastcall sub_987EF0(__int64 a1, int a2)
{
  int v2; // esi
  __int64 result; // rax

  v2 = *(_DWORD *)a1 & ~a2;
  result = v2 & 0x3FF;
  *(_DWORD *)a1 = result;
  if ( (v2 & 3) == 0 && !*(_BYTE *)(a1 + 5) )
  {
    if ( (v2 & 0x3C) != 0 )
    {
      if ( (v2 & 0x3C0) == 0 )
      {
        result = 257;
        *(_WORD *)(a1 + 4) = 257;
      }
    }
    else
    {
      *(_WORD *)(a1 + 4) = 256;
    }
  }
  return result;
}
