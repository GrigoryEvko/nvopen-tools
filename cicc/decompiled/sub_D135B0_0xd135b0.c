// Function: sub_D135B0
// Address: 0xd135b0
//
__int64 __fastcall sub_D135B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( **(_BYTE **)(a2 + 24) != 30 || (result = 2, *(_BYTE *)(a1 + 8)) )
  {
    *(_BYTE *)(a1 + 9) = 1;
    return 0;
  }
  return result;
}
