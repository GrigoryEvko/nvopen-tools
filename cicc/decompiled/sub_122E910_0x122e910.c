// Function: sub_122E910
// Address: 0x122e910
//
__int64 __fastcall sub_122E910(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  while ( *(_DWORD *)(a1 + 240) == 511 )
  {
    result = sub_122E8D0(a1, a2);
    if ( (_BYTE)result )
      return result;
  }
  return 0;
}
