// Function: sub_141FE00
// Address: 0x141fe00
//
__int64 __fastcall sub_141FE00(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (_DWORD)result == 21 )
  {
    *(_DWORD *)(a2 + 84) = -1;
  }
  else if ( (_DWORD)result == 22 )
  {
    *(_DWORD *)(a2 + 88) = -1;
  }
  return result;
}
