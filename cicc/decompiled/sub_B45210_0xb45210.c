// Function: sub_B45210
// Address: 0xb45210
//
__int64 __fastcall sub_B45210(__int64 a1)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 1) >> 1;
  if ( (_DWORD)result == 127 )
    return 0xFFFFFFFFLL;
  return result;
}
