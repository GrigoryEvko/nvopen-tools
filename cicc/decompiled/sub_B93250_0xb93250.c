// Function: sub_B93250
// Address: 0xb93250
//
__int64 __fastcall sub_B93250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 1) & 0x7F;
  if ( (_BYTE)result != 2 && (*(_DWORD *)(a1 - 8))-- == 1 )
    return sub_B93110(a1, a2, a3, a4, a5);
  return result;
}
