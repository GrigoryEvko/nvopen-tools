// Function: sub_6426B0
// Address: 0x6426b0
//
__int64 __fastcall sub_6426B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 140) & 0xFB;
  if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
  {
    result = sub_8D4C10(a1, dword_4F077C4 != 2);
    if ( (_DWORD)result )
      return sub_684B30(21, a2);
  }
  return result;
}
