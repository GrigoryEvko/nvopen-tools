// Function: sub_6729D0
// Address: 0x6729d0
//
__int64 __fastcall sub_6729D0(__int64 a1)
{
  __int64 result; // rax

  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(result + 4) == 6 && (*(_BYTE *)(result + 6) & 0x82) == 2 )
  {
    *(_BYTE *)(result + 6) |= 0x80u;
    return sub_643E40((__int64)sub_667F10, a1, 1);
  }
  return result;
}
