// Function: sub_667F10
// Address: 0x667f10
//
__int64 __fastcall sub_667F10(__int64 a1)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 132) & 0x10) != 0 )
  {
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(result + 6) &= ~0x80u;
  }
  return result;
}
