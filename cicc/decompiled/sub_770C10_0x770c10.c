// Function: sub_770C10
// Address: 0x770c10
//
__int64 __fastcall sub_770C10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // al
  __int64 v4; // rdx

  if ( !qword_4F04C50 || (result = *(_QWORD *)(qword_4F04C50 + 32LL), (*(_BYTE *)(result + 193) & 4) == 0) )
  {
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(result + 12) & 0x10) == 0 )
    {
      v3 = *(_BYTE *)(a1 + 89);
      v4 = 0;
      if ( (v3 & 0x40) == 0 )
      {
        if ( (v3 & 8) != 0 )
          v4 = *(_QWORD *)(a1 + 24);
        else
          v4 = *(_QWORD *)(a1 + 8);
      }
      return sub_684B10(0xBF2u, (_DWORD *)(a2 + 28), v4);
    }
  }
  return result;
}
