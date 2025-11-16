// Function: sub_64EBC0
// Address: 0x64ebc0
//
__int64 __fastcall sub_64EBC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _BOOL4 v4; // r8d

  if ( (*(_QWORD *)(a1 + 16) & 0x430LL) == 0 && (*(_BYTE *)(a3 + 16) & 0x10) == 0 && (*(_BYTE *)(a1 + 130) & 0x20) == 0 )
  {
    result = dword_4F077C0;
    if ( !dword_4F077C0
      || (result = (__int64)&qword_4F077A8, qword_4F077A8 > 0x7593u)
      || (result = (__int64)word_4F06418, word_4F06418[0] != 56)
      || *(_BYTE *)(a1 + 269) != 4 )
    {
      if ( (*(_BYTE *)(a3 + 17) & 0x20) == 0 )
      {
        v4 = 0;
        if ( a2 )
          v4 = (*(_BYTE *)(a2 + 64) & 0x20) != 0;
        return sub_64E990(a1 + 40, *(_QWORD *)(a1 + 288), a2 != 0, 0, v4, ((*(_BYTE *)(a1 + 126) >> 3) ^ 1) & 1);
      }
    }
  }
  return result;
}
