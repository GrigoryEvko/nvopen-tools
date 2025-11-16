// Function: sub_7AFEC0
// Address: 0x7afec0
//
_DWORD *__fastcall sub_7AFEC0(char a1)
{
  _DWORD *result; // rax

  result = (_DWORD *)unk_4F064B0;
  if ( unk_4F064B0 && (*(_BYTE *)(unk_4F064B0 + 88LL) & 1) != 0 )
  {
    *(_BYTE *)(unk_4F064B0 + 89LL) = a1;
    if ( a1 == 1 )
    {
      dword_4F064B8[0] = 0;
      return dword_4F064B8;
    }
  }
  return result;
}
