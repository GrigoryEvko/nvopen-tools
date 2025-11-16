// Function: sub_8D4E00
// Address: 0x8d4e00
//
_BOOL8 __fastcall sub_8D4E00(__int64 a1)
{
  __int64 v1; // r12
  char v2; // r8
  _BOOL8 result; // rax

  v1 = a1;
  if ( sub_8D3410(a1) )
    v1 = sub_8D40F0(a1);
  if ( (*(_BYTE *)(v1 + 140) & 0xFB) != 8 || (v2 = sub_8D4C10(v1, dword_4F077C4 != 2), result = 1, (v2 & 2) == 0) )
  {
    result = sub_8D3A70(v1);
    if ( result )
    {
      for ( ; *(_BYTE *)(v1 + 140) == 12; v1 = *(_QWORD *)(v1 + 160) )
        ;
      return (*(_BYTE *)(v1 + 176) & 4) != 0;
    }
  }
  return result;
}
