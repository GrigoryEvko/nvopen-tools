// Function: sub_8D25A0
// Address: 0x8d25a0
//
_BOOL8 __fastcall sub_8D25A0(__int64 a1)
{
  __int64 i; // rbx
  _BOOL8 result; // rax

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = sub_8D2530(i);
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(i + 141) & 0x20) != 0 )
    return 0;
  return result;
}
