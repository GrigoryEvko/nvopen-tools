// Function: sub_8D5DF0
// Address: 0x8d5df0
//
_BOOL8 __fastcall sub_8D5DF0(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 v3; // rdx

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  result = 1;
  if ( a2 != a1 )
  {
    if ( !dword_4F07588 )
      return sub_8D5CE0(a1, a2) != 0;
    v3 = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a2 + 32) != v3 || !v3 )
      return sub_8D5CE0(a1, a2) != 0;
  }
  return result;
}
