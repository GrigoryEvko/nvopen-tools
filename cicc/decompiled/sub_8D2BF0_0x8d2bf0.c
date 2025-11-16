// Function: sub_8D2BF0
// Address: 0x8d2bf0
//
_BOOL8 __fastcall sub_8D2BF0(__int64 a1)
{
  __int64 i; // rbx
  _BOOL4 v2; // r8d
  _BOOL8 result; // rax

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = sub_8D2BB0(i);
  result = 1;
  if ( !v2 )
    return *(_BYTE *)(i + 140) == 18;
  return result;
}
