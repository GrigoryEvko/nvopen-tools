// Function: sub_8D3FE0
// Address: 0x8d3fe0
//
_BOOL8 __fastcall sub_8D3FE0(__int64 a1)
{
  __int64 i; // rbx
  _BOOL4 v2; // r8d
  _BOOL8 result; // rax
  char v4; // dl

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = sub_8D2530(i);
  result = 1;
  if ( !v2 )
  {
    v4 = *(_BYTE *)(i + 140);
    result = v4 == 7;
    if ( v4 == 6 )
      return *(_BYTE *)(i + 168) & 1;
  }
  return result;
}
