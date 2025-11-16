// Function: sub_392A370
// Address: 0x392a370
//
__int64 __fastcall sub_392A370(__int64 *a1, unsigned int a2)
{
  __int64 v3; // rsi
  __int64 i; // rax
  char v5; // dl
  char v6; // dl

  v3 = 0;
  for ( i = *a1 + 1; ; ++i )
  {
    v5 = *(_BYTE *)(i - 1);
    if ( (unsigned __int8)(v5 - 48) <= 9u )
      continue;
    v6 = v5 & 0xDF;
    if ( (unsigned __int8)(v6 - 65) > 5u )
      break;
    if ( !v3 )
      v3 = i - 1;
  }
  if ( !v3 || v6 == 72 )
  {
    *a1 = i - 1;
    if ( v6 == 72 )
      return 16;
    return a2;
  }
  else
  {
    *a1 = v3;
    return a2;
  }
}
