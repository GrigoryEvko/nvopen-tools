// Function: sub_10953C0
// Address: 0x10953c0
//
__int64 __fastcall sub_10953C0(__int64 *a1, unsigned int a2, char a3)
{
  __int64 v5; // rsi
  __int64 i; // rax
  unsigned __int8 v7; // dl

  v5 = 0;
  for ( i = *a1 + 1; ; ++i )
  {
    v7 = *(_BYTE *)(i - 1);
    if ( (unsigned __int8)(v7 - 48) > 9u )
    {
      if ( !v5 )
        v5 = i - 1;
      if ( !a3 )
        goto LABEL_8;
      if ( word_3F64060[v7] == 0xFFFF )
        break;
    }
  }
  if ( (v7 & 0xDF) != 0x48 )
  {
LABEL_8:
    *a1 = v5;
    return a2;
  }
  *a1 = i - 1;
  return 16;
}
