// Function: sub_917A60
// Address: 0x917a60
//
__int64 __fastcall sub_917A60(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  char i; // dl
  __int64 result; // rax

  if ( !a1 )
    return 0;
  v1 = a1;
  while ( (*(_BYTE *)(v1 + 144) & 4) == 0 )
  {
    v2 = *(_QWORD *)(v1 + 120);
    for ( i = *(_BYTE *)(v2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
      v2 = *(_QWORD *)(v2 + 160);
    while ( i == 8 )
    {
      do
      {
        v2 = *(_QWORD *)(v2 + 160);
        i = *(_BYTE *)(v2 + 140);
      }
      while ( i == 12 );
    }
    if ( (unsigned __int8)(i - 10) <= 1u )
    {
      result = sub_917A60(*(_QWORD *)(v2 + 160));
      if ( (_BYTE)result )
        return result;
    }
    v1 = *(_QWORD *)(v1 + 112);
    if ( !v1 )
      return 0;
  }
  return 1;
}
