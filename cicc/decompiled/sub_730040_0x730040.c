// Function: sub_730040
// Address: 0x730040
//
__int64 __fastcall sub_730040(char a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  char i; // bl
  unsigned int v6; // r12d
  __int64 v7; // [rsp+8h] [rbp-28h]

  v3 = *(unsigned __int8 *)(a2 + 140);
  if ( (_BYTE)v3 != 12 )
    goto LABEL_5;
  do
  {
    a2 = *(_QWORD *)(a2 + 160);
    v3 = *(unsigned __int8 *)(a2 + 140);
  }
  while ( (_BYTE)v3 == 12 );
  for ( i = *(_BYTE *)(a3 + 140); i == 12; i = *(_BYTE *)(a3 + 140) )
  {
    a3 = *(_QWORD *)(a3 + 160);
LABEL_5:
    ;
  }
  if ( (unsigned __int8)(a1 - 87) <= 1u )
    return 2;
  if ( a1 == 91 )
    return 21;
  if ( i == (_BYTE)v3 )
  {
    if ( i == 4 )
    {
      if ( a1 == 41 )
        return 4;
      return 3;
    }
    if ( (v3 & 0xFD) == 9 )
      return 10;
    return v3;
  }
  else
  {
    if ( (_BYTE)v3 && i )
    {
      if ( (_BYTE)v3 == 14 )
        return 14;
      if ( i == 14 )
        return 14;
      v7 = a3;
      if ( (unsigned int)sub_8DBE70(a2) || (unsigned int)sub_8DBE70(v7) )
        return 14;
      if ( (_BYTE)v3 == 6 || i == 6 )
        return 6;
      if ( (_BYTE)v3 == 13 || i == 13 )
        return 13;
      if ( (_BYTE)v3 != 4 && i != 4 )
      {
        if ( (_BYTE)v3 == 15 || i == 15 )
        {
          return 15;
        }
        else
        {
          if ( (_BYTE)v3 != 19 && i != 19 )
            sub_721090();
          return 19;
        }
      }
      if ( (_BYTE)v3 == 5 || i == 5 )
        return 5;
      v6 = 5;
      if ( (unsigned __int8)(a1 - 46) > 3u )
      {
        if ( i == 4 && (a1 == 42 || a1 == 77) )
          return 4;
        return 3;
      }
      return v6;
    }
    return 0;
  }
}
