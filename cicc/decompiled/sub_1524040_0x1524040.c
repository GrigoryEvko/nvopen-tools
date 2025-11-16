// Function: sub_1524040
// Address: 0x1524040
//
__int64 __fastcall sub_1524040(char *a1, __int64 a2)
{
  char *v2; // rsi
  char v3; // al
  char *v4; // rdi
  char v5; // al
  char *v6; // rdi

  v2 = &a1[a2];
  if ( a1 == v2 )
    return 0;
  v3 = *a1;
  v4 = a1 + 1;
  while ( (unsigned __int8)(v3 - 97) <= 0x19u
       || (unsigned __int8)(v3 - 65) <= 0x19u
       || (unsigned __int8)(v3 - 48) <= 9u
       || v3 == 46
       || v3 == 95 )
  {
    if ( v2 == v4 )
      return 0;
    v3 = *v4++;
  }
  if ( v3 < 0 )
    return 2;
  if ( v2 != v4 )
  {
    v5 = *v4;
    v6 = v4 + 1;
    while ( v5 >= 0 )
    {
      if ( v2 == v6 )
        return 1;
      v5 = *v6++;
    }
    return 2;
  }
  return 1;
}
