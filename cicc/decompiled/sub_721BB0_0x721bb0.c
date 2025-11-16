// Function: sub_721BB0
// Address: 0x721bb0
//
char *__fastcall sub_721BB0(char *a1, int a2)
{
  char v2; // al
  char *v3; // rbx

  v2 = *a1;
  if ( !*a1 )
    return 0;
  v3 = a1;
  while ( v2 != a2 )
  {
    if ( v2 >= 0 )
    {
      v2 = *++v3;
      if ( !*v3 )
        return 0;
    }
    else
    {
      v3 += (int)sub_721AB0(v3, 0, 0);
      v2 = *v3;
      if ( !*v3 )
        return 0;
    }
  }
  return v3;
}
