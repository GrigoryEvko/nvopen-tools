// Function: sub_B6EAD0
// Address: 0xb6ead0
//
char *__fastcall sub_B6EAD0(char a1)
{
  if ( a1 != 2 )
  {
    if ( a1 > 2 )
    {
      if ( a1 == 3 )
        return "note";
    }
    else
    {
      if ( !a1 )
        return "error";
      if ( a1 == 1 )
        return "warning";
    }
    BUG();
  }
  return "remark";
}
