// Function: sub_16027C0
// Address: 0x16027c0
//
char *__fastcall sub_16027C0(char a1)
{
  char *result; // rax

  result = "remark";
  if ( a1 != 2 )
  {
    result = "note";
    if ( a1 <= 2 )
    {
      result = "warning";
      if ( !a1 )
        return "error";
    }
  }
  return result;
}
