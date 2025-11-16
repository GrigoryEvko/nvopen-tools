// Function: sub_AF33B0
// Address: 0xaf33b0
//
const char *__fastcall sub_AF33B0(unsigned int a1)
{
  const char *result; // rax

  result = "LineTablesOnly";
  if ( a1 != 2 )
  {
    if ( a1 > 2 )
    {
      result = "DebugDirectivesOnly";
      if ( a1 != 3 )
        return 0;
    }
    else
    {
      result = "FullDebug";
      if ( !a1 )
        return "NoDebug";
    }
  }
  return result;
}
