// Function: sub_15B0FC0
// Address: 0x15b0fc0
//
const char *__fastcall sub_15B0FC0(unsigned int a1)
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
