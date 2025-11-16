// Function: sub_14E9610
// Address: 0x14e9610
//
const char *__fastcall sub_14E9610(unsigned int a1)
{
  const char *result; // rax

  if ( a1 == 4 )
    return "DW_MACINFO_end_file";
  if ( a1 > 4 )
  {
    result = "DW_MACINFO_vendor_ext";
    if ( a1 != 255 )
    {
      if ( a1 != -1 )
        return 0;
      return "DW_MACINFO_invalid";
    }
  }
  else if ( a1 == 2 )
  {
    return "DW_MACINFO_undef";
  }
  else
  {
    result = "DW_MACINFO_start_file";
    if ( a1 != 3 )
    {
      result = "DW_MACINFO_define";
      if ( a1 != 1 )
        return 0;
    }
  }
  return result;
}
