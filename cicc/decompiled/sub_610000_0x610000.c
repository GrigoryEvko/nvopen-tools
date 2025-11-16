// Function: sub_610000
// Address: 0x610000
//
unsigned int *sub_610000()
{
  unsigned int *result; // rax

  result = &dword_4D0446C;
  if ( !unk_4D04474 )
  {
    if ( dword_4D0446C )
    {
      if ( !byte_4CF8148 )
      {
        if ( byte_4CF814A )
        {
LABEL_5:
          unk_4D04474 = 1;
          goto LABEL_6;
        }
        if ( !unk_4D04470 )
        {
LABEL_15:
          dword_4D0446C = 0;
          unk_4D04470 = 0;
          return result;
        }
LABEL_22:
        if ( byte_4CF8149 )
          goto LABEL_5;
        goto LABEL_15;
      }
      if ( byte_4CF814A )
LABEL_24:
        sub_6849E0(2325);
      if ( !unk_4D04470 )
        goto LABEL_15;
    }
    else
    {
      if ( !unk_4D04470 )
        return result;
      if ( !byte_4CF8148 )
        goto LABEL_22;
    }
    if ( !byte_4CF8149 )
      goto LABEL_15;
    goto LABEL_24;
  }
LABEL_6:
  if ( dword_4D0446C && unk_4D04470 )
  {
    if ( byte_4CF814A )
    {
      if ( byte_4CF8149 )
        sub_6849E0(2324);
    }
    else if ( byte_4CF8149 )
    {
      dword_4D0446C = 0;
      return result;
    }
    unk_4D04470 = 0;
  }
  return result;
}
