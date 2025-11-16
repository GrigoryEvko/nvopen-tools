// Function: sub_60E530
// Address: 0x60e530
//
_DWORD *sub_60E530()
{
  _DWORD *result; // rax

  sub_60DFC0();
  if ( !byte_4CF810F )
    unk_4D047EC = 1;
  unk_4D042DC = 1;
  unk_4D0421C = 1;
  unk_4D04390 = 1;
  if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 || byte_4CF814F || qword_4F077A8 <= 0x9D6Bu )
  {
    unk_4D04374 = 0;
    unk_4D04370 = 1;
    unk_4F072F2 = 1;
  }
  else
  {
    unk_4D04374 = 1;
    unk_4D04370 = 0;
  }
  unk_4D04398 = 1;
  if ( dword_4F077B8 )
  {
    if ( !dword_4F077B4 )
    {
      result = (_DWORD *)qword_4F077A8;
      if ( qword_4F077A8 > 0x9EFBu )
      {
        unk_4D043F0 = 1;
        unk_4D043F4 = 1;
        unk_4F07764 = 1;
LABEL_22:
        unk_4D044C4 = 1;
        if ( (unsigned __int64)result > 0x9FC3 )
        {
          unk_4D04220 = 1;
          unk_4D044D0 = 1;
          if ( (unsigned __int64)result > 0xEA5F )
          {
            unk_4D043CC = 1;
            if ( (unsigned __int64)result > 0x1116F )
            {
              unk_4D0428C = 1;
              if ( (unsigned __int64)result > 0x15F8F )
              {
                dword_4F07760 = 1;
                if ( (unsigned __int64)result > 0x1869F )
                {
                  dword_4D043F8 = 1;
                  dword_4D041E8 = 1;
                  return &dword_4D041E8;
                }
              }
            }
          }
        }
        return result;
      }
LABEL_20:
      if ( (unsigned __int64)result <= 0x9E97 )
        return result;
      unk_4F07764 = 1;
      if ( (unsigned __int64)result <= 0x9EFB )
        return result;
      goto LABEL_22;
    }
  }
  else if ( !dword_4F077B4 )
  {
    result = (_DWORD *)qword_4F077A8;
    goto LABEL_20;
  }
  unk_4D043CC = 1;
  result = (_DWORD *)unk_4F077A0;
  if ( unk_4F077A0 > 0x7593u )
  {
    unk_4D044D0 = 1;
    unk_4D04220 = 1;
    if ( unk_4F077A0 > 0x765Bu )
    {
      unk_4D044C4 = 1;
      if ( unk_4F077A0 > 0x7723u )
      {
        result = &dword_4F07760;
        dword_4F07760 = 1;
      }
    }
  }
  return result;
}
