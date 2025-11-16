// Function: sub_615370
// Address: 0x615370
//
__int64 sub_615370()
{
  __int64 result; // rax

  if ( (unsigned __int8)byte_4CF816F
     | (unsigned __int8)(byte_4CF816E | byte_4CF8169 | byte_4CF8163 | byte_4CF8157 | byte_4CF813F) )
  {
    goto LABEL_6;
  }
  if ( !dword_4F077B4 )
  {
    if ( !dword_4F077B8 )
      goto LABEL_4;
    if ( qword_4F077A8 > 0x1ADAFu )
    {
      unk_4F07778 = 201703;
      goto LABEL_6;
    }
    if ( qword_4F077A8 <= 0xEA5Fu )
      goto LABEL_4;
LABEL_45:
    unk_4F07778 = 201402;
    goto LABEL_6;
  }
  if ( unk_4F077A0 > 0xEA5Fu )
    goto LABEL_45;
LABEL_4:
  if ( !unk_4F07778 )
    unk_4F07778 = 199711;
LABEL_6:
  unk_4D0436C = 0;
  unk_4D042D8 = unk_4F069A0;
  unk_4D042DC = 1;
  if ( !byte_4CF810B )
    unk_4D0475C = 0;
  if ( !byte_4CF810F )
    unk_4D047EC = 0;
  unk_4D047E8 = 0;
  unk_4D04798 = 0;
  if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 && !byte_4CF811C )
  {
    unk_4D04798 = 1;
    unk_4D04790 = 1;
  }
  if ( byte_4CF811D )
    sub_6849E0(958);
  unk_4D04794 = 0;
  if ( byte_4CF8121 )
    sub_6849E0(974);
  unk_4D0477C = 0;
  if ( byte_4CF8136 )
    sub_6849E0(1343);
  if ( byte_4CF8153 )
    sub_6849E0(3300);
  unk_4D042A8 = 1;
  unk_4D042A0 = 1;
  unk_4D04234 = 1;
  if ( !byte_4CF8156 && !dword_4F077BC )
    unk_4D04408 = 0;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
    sub_60D650(1u);
  }
  else if ( byte_4CF8157 )
  {
    sub_60D650(0);
  }
  else if ( !((unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C) | dword_4F077BC | dword_4D04964) )
  {
    unk_4F07770 = 0;
    unk_4F07768 = 1;
    unk_4F0776C = 1;
    unk_4D04788 = 1;
    unk_4D04784 = 1;
    unk_4D04218 = 1;
    unk_4D04210 = 1;
  }
  if ( unk_4D0448C )
    unk_4D04488 = 1;
  if ( byte_4CF814F )
    sub_6849E0(1852);
  result = (__int64)&dword_4D04360;
  if ( dword_4D04360 )
  {
    if ( byte_4CF8158 )
      sub_6849E0(2359);
    dword_4D04360 = 0;
  }
  if ( !byte_4CF8172 )
  {
    result = (__int64)&dword_4D041AC;
    dword_4D041AC = 1;
  }
  if ( !byte_4CF8108 )
  {
    result = 1;
    if ( dword_4F077C4 == 2 )
    {
      result = 0;
      if ( unk_4F07778 <= 201102 )
        result = dword_4F07774 == 0;
    }
    unk_4D04338 = result;
  }
  return result;
}
