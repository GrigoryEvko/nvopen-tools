// Function: sub_60F990
// Address: 0x60f990
//
char *sub_60F990()
{
  char v0; // dl
  char *result; // rax
  _BOOL4 v2; // edx
  _BOOL4 v3; // edx

  if ( unk_4D04950 )
  {
    if ( byte_4CF806B )
      goto LABEL_129;
    unk_4D04950 = 0;
  }
  if ( !unk_4D0435C )
    goto LABEL_7;
  if ( byte_4CF80F5 )
LABEL_129:
    sub_6849E0(600);
  unk_4D0435C = 0;
LABEL_7:
  if ( unk_4D0475C )
  {
    if ( byte_4CF810B )
      sub_6849E0(814);
    unk_4D0475C = 0;
  }
  if ( !byte_4CF8138 )
  {
    if ( dword_4F077C4 != 2 )
    {
      if ( unk_4F07778 > 202310 )
        goto LABEL_14;
      goto LABEL_13;
    }
    if ( unk_4F07778 <= 201702 )
LABEL_13:
      unk_4D04384 = 1;
  }
LABEL_14:
  if ( !byte_4CF811D )
    unk_4D04794 = 0;
  if ( !byte_4CF811F )
    unk_4D04780 = 0;
  if ( !byte_4CF8073 )
    unk_4D04748 = 0;
  if ( byte_4CF813D )
    goto LABEL_33;
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || dword_4F07774 )
    {
      if ( !byte_4CF8145 )
        goto LABEL_49;
LABEL_50:
      unk_4D04308 = 0;
      unk_4D04800 = unk_4F07778 > 202001;
      unk_4D04764 = 0;
      if ( !byte_4CF80F0 )
        unk_4D04388 = 1;
      if ( !byte_4CF80EE )
        unk_4D043A4 = 1;
      if ( !byte_4CF80F9 )
        unk_4D0439C = 1;
      if ( !byte_4CF80ED )
        unk_4D044B4 = 0;
      if ( !byte_4CF80F7 )
        unk_4D0484C = 1;
      if ( !byte_4CF80FA )
        unk_4D04844 = 1;
      if ( !byte_4CF80FB )
        unk_4D04840 = 1;
      if ( !byte_4CF80FC )
        unk_4D04838 = 1;
      if ( !byte_4CF8100 )
        unk_4D0482C = 0;
      if ( !byte_4CF80FF )
        unk_4D04830 = 1;
      if ( !byte_4CF8101 )
        unk_4D04760 = 0;
      if ( !byte_4CF8104 )
        unk_4D04350 = 0;
      if ( !byte_4CF8105 )
        unk_4D04348 = 0;
      if ( !byte_4CF8107 )
        unk_4D04340 = 0;
      if ( !byte_4CF8108 )
        unk_4D04338 = 0;
      if ( !byte_4CF810A )
        unk_4D04334 = 0;
      if ( !byte_4CF810C )
        dword_4D04824 = 1;
      if ( !byte_4CF8110 )
        unk_4D047E4 = 1;
      if ( !byte_4CF8116 )
        unk_4D047E0 = 1;
      if ( !byte_4CF815F )
      {
        v2 = 0;
        if ( unk_4F07778 <= 201102 )
          v2 = dword_4F07774 == 0;
        unk_4F0771C = v2;
      }
      unk_4F06970 = 1;
      if ( !byte_4CF8117 )
        unk_4D047DC = 1;
      if ( !byte_4CF8118 )
        unk_4D047D8 = 1;
      if ( !byte_4CF8119 )
      {
        unk_4D047D0 = 0;
        unk_4D047CC = 0;
      }
      if ( !byte_4CF8123 )
        unk_4D047C8 = 1;
      v3 = 1;
      if ( unk_4F07778 <= 201102 )
        v3 = dword_4F07774 != 0;
      unk_4D047D4 = v3;
      if ( !byte_4CF8125 )
        dword_4D047B0 = 1;
      if ( !byte_4CF8128 )
        dword_4D04278 = 0;
      if ( !byte_4CF811A )
        unk_4D0479C = 0;
      if ( !byte_4CF8111 )
        unk_4D04318 = 0;
      if ( !byte_4CF8139 )
        unk_4D04314 = 0;
      if ( !byte_4CF814D )
        unk_4D047C4 = 0;
      if ( !byte_4CF8112 )
        unk_4D0430C = 0;
      if ( !byte_4CF8122 )
        unk_4D04344 = 0;
      if ( !byte_4CF807B )
        dword_4D048B8 = 1;
      if ( !byte_4CF8154 && unk_4F07778 <= 201102 && !dword_4F07774 )
      {
        unk_4F07734 = 0;
        unk_4F07730 = 0;
      }
      if ( unk_4D042AC )
      {
        if ( byte_4CF8124 )
          sub_6849E0(1408);
        unk_4D042AC = 0;
      }
      goto LABEL_27;
    }
    goto LABEL_32;
  }
  if ( unk_4F07778 <= 201111 )
  {
LABEL_32:
    unk_4D043A8 = 0;
LABEL_33:
    if ( !byte_4CF8145 )
      goto LABEL_49;
    if ( dword_4F077C4 != 2 )
      goto LABEL_35;
    goto LABEL_50;
  }
  if ( byte_4CF8145 )
  {
    v0 = byte_4CF80F0;
    goto LABEL_25;
  }
LABEL_49:
  unk_4D0420C = 1;
  if ( dword_4F077C4 == 2 )
    goto LABEL_50;
LABEL_35:
  v0 = byte_4CF80F0;
  if ( unk_4F07778 > 199900 )
  {
LABEL_25:
    if ( !v0 )
      unk_4D04388 = 1;
    goto LABEL_27;
  }
  unk_4D042A4 = 0;
  if ( !byte_4CF811E )
    unk_4D04788 = 0;
  if ( !byte_4CF80EC )
    unk_4D044C8 = 0;
  if ( !byte_4CF80F0 )
    unk_4D04388 = 1;
  if ( !byte_4CF810F )
    unk_4D047EC = 0;
  if ( !byte_4CF811C )
    unk_4D04798 = 0;
  if ( !byte_4CF8121 )
    unk_4D0477C = 0;
LABEL_27:
  result = (char *)&dword_4D0443C;
  dword_4D0443C = 0;
  if ( !byte_4CF8162 )
  {
    result = (char *)&qword_4D043AC + 4;
    HIDWORD(qword_4D043AC) = 0;
  }
  return result;
}
