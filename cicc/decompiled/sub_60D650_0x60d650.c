// Function: sub_60D650
// Address: 0x60d650
//
_DWORD *__fastcall sub_60D650(unsigned int a1)
{
  unsigned int v1; // eax
  unsigned int v3; // esi

  if ( !byte_4CF80F0 )
    unk_4D04388 = 1;
  if ( !byte_4CF807B )
    dword_4D048B8 = a1;
  unk_4F07770 = a1;
  unk_4F0776C = a1;
  unk_4F07768 = a1;
  if ( !byte_4CF8132 )
    unk_4D04298 = a1;
  unk_4D04294 = a1;
  if ( !byte_4CF811E )
    unk_4D04788 = a1;
  unk_4D04784 = a1;
  unk_4F07764 = a1;
  if ( !byte_4CF814B )
    unk_4F0775C = a1;
  if ( !byte_4CF814C )
    unk_4F07758 = a1 ^ 1;
  if ( !byte_4CF8123 )
    unk_4D047C8 = a1;
  if ( !byte_4CF8125 )
    dword_4D047B0 = a1;
  unk_4D04218 = a1;
  unk_4D04210 = a1;
  unk_4F07754 = a1;
  unk_4F07740 = a1;
  unk_4D044A8 = a1;
  unk_4D044A0 = a1;
  unk_4D0449C = a1;
  if ( !byte_4CF8147 )
    unk_4D0448C = a1;
  if ( byte_4CF8148 )
  {
    v1 = unk_4D04474;
  }
  else
  {
    unk_4D04474 = a1;
    v1 = a1;
  }
  if ( v1 )
  {
    if ( !unk_4D04470 || !byte_4CF8149 )
    {
      dword_4D0446C = 1;
      unk_4D04470 = 0;
    }
  }
  else
  {
    unk_4D04470 = 0;
    dword_4D0446C = 0;
  }
  unk_4D04440 = a1;
  unk_4D04438 = a1;
  if ( !byte_4CF815A )
    dword_4D04434 = a1;
  unk_4D04468 = a1;
  dword_4D04464 = a1;
  unk_4D04430 = a1;
  unk_4D0442C = a1;
  dword_4D04428 = a1;
  unk_4D04424 = a1;
  dword_4D043F8 = a1;
  unk_4D043F4 = a1;
  unk_4D043F0 = a1;
  unk_4D0440C = a1;
  unk_4D043E4 = a1;
  if ( !byte_4CF8156 )
    unk_4D04408 = a1;
  if ( !byte_4CF814E )
    unk_4F0773C = a1;
  if ( !byte_4CF8154 )
    unk_4F07734 = a1;
  if ( !byte_4CF8155 && unk_4F07734 )
    unk_4F07730 = 0;
  if ( !byte_4CF8128 )
    dword_4D04278 = 0;
  unk_4D044AC = a1;
  if ( !byte_4CF813D )
    unk_4D043A8 = a1;
  unk_4F07728 = a1;
  if ( byte_4CF8119 )
  {
    dword_4D048B4 = a1;
    if ( dword_4D048B8 && a1 && !byte_4CF8159 )
LABEL_52:
      dword_4D048B0 = 1;
  }
  else
  {
    if ( !a1 )
    {
      dword_4D048B4 = 0;
      goto LABEL_41;
    }
    unk_4D047D0 = 0;
    unk_4D047CC = 0;
    dword_4D048B4 = 1;
    if ( dword_4D048B8 && !byte_4CF8159 )
      goto LABEL_52;
  }
LABEL_41:
  unk_4D048A0 = a1;
  dword_4D0489C = a1;
  unk_4D048A8 = v1;
  word_4D04898 = a1;
  if ( !byte_4CF815F )
    unk_4F0771C = 1;
  if ( !byte_4CF8160 )
    unk_4F07718 = a1;
  unk_4F07710 = a1;
  unk_4D04774 = 1;
  unk_4D04220 = a1;
  unk_4D043E8 = a1;
  unk_4D0487C = a1;
  if ( dword_4F077C4 == 2 && unk_4F07778 > 201401 )
  {
    if ( unk_4F0775C )
      unk_4F07750 = 1;
    unk_4F0774C = 1;
    unk_4D04484 = 1;
    unk_4D04480 = 1;
    unk_4D0447C = 1;
    if ( a1 )
      unk_4D0441C = 1;
    unk_4D0427C = 1;
    if ( !byte_4CF8167 )
      unk_4F0770C = 1;
    v3 = dword_4F077BC;
    if ( v1 )
    {
      if ( !dword_4F077BC )
      {
        if ( !dword_4F077B4 )
          unk_4D04410 = 1;
        dword_4D0488C = 1;
        unk_4D04478 = 1;
        goto LABEL_73;
      }
      dword_4D0488C = 1;
      unk_4D04478 = 1;
    }
    else
    {
      dword_4D0488C = 1;
      unk_4D04478 = 1;
      if ( !dword_4F077BC )
        goto LABEL_72;
    }
    if ( !dword_4F077B4 && qword_4F077A8 <= 0xC34Fu )
    {
      v3 = 1;
LABEL_73:
      unk_4D0487C = v3;
      unk_4D043C8 = 1;
      unk_4D04854 = 1;
      if ( unk_4F07778 > 201702 )
      {
        unk_4D043D8 = 1;
        unk_4D043D4 = 1;
        unk_4D043CC = 1;
        unk_4F07760 = 1;
        if ( !byte_4CF816A )
          unk_4D041F0 = 1;
        if ( a1 )
          unk_4F07724 = 1;
        unk_4D04770 = 1;
        unk_4D0476C = 0;
        unk_4D041EC = 1;
        unk_4D041E8 = 1;
        unk_4D04280 = 1;
        dword_4D0485C = 1;
        unk_4D04858 = 1;
        unk_4D041E4 = 1;
        unk_4D043C4 = 1;
        dword_4D041E0 = 1;
        unk_4D041DC = 1;
        unk_4D04404 = 1;
        unk_4D04400 = 1;
        dword_4D04820 = 1;
        if ( !byte_4CF816C )
          unk_4F06978 = 1;
        if ( !byte_4CF816D )
          unk_4D04818 = 1;
        unk_4D04460 = 1;
        unk_4D0480C = 1;
        unk_4D04810 = 1;
        if ( !byte_4CF8138 )
          unk_4D04384 = 0;
        unk_4D04418 = 1;
        unk_4D04808 = 1;
        unk_4D04804 = 1;
        if ( unk_4F07778 > 202001 )
        {
          unk_4D0483C = 1;
          dword_4D04888 = 1;
          unk_4D04884 = 1;
          unk_4D04880 = 1;
          unk_4D04894 = 1;
          unk_4D04890 = 1;
          if ( !unk_4D04498 )
            unk_4D04498 = 1;
          unk_4D041D8 = 1;
          unk_4D041D4 = 1;
          unk_4D041C4 = 1;
          unk_4D04414 = 0;
          unk_4D041C0 = 1;
          unk_4D04878 = 1;
          unk_4D04874 = 1;
          unk_4D04870 = 1;
          unk_4D0486C = 1;
          unk_4D04868 = 1;
          unk_4D04864 = 1;
          unk_4D04860 = 1;
          unk_4D041BC = 1;
          unk_4D041B8 = 1;
          unk_4D043D0 = 1;
          unk_4D0478C = 1;
          unk_4D041B0 = 1;
          if ( !byte_4CF8171 )
            unk_4D041B4 = 1;
          unk_4D04814 = 1;
          if ( !byte_4CF817D )
            unk_4D04494 = 1;
          dword_4D04490 = 1;
          unk_4D04458 = 1;
          if ( !byte_4CF817B )
            unk_4D0445C = 0;
          unk_4D04274 = 0;
          unk_4D04800 = 1;
          if ( unk_4F07778 > 202301 )
          {
            unk_4D043C0 = 1;
            unk_4D048A4 = 1;
            unk_4D04190 = 1;
            unk_4D04188 = 1;
            unk_4D04184 = 1;
          }
        }
      }
      goto LABEL_46;
    }
LABEL_72:
    v3 = 0;
    goto LABEL_73;
  }
LABEL_46:
  dword_4D04394 = a1;
  return &dword_4D04394;
}
