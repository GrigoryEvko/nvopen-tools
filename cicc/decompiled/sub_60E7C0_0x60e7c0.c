// Function: sub_60E7C0
// Address: 0x60e7c0
//
char sub_60E7C0()
{
  _DWORD *v0; // rax
  unsigned int v1; // edx
  _BOOL4 v2; // esi
  unsigned int v3; // esi
  int v4; // esi
  unsigned int v5; // esi
  bool v6; // si
  unsigned int v7; // edi
  int v8; // esi
  bool v9; // si
  bool v10; // si
  int v11; // esi
  unsigned __int64 v12; // rdx
  _BOOL4 v13; // esi

  sub_60DFC0();
  if ( !byte_4CF807B )
    dword_4D048B8 = 1;
  if ( !byte_4CF810C )
    dword_4D04824 = 1;
  if ( !byte_4CF80F0 )
    unk_4D04388 = 1;
  LOBYTE(v0) = qword_4F077A8;
  if ( !byte_4CF8108 )
  {
    v1 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      v1 = 0;
      if ( !dword_4F077B4 )
        v1 = qword_4F077A8 <= 0x76BFu;
    }
    unk_4D04338 = v1;
  }
  if ( qword_4F077A8 > 0x76BFu )
  {
    if ( !byte_4CF8123 )
      unk_4D047C8 = 1;
    if ( !byte_4CF8125 )
      dword_4D047B0 = 1;
  }
  if ( !byte_4CF8139 )
    unk_4D04314 = qword_4F077A8 <= 0x9D6Bu;
  if ( !byte_4CF814D )
    unk_4D047C4 = 0;
  if ( !byte_4CF8119 )
  {
    unk_4D047D0 = qword_4F077A8 <= 0x9C40u;
    unk_4D047CC = qword_4F077A8 <= 0x9CA3u;
  }
  if ( !byte_4CF810F )
    unk_4D047EC = 1;
  unk_4D047C0 = 1;
  unk_4D047BC = qword_4F077A8 <= 0x1D4BFu;
  v2 = 0;
  if ( qword_4F077A8 <= 0x9DCFu )
    v2 = (unk_4D0448C | unk_4D04430) == 0;
  unk_4D047B8 = v2;
  unk_4D044B8 = qword_4F077A8 <= 0x9D6Bu;
  unk_4D047B4 = 1;
  unk_4F07584 = 0;
  if ( !byte_4CF8107 )
    unk_4D04340 = 0;
  if ( !byte_4CF8116 )
    unk_4D047E0 = 1;
  if ( !byte_4CF815F )
    unk_4F0771C = 1;
  if ( byte_4CF8160 )
  {
    v3 = unk_4F07718;
    if ( !unk_4F07718 )
      goto LABEL_36;
    v3 = 0;
    if ( dword_4F077C4 != 2 )
      goto LABEL_36;
  }
  else
  {
    if ( dword_4F077C4 != 2 )
    {
      unk_4F07718 = 0;
      v3 = 0;
      goto LABEL_36;
    }
    if ( unk_4F07778 <= 201102 )
    {
      v3 = dword_4F07774;
      if ( !dword_4F07774 )
      {
        unk_4F07718 = 0;
        goto LABEL_36;
      }
    }
    unk_4F07718 = qword_4F077A8 > 0x9EFBu;
    if ( qword_4F077A8 <= 0x9EFBu )
    {
      v3 = 0;
      goto LABEL_36;
    }
  }
  if ( unk_4F07778 > 201401 )
  {
    if ( dword_4F077B8 )
    {
      LOBYTE(v4) = qword_4F077A8 > 0x9FC3u && dword_4F077B4 == 0;
      if ( (_BYTE)v4 )
      {
        unk_4F07700 = 1;
        goto LABEL_158;
      }
    }
    goto LABEL_232;
  }
  if ( unk_4F07778 > 201102 || (v3 = dword_4F07774) != 0 )
  {
LABEL_232:
    v3 = 0;
    if ( dword_4F077B4 )
      v3 = unk_4F077A0 > 0x76BFu;
  }
LABEL_36:
  unk_4F07700 = v3;
  v4 = 0;
  if ( qword_4F077A8 <= 0x9F5Fu )
    goto LABEL_37;
  LOBYTE(v4) = dword_4F077B4 == 0;
LABEL_158:
  v4 = (unsigned __int8)v4;
LABEL_37:
  unk_4F07714 = v4;
  if ( !byte_4CF813E )
    unk_4D044B0 = qword_4F077A8 > 0x9D6Bu;
  unk_4F06904 = 0;
  unk_4D0423C = 1;
  if ( qword_4F077A8 <= 0x1ADAFu || dword_4F077C4 != 2 || unk_4F07778 <= 202001 )
    unk_4D04800 = 0;
  unk_4D04208 = 0;
  unk_4D04758 = qword_4F077A8 <= 0x765Cu;
  unk_4D04764 = qword_4F077A8 <= 0x76BFu;
  unk_4D04218 = 1;
  unk_4D04214 = 1;
  unk_4F06970 = 0;
  unk_4F06964 = 0;
  if ( !((unsigned __int8)byte_4CF813F | (unsigned __int8)byte_4CF8154) )
    unk_4F07734 = qword_4F077A8 > 0x76BFu;
  if ( !byte_4CF8155 && unk_4F07734 )
  {
    unk_4F07730 = qword_4F077A8 <= 0x9F5Fu;
    if ( dword_4F077C4 != 2 )
      goto LABEL_47;
LABEL_136:
    if ( unk_4F07778 > 201102 || dword_4F07774 )
      goto LABEL_49;
    goto LABEL_47;
  }
  if ( dword_4F077C4 == 2 )
    goto LABEL_136;
LABEL_47:
  if ( qword_4F077A8 <= 0x9D6Bu )
    goto LABEL_66;
  unk_4F07754 = 1;
  unk_4F07744 = 1;
LABEL_49:
  if ( qword_4F077A8 > 0x9E33u )
  {
    unk_4D044AC = 1;
    if ( byte_4CF8149 || qword_4F077A8 > 0x9E97u )
      goto LABEL_68;
    goto LABEL_52;
  }
LABEL_66:
  if ( !byte_4CF8149 )
  {
    if ( qword_4F077A8 > 0x9E97u )
      goto LABEL_68;
LABEL_52:
    unk_4D04470 = 0;
  }
  if ( qword_4F077A8 <= 0x9DCFu )
  {
    if ( dword_4F077C4 != 2 )
      goto LABEL_55;
    goto LABEL_70;
  }
LABEL_68:
  unk_4D043E4 = 1;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
LABEL_196:
    if ( !byte_4CF814A )
    {
      v13 = 0;
      if ( qword_4F077A8 > 0x9E97u )
        v13 = unk_4D04470 == 0;
      dword_4D0446C = v13;
      if ( qword_4F077A8 <= 0x9D6Bu )
        goto LABEL_75;
      goto LABEL_73;
    }
LABEL_72:
    if ( qword_4F077A8 <= 0x9D6Bu )
      goto LABEL_75;
LABEL_73:
    if ( byte_4CF8156 )
      goto LABEL_75;
    goto LABEL_74;
  }
  unk_4D044A8 = 1;
  unk_4D044A4 = 1;
  if ( dword_4F077C4 == 2 )
  {
LABEL_70:
    if ( unk_4F07778 <= 201102 && !dword_4F07774 )
      goto LABEL_72;
    goto LABEL_196;
  }
LABEL_55:
  if ( qword_4F077A8 <= 0x9D6Bu )
    goto LABEL_56;
  if ( byte_4CF8156 )
    goto LABEL_89;
LABEL_74:
  unk_4D04408 = 1;
  if ( dword_4F077C4 != 2 )
    goto LABEL_89;
LABEL_75:
  if ( unk_4F07778 <= 201102 )
  {
    if ( !dword_4F07774 )
      goto LABEL_89;
    if ( qword_4F077A8 > 0x9E33u )
    {
      word_4D04898 = 1;
      if ( !dword_4F07774 )
        goto LABEL_89;
    }
  }
  else if ( qword_4F077A8 > 0x9E33u )
  {
    word_4D04898 = 1;
  }
  v7 = dword_4F077BC;
  if ( dword_4F077BC && qword_4F077A8 > 0x1116Fu )
  {
    if ( !dword_4F077B4 )
    {
LABEL_82:
      unk_4D043C4 = 1;
      goto LABEL_83;
    }
  }
  else if ( !dword_4F077B4 )
  {
    goto LABEL_83;
  }
  if ( unk_4F077A0 > 0x9C3Fu )
    goto LABEL_82;
LABEL_83:
  if ( unk_4F07778 <= 201102 )
  {
    if ( !dword_4F07774 )
      goto LABEL_89;
    v7 = dword_4F077BC;
  }
  if ( v7 && qword_4F077A8 > 0x15F8Fu )
  {
    if ( !dword_4F077B4 )
    {
LABEL_88:
      unk_4D0486C = 1;
      goto LABEL_89;
    }
LABEL_87:
    if ( unk_4F077A0 <= 0x15F8Fu )
      goto LABEL_89;
    goto LABEL_88;
  }
  if ( dword_4F077B4 )
    goto LABEL_87;
LABEL_89:
  if ( qword_4F077A8 <= 0x9EFBu )
  {
LABEL_56:
    if ( qword_4F077A8 <= 0x76BFu )
      goto LABEL_57;
    goto LABEL_160;
  }
  v8 = unk_4D04408;
  if ( !unk_4D04408 )
  {
LABEL_160:
    if ( byte_4CF813C )
      goto LABEL_57;
    if ( dword_4D047B0 )
    {
      dword_4D047AC = 1;
      goto LABEL_57;
    }
    v8 = unk_4D04408;
    goto LABEL_211;
  }
  if ( !byte_4CF8125 || dword_4D047B0 )
  {
    unk_4D043FC = 1;
    goto LABEL_160;
  }
  if ( !byte_4CF813C )
LABEL_211:
    dword_4D047AC = v8 != 0;
LABEL_57:
  if ( !byte_4CF8159 )
  {
    v5 = dword_4D048B8;
    if ( dword_4D048B8 )
      v5 = dword_4D048B4 != 0 && qword_4F077A8 > 0x9F5Fu;
    dword_4D048B0 = v5;
  }
  if ( dword_4F077B4 )
  {
    if ( !byte_4CF811C )
      unk_4D04790 = 0;
    if ( dword_4F077C4 != 2 )
    {
LABEL_65:
      v6 = unk_4F077A0 > 0x752Fu;
      goto LABEL_97;
    }
  }
  else if ( dword_4F077C4 != 2 )
  {
    goto LABEL_96;
  }
  v11 = unk_4F07778;
  if ( unk_4F07778 <= 201102 )
  {
    if ( dword_4F07774 )
      goto LABEL_117;
    if ( dword_4F077B4 )
      goto LABEL_65;
LABEL_96:
    v6 = qword_4F077A8 > 0x9DCFu;
LABEL_97:
    if ( v6 )
    {
      dword_4D04464 = 1;
      unk_4D04468 = 1;
    }
    if ( byte_4CF8147 )
    {
      if ( qword_4F077A8 > 0x9EFBu )
      {
        if ( !dword_4F077B4 )
        {
LABEL_103:
          if ( !byte_4CF815D )
            unk_4D048A0 = 1;
          goto LABEL_105;
        }
        goto LABEL_140;
      }
    }
    else if ( qword_4F077A8 > 0x9E33u )
    {
      if ( !dword_4F077B4 )
      {
        unk_4D0448C = 1;
        if ( qword_4F077A8 > 0x9EFBu )
          goto LABEL_103;
LABEL_105:
        v9 = qword_4F077A8 > 0x9EFBu;
        goto LABEL_106;
      }
LABEL_140:
      v9 = unk_4F077A0 > 0x752Fu;
LABEL_106:
      if ( v9 )
      {
        unk_4D04424 = 1;
        unk_4D043E8 = 1;
      }
      if ( dword_4F077B4 )
        v10 = unk_4F077A0 > 0x752Fu;
      else
        v10 = qword_4F077A8 > 0x9F5Fu;
      if ( v10 && !byte_4CF8148 )
        unk_4D04474 = 1;
      if ( qword_4F077A8 <= 0x9F5Fu )
      {
        if ( qword_4F077A8 <= 0x9EFBu && !dword_4F077B4 )
          goto LABEL_116;
      }
      else if ( !dword_4F077B4 )
      {
        if ( unk_4D04474 )
          dword_4D0489C = 1;
        dword_4D043F8 = 1;
      }
      unk_4F0776C = 1;
      unk_4F07740 = 1;
LABEL_116:
      if ( dword_4F077C4 != 2 )
        goto LABEL_117;
      v11 = unk_4F07778;
      goto LABEL_146;
    }
    if ( !dword_4F077B4 )
      goto LABEL_105;
    goto LABEL_140;
  }
LABEL_146:
  if ( v11 > 201401 )
  {
    if ( qword_4F077A8 > 0x9FC3u && dword_4F077BC != 0 && !dword_4F077B4 )
    {
      dword_4D04490 = 1;
      if ( v11 <= 201702 )
      {
LABEL_124:
        unk_4D04854 = 0;
        goto LABEL_125;
      }
      if ( !unk_4F06978 )
        goto LABEL_125;
LABEL_151:
      unk_4F06974 = 1;
LABEL_152:
      if ( dword_4F077B4 )
      {
        unk_4D043CC = 1;
        unk_4D0418C = 1;
        v12 = unk_4F077A0;
        unk_4D04740 = unk_4F077A0 > 0x1116Fu;
LABEL_154:
        dword_4F07760 = 1;
        goto LABEL_167;
      }
      goto LABEL_125;
    }
    if ( v11 <= 201702 )
      goto LABEL_235;
LABEL_150:
    if ( !unk_4F06978 )
      goto LABEL_152;
    goto LABEL_151;
  }
LABEL_117:
  if ( unk_4D0448C )
    unk_4D0447C = 1;
  if ( !(unk_4F0774C | dword_4F077B4) )
  {
    if ( unk_4F0775C && (unsigned __int64)(qword_4F077A8 - 40800LL) <= 0x63 )
    {
      unk_4F0774C = 1;
      unk_4F07748 = 1;
    }
    if ( dword_4F077C4 != 2 )
      goto LABEL_124;
LABEL_234:
    v11 = unk_4F07778;
    if ( unk_4F07778 <= 201702 )
    {
LABEL_235:
      unk_4D04854 = 0;
      if ( dword_4F077B4 )
      {
        unk_4D043CC = 1;
        unk_4D0418C = 1;
        v12 = unk_4F077A0;
        unk_4D04740 = unk_4F077A0 > 0x1116Fu;
        if ( v11 > 201102 || dword_4F07774 )
          goto LABEL_154;
LABEL_166:
        unk_4D0440C = 1;
LABEL_167:
        if ( v12 > 0x752F )
        {
          unk_4D043E4 = 1;
          if ( v12 > 0x7593 )
            unk_4D044D0 = 1;
          if ( v12 > 0x75F7 )
            unk_4D043F0 = 1;
          if ( v12 > 0x765B )
            unk_4D04220 = 1;
          if ( v12 > 0x7787 )
            unk_4D043D4 = 1;
          if ( v12 > 0xEA5F )
          {
            unk_4D041EC = 1;
            unk_4D043D8 = 1;
            unk_4D04404 = 1;
            if ( v12 > 0x1387F )
            {
              unk_4D043D0 = 1;
              if ( v12 > 0x1FBCF )
                unk_4D04184 = 1;
            }
          }
        }
LABEL_128:
        if ( !byte_4CF8172 )
        {
          LOBYTE(v0) = qword_4F077A8 <= 0x1ADAFu;
          if ( qword_4F077A8 <= 0x1ADAFu || dword_4F077BC == 0 || dword_4F077B4 )
            goto LABEL_131;
        }
        return (char)v0;
      }
      goto LABEL_125;
    }
    goto LABEL_150;
  }
  if ( dword_4F077C4 == 2 )
    goto LABEL_234;
  unk_4D04854 = 0;
  if ( dword_4F077B4 )
  {
    unk_4D043CC = 1;
    unk_4D0418C = 1;
    v12 = unk_4F077A0;
    unk_4D04740 = unk_4F077A0 > 0x1116Fu;
    goto LABEL_166;
  }
LABEL_125:
  unk_4D04304 = qword_4F077A8 > 0x9EFBu;
  if ( qword_4F077A8 > 0xEA5Fu )
  {
    unk_4D043CC = 1;
    unk_4D04404 = 1;
    unk_4D043D8 = 1;
    unk_4D043D4 = 1;
    if ( dword_4F077C4 != 2 )
      goto LABEL_188;
    if ( unk_4F07778 > 201102 )
      goto LABEL_213;
  }
  else
  {
    dword_4D04394 = 0;
    if ( dword_4F077C4 != 2 || unk_4F07778 > 201102 )
      goto LABEL_127;
  }
  if ( dword_4F07774 )
  {
    if ( qword_4F077A8 > 0xEA5Fu )
    {
LABEL_213:
      dword_4F07760 = 1;
      if ( qword_4F077A8 <= 0x1116Fu )
      {
        unk_4D04774 = 0;
        unk_4D04770 = 0;
        unk_4D0476C = 1;
        unk_4D04740 = 0;
        goto LABEL_191;
      }
      goto LABEL_189;
    }
LABEL_127:
    unk_4D04774 = 0;
    unk_4D04770 = 0;
    unk_4D0476C = 1;
    unk_4D04740 = 0;
    goto LABEL_128;
  }
LABEL_188:
  if ( qword_4F077A8 > 0x1116Fu )
  {
LABEL_189:
    unk_4D041DC = 1;
    unk_4D04858 = 1;
    unk_4D041EC = 1;
    unk_4D04774 = 0;
    unk_4D04770 = 0;
    unk_4D0476C = 1;
    unk_4D04740 = 0;
    if ( qword_4F077A8 > 0x15F8Fu )
    {
      unk_4D043D0 = 1;
      unk_4D041B0 = 1;
      unk_4D04814 = 1;
    }
    goto LABEL_191;
  }
  unk_4D04774 = 0;
  unk_4D04770 = 0;
  unk_4D0476C = 1;
  unk_4D04740 = 0;
  if ( qword_4F077A8 <= 0xEA5Fu )
    goto LABEL_128;
LABEL_191:
  unk_4D04870 = 1;
  if ( qword_4F077A8 > 0x1ADAFu )
  {
    unk_4D04184 = 1;
    if ( qword_4F077A8 > 0x1D4BFu )
      unk_4D0428C = 1;
    goto LABEL_128;
  }
  if ( !byte_4CF8172 )
  {
LABEL_131:
    v0 = &dword_4D041AC;
    dword_4D041AC = 0;
  }
  return (char)v0;
}
