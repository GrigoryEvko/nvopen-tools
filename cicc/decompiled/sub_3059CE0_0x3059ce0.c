// Function: sub_3059CE0
// Address: 0x3059ce0
//
__int64 __fastcall sub_3059CE0(
        __int64 a1,
        _DWORD *a2,
        size_t a3,
        _DWORD *a4,
        size_t a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 result; // rax

  sub_EA0C90(a1, a2, a3, a4, a5, a6, a7, a8);
  v8 = *(_QWORD *)(a1 + 232);
  if ( (v8 & 1) != 0 && *(int *)(a1 + 352) < 0 )
  {
    *(_DWORD *)(a1 + 352) = 0;
    if ( (v8 & 2) == 0 )
    {
      if ( (v8 & 4) == 0 )
        goto LABEL_6;
      goto LABEL_251;
    }
    goto LABEL_263;
  }
  if ( (v8 & 2) != 0 && *(int *)(a1 + 352) <= 0 )
  {
LABEL_263:
    *(_DWORD *)(a1 + 352) = 1;
    if ( (v8 & 4) != 0 )
      *(_DWORD *)(a1 + 352) = 2;
    goto LABEL_6;
  }
  if ( (v8 & 4) != 0 && *(int *)(a1 + 352) <= 1 )
LABEL_251:
    *(_DWORD *)(a1 + 352) = 2;
LABEL_6:
  if ( (v8 & 8) != 0 && *(_DWORD *)(a1 + 336) <= 0x1Fu )
  {
    *(_DWORD *)(a1 + 336) = 32;
    if ( (v8 & 0x10) != 0 )
      goto LABEL_9;
    if ( (v8 & 0x20) != 0 )
      goto LABEL_10;
LABEL_103:
    if ( (v8 & 0x40) == 0 || *(_DWORD *)(a1 + 336) > 0x29u )
      goto LABEL_105;
    goto LABEL_11;
  }
  if ( (v8 & 0x10) != 0 && *(_DWORD *)(a1 + 336) <= 0x27u )
  {
LABEL_9:
    *(_DWORD *)(a1 + 336) = 40;
    if ( (v8 & 0x20) != 0 )
      goto LABEL_10;
    if ( (v8 & 0x40) == 0 )
    {
LABEL_105:
      if ( (v8 & 0x80u) == 0LL || *(_DWORD *)(a1 + 336) > 0x2Au )
        goto LABEL_107;
      goto LABEL_12;
    }
LABEL_11:
    *(_DWORD *)(a1 + 336) = 42;
    if ( (v8 & 0x80u) != 0LL )
      goto LABEL_12;
    if ( (v8 & 0x100) == 0 )
    {
LABEL_109:
      if ( (v8 & 0x200) == 0 || *(_DWORD *)(a1 + 336) > 0x3Bu )
        goto LABEL_111;
      goto LABEL_14;
    }
LABEL_13:
    *(_DWORD *)(a1 + 336) = 50;
    if ( (v8 & 0x200) != 0 )
      goto LABEL_14;
    if ( (v8 & 0x400) == 0 )
    {
LABEL_113:
      if ( (v8 & 0x800) == 0 || *(_DWORD *)(a1 + 336) > 0x3Du )
        goto LABEL_115;
      goto LABEL_16;
    }
LABEL_15:
    *(_DWORD *)(a1 + 336) = 61;
    if ( (v8 & 0x800) != 0 )
      goto LABEL_16;
    if ( (v8 & 0x1000) == 0 )
    {
LABEL_117:
      if ( (v8 & 0x2000) == 0 || *(_DWORD *)(a1 + 336) > 0x3Fu )
        goto LABEL_119;
      goto LABEL_18;
    }
LABEL_17:
    *(_DWORD *)(a1 + 336) = 63;
    if ( (v8 & 0x2000) != 0 )
      goto LABEL_18;
    if ( (v8 & 0x4000) == 0 )
    {
LABEL_121:
      if ( (v8 & 0x8000) == 0 || *(_DWORD *)(a1 + 336) > 0x45u )
        goto LABEL_123;
      goto LABEL_20;
    }
LABEL_19:
    *(_DWORD *)(a1 + 336) = 65;
    if ( (v8 & 0x8000) != 0 )
      goto LABEL_20;
    if ( (v8 & 0x10000) == 0 )
    {
LABEL_125:
      if ( (v8 & 0x20000) == 0 || *(_DWORD *)(a1 + 336) > 0x47u )
        goto LABEL_127;
      goto LABEL_22;
    }
LABEL_21:
    *(_DWORD *)(a1 + 336) = 71;
    if ( (v8 & 0x20000) != 0 )
      goto LABEL_22;
    if ( (v8 & 0x40000) == 0 )
    {
LABEL_129:
      if ( (v8 & 0x80000) == 0 || *(_DWORD *)(a1 + 336) > 0x49u )
        goto LABEL_131;
      goto LABEL_24;
    }
LABEL_23:
    *(_DWORD *)(a1 + 336) = 73;
    if ( (v8 & 0x80000) != 0 )
      goto LABEL_24;
    if ( (v8 & 0x100000) == 0 )
    {
LABEL_133:
      if ( (v8 & 0x200000) == 0 || *(_DWORD *)(a1 + 336) > 0x4Bu )
        goto LABEL_135;
      goto LABEL_26;
    }
LABEL_25:
    *(_DWORD *)(a1 + 336) = 75;
    if ( (v8 & 0x200000) != 0 )
      goto LABEL_26;
    if ( (v8 & 0x400000) == 0 )
    {
LABEL_137:
      if ( (v8 & 0x800000) == 0 || *(_DWORD *)(a1 + 336) > 0x4Du )
        goto LABEL_139;
      goto LABEL_28;
    }
LABEL_27:
    *(_DWORD *)(a1 + 336) = 77;
    if ( (v8 & 0x800000) != 0 )
      goto LABEL_28;
    if ( (v8 & 0x1000000) == 0 )
    {
LABEL_141:
      if ( (v8 & 0x2000000) == 0 || *(_DWORD *)(a1 + 336) > 0x50u )
        goto LABEL_143;
      goto LABEL_30;
    }
LABEL_29:
    *(_DWORD *)(a1 + 336) = 80;
    if ( (v8 & 0x2000000) != 0 )
      goto LABEL_30;
    if ( (v8 & 0x4000000) == 0 )
    {
LABEL_145:
      if ( (v8 & 0x8000000) == 0 || *(_DWORD *)(a1 + 336) > 0x52u )
        goto LABEL_147;
      goto LABEL_32;
    }
LABEL_31:
    *(_DWORD *)(a1 + 336) = 82;
    if ( (v8 & 0x8000000) != 0 )
      goto LABEL_32;
    if ( (v8 & 0x10000000) == 0 )
    {
LABEL_149:
      if ( (v8 & 0x20000000) == 0 || *(_DWORD *)(a1 + 336) > 0x54u )
        goto LABEL_151;
      goto LABEL_34;
    }
LABEL_33:
    *(_DWORD *)(a1 + 336) = 84;
    if ( (v8 & 0x20000000) != 0 )
      goto LABEL_34;
    if ( (v8 & 0x40000000) == 0 )
    {
LABEL_153:
      if ( (int)v8 >= 0 || *(_DWORD *)(a1 + 336) > 0x56u )
        goto LABEL_155;
LABEL_36:
      *(_DWORD *)(a1 + 336) = 87;
      if ( (v8 & 0x100000000LL) == 0 )
        goto LABEL_37;
LABEL_157:
      *(_DWORD *)(a1 + 336) = 88;
      goto LABEL_37;
    }
LABEL_35:
    *(_DWORD *)(a1 + 336) = 86;
    if ( (int)v8 >= 0 )
    {
      if ( (v8 & 0x100000000LL) == 0 )
        goto LABEL_37;
      goto LABEL_157;
    }
    goto LABEL_36;
  }
  if ( (v8 & 0x20) == 0 || *(_DWORD *)(a1 + 336) > 0x28u )
    goto LABEL_103;
LABEL_10:
  *(_DWORD *)(a1 + 336) = 41;
  if ( (v8 & 0x40) != 0 )
    goto LABEL_11;
  if ( (v8 & 0x80u) == 0LL )
  {
LABEL_107:
    if ( (v8 & 0x100) == 0 || *(_DWORD *)(a1 + 336) > 0x31u )
      goto LABEL_109;
    goto LABEL_13;
  }
LABEL_12:
  *(_DWORD *)(a1 + 336) = 43;
  if ( (v8 & 0x100) != 0 )
    goto LABEL_13;
  if ( (v8 & 0x200) == 0 )
  {
LABEL_111:
    if ( (v8 & 0x400) == 0 || *(_DWORD *)(a1 + 336) > 0x3Cu )
      goto LABEL_113;
    goto LABEL_15;
  }
LABEL_14:
  *(_DWORD *)(a1 + 336) = 60;
  if ( (v8 & 0x400) != 0 )
    goto LABEL_15;
  if ( (v8 & 0x800) == 0 )
  {
LABEL_115:
    if ( (v8 & 0x1000) == 0 || *(_DWORD *)(a1 + 336) > 0x3Eu )
      goto LABEL_117;
    goto LABEL_17;
  }
LABEL_16:
  *(_DWORD *)(a1 + 336) = 62;
  if ( (v8 & 0x1000) != 0 )
    goto LABEL_17;
  if ( (v8 & 0x2000) == 0 )
  {
LABEL_119:
    if ( (v8 & 0x4000) == 0 || *(_DWORD *)(a1 + 336) > 0x40u )
      goto LABEL_121;
    goto LABEL_19;
  }
LABEL_18:
  *(_DWORD *)(a1 + 336) = 64;
  if ( (v8 & 0x4000) != 0 )
    goto LABEL_19;
  if ( (v8 & 0x8000) == 0 )
  {
LABEL_123:
    if ( (v8 & 0x10000) == 0 || *(_DWORD *)(a1 + 336) > 0x46u )
      goto LABEL_125;
    goto LABEL_21;
  }
LABEL_20:
  *(_DWORD *)(a1 + 336) = 70;
  if ( (v8 & 0x10000) != 0 )
    goto LABEL_21;
  if ( (v8 & 0x20000) == 0 )
  {
LABEL_127:
    if ( (v8 & 0x40000) == 0 || *(_DWORD *)(a1 + 336) > 0x48u )
      goto LABEL_129;
    goto LABEL_23;
  }
LABEL_22:
  *(_DWORD *)(a1 + 336) = 72;
  if ( (v8 & 0x40000) != 0 )
    goto LABEL_23;
  if ( (v8 & 0x80000) == 0 )
  {
LABEL_131:
    if ( (v8 & 0x100000) == 0 || *(_DWORD *)(a1 + 336) > 0x4Au )
      goto LABEL_133;
    goto LABEL_25;
  }
LABEL_24:
  *(_DWORD *)(a1 + 336) = 74;
  if ( (v8 & 0x100000) != 0 )
    goto LABEL_25;
  if ( (v8 & 0x200000) == 0 )
  {
LABEL_135:
    if ( (v8 & 0x400000) == 0 || *(_DWORD *)(a1 + 336) > 0x4Cu )
      goto LABEL_137;
    goto LABEL_27;
  }
LABEL_26:
  *(_DWORD *)(a1 + 336) = 76;
  if ( (v8 & 0x400000) != 0 )
    goto LABEL_27;
  if ( (v8 & 0x800000) == 0 )
  {
LABEL_139:
    if ( (v8 & 0x1000000) == 0 || *(_DWORD *)(a1 + 336) > 0x4Fu )
      goto LABEL_141;
    goto LABEL_29;
  }
LABEL_28:
  *(_DWORD *)(a1 + 336) = 78;
  if ( (v8 & 0x1000000) != 0 )
    goto LABEL_29;
  if ( (v8 & 0x2000000) == 0 )
  {
LABEL_143:
    if ( (v8 & 0x4000000) == 0 || *(_DWORD *)(a1 + 336) > 0x51u )
      goto LABEL_145;
    goto LABEL_31;
  }
LABEL_30:
  *(_DWORD *)(a1 + 336) = 81;
  if ( (v8 & 0x4000000) != 0 )
    goto LABEL_31;
  if ( (v8 & 0x8000000) == 0 )
  {
LABEL_147:
    if ( (v8 & 0x10000000) == 0 || *(_DWORD *)(a1 + 336) > 0x53u )
      goto LABEL_149;
    goto LABEL_33;
  }
LABEL_32:
  *(_DWORD *)(a1 + 336) = 83;
  if ( (v8 & 0x10000000) != 0 )
    goto LABEL_33;
  if ( (v8 & 0x20000000) == 0 )
  {
LABEL_151:
    if ( (v8 & 0x40000000) == 0 || *(_DWORD *)(a1 + 336) > 0x55u )
      goto LABEL_153;
    goto LABEL_35;
  }
LABEL_34:
  *(_DWORD *)(a1 + 336) = 85;
  if ( (v8 & 0x40000000) != 0 )
    goto LABEL_35;
  if ( (int)v8 < 0 )
    goto LABEL_36;
LABEL_155:
  if ( (v8 & 0x100000000LL) != 0 && *(_DWORD *)(a1 + 336) <= 0x57u )
    goto LABEL_157;
LABEL_37:
  v9 = v8 & 0x400000000LL;
  if ( (v8 & 0x200000000LL) != 0 && *(int *)(a1 + 356) < 0 )
  {
    *(_DWORD *)(a1 + 356) = 0;
    if ( !v9 )
    {
      if ( (v8 & 0x800000000LL) == 0 )
        goto LABEL_43;
LABEL_258:
      *(_DWORD *)(a1 + 356) = 2;
      if ( (v8 & 0x1000000000LL) != 0 )
        *(_DWORD *)(a1 + 356) = 3;
      goto LABEL_44;
    }
LABEL_257:
    *(_DWORD *)(a1 + 356) = 1;
    if ( (v8 & 0x800000000LL) == 0 )
    {
      if ( (v8 & 0x1000000000LL) == 0 )
        goto LABEL_44;
      goto LABEL_249;
    }
    goto LABEL_258;
  }
  if ( v9 && *(int *)(a1 + 356) <= 0 )
    goto LABEL_257;
  if ( (v8 & 0x800000000LL) != 0 && *(int *)(a1 + 356) <= 1 )
    goto LABEL_258;
LABEL_43:
  if ( (v8 & 0x1000000000LL) != 0 && *(int *)(a1 + 356) <= 2 )
LABEL_249:
    *(_DWORD *)(a1 + 356) = 3;
LABEL_44:
  v10 = v8 & 0x4000000000LL;
  if ( (v8 & 0x2000000000LL) != 0 && *(int *)(a1 + 360) < 0 )
  {
    *(_DWORD *)(a1 + 360) = 0;
    if ( v10 )
      *(_DWORD *)(a1 + 360) = 1;
  }
  else if ( v10 && *(int *)(a1 + 360) <= 0 )
  {
    *(_DWORD *)(a1 + 360) = 1;
  }
  v11 = v8 & 0x10000000000LL;
  if ( (v8 & 0x8000000000LL) != 0 && *(_DWORD *)(a1 + 340) <= 0xC7u )
  {
    *(_DWORD *)(a1 + 340) = 200;
    if ( v11 )
      goto LABEL_50;
    if ( (v8 & 0x20000000000LL) != 0 )
      goto LABEL_51;
LABEL_162:
    if ( (v8 & 0x40000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x13Fu )
      goto LABEL_164;
    goto LABEL_52;
  }
  if ( v11 && *(_DWORD *)(a1 + 340) <= 0xD1u )
  {
LABEL_50:
    *(_DWORD *)(a1 + 340) = 210;
    if ( (v8 & 0x20000000000LL) != 0 )
      goto LABEL_51;
    if ( (v8 & 0x40000000000LL) == 0 )
    {
LABEL_164:
      if ( (v8 & 0x80000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x15Du )
        goto LABEL_166;
      goto LABEL_53;
    }
LABEL_52:
    *(_DWORD *)(a1 + 340) = 320;
    if ( (v8 & 0x80000000000LL) != 0 )
      goto LABEL_53;
    if ( (v8 & 0x100000000000LL) == 0 )
    {
LABEL_168:
      if ( (v8 & 0x200000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x1F3u )
        goto LABEL_170;
      goto LABEL_55;
    }
LABEL_54:
    *(_DWORD *)(a1 + 340) = 370;
    if ( (v8 & 0x200000000000LL) != 0 )
      goto LABEL_55;
    if ( (v8 & 0x400000000000LL) == 0 )
    {
LABEL_172:
      if ( (v8 & 0x800000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x211u )
        goto LABEL_174;
      goto LABEL_57;
    }
LABEL_56:
    *(_DWORD *)(a1 + 340) = 520;
    if ( (v8 & 0x800000000000LL) != 0 )
      goto LABEL_57;
    if ( (v8 & 0x1000000000000LL) == 0 )
    {
LABEL_176:
      if ( (v8 & 0x2000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x261u )
        goto LABEL_178;
      goto LABEL_59;
    }
LABEL_58:
    *(_DWORD *)(a1 + 340) = 600;
    if ( (v8 & 0x2000000000000LL) != 0 )
      goto LABEL_59;
    if ( (v8 & 0x4000000000000LL) == 0 )
    {
LABEL_180:
      if ( (v8 & 0x8000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x2BBu )
        goto LABEL_182;
      goto LABEL_61;
    }
LABEL_60:
    *(_DWORD *)(a1 + 340) = 620;
    if ( (v8 & 0x8000000000000LL) != 0 )
      goto LABEL_61;
    if ( (v8 & 0x10000000000000LL) == 0 )
    {
LABEL_184:
      if ( (v8 & 0x20000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x2D9u )
        goto LABEL_186;
      goto LABEL_63;
    }
LABEL_62:
    *(_DWORD *)(a1 + 340) = 720;
    if ( (v8 & 0x20000000000000LL) != 0 )
      goto LABEL_63;
    if ( (v8 & 0x40000000000000LL) == 0 )
    {
LABEL_188:
      if ( (v8 & 0x80000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x31Fu )
        goto LABEL_190;
      goto LABEL_65;
    }
LABEL_64:
    *(_DWORD *)(a1 + 340) = 750;
    if ( (v8 & 0x80000000000000LL) != 0 )
      goto LABEL_65;
    if ( (v8 & 0x100000000000000LL) == 0 )
    {
LABEL_192:
      if ( (v8 & 0x200000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x35Bu )
        goto LABEL_194;
      goto LABEL_67;
    }
LABEL_66:
    *(_DWORD *)(a1 + 340) = 820;
    if ( (v8 & 0x200000000000000LL) != 0 )
      goto LABEL_67;
    if ( (v8 & 0x400000000000000LL) == 0 )
    {
LABEL_196:
      if ( (v8 & 0x800000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x36Fu )
        goto LABEL_198;
      goto LABEL_69;
    }
LABEL_68:
    *(_DWORD *)(a1 + 340) = 870;
    if ( (v8 & 0x800000000000000LL) != 0 )
      goto LABEL_69;
    if ( (v8 & 0x1000000000000000LL) == 0 )
    {
LABEL_200:
      if ( (v8 & 0x2000000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x383u )
        goto LABEL_202;
      goto LABEL_71;
    }
LABEL_70:
    *(_DWORD *)(a1 + 340) = 890;
    if ( (v8 & 0x2000000000000000LL) != 0 )
      goto LABEL_71;
    if ( (v8 & 0x4000000000000000LL) == 0 )
    {
LABEL_204:
      if ( v8 < 0 && *(_DWORD *)(a1 + 340) <= 0x3E7u )
        goto LABEL_253;
      goto LABEL_205;
    }
LABEL_72:
    *(_DWORD *)(a1 + 340) = 901;
    if ( v8 >= 0 )
    {
      result = *(_QWORD *)(a1 + 240);
      if ( (result & 1) != 0 )
        goto LABEL_74;
LABEL_207:
      if ( (result & 2) == 0 || *(_DWORD *)(a1 + 340) > 0x3E9u )
      {
LABEL_209:
        if ( (result & 4) != 0 && *(_DWORD *)(a1 + 340) <= 0x3F1u )
          goto LABEL_76;
LABEL_211:
        if ( (result & 8) != 0 && *(_DWORD *)(a1 + 340) <= 0x3F2u )
          goto LABEL_77;
        goto LABEL_213;
      }
LABEL_75:
      *(_DWORD *)(a1 + 340) = 1002;
      if ( (result & 4) != 0 )
        goto LABEL_76;
      if ( (result & 8) != 0 )
      {
LABEL_77:
        *(_DWORD *)(a1 + 340) = 1011;
        if ( (result & 0x10) != 0 )
          goto LABEL_78;
        if ( (result & 0x20) != 0 )
        {
LABEL_79:
          *(_DWORD *)(a1 + 340) = 1020;
          if ( (result & 0x40) != 0 )
            goto LABEL_80;
          if ( (result & 0x80u) != 0LL )
          {
LABEL_81:
            *(_DWORD *)(a1 + 340) = 1022;
            if ( (result & 0x100) != 0 )
              goto LABEL_82;
            if ( (result & 0x200) != 0 )
            {
LABEL_83:
              *(_DWORD *)(a1 + 340) = 1031;
              if ( (result & 0x400) != 0 )
                goto LABEL_84;
              if ( (result & 0x800) != 0 )
              {
LABEL_85:
                *(_DWORD *)(a1 + 340) = 1100;
                if ( (result & 0x1000) != 0 )
                  goto LABEL_86;
                if ( (result & 0x2000) != 0 )
                {
LABEL_87:
                  *(_DWORD *)(a1 + 340) = 1102;
                  if ( (result & 0x4000) != 0 )
                    goto LABEL_88;
                  if ( (result & 0x8000) != 0 )
                  {
LABEL_89:
                    *(_DWORD *)(a1 + 340) = 1201;
                    if ( (result & 0x10000) != 0 )
                      goto LABEL_90;
                    if ( (result & 0x20000) != 0 )
                    {
LABEL_91:
                      *(_DWORD *)(a1 + 340) = 1210;
                      if ( (result & 0x40000) == 0 )
                      {
                        if ( (result & 0x80000) == 0 )
                          goto LABEL_93;
                        goto LABEL_245;
                      }
                      goto LABEL_92;
                    }
LABEL_241:
                    if ( (result & 0x40000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4BAu )
                      goto LABEL_92;
                    goto LABEL_243;
                  }
LABEL_237:
                  if ( (result & 0x10000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4B1u )
                    goto LABEL_90;
LABEL_239:
                  if ( (result & 0x20000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4B9u )
                    goto LABEL_91;
                  goto LABEL_241;
                }
LABEL_233:
                if ( (result & 0x4000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4AFu )
                  goto LABEL_88;
LABEL_235:
                if ( (result & 0x8000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4B0u )
                  goto LABEL_89;
                goto LABEL_237;
              }
LABEL_229:
              if ( (result & 0x1000) != 0 && *(_DWORD *)(a1 + 340) <= 0x44Cu )
                goto LABEL_86;
LABEL_231:
              if ( (result & 0x2000) != 0 && *(_DWORD *)(a1 + 340) <= 0x44Du )
                goto LABEL_87;
              goto LABEL_233;
            }
LABEL_225:
            if ( (result & 0x400) != 0 && *(_DWORD *)(a1 + 340) <= 0x407u )
              goto LABEL_84;
LABEL_227:
            if ( (result & 0x800) != 0 && *(_DWORD *)(a1 + 340) <= 0x44Bu )
              goto LABEL_85;
            goto LABEL_229;
          }
LABEL_221:
          if ( (result & 0x100) != 0 && *(_DWORD *)(a1 + 340) <= 0x405u )
            goto LABEL_82;
LABEL_223:
          if ( (result & 0x200) != 0 && *(_DWORD *)(a1 + 340) <= 0x406u )
            goto LABEL_83;
          goto LABEL_225;
        }
LABEL_217:
        if ( (result & 0x40) != 0 && *(_DWORD *)(a1 + 340) <= 0x3FCu )
          goto LABEL_80;
LABEL_219:
        if ( (result & 0x80u) != 0LL && *(_DWORD *)(a1 + 340) <= 0x3FDu )
          goto LABEL_81;
        goto LABEL_221;
      }
LABEL_213:
      if ( (result & 0x10) != 0 && *(_DWORD *)(a1 + 340) <= 0x3F3u )
        goto LABEL_78;
LABEL_215:
      if ( (result & 0x20) != 0 && *(_DWORD *)(a1 + 340) <= 0x3FBu )
        goto LABEL_79;
      goto LABEL_217;
    }
    goto LABEL_253;
  }
  if ( (v8 & 0x20000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x12Bu )
    goto LABEL_162;
LABEL_51:
  *(_DWORD *)(a1 + 340) = 300;
  if ( (v8 & 0x40000000000LL) != 0 )
    goto LABEL_52;
  if ( (v8 & 0x80000000000LL) == 0 )
  {
LABEL_166:
    if ( (v8 & 0x100000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x171u )
      goto LABEL_168;
    goto LABEL_54;
  }
LABEL_53:
  *(_DWORD *)(a1 + 340) = 350;
  if ( (v8 & 0x100000000000LL) != 0 )
    goto LABEL_54;
  if ( (v8 & 0x200000000000LL) == 0 )
  {
LABEL_170:
    if ( (v8 & 0x400000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x207u )
      goto LABEL_172;
    goto LABEL_56;
  }
LABEL_55:
  *(_DWORD *)(a1 + 340) = 500;
  if ( (v8 & 0x400000000000LL) != 0 )
    goto LABEL_56;
  if ( (v8 & 0x800000000000LL) == 0 )
  {
LABEL_174:
    if ( (v8 & 0x1000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x257u )
      goto LABEL_176;
    goto LABEL_58;
  }
LABEL_57:
  *(_DWORD *)(a1 + 340) = 530;
  if ( (v8 & 0x1000000000000LL) != 0 )
    goto LABEL_58;
  if ( (v8 & 0x2000000000000LL) == 0 )
  {
LABEL_178:
    if ( (v8 & 0x4000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x26Bu )
      goto LABEL_180;
    goto LABEL_60;
  }
LABEL_59:
  *(_DWORD *)(a1 + 340) = 610;
  if ( (v8 & 0x4000000000000LL) != 0 )
    goto LABEL_60;
  if ( (v8 & 0x8000000000000LL) == 0 )
  {
LABEL_182:
    if ( (v8 & 0x10000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x2CFu )
      goto LABEL_184;
    goto LABEL_62;
  }
LABEL_61:
  *(_DWORD *)(a1 + 340) = 700;
  if ( (v8 & 0x10000000000000LL) != 0 )
    goto LABEL_62;
  if ( (v8 & 0x20000000000000LL) == 0 )
  {
LABEL_186:
    if ( (v8 & 0x40000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x2EDu )
      goto LABEL_188;
    goto LABEL_64;
  }
LABEL_63:
  *(_DWORD *)(a1 + 340) = 730;
  if ( (v8 & 0x40000000000000LL) != 0 )
    goto LABEL_64;
  if ( (v8 & 0x80000000000000LL) == 0 )
  {
LABEL_190:
    if ( (v8 & 0x100000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x333u )
      goto LABEL_192;
    goto LABEL_66;
  }
LABEL_65:
  *(_DWORD *)(a1 + 340) = 800;
  if ( (v8 & 0x100000000000000LL) != 0 )
    goto LABEL_66;
  if ( (v8 & 0x200000000000000LL) == 0 )
  {
LABEL_194:
    if ( (v8 & 0x400000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x365u )
      goto LABEL_196;
    goto LABEL_68;
  }
LABEL_67:
  *(_DWORD *)(a1 + 340) = 860;
  if ( (v8 & 0x400000000000000LL) != 0 )
    goto LABEL_68;
  if ( (v8 & 0x800000000000000LL) == 0 )
  {
LABEL_198:
    if ( (v8 & 0x1000000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x379u )
      goto LABEL_200;
    goto LABEL_70;
  }
LABEL_69:
  *(_DWORD *)(a1 + 340) = 880;
  if ( (v8 & 0x1000000000000000LL) != 0 )
    goto LABEL_70;
  if ( (v8 & 0x2000000000000000LL) == 0 )
  {
LABEL_202:
    if ( (v8 & 0x4000000000000000LL) == 0 || *(_DWORD *)(a1 + 340) > 0x384u )
      goto LABEL_204;
    goto LABEL_72;
  }
LABEL_71:
  *(_DWORD *)(a1 + 340) = 900;
  if ( (v8 & 0x4000000000000000LL) != 0 )
    goto LABEL_72;
  if ( v8 < 0 )
  {
LABEL_253:
    result = *(_QWORD *)(a1 + 240);
    *(_DWORD *)(a1 + 340) = 1000;
    if ( (result & 1) == 0 )
    {
      if ( (result & 2) == 0 )
        goto LABEL_209;
      goto LABEL_75;
    }
    goto LABEL_74;
  }
LABEL_205:
  result = *(_QWORD *)(a1 + 240);
  if ( (result & 1) == 0 || *(_DWORD *)(a1 + 340) > 0x3E8u )
    goto LABEL_207;
LABEL_74:
  *(_DWORD *)(a1 + 340) = 1001;
  if ( (result & 2) != 0 )
    goto LABEL_75;
  if ( (result & 4) == 0 )
    goto LABEL_211;
LABEL_76:
  *(_DWORD *)(a1 + 340) = 1010;
  if ( (result & 8) != 0 )
    goto LABEL_77;
  if ( (result & 0x10) == 0 )
    goto LABEL_215;
LABEL_78:
  *(_DWORD *)(a1 + 340) = 1012;
  if ( (result & 0x20) != 0 )
    goto LABEL_79;
  if ( (result & 0x40) == 0 )
    goto LABEL_219;
LABEL_80:
  *(_DWORD *)(a1 + 340) = 1021;
  if ( (result & 0x80u) != 0LL )
    goto LABEL_81;
  if ( (result & 0x100) == 0 )
    goto LABEL_223;
LABEL_82:
  *(_DWORD *)(a1 + 340) = 1030;
  if ( (result & 0x200) != 0 )
    goto LABEL_83;
  if ( (result & 0x400) == 0 )
    goto LABEL_227;
LABEL_84:
  *(_DWORD *)(a1 + 340) = 1032;
  if ( (result & 0x800) != 0 )
    goto LABEL_85;
  if ( (result & 0x1000) == 0 )
    goto LABEL_231;
LABEL_86:
  *(_DWORD *)(a1 + 340) = 1101;
  if ( (result & 0x2000) != 0 )
    goto LABEL_87;
  if ( (result & 0x4000) == 0 )
    goto LABEL_235;
LABEL_88:
  *(_DWORD *)(a1 + 340) = 1200;
  if ( (result & 0x8000) != 0 )
    goto LABEL_89;
  if ( (result & 0x10000) == 0 )
    goto LABEL_239;
LABEL_90:
  *(_DWORD *)(a1 + 340) = 1202;
  if ( (result & 0x20000) != 0 )
    goto LABEL_91;
  if ( (result & 0x40000) != 0 )
  {
LABEL_92:
    *(_DWORD *)(a1 + 340) = 1211;
    if ( (result & 0x80000) == 0 )
      goto LABEL_93;
LABEL_245:
    *(_DWORD *)(a1 + 340) = 1212;
    goto LABEL_93;
  }
LABEL_243:
  if ( (result & 0x80000) != 0 && *(_DWORD *)(a1 + 340) <= 0x4BBu )
    goto LABEL_245;
LABEL_93:
  if ( (result & 0x100000) != 0 && *(int *)(a1 + 364) <= 0 )
    *(_DWORD *)(a1 + 364) = 1;
  if ( (result & 0x200000) != 0 )
    *(_BYTE *)(a1 + 368) = 1;
  return result;
}
