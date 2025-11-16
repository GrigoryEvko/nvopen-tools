// Function: sub_21642F0
// Address: 0x21642f0
//
__int64 __fastcall sub_21642F0(__int64 a1)
{
  __int64 result; // rax

  sub_38E1E90();
  result = *(_QWORD *)(a1 + 192);
  if ( (result & 1) != 0 && *(int *)(a1 + 82308) < 0 )
  {
    *(_DWORD *)(a1 + 82308) = 0;
    if ( (result & 2) == 0 )
    {
      if ( (result & 4) == 0 )
        goto LABEL_6;
      goto LABEL_143;
    }
    goto LABEL_157;
  }
  if ( (result & 2) != 0 && *(int *)(a1 + 82308) <= 0 )
  {
LABEL_157:
    *(_DWORD *)(a1 + 82308) = 1;
    if ( (result & 4) != 0 )
      *(_DWORD *)(a1 + 82308) = 2;
    goto LABEL_6;
  }
  if ( (result & 4) != 0 && *(int *)(a1 + 82308) <= 1 )
LABEL_143:
    *(_DWORD *)(a1 + 82308) = 2;
LABEL_6:
  if ( (result & 8) != 0 && *(_DWORD *)(a1 + 248) <= 0x1Fu )
  {
    *(_DWORD *)(a1 + 248) = 32;
    if ( (result & 0x10) != 0 )
      goto LABEL_9;
    if ( (result & 0x20) != 0 )
      goto LABEL_10;
LABEL_67:
    if ( (result & 0x40) == 0 || *(_DWORD *)(a1 + 248) > 0x29u )
      goto LABEL_69;
    goto LABEL_11;
  }
  if ( (result & 0x10) != 0 && *(_DWORD *)(a1 + 248) <= 0x27u )
  {
LABEL_9:
    *(_DWORD *)(a1 + 248) = 40;
    if ( (result & 0x20) != 0 )
      goto LABEL_10;
    if ( (result & 0x40) == 0 )
    {
LABEL_69:
      if ( (result & 0x80u) == 0LL || *(_DWORD *)(a1 + 248) > 0x2Au )
        goto LABEL_71;
      goto LABEL_12;
    }
LABEL_11:
    *(_DWORD *)(a1 + 248) = 42;
    if ( (result & 0x80u) != 0LL )
      goto LABEL_12;
    if ( (result & 0x100) == 0 )
    {
LABEL_73:
      if ( (result & 0x200) == 0 || *(_DWORD *)(a1 + 248) > 0x3Bu )
        goto LABEL_75;
LABEL_14:
      *(_DWORD *)(a1 + 248) = 60;
      if ( (result & 0x400) == 0 )
        goto LABEL_15;
LABEL_77:
      *(_DWORD *)(a1 + 248) = 61;
      goto LABEL_15;
    }
LABEL_13:
    *(_DWORD *)(a1 + 248) = 50;
    if ( (result & 0x200) == 0 )
    {
      if ( (result & 0x400) == 0 )
        goto LABEL_15;
      goto LABEL_77;
    }
    goto LABEL_14;
  }
  if ( (result & 0x20) == 0 || *(_DWORD *)(a1 + 248) > 0x28u )
    goto LABEL_67;
LABEL_10:
  *(_DWORD *)(a1 + 248) = 41;
  if ( (result & 0x40) != 0 )
    goto LABEL_11;
  if ( (result & 0x80u) == 0LL )
  {
LABEL_71:
    if ( (result & 0x100) == 0 || *(_DWORD *)(a1 + 248) > 0x31u )
      goto LABEL_73;
    goto LABEL_13;
  }
LABEL_12:
  *(_DWORD *)(a1 + 248) = 43;
  if ( (result & 0x100) != 0 )
    goto LABEL_13;
  if ( (result & 0x200) != 0 )
    goto LABEL_14;
LABEL_75:
  if ( (result & 0x400) != 0 && *(_DWORD *)(a1 + 248) <= 0x3Cu )
    goto LABEL_77;
LABEL_15:
  if ( (result & 0x800) != 0 && *(int *)(a1 + 82312) < 0 )
  {
    *(_DWORD *)(a1 + 82312) = 0;
    if ( (result & 0x1000) == 0 )
    {
      if ( (result & 0x2000) == 0 )
        goto LABEL_21;
LABEL_154:
      *(_DWORD *)(a1 + 82312) = 2;
      if ( (result & 0x4000) != 0 )
        *(_DWORD *)(a1 + 82312) = 3;
      goto LABEL_22;
    }
LABEL_153:
    *(_DWORD *)(a1 + 82312) = 1;
    if ( (result & 0x2000) == 0 )
    {
      if ( (result & 0x4000) == 0 )
        goto LABEL_22;
      goto LABEL_141;
    }
    goto LABEL_154;
  }
  if ( (result & 0x1000) != 0 && *(int *)(a1 + 82312) <= 0 )
    goto LABEL_153;
  if ( (result & 0x2000) != 0 && *(int *)(a1 + 82312) <= 1 )
    goto LABEL_154;
LABEL_21:
  if ( (result & 0x4000) != 0 && *(int *)(a1 + 82312) <= 2 )
LABEL_141:
    *(_DWORD *)(a1 + 82312) = 3;
LABEL_22:
  if ( (result & 0x8000) != 0 && *(int *)(a1 + 82316) < 0 )
  {
    *(_DWORD *)(a1 + 82316) = 0;
    if ( (result & 0x10000) != 0 )
      *(_DWORD *)(a1 + 82316) = 1;
  }
  else if ( (result & 0x10000) != 0 && *(int *)(a1 + 82316) <= 0 )
  {
    *(_DWORD *)(a1 + 82316) = 1;
  }
  if ( (result & 0x20000) != 0 && *(_DWORD *)(a1 + 252) <= 0x13u )
  {
    *(_DWORD *)(a1 + 252) = 20;
    if ( (result & 0x40000) != 0 )
      goto LABEL_28;
    if ( (result & 0x80000) != 0 )
      goto LABEL_29;
LABEL_82:
    if ( (result & 0x100000) == 0 || *(_DWORD *)(a1 + 252) > 0x1Fu )
      goto LABEL_84;
    goto LABEL_30;
  }
  if ( (result & 0x40000) != 0 && *(_DWORD *)(a1 + 252) <= 0x14u )
  {
LABEL_28:
    *(_DWORD *)(a1 + 252) = 21;
    if ( (result & 0x80000) != 0 )
      goto LABEL_29;
    if ( (result & 0x100000) == 0 )
    {
LABEL_84:
      if ( (result & 0x200000) == 0 || *(_DWORD *)(a1 + 252) > 0x22u )
        goto LABEL_86;
      goto LABEL_31;
    }
LABEL_30:
    *(_DWORD *)(a1 + 252) = 32;
    if ( (result & 0x200000) != 0 )
      goto LABEL_31;
    if ( (result & 0x400000) == 0 )
    {
LABEL_88:
      if ( (result & 0x800000) == 0 || *(_DWORD *)(a1 + 252) > 0x31u )
        goto LABEL_90;
      goto LABEL_33;
    }
LABEL_32:
    *(_DWORD *)(a1 + 252) = 37;
    if ( (result & 0x800000) != 0 )
      goto LABEL_33;
    if ( (result & 0x1000000) == 0 )
    {
LABEL_92:
      if ( (result & 0x2000000) == 0 || *(_DWORD *)(a1 + 252) > 0x34u )
        goto LABEL_94;
      goto LABEL_35;
    }
LABEL_34:
    *(_DWORD *)(a1 + 252) = 52;
    if ( (result & 0x2000000) != 0 )
      goto LABEL_35;
    if ( (result & 0x4000000) == 0 )
    {
LABEL_96:
      if ( (result & 0x8000000) == 0 || *(_DWORD *)(a1 + 252) > 0x3Cu )
        goto LABEL_98;
      goto LABEL_37;
    }
LABEL_36:
    *(_DWORD *)(a1 + 252) = 60;
    if ( (result & 0x8000000) != 0 )
      goto LABEL_37;
    if ( (result & 0x10000000) == 0 )
    {
LABEL_100:
      if ( (result & 0x20000000) == 0 || *(_DWORD *)(a1 + 252) > 0x45u )
        goto LABEL_102;
      goto LABEL_39;
    }
LABEL_38:
    *(_DWORD *)(a1 + 252) = 62;
    if ( (result & 0x20000000) != 0 )
      goto LABEL_39;
    if ( (result & 0x40000000) == 0 )
    {
LABEL_104:
      if ( (int)result >= 0 || *(_DWORD *)(a1 + 252) > 0x48u )
        goto LABEL_106;
      goto LABEL_41;
    }
LABEL_40:
    *(_DWORD *)(a1 + 252) = 72;
    if ( (int)result < 0 )
      goto LABEL_41;
    if ( (result & 0x100000000LL) == 0 )
    {
LABEL_108:
      if ( (result & 0x200000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x4Fu )
        goto LABEL_110;
      goto LABEL_43;
    }
LABEL_42:
    *(_DWORD *)(a1 + 252) = 75;
    if ( (result & 0x200000000LL) != 0 )
      goto LABEL_43;
    if ( (result & 0x400000000LL) == 0 )
    {
LABEL_112:
      if ( (result & 0x800000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x55u )
        goto LABEL_114;
      goto LABEL_45;
    }
LABEL_44:
    *(_DWORD *)(a1 + 252) = 82;
    if ( (result & 0x800000000LL) != 0 )
      goto LABEL_45;
    if ( (result & 0x1000000000LL) == 0 )
    {
LABEL_116:
      if ( (result & 0x2000000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x57u )
        goto LABEL_118;
      goto LABEL_47;
    }
LABEL_46:
    *(_DWORD *)(a1 + 252) = 87;
    if ( (result & 0x2000000000LL) != 0 )
      goto LABEL_47;
    if ( (result & 0x4000000000LL) == 0 )
    {
LABEL_120:
      if ( (result & 0x8000000000LL) == 0 )
        goto LABEL_145;
      if ( *(_DWORD *)(a1 + 252) > 0x59u )
        goto LABEL_122;
      goto LABEL_49;
    }
LABEL_48:
    *(_DWORD *)(a1 + 252) = 89;
    if ( (result & 0x8000000000LL) != 0 )
      goto LABEL_49;
    if ( (result & 0x10000000000LL) == 0 )
    {
LABEL_122:
      if ( (result & 0x20000000000LL) == 0 )
        goto LABEL_50;
      goto LABEL_123;
    }
LABEL_147:
    *(_DWORD *)(a1 + 252) = 90;
    if ( (result & 0x20000000000LL) == 0 )
    {
      if ( (result & 0x40000000000LL) != 0 )
        goto LABEL_52;
      goto LABEL_124;
    }
LABEL_174:
    *(_DWORD *)(a1 + 252) = 100;
    goto LABEL_124;
  }
  if ( (result & 0x80000) == 0 || *(_DWORD *)(a1 + 252) > 0x1Du )
    goto LABEL_82;
LABEL_29:
  *(_DWORD *)(a1 + 252) = 30;
  if ( (result & 0x100000) != 0 )
    goto LABEL_30;
  if ( (result & 0x200000) == 0 )
  {
LABEL_86:
    if ( (result & 0x400000) == 0 || *(_DWORD *)(a1 + 252) > 0x24u )
      goto LABEL_88;
    goto LABEL_32;
  }
LABEL_31:
  *(_DWORD *)(a1 + 252) = 35;
  if ( (result & 0x400000) != 0 )
    goto LABEL_32;
  if ( (result & 0x800000) == 0 )
  {
LABEL_90:
    if ( (result & 0x1000000) == 0 || *(_DWORD *)(a1 + 252) > 0x33u )
      goto LABEL_92;
    goto LABEL_34;
  }
LABEL_33:
  *(_DWORD *)(a1 + 252) = 50;
  if ( (result & 0x1000000) != 0 )
    goto LABEL_34;
  if ( (result & 0x2000000) == 0 )
  {
LABEL_94:
    if ( (result & 0x4000000) == 0 || *(_DWORD *)(a1 + 252) > 0x3Bu )
      goto LABEL_96;
    goto LABEL_36;
  }
LABEL_35:
  *(_DWORD *)(a1 + 252) = 53;
  if ( (result & 0x4000000) != 0 )
    goto LABEL_36;
  if ( (result & 0x8000000) == 0 )
  {
LABEL_98:
    if ( (result & 0x10000000) == 0 || *(_DWORD *)(a1 + 252) > 0x3Du )
      goto LABEL_100;
    goto LABEL_38;
  }
LABEL_37:
  *(_DWORD *)(a1 + 252) = 61;
  if ( (result & 0x10000000) != 0 )
    goto LABEL_38;
  if ( (result & 0x20000000) == 0 )
  {
LABEL_102:
    if ( (result & 0x40000000) == 0 || *(_DWORD *)(a1 + 252) > 0x47u )
      goto LABEL_104;
    goto LABEL_40;
  }
LABEL_39:
  *(_DWORD *)(a1 + 252) = 70;
  if ( (result & 0x40000000) != 0 )
    goto LABEL_40;
  if ( (int)result >= 0 )
  {
LABEL_106:
    if ( (result & 0x100000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x4Au )
      goto LABEL_108;
    goto LABEL_42;
  }
LABEL_41:
  *(_DWORD *)(a1 + 252) = 73;
  if ( (result & 0x100000000LL) != 0 )
    goto LABEL_42;
  if ( (result & 0x200000000LL) == 0 )
  {
LABEL_110:
    if ( (result & 0x400000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x51u )
      goto LABEL_112;
    goto LABEL_44;
  }
LABEL_43:
  *(_DWORD *)(a1 + 252) = 80;
  if ( (result & 0x400000000LL) != 0 )
    goto LABEL_44;
  if ( (result & 0x800000000LL) == 0 )
  {
LABEL_114:
    if ( (result & 0x1000000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x56u )
      goto LABEL_116;
    goto LABEL_46;
  }
LABEL_45:
  *(_DWORD *)(a1 + 252) = 86;
  if ( (result & 0x1000000000LL) != 0 )
    goto LABEL_46;
  if ( (result & 0x2000000000LL) == 0 )
  {
LABEL_118:
    if ( (result & 0x4000000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x58u )
      goto LABEL_120;
    goto LABEL_48;
  }
LABEL_47:
  *(_DWORD *)(a1 + 252) = 88;
  if ( (result & 0x4000000000LL) != 0 )
    goto LABEL_48;
  if ( (result & 0x8000000000LL) == 0 )
  {
LABEL_145:
    if ( (result & 0x10000000000LL) == 0 || *(_DWORD *)(a1 + 252) > 0x59u )
      goto LABEL_122;
    goto LABEL_147;
  }
LABEL_49:
  *(_DWORD *)(a1 + 252) = 90;
  if ( (result & 0x20000000000LL) != 0 )
  {
LABEL_123:
    if ( *(_DWORD *)(a1 + 252) > 0x63u )
      goto LABEL_124;
    goto LABEL_174;
  }
LABEL_50:
  if ( (result & 0x40000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x63u )
  {
LABEL_52:
    *(_DWORD *)(a1 + 252) = 100;
    if ( (result & 0x80000000000LL) == 0 )
    {
      if ( (result & 0x100000000000LL) != 0 )
        goto LABEL_54;
      goto LABEL_126;
    }
LABEL_176:
    *(_DWORD *)(a1 + 252) = 101;
    goto LABEL_126;
  }
LABEL_124:
  if ( (result & 0x80000000000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 252) > 0x64u )
      goto LABEL_126;
    goto LABEL_176;
  }
  if ( (result & 0x100000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x64u )
  {
LABEL_54:
    *(_DWORD *)(a1 + 252) = 101;
    if ( (result & 0x200000000000LL) == 0 )
    {
      if ( (result & 0x400000000000LL) != 0 )
        goto LABEL_56;
      goto LABEL_128;
    }
LABEL_175:
    *(_DWORD *)(a1 + 252) = 103;
    goto LABEL_128;
  }
LABEL_126:
  if ( (result & 0x200000000000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 252) > 0x66u )
      goto LABEL_128;
    goto LABEL_175;
  }
  if ( (result & 0x400000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x66u )
  {
LABEL_56:
    *(_DWORD *)(a1 + 252) = 103;
    if ( (result & 0x800000000000LL) == 0 )
    {
      if ( (result & 0x1000000000000LL) != 0 )
        goto LABEL_58;
      goto LABEL_130;
    }
LABEL_178:
    *(_DWORD *)(a1 + 252) = 110;
    goto LABEL_130;
  }
LABEL_128:
  if ( (result & 0x800000000000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 252) > 0x6Du )
      goto LABEL_130;
    goto LABEL_178;
  }
  if ( (result & 0x1000000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x6Du )
  {
LABEL_58:
    *(_DWORD *)(a1 + 252) = 110;
    if ( (result & 0x2000000000000LL) == 0 )
    {
      if ( (result & 0x4000000000000LL) != 0 )
        goto LABEL_60;
      goto LABEL_132;
    }
LABEL_177:
    *(_DWORD *)(a1 + 252) = 120;
    goto LABEL_132;
  }
LABEL_130:
  if ( (result & 0x2000000000000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 252) > 0x77u )
      goto LABEL_132;
    goto LABEL_177;
  }
  if ( (result & 0x4000000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x77u )
  {
LABEL_60:
    *(_DWORD *)(a1 + 252) = 120;
    if ( (result & 0x8000000000000LL) != 0 || (result & 0x10000000000000LL) != 0 )
      goto LABEL_62;
    goto LABEL_134;
  }
LABEL_132:
  if ( (result & 0x8000000000000LL) != 0 )
  {
    if ( *(_DWORD *)(a1 + 252) <= 0x78u )
LABEL_62:
      *(_DWORD *)(a1 + 252) = 121;
  }
  else if ( (result & 0x10000000000000LL) != 0 && *(_DWORD *)(a1 + 252) <= 0x78u )
  {
    *(_DWORD *)(a1 + 252) = 121;
  }
LABEL_134:
  if ( (result & 0x20000000000000LL) != 0 )
  {
    result = *(unsigned int *)(a1 + 82320);
    if ( (int)result <= 0 )
      *(_DWORD *)(a1 + 82320) = 1;
  }
  return result;
}
