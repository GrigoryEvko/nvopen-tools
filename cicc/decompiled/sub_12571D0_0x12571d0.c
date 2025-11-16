// Function: sub_12571D0
// Address: 0x12571d0
//
char *sub_12571D0()
{
  int v0; // eax
  int v1; // r8d
  unsigned int v11; // r10d
  unsigned int v12; // r15d
  int v13; // r14d
  int v14; // r12d
  int v15; // r13d
  int v16; // edi
  int v17; // ebx
  int v18; // r11d
  int v19; // r9d
  int v20; // ecx
  int v21; // esi
  int v22; // eax
  int v23; // eax
  char v24; // al
  char v25; // r12
  bool v26; // r9
  __int16 v27; // r11
  int v68; // eax
  int v69; // eax
  int v70; // eax
  int v71; // eax
  char *result; // rax
  int v73; // edi
  unsigned int v74; // [rsp+8h] [rbp-58h]
  int v75; // [rsp+Ch] [rbp-54h]
  int v76; // [rsp+10h] [rbp-50h]
  int v77; // [rsp+14h] [rbp-4Ch]
  int v78; // [rsp+18h] [rbp-48h]
  unsigned int v79[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v79[0] = 0;
  v0 = sub_1257130((char *)v79);
  if ( !v0 )
    return "generic";
  v1 = v0;
  _RAX = 0;
  __asm { cpuid }
  if ( (_DWORD)_RAX )
  {
    _RAX = 1;
    __asm { cpuid }
    v11 = ((unsigned int)_RAX >> 8) & 0xF;
    v12 = (unsigned __int8)_RAX >> 4;
    if ( v11 == 6 )
    {
LABEL_4:
      v78 = _RCX & 2;
      v77 = _RCX & 0x200;
      v12 += ((unsigned int)_RAX >> 12) & 0xF0;
      v76 = _RCX & 0x1000;
      LODWORD(_RAX) = _RDX & 0x800000;
      v75 = _RCX & 0x80000;
      v13 = _RDX & 0x2000000;
      v14 = _RCX & 1;
      v15 = _RDX & 0x4000000;
      v16 = _RCX & 0x100000;
      v17 = _RCX & 0x800000;
      v18 = _RCX & 0x2000000;
      v19 = _RCX & 0x400000;
      v20 = _RCX & 0x18000000;
      v21 = (_RDX & 0x8000) != 0;
      goto LABEL_5;
    }
    if ( v11 == 15 )
    {
      v11 = (unsigned __int8)((unsigned int)_RAX >> 20) + 15;
      goto LABEL_4;
    }
    v78 = _RCX & 2;
    LODWORD(_RAX) = _RDX & 0x800000;
    v13 = _RDX & 0x2000000;
    v77 = _RCX & 0x200;
    v15 = _RDX & 0x4000000;
    v14 = _RCX & 1;
    v17 = _RCX & 0x800000;
    v21 = ((unsigned int)_RDX >> 15) & 1;
    v18 = _RCX & 0x2000000;
    v76 = _RCX & 0x1000;
    v19 = _RCX & 0x400000;
    v75 = _RCX & 0x80000;
    v73 = _RCX;
    v20 = _RCX & 0x18000000;
    v16 = v73 & 0x100000;
  }
  else
  {
    v75 = 0;
    v21 = 0;
    v20 = 0;
    v19 = 0;
    v76 = 0;
    v18 = 0;
    v17 = 0;
    v16 = 0;
    v77 = 0;
    v14 = 0;
    v15 = 0;
    v13 = 0;
    v78 = 0;
    v12 = 0;
    v11 = 0;
  }
LABEL_5:
  if ( (_DWORD)_RAX )
    v21 |= 2u;
  if ( v13 )
    v21 |= 8u;
  if ( v15 )
    v21 |= 0x10u;
  if ( v14 )
    v21 |= 0x20u;
  if ( v78 )
    v21 |= 0x80000u;
  if ( v77 )
    v21 |= 0x40u;
  v22 = v21;
  if ( v76 )
  {
    BYTE1(v22) = BYTE1(v21) | 0x40;
    v21 = v22;
  }
  v23 = v21;
  if ( v75 )
  {
    LOBYTE(v23) = v21 | 0x80;
    v21 = v23;
  }
  if ( v16 )
  {
    v21 |= 0x100u;
    v16 = 0x80000;
  }
  if ( v17 )
    v21 |= 4u;
  if ( v18 )
    v21 |= 0x40000u;
  v24 = v16;
  if ( v19 )
    v16 |= 0x4000000u;
  v25 = 0;
  v26 = 0;
  if ( v20 == 402653184 )
  {
    __asm { xgetbv }
    if ( (v24 & 6) == 6 )
    {
      v25 = 1;
      v26 = (v24 & 0xE0) == 224;
      v21 |= 0x200u;
    }
  }
  v27 = 0;
  if ( v79[0] > 6 )
  {
    _RAX = 0;
    __asm { cpuid }
    if ( (unsigned int)_RAX > 6 )
    {
      _RAX = 7;
      __asm { cpuid }
      if ( (_RBX & 8) != 0 )
        v21 |= 0x10000u;
      if ( (_RBX & 0x20) != 0 && v25 )
        v21 |= 0x400u;
      if ( (_RBX & 0x100) != 0 )
        v21 |= 0x20000u;
      if ( (_RBX & 0x10000) != 0 && v26 )
      {
        v21 |= 0x8000u;
        v27 = 2;
      }
      else
      {
        v27 = 0;
      }
      if ( (_RBX & 0x20000) != 0 && v26 )
        v21 |= 0x400000u;
      if ( (_RBX & 0x80000) != 0 )
        v16 |= 0x100u;
      if ( (_RBX & 0x200000) != 0 && v26 )
        v21 |= 0x8000000u;
      if ( (_RBX & 0x800000) != 0 )
        v16 |= 0x800u;
      if ( (_RBX & 0x10000000) != 0 && v26 )
        v21 |= 0x800000u;
      if ( (_RBX & 0x20000000) != 0 )
        v27 |= 0x400u;
      if ( (_RBX & 0x40000000) != 0 && v26 )
        v21 |= 0x200000u;
      if ( (int)_RBX < 0 && v26 )
        v21 |= 0x100000u;
      if ( (_RCX & 2) != 0 && v26 )
        v21 |= 0x4000000u;
      if ( (_RCX & 0x40) != 0 && v26 )
        v21 |= 0x80000000;
      if ( (_RCX & 0x100) != 0 )
        v16 |= 1u;
      if ( (_RCX & 0x400) != 0 && v25 )
        v16 |= 2u;
      if ( (_RCX & 0x800) != 0 && v26 )
        v16 |= 4u;
      if ( (_RCX & 0x1000) != 0 && v26 )
        v16 |= 8u;
      if ( (_RCX & 0x4000) != 0 && v26 )
        v21 |= 0x40000000u;
      if ( (_RDX & 4) != 0 && v26 )
        v21 |= 0x10000000u;
      if ( (_RDX & 8) != 0 && v26 )
        v21 |= 0x20000000u;
      if ( (_RDX & 0x100) != 0 && v26 )
        v16 |= 0x20u;
      if ( (_DWORD)_RAX )
      {
        _RAX = 0;
        __asm { cpuid }
        if ( (unsigned int)_RAX > 6 )
        {
          _RAX = 7;
          __asm { cpuid }
          if ( (_RAX & 0x20) != 0 && v26 )
            v16 |= 0x10u;
        }
      }
    }
  }
  _RAX = 0x80000000LL;
  __asm { cpuid }
  if ( (int)_RAX < 0 )
  {
    _RAX = 0x80000000LL;
    __asm { cpuid }
    v74 = _RAX;
  }
  if ( v74 > 0x80000000 )
  {
    _RAX = 0x80000000LL;
    __asm { cpuid }
    if ( (unsigned int)_RAX > 0x80000000 )
    {
      _RAX = 2147483649LL;
      __asm { cpuid }
      v68 = v21;
      if ( (_RCX & 0x40) != 0 )
      {
        BYTE1(v68) = BYTE1(v21) | 8;
        v21 = v68;
      }
      v69 = v21;
      if ( (_RCX & 0x800) != 0 )
      {
        BYTE1(v69) = BYTE1(v21) | 0x20;
        v21 = v69;
      }
      v70 = v21;
      if ( (_RCX & 0x10000) != 0 )
      {
        BYTE1(v70) = BYTE1(v21) | 0x10;
        v21 = v70;
      }
      v71 = v16;
      if ( (_RDX & 0x20000000) != 0 )
      {
        BYTE1(v71) = BYTE1(v16) | 2;
        v16 = v71;
      }
    }
  }
  if ( v1 == 1 )
  {
    switch ( v11 )
    {
      case 3u:
        result = "i386";
        break;
      case 4u:
LABEL_132:
        result = "i486";
        break;
      case 5u:
        result = "pentium-mmx";
        if ( (v21 & 2) == 0 )
          result = "pentium";
        break;
      case 6u:
        switch ( v12 )
        {
          case 0xFu:
          case 0x16u:
            goto LABEL_216;
          case 0x17u:
          case 0x1Du:
            goto LABEL_217;
          case 0x1Au:
          case 0x1Eu:
          case 0x1Fu:
          case 0x2Eu:
            return "nehalem";
          case 0x1Cu:
          case 0x26u:
          case 0x27u:
          case 0x35u:
          case 0x36u:
            return "bonnell";
          case 0x25u:
          case 0x2Cu:
          case 0x2Fu:
            return "westmere";
          case 0x2Au:
          case 0x2Du:
            goto LABEL_205;
          case 0x37u:
          case 0x4Au:
          case 0x4Cu:
          case 0x4Du:
          case 0x5Au:
          case 0x5Du:
            return "silvermont";
          case 0x3Au:
          case 0x3Eu:
            return "ivybridge";
          case 0x3Cu:
          case 0x3Fu:
          case 0x45u:
          case 0x46u:
            goto LABEL_182;
          case 0x3Du:
          case 0x47u:
          case 0x4Fu:
          case 0x56u:
            goto LABEL_181;
          case 0x4Eu:
          case 0x5Eu:
          case 0x8Eu:
          case 0x9Eu:
          case 0xA5u:
          case 0xA6u:
            return "skylake";
          case 0x55u:
            if ( (v16 & 0x10) != 0 )
              goto LABEL_222;
            result = "skylake-avx512";
            if ( (v16 & 4) != 0 )
              result = "cascadelake";
            break;
          case 0x57u:
            return "knl";
          case 0x5Cu:
          case 0x5Fu:
            return "goldmont";
          case 0x66u:
            goto LABEL_207;
          case 0x6Au:
          case 0x6Cu:
            return "icelake-server";
          case 0x7Au:
            return "goldmont-plus";
          case 0x7Du:
          case 0x7Eu:
            goto LABEL_188;
          case 0x85u:
            return "knm";
          case 0x86u:
          case 0x8Au:
          case 0x96u:
          case 0x9Cu:
            return "tremont";
          case 0x8Cu:
          case 0x8Du:
            goto LABEL_187;
          case 0x8Fu:
            return "sapphirerapids";
          case 0x97u:
          case 0x9Au:
            return "alderlake";
          case 0xA7u:
            return "rocketlake";
          case 0xAAu:
          case 0xACu:
            return "meteorlake";
          case 0xADu:
            return "graniterapids";
          case 0xAEu:
            return "graniterapids-d";
          case 0xAFu:
            return "sierraforest";
          case 0xB5u:
          case 0xC5u:
            return "arrowlake";
          case 0xB6u:
            return "grandridge";
          case 0xB7u:
          case 0xBAu:
          case 0xBFu:
            return "raptorlake";
          case 0xBDu:
            return "lunarlake";
          case 0xBEu:
            return "gracemont";
          case 0xC6u:
            return "arrowlake-s";
          case 0xCCu:
            return "pantherlake";
          case 0xCFu:
            return "emeraldrapids";
          case 0xDDu:
            return "clearwaterforest";
          default:
            if ( (v16 & 0x20) != 0 )
            {
LABEL_187:
              result = "tigerlake";
            }
            else if ( v21 < 0 )
            {
LABEL_188:
              result = "icelake-client";
            }
            else if ( (v21 & 0x4000000) != 0 )
            {
LABEL_207:
              result = "cannonlake";
            }
            else if ( (v16 & 0x10) != 0 )
            {
LABEL_222:
              result = "cooperlake";
            }
            else if ( (v16 & 4) != 0 )
            {
              result = "cascadelake";
            }
            else if ( (v21 & 0x100000) != 0 )
            {
              result = "skylake-avx512";
            }
            else if ( (v16 & 0x800) != 0 )
            {
              result = "goldmont";
              if ( (v27 & 0x400) == 0 )
                result = "skylake";
            }
            else if ( (v16 & 0x100) != 0 )
            {
LABEL_181:
              result = "broadwell";
            }
            else if ( (v21 & 0x400) != 0 )
            {
LABEL_182:
              result = "haswell";
            }
            else if ( (v21 & 0x200) != 0 )
            {
LABEL_205:
              result = "sandybridge";
            }
            else if ( (v21 & 0x100) != 0 )
            {
              result = "silvermont";
              if ( (v16 & 0x4000000) == 0 )
                result = "nehalem";
            }
            else if ( (v21 & 0x80u) != 0 )
            {
LABEL_217:
              result = "penryn";
            }
            else if ( (v21 & 0x40) != 0 )
            {
              result = "core2";
              if ( (v16 & 0x4000000) != 0 )
                result = "bonnell";
            }
            else if ( (v16 & 0x200) != 0 )
            {
LABEL_216:
              result = "core2";
            }
            else if ( (v21 & 0x20) != 0 )
            {
              result = "yonah";
            }
            else if ( (v21 & 0x10) != 0 )
            {
              result = "pentium-m";
            }
            else if ( (v21 & 8) != 0 )
            {
              result = "pentium3";
            }
            else
            {
              result = "pentium2";
              if ( (v21 & 2) == 0 )
                result = "pentiumpro";
            }
            break;
        }
        break;
      case 0xFu:
        result = "nocona";
        if ( (v16 & 0x200) == 0 )
        {
          result = "prescott";
          if ( (v21 & 0x20) == 0 )
            result = "pentium4";
        }
        break;
      case 0x13u:
        if ( v12 != 1 )
          return "generic";
        result = "diamondrapids";
        break;
      default:
        return "generic";
    }
  }
  else
  {
    if ( v1 != 2 )
      return "generic";
    switch ( v11 )
    {
      case 4u:
        goto LABEL_132;
      case 5u:
        switch ( v12 )
        {
          case 6u:
          case 7u:
            result = "k6";
            break;
          case 8u:
            result = "k6-2";
            break;
          case 9u:
          case 0xDu:
            result = "k6-3";
            break;
          case 0xAu:
            result = "geode";
            break;
          default:
            result = "pentium";
            break;
        }
        break;
      case 6u:
        result = "athlon-xp";
        if ( (v21 & 8) == 0 )
          result = "athlon";
        break;
      case 0xFu:
        result = "k8-sse3";
        if ( (v21 & 0x20) == 0 )
          result = "k8";
        break;
      case 0x10u:
      case 0x12u:
        result = "amdfam10";
        break;
      case 0x14u:
        result = "btver1";
        break;
      case 0x15u:
        result = "bdver4";
        if ( v12 - 96 > 0x1F )
        {
          result = "bdver3";
          if ( v12 - 48 > 0xF )
          {
            if ( v12 - 16 <= 0xF || (result = "bdver1", v12 == 2) )
              result = "bdver2";
          }
        }
        break;
      case 0x16u:
        result = "btver2";
        break;
      case 0x17u:
        if ( v12 - 48 <= 0xF || v12 == 71 )
        {
          result = "znver2";
        }
        else
        {
          result = "znver2";
          if ( v12 - 96 > 0x1F && v12 - 132 > 3 && v12 - 144 >= 0x20 )
            result = "znver1";
        }
        break;
      case 0x19u:
        if ( v12 - 32 <= 0x3F || v12 <= 0xF )
        {
          result = "znver3";
        }
        else if ( v12 - 16 <= 0xF || v12 - 96 <= 0x1F )
        {
          result = "znver4";
        }
        else
        {
          result = "znver3";
          if ( v12 - 160 <= 0xF )
            result = "znver4";
        }
        break;
      case 0x1Au:
        result = "znver5";
        break;
      default:
        return "generic";
    }
  }
  return result;
}
