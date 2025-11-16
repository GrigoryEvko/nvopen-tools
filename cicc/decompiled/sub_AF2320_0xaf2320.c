// Function: sub_AF2320
// Address: 0xaf2320
//
const char *__fastcall sub_AF2320(unsigned int a1)
{
  const char *result; // rax

  if ( a1 == 0x2000 )
    return "DIFlagLValueReference";
  if ( a1 <= 0x2000 )
  {
    switch ( a1 )
    {
      case 0u:
        return "DIFlagZero";
      case 1u:
        return "DIFlagPrivate";
      case 2u:
        return "DIFlagProtected";
      case 3u:
        return "DIFlagPublic";
      case 4u:
        return "DIFlagFwdDecl";
      case 5u:
      case 6u:
      case 7u:
      case 9u:
      case 0xAu:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0xFu:
      case 0x11u:
      case 0x12u:
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x21u:
      case 0x22u:
      case 0x23u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x29u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Eu:
      case 0x2Fu:
      case 0x30u:
      case 0x31u:
      case 0x32u:
      case 0x33u:
      case 0x34u:
      case 0x35u:
      case 0x36u:
      case 0x37u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
      case 0x3Cu:
      case 0x3Du:
      case 0x3Eu:
      case 0x3Fu:
        return byte_3F871B3;
      case 8u:
        return "DIFlagAppleBlock";
      case 0x10u:
        return "DIFlagReservedBit4";
      case 0x20u:
        return "DIFlagVirtual";
      case 0x24u:
        return "DIFlagIndirectVirtualBase";
      case 0x40u:
        return "DIFlagArtificial";
      default:
        if ( a1 == 1024 )
          return "DIFlagObjectPointer";
        if ( a1 <= 0x400 )
        {
          result = "DIFlagPrototyped";
          switch ( a1 )
          {
            case 0x100u:
              return result;
            case 0x200u:
              return "DIFlagObjcClassComplete";
            case 0x80u:
              return "DIFlagExplicit";
          }
        }
        else
        {
          result = "DIFlagVector";
          if ( a1 == 2048 )
            return result;
          if ( a1 == 4096 )
            return "DIFlagStaticMember";
        }
        break;
    }
    return byte_3F871B3;
  }
  if ( a1 == (_DWORD)&dword_400000 )
    return "DIFlagTypePassByValue";
  if ( a1 <= (unsigned int)&dword_400000 )
  {
    if ( a1 == 196608 )
    {
      return "DIFlagVirtualInheritance";
    }
    else if ( a1 <= 0x30000 )
    {
      if ( a1 == 0x10000 )
      {
        return "DIFlagSingleInheritance";
      }
      else if ( a1 <= 0x10000 )
      {
        result = "DIFlagRValueReference";
        if ( a1 != 0x4000 )
        {
          if ( a1 == 0x8000 )
            return "DIFlagExportSymbols";
          return byte_3F871B3;
        }
      }
      else
      {
        result = "DIFlagMultipleInheritance";
        if ( a1 != 0x20000 )
          return byte_3F871B3;
      }
    }
    else
    {
      result = "DIFlagBitField";
      if ( a1 != 0x80000 )
      {
        if ( a1 == 0x100000 )
          return "DIFlagNoReturn";
        if ( a1 == 0x40000 )
          return "DIFlagIntroducedVirtual";
        return byte_3F871B3;
      }
    }
  }
  else if ( a1 == 0x4000000 )
  {
    return "DIFlagNonTrivial";
  }
  else if ( a1 <= 0x4000000 )
  {
    result = "DIFlagEnumClass";
    if ( a1 != (_DWORD)&loc_1000000 )
    {
      if ( a1 == 0x2000000 )
        return "DIFlagThunk";
      if ( a1 == 0x800000 )
        return "DIFlagTypePassByReference";
      return byte_3F871B3;
    }
  }
  else
  {
    result = "DIFlagLittleEndian";
    if ( a1 != 0x10000000 )
    {
      if ( a1 == 0x20000000 )
        return "DIFlagAllCallsDescribed";
      if ( a1 == 0x8000000 )
        return "DIFlagBigEndian";
      return byte_3F871B3;
    }
  }
  return result;
}
