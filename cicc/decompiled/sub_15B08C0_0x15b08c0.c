// Function: sub_15B08C0
// Address: 0x15b08c0
//
const char *__fastcall sub_15B08C0(unsigned int a1)
{
  const char *result; // rax

  if ( a1 == 4096 )
    return "DIFlagStaticMember";
  if ( a1 > 0x1000 )
  {
    if ( a1 == 0x80000 )
      return "DIFlagBitField";
    if ( a1 <= 0x80000 )
    {
      if ( a1 == 0x10000 )
        return "DIFlagSingleInheritance";
      if ( a1 <= 0x10000 )
      {
        result = "DIFlagRValueReference";
        switch ( a1 )
        {
          case 0x4000u:
            return result;
          case 0x8000u:
            return "DIFlagReserved";
          case 0x2000u:
            return "DIFlagLValueReference";
        }
      }
      else
      {
        result = "DIFlagVirtualInheritance";
        switch ( a1 )
        {
          case 0x30000u:
            return result;
          case 0x40000u:
            return "DIFlagIntroducedVirtual";
          case 0x20000u:
            return "DIFlagMultipleInheritance";
        }
      }
    }
    else
    {
      if ( a1 == 0x800000 )
        return "DIFlagTypePassByReference";
      if ( a1 <= 0x800000 )
      {
        result = "DIFlagMainSubprogram";
        if ( a1 == 0x200000 )
          return result;
        if ( a1 == (_DWORD)&dword_400000 )
          return "DIFlagTypePassByValue";
        if ( a1 == 0x100000 )
          return "DIFlagNoReturn";
      }
      else
      {
        result = "DIFlagThunk";
        if ( a1 == 0x2000000 )
          return result;
        if ( a1 == 0x4000000 )
          return "DIFlagTrivial";
        if ( a1 == (_DWORD)&loc_1000000 )
          return "DIFlagFixedEnum";
      }
    }
  }
  else
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
        return "DIFlagBlockByrefStruct";
      case 0x20u:
        return "DIFlagVirtual";
      case 0x24u:
        return "DIFlagIndirectVirtualBase";
      case 0x40u:
        return "DIFlagArtificial";
      default:
        if ( a1 == 512 )
          return "DIFlagObjcClassComplete";
        if ( a1 <= 0x200 )
        {
          result = "DIFlagExplicit";
          if ( a1 == 128 )
            return result;
          if ( a1 == 256 )
            return "DIFlagPrototyped";
        }
        else
        {
          result = "DIFlagObjectPointer";
          if ( a1 == 1024 )
            return result;
          if ( a1 == 2048 )
            return "DIFlagVector";
        }
        break;
    }
  }
  return byte_3F871B3;
}
