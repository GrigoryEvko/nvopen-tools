// Function: sub_AF3870
// Address: 0xaf3870
//
const char *__fastcall sub_AF3870(unsigned int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0u:
      return "DISPFlagZero";
    case 1u:
      return "DISPFlagVirtual";
    case 2u:
      return "DISPFlagPureVirtual";
    case 3u:
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
    case 0x24u:
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
    case 4u:
      return "DISPFlagLocalToUnit";
    case 8u:
      return "DISPFlagDefinition";
    case 0x10u:
      return "DISPFlagOptimized";
    case 0x20u:
      return "DISPFlagPure";
    case 0x40u:
      return "DISPFlagElemental";
    default:
      if ( a1 == 512 )
        return "DISPFlagDeleted";
      if ( a1 <= 0x200 )
      {
        result = "DISPFlagRecursive";
        if ( a1 == 128 )
          return result;
        if ( a1 == 256 )
          return "DISPFlagMainSubprogram";
      }
      else
      {
        result = "DISPFlagObjCDirect";
        if ( a1 == 2048 )
          return result;
      }
      return byte_3F871B3;
  }
}
