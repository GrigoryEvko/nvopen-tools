// Function: sub_370C640
// Address: 0x370c640
//
char *__fastcall sub_370C640(unsigned __int16 a1)
{
  char *result; // rax

  if ( a1 > 0x151Du )
  {
    switch ( a1 )
    {
      case 0x1601u:
        result = "FuncId";
        break;
      case 0x1602u:
        result = "MemberFuncId";
        break;
      case 0x1603u:
        result = "BuildInfo";
        break;
      case 0x1604u:
        result = "StringList";
        break;
      case 0x1605u:
        result = "StringId";
        break;
      case 0x1606u:
        result = "UdtSourceLine";
        break;
      case 0x1607u:
        result = "UdtModSourceLine";
        break;
      default:
        return "UnknownLeaf";
    }
  }
  else
  {
    if ( a1 <= 0x1501u )
    {
      if ( a1 == 4611 )
        return "FieldList";
      if ( a1 <= 0x1203u )
      {
        if ( a1 == 4098 )
          return "Pointer";
        if ( a1 <= 0x1002u )
        {
          if ( a1 == 20 )
            return "EndPrecomp";
          if ( a1 <= 0x14u )
          {
            result = "VFTableShape";
            if ( a1 == 10 )
              return result;
            if ( a1 == 14 )
              return "Label";
          }
          else
          {
            result = "Modifier";
            if ( a1 == 4097 )
              return result;
          }
        }
        else
        {
          result = "MemberFunction";
          switch ( a1 )
          {
            case 0x1009u:
              return result;
            case 0x1201u:
              return "ArgList";
            case 0x1008u:
              return "Procedure";
          }
        }
      }
      else if ( a1 <= 0x1409u )
      {
        if ( a1 > 0x13FFu )
        {
          switch ( a1 )
          {
            case 0x1400u:
              result = "BaseClass";
              break;
            case 0x1401u:
              result = "VirtualBaseClass";
              break;
            case 0x1402u:
              result = "IndirectVirtualBaseClass";
              break;
            case 0x1404u:
              result = "ListContinuation";
              break;
            case 0x1409u:
              result = "VFPtr";
              break;
            default:
              return "UnknownLeaf";
          }
          return result;
        }
        result = "BitField";
        if ( a1 == 4613 )
          return result;
        if ( a1 == 4614 )
          return "MethodOverloadList";
      }
      return "UnknownLeaf";
    }
    switch ( a1 )
    {
      case 0x1502u:
        result = "Enumerator";
        break;
      case 0x1503u:
        result = "Array";
        break;
      case 0x1504u:
        result = "Class";
        break;
      case 0x1505u:
        result = "Struct";
        break;
      case 0x1506u:
        result = "Union";
        break;
      case 0x1507u:
        result = "Enum";
        break;
      case 0x1509u:
        result = "Precomp";
        break;
      case 0x150Du:
        result = "DataMember";
        break;
      case 0x150Eu:
        result = "StaticDataMember";
        break;
      case 0x150Fu:
        result = "OverloadedMethod";
        break;
      case 0x1510u:
        result = "NestedType";
        break;
      case 0x1511u:
        result = "OneMethod";
        break;
      case 0x1515u:
        result = "TypeServer2";
        break;
      case 0x1519u:
        result = "Interface";
        break;
      case 0x151Au:
        result = "BaseInterface";
        break;
      case 0x151Du:
        result = "VFTable";
        break;
      default:
        return "UnknownLeaf";
    }
  }
  return result;
}
