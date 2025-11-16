// Function: sub_DFD690
// Address: 0xdfd690
//
__int64 __fastcall sub_DFD690(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rax
  unsigned int v3; // ecx
  __int64 result; // rax

  v2 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1360LL);
  if ( (char *)v2 != (char *)sub_DF6F70 )
    return v2();
  v3 = *(_DWORD *)(a2 + 16);
  if ( v3 > 0xD3 )
  {
    if ( v3 > 0x178 )
    {
      return 1;
    }
    else if ( v3 > 0x143 )
    {
      return ((1LL << ((unsigned __int8)v3 - 68)) & 0x10000020401001LL) == 0;
    }
    else
    {
      return v3 != 282 && v3 - 291 >= 2;
    }
  }
  else
  {
    if ( v3 <= 0x94 )
    {
      switch ( v3 )
      {
        case 5u:
        case 6u:
        case 7u:
        case 8u:
        case 0xBu:
        case 0x1Bu:
        case 0x1Cu:
        case 0x27u:
        case 0x28u:
        case 0x2Bu:
        case 0x2Eu:
        case 0x2Fu:
        case 0x3Au:
        case 0x3Bu:
        case 0x3Cu:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
          return 0;
        default:
          return 1;
      }
    }
    switch ( v3 )
    {
      case 0x95u:
      case 0x96u:
      case 0x9Bu:
      case 0xA9u:
      case 0xCCu:
      case 0xCDu:
      case 0xCEu:
      case 0xD0u:
      case 0xD2u:
      case 0xD3u:
        return 0;
      case 0xA1u:
        result = 0;
        break;
      default:
        return 1;
    }
  }
  return result;
}
