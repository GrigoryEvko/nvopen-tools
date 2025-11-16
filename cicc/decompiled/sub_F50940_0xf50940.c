// Function: sub_F50940
// Address: 0xf50940
//
__int64 __fastcall sub_F50940(_QWORD *a1, unsigned int a2)
{
  __int64 result; // rax

  if ( (a1[((unsigned __int64)a2 >> 6) + 1] & (1LL << a2)) != 0
    || (((int)*(unsigned __int8 *)(*a1 + (a2 >> 2)) >> (2 * (a2 & 3))) & 3) == 0 )
  {
    return 0;
  }
  switch ( a2 )
  {
    case 0x8Du:
    case 0x8Eu:
    case 0x8Fu:
    case 0xA0u:
    case 0xA1u:
    case 0xA5u:
    case 0xA7u:
    case 0xA8u:
    case 0xACu:
    case 0xADu:
    case 0xAEu:
    case 0xAFu:
    case 0xB0u:
    case 0xB1u:
    case 0xB5u:
    case 0xBAu:
    case 0xC4u:
    case 0xC5u:
    case 0xC6u:
    case 0xCBu:
    case 0xCCu:
    case 0xCDu:
    case 0xCEu:
    case 0xCFu:
    case 0xD0u:
    case 0xD1u:
    case 0xD2u:
    case 0xD3u:
    case 0xE4u:
    case 0xE5u:
    case 0xE6u:
    case 0xE7u:
    case 0xE8u:
    case 0xE9u:
    case 0xEFu:
    case 0xF0u:
    case 0xF1u:
    case 0x102u:
    case 0x103u:
    case 0x104u:
    case 0x108u:
    case 0x109u:
    case 0x10Au:
    case 0x10Bu:
    case 0x10Cu:
    case 0x10Du:
    case 0x149u:
    case 0x14Au:
    case 0x14Bu:
    case 0x154u:
    case 0x155u:
    case 0x156u:
    case 0x164u:
    case 0x165u:
    case 0x166u:
    case 0x167u:
    case 0x168u:
    case 0x16Au:
    case 0x176u:
    case 0x177u:
    case 0x178u:
    case 0x1A0u:
    case 0x1A1u:
    case 0x1A2u:
    case 0x1A4u:
    case 0x1A8u:
    case 0x1A9u:
    case 0x1B4u:
    case 0x1B5u:
    case 0x1B6u:
    case 0x1B7u:
    case 0x1B8u:
    case 0x1B9u:
    case 0x1C0u:
    case 0x1C1u:
    case 0x1C2u:
    case 0x1C8u:
    case 0x1CDu:
    case 0x1CFu:
    case 0x1D4u:
    case 0x1DAu:
    case 0x1EAu:
    case 0x1EBu:
    case 0x1ECu:
    case 0x1EDu:
    case 0x1EEu:
    case 0x1EFu:
    case 0x1F4u:
    case 0x1F5u:
    case 0x1F6u:
      result = 1;
      break;
    default:
      return 0;
  }
  return result;
}
