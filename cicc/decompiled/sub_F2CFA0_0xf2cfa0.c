// Function: sub_F2CFA0
// Address: 0xf2cfa0
//
__int64 __fastcall sub_F2CFA0(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 result; // rax

  switch ( *(_BYTE *)a2 )
  {
    case 0x1E:
      result = (__int64)sub_F29A20((__int64)a1, a2);
      break;
    case 0x1F:
      if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
        result = sub_F109E0((__int64)a1, a2);
      else
        result = sub_F2AF70(a1, a2);
      break;
    case 0x20:
      result = sub_F2B940(a1, a2);
      break;
    case 0x21:
    case 0x23:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x41:
    case 0x50:
    case 0x51:
    case 0x57:
    case 0x58:
    case 0x59:
      result = 0;
      break;
    case 0x22:
      result = sub_10ECB80(a1);
      break;
    case 0x24:
      sub_F23800((__int64)a1, a2);
      result = 0;
      break;
    case 0x28:
      result = sub_10ECB90(a1);
      break;
    case 0x29:
      result = sub_10A6920(a1);
      break;
    case 0x2A:
      result = sub_10B23F0(a1);
      break;
    case 0x2B:
      result = sub_10B79B0(a1);
      break;
    case 0x2C:
      result = sub_10ACA40(a1);
      break;
    case 0x2D:
      result = sub_10B62C0(a1);
      break;
    case 0x2E:
      result = sub_115D7A0(a1);
      break;
    case 0x2F:
      result = sub_115F6E0(a1);
      break;
    case 0x30:
      result = sub_1167470(a1);
      break;
    case 0x31:
      result = sub_1167D00(a1);
      break;
    case 0x32:
      result = sub_1162F40(a1);
      break;
    case 0x33:
      result = sub_1166410(a1);
      break;
    case 0x34:
      result = sub_1166AC0(a1);
      break;
    case 0x35:
      result = sub_11596F0(a1);
      break;
    case 0x36:
      result = sub_119B990(a1);
      break;
    case 0x37:
      result = sub_119DD20(a1);
      break;
    case 0x38:
      result = sub_119A9E0(a1);
      break;
    case 0x39:
      result = sub_10DC830(a1);
      break;
    case 0x3A:
      result = sub_10D8BB0(a1);
      break;
    case 0x3B:
      result = sub_10D48A0(a1);
      break;
    case 0x3C:
      result = sub_1152CF0(a1);
      break;
    case 0x3D:
      result = sub_1150D90(a1);
      break;
    case 0x3E:
      result = sub_114D150(a1);
      break;
    case 0x3F:
      result = sub_F20C20(a1, a2, a3, a4, a5, a6);
      break;
    case 0x40:
      result = sub_10E9040(a1);
      break;
    case 0x42:
      result = sub_1110120(a1);
      break;
    case 0x43:
      result = sub_1108070(a1);
      break;
    case 0x44:
      result = sub_1107060(a1);
      break;
    case 0x45:
      result = sub_110A4B0(a1);
      break;
    case 0x46:
      result = sub_1101580(a1);
      break;
    case 0x47:
      result = sub_11015C0(a1);
      break;
    case 0x48:
      result = sub_1100B60(a1);
      break;
    case 0x49:
      result = sub_1100BD0(a1);
      break;
    case 0x4A:
      result = sub_1102530(a1);
      break;
    case 0x4B:
      result = sub_1101320(a1);
      break;
    case 0x4C:
      result = sub_1101600(a1);
      break;
    case 0x4D:
      result = sub_1100C60(a1);
      break;
    case 0x4E:
      result = sub_110CA10(a1);
      break;
    case 0x4F:
      result = sub_1100F40(a1);
      break;
    case 0x52:
      result = sub_1147470(a1);
      break;
    case 0x53:
      result = sub_113DAC0(a1);
      break;
    case 0x54:
      result = sub_1175E90(a1);
      break;
    case 0x55:
      v6 = *(_QWORD *)(a2 - 32);
      if ( v6 && !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v6 + 36) == 374 )
        result = sub_10E8FE0(a1);
      else
        result = sub_10EE7A0(a1);
      break;
    case 0x56:
      result = sub_1190310(a1);
      break;
    case 0x5A:
      result = sub_11B67C0(a1);
      break;
    case 0x5B:
      result = sub_11B8020(a1);
      break;
    case 0x5C:
      result = sub_11BABC0(a1);
      break;
    case 0x5D:
      result = (__int64)sub_F283A0(a1, a2);
      break;
    case 0x5E:
      result = sub_11B4850(a1);
      break;
    case 0x5F:
      result = sub_F1B280((__int64)a1, a2);
      break;
    case 0x60:
      result = (__int64)sub_F29590((__m128i *)a1, a2);
      break;
    default:
      BUG();
  }
  return result;
}
