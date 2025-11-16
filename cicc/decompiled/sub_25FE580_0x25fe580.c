// Function: sub_25FE580
// Address: 0x25fe580
//
__int64 __fastcall sub_25FE580(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // eax
  __int64 result; // rax

  switch ( *(_BYTE *)a2 )
  {
    case 0x1E:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x3C:
    case 0x50:
    case 0x51:
    case 0x59:
    case 0x5F:
    case 0x60:
      result = 0;
      break;
    case 0x1F:
    case 0x54:
      result = *(unsigned __int8 *)a1;
      break;
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x4D:
    case 0x4E:
    case 0x4F:
    case 0x52:
    case 0x53:
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x5A:
    case 0x5B:
    case 0x5C:
    case 0x5D:
    case 0x5E:
      result = 1;
      break;
    case 0x55:
      v2 = *(_QWORD *)(a2 - 32);
      if ( v2
        && !*(_BYTE *)v2
        && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v2 + 24)
        && (v3 = *(_DWORD *)(v2 + 36)) != 0
        && (unsigned int)(v3 - 69) <= 2 )
      {
        result = 1;
      }
      else
      {
        result = sub_25FB9B0(a1, a2);
      }
      break;
    default:
      BUG();
  }
  return result;
}
