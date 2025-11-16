// Function: sub_16DE120
// Address: 0x16de120
//
unsigned __int64 __fastcall sub_16DE120(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // ecx

  switch ( *(_DWORD *)(a1 + 32) )
  {
    case 0:
    case 1:
    case 3:
    case 0x1D:
    case 0x1F:
    case 0x20:
      v2 = *(unsigned int *)(a1 + 44);
      result = 2;
      if ( (unsigned int)v2 <= 0x1E )
      {
        v3 = 1610614920;
        result = 3;
        if ( !_bittest64(&v3, v2) )
          result = (unsigned int)((_DWORD)v2 != 15) + 1;
      }
      break;
    case 2:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1E:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x31:
    case 0x32:
    case 0x33:
      result = 2;
      break;
    case 0x10:
    case 0x11:
      v4 = *(_DWORD *)(a1 + 44);
      result = 2;
      if ( v4 <= 0x1E )
        result = ((0x60000888uLL >> v4) & 1) + 2;
      break;
    case 0x2F:
    case 0x30:
      result = 4;
      break;
  }
  return result;
}
