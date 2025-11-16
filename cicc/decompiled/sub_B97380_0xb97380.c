// Function: sub_B97380
// Address: 0xb97380
//
__int64 __fastcall sub_B97380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdi

  switch ( *(_BYTE *)a1 )
  {
    case 5:
    case 6:
    case 9:
    case 0x1E:
      sub_B972A0(a1, a2, a3, a4, a5);
      break;
    case 7:
      v7 = *(_QWORD *)(a1 + 16);
      if ( v7 )
      {
        a2 = *(_QWORD *)(a1 + 32) - v7;
        j_j___libc_free_0(v7, a2);
      }
      break;
    case 8:
    case 0xA:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
    case 0x10:
    case 0x11:
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
    case 0x1D:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
      break;
    case 0xB:
      sub_969240((__int64 *)(a1 + 16));
      break;
    default:
      BUG();
  }
  sub_B706B0((__int64 *)(a1 + 8));
  return sub_B914E0(a1, a2);
}
