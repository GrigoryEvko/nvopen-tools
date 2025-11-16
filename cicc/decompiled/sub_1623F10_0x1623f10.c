// Function: sub_1623F10
// Address: 0x1623f10
//
__int64 __fastcall sub_1623F10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdi

  switch ( *(_BYTE *)a1 )
  {
    case 4:
    case 5:
    case 8:
      sub_1623E60(a1, a2, a3, a4, a5);
      break;
    case 6:
      v7 = *(_QWORD *)(a1 + 24);
      if ( v7 )
        j_j___libc_free_0(v7, *(_QWORD *)(a1 + 40) - v7);
      break;
    case 7:
    case 9:
    case 0xA:
    case 0xB:
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
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
      break;
  }
  sub_1604260((__int64 *)(a1 + 16));
  return sub_161E9C0(a1);
}
