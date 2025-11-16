// Function: sub_BD72D0
// Address: 0xbd72d0
//
__int64 __fastcall sub_BD72D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi

  switch ( *(_BYTE *)a1 )
  {
    case 0x16:
      sub_BD7260(a1, a2);
      return j_j___libc_free_0(a1, 40);
    case 0x17:
      sub_AA5290(a1);
      return j_j___libc_free_0(a1, 80);
    case 0x18:
      sub_B91290(a1);
      return j_j___libc_free_0(a1, 32);
    case 0x19:
      v5 = *(_QWORD *)(a1 + 56);
      if ( v5 != a1 + 72 )
      {
        a2 = *(_QWORD *)(a1 + 72) + 1LL;
        j_j___libc_free_0(v5, a2);
      }
      v6 = *(_QWORD *)(a1 + 24);
      if ( v6 != a1 + 40 )
      {
        a2 = *(_QWORD *)(a1 + 40) + 1LL;
        j_j___libc_free_0(v6, a2);
      }
      sub_BD7260(a1, a2);
      return j_j___libc_free_0(a1, 112);
    case 0x1A:
    case 0x1B:
    case 0x1C:
      return (*(__int64 (**)(void))(a1 + 24))();
    case 0x1E:
    case 0x1F:
    case 0x20:
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
    case 0x3C:
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
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x59:
    case 0x5A:
    case 0x5B:
    case 0x5F:
    case 0x60:
      goto LABEL_4;
    case 0x5C:
    case 0x5D:
    case 0x5E:
      v3 = *(_QWORD *)(a1 + 72);
      if ( v3 != a1 + 88 )
        _libc_free(v3, a2);
LABEL_4:
      sub_B43C40(a1);
      return sub_BD2DD0(a1);
    default:
      BUG();
  }
}
