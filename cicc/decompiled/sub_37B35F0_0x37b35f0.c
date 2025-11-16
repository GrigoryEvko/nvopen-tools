// Function: sub_37B35F0
// Address: 0x37b35f0
//
void __fastcall sub_37B35F0(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r9
  __m128i v12; // [rsp+0h] [rbp-40h] BYREF
  __m128i v13[3]; // [rsp+10h] [rbp-30h] BYREF

  v4 = a3;
  v5 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v12.m128i_i32[2] = 0;
  v13[0].m128i_i32[2] = 0;
  v6 = *(_QWORD *)(v5 + 8);
  v12.m128i_i64[0] = 0;
  v13[0].m128i_i64[0] = 0;
  if ( !(unsigned __int8)sub_3761870(a1, a2, *(_WORD *)v5, v6, 1) )
  {
    switch ( *(_DWORD *)(a2 + 24) )
    {
      case 4:
        sub_384A180(a1, a2, &v12, v13);
        goto LABEL_4;
      case 0x33:
        sub_384A050(a1, a2, &v12, v13);
        goto LABEL_4;
      case 0x34:
      case 0x9A:
      case 0xBD:
      case 0xC5:
      case 0xC6:
      case 0xC7:
      case 0xC8:
      case 0xC9:
      case 0xCB:
      case 0xCC:
      case 0xD8:
      case 0xDC:
      case 0xDD:
      case 0xE2:
      case 0xE3:
      case 0xE6:
      case 0xE9:
      case 0xF4:
      case 0xF5:
      case 0xF6:
      case 0xF8:
      case 0xF9:
      case 0xFA:
      case 0xFB:
      case 0xFC:
      case 0xFD:
      case 0xFE:
      case 0xFF:
      case 0x100:
      case 0x106:
      case 0x107:
      case 0x108:
      case 0x109:
      case 0x10A:
      case 0x10B:
      case 0x10C:
      case 0x10D:
      case 0x10E:
      case 0x10F:
      case 0x110:
      case 0x111:
      case 0x112:
      case 0x113:
      case 0x114:
      case 0x115:
      case 0x116:
      case 0x14F:
      case 0x19C:
      case 0x19D:
      case 0x19E:
      case 0x19F:
      case 0x1A0:
      case 0x1A1:
      case 0x1A2:
      case 0x1A3:
      case 0x1B1:
      case 0x1B2:
      case 0x1B3:
      case 0x1BB:
      case 0x1BC:
      case 0x1BD:
      case 0x1BE:
      case 0x1BF:
      case 0x1C0:
      case 0x1C1:
      case 0x1C2:
      case 0x1C3:
      case 0x1C4:
      case 0x1C5:
      case 0x1C6:
      case 0x1C7:
      case 0x1C8:
      case 0x1C9:
      case 0x1CA:
        sub_3781AE0(a1, a2, (__int64)&v12, v13, a4);
        goto LABEL_4;
      case 0x37:
        sub_3849100(a1, a2, (unsigned int)v4, &v12, v13);
        goto LABEL_4;
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x3B:
      case 0x3C:
      case 0x3D:
      case 0x3E:
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
      case 0x56:
      case 0x57:
      case 0x60:
      case 0x61:
      case 0x62:
      case 0x63:
      case 0x64:
      case 0xAC:
      case 0xAD:
      case 0xAE:
      case 0xAF:
      case 0xB0:
      case 0xB1:
      case 0xB2:
      case 0xB3:
      case 0xB4:
      case 0xB5:
      case 0xB6:
      case 0xB7:
      case 0xBA:
      case 0xBB:
      case 0xBC:
      case 0xBE:
      case 0xBF:
      case 0xC0:
      case 0xC1:
      case 0xC2:
      case 0x101:
      case 0x104:
      case 0x117:
      case 0x118:
      case 0x119:
      case 0x11A:
      case 0x11B:
      case 0x11C:
      case 0x11D:
      case 0x11E:
      case 0x18B:
      case 0x18C:
      case 0x18D:
      case 0x18E:
      case 0x18F:
      case 0x190:
      case 0x191:
      case 0x192:
      case 0x193:
      case 0x194:
      case 0x195:
      case 0x196:
      case 0x197:
      case 0x198:
      case 0x199:
      case 0x19A:
      case 0x19B:
      case 0x1A8:
      case 0x1A9:
      case 0x1AA:
      case 0x1AB:
      case 0x1AC:
      case 0x1AD:
      case 0x1AE:
      case 0x1AF:
      case 0x1B0:
      case 0x1B6:
      case 0x1B7:
      case 0x1B8:
      case 0x1B9:
      case 0x1BA:
        sub_3777A10((__int64)a1, a2, (unsigned __int8 **)&v12, (unsigned __int8 **)v13, a4);
        goto LABEL_4;
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
      case 0x50:
      case 0x51:
        sub_377C060(a1, a2, v4, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x58:
      case 0x59:
      case 0x5A:
      case 0x5B:
      case 0x5C:
      case 0x5D:
      case 0x5E:
      case 0x5F:
        sub_3778790((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x65:
      case 0x66:
      case 0x67:
      case 0x68:
      case 0x69:
      case 0x6A:
      case 0x6B:
      case 0x6C:
      case 0x6D:
      case 0x6E:
      case 0x6F:
      case 0x70:
      case 0x71:
      case 0x72:
      case 0x73:
      case 0x74:
      case 0x75:
      case 0x76:
      case 0x77:
      case 0x78:
      case 0x79:
      case 0x7A:
      case 0x7B:
      case 0x7C:
      case 0x7D:
      case 0x7E:
      case 0x7F:
      case 0x80:
      case 0x81:
      case 0x82:
      case 0x83:
      case 0x84:
      case 0x85:
      case 0x86:
      case 0x87:
      case 0x88:
      case 0x89:
      case 0x8A:
      case 0x8B:
      case 0x8C:
      case 0x8D:
      case 0x8E:
      case 0x8F:
      case 0x90:
      case 0x91:
      case 0x92:
      case 0x93:
      case 0x94:
        sub_377AE80(a1, a2, (__int64)&v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0x96:
      case 0xC3:
      case 0xC4:
      case 0x1A6:
      case 0x1A7:
      case 0x1B4:
        sub_3777E20((__int64)a1, a2, &v12, v13, a4);
        goto LABEL_4;
      case 0x98:
      case 0x102:
      case 0x103:
        sub_3779DC0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x9B:
        sub_377A280(a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x9C:
        sub_3779320((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x9D:
        sub_377C6C0(a1, a2, (unsigned int *)&v12, (unsigned int *)v13);
        goto LABEL_4;
      case 0x9F:
        sub_3779790((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xA0:
        sub_379AF20(a1, a2, (__int64)&v12, (unsigned int *)v13);
        goto LABEL_4;
      case 0xA1:
        sub_3779BC0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xA2:
        sub_3784280((__int64)a1, a2, v7, v8, v9, v10);
        return;
      case 0xA3:
        sub_37846E0((__int64)a1, a2, v7, v8, v9, v10);
        return;
      case 0xA4:
        sub_37836C0((__int64)a1, a2, (__int64)&v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0xA5:
        sub_37B2DF0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xA6:
        sub_37837E0(a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xA7:
      case 0xA8:
        sub_377DAA0((__int64)a1, a2, (__int64)&v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0xAA:
        sub_377D640((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xAB:
        sub_377E430(a1, a2, v12.m128i_i64, (__int64)v13);
        goto LABEL_4;
      case 0xB8:
      case 0xB9:
        sub_37782C0(a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xCD:
      case 0xCE:
      case 0x1E8:
      case 0x1E9:
        sub_3849200(a1, a2, &v12, v13);
        goto LABEL_4;
      case 0xCF:
        sub_3849C90(a1, a2, &v12, v13);
        goto LABEL_4;
      case 0xD0:
      case 0x1CF:
        sub_377EF80(a1, a2, (__int64)&v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0xD5:
      case 0xD6:
      case 0xD7:
      case 0x1CB:
      case 0x1CC:
        sub_3782790(a1, a2, (__int64)&v12, v13, a4);
        goto LABEL_4;
      case 0xDE:
        sub_377A5E0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xDF:
      case 0xE0:
      case 0xE1:
        sub_377A7C0(a1, a2, (__int64)&v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0xE4:
      case 0xE5:
        sub_3783390(a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0xEA:
        sub_3778950(a1, a2, &v12, (__int64)v13, a4);
        goto LABEL_4;
      case 0xEB:
        sub_37820A0(a1, a2, (unsigned __int64 *)&v12, (unsigned __int64 *)v13);
        goto LABEL_4;
      case 0x105:
      case 0x11F:
      case 0x120:
      case 0x121:
        sub_3782360(a1, a2, v4, &v12, v13, a4);
        goto LABEL_4;
      case 0x12A:
        sub_377DDB0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x13D:
        sub_37830B0((__int64)a1, a2, (__int64)&v12, (__int64)v13, v9);
        goto LABEL_4;
      case 0x16A:
        sub_37805F0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x16C:
      case 0x1D6:
        sub_3780EC0(a1, a2, (__m128i **)&v12, (__m128i **)v13, 1, a4);
        goto LABEL_4;
      case 0x187:
      case 0x188:
        sub_3784150(a1, a2, (__int64)&v12, (__int64)v13, v9, a4);
        goto LABEL_4;
      case 0x1D4:
        sub_377FDD0((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x1D5:
        sub_377F660((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x1EB:
        sub_3783910((__int64)a1, a2, (__int64)&v12, (__int64)v13);
        goto LABEL_4;
      case 0x1EC:
        sub_377DC10(a1, a2, (__int64)&v12, (__int64)v13, a4);
LABEL_4:
        if ( v12.m128i_i64[0] )
          sub_3760810(
            (__int64)a1,
            a2,
            v4,
            v12.m128i_u64[0],
            v12.m128i_i64[1],
            v11,
            v13[0].m128i_u64[0],
            v13[0].m128i_i64[1]);
        break;
      default:
        sub_C64ED0("Do not know how to split the result of this operator!\n", 1u);
    }
  }
}
