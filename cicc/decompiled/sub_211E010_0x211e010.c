// Function: sub_211E010
// Address: 0x211e010
//
void __fastcall sub_211E010(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  const __m128i *v9; // r9
  __m128i v10; // [rsp+8h] [rbp-40h] BYREF
  __m128i v11[3]; // [rsp+18h] [rbp-30h] BYREF

  v6 = a3;
  v7 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v11[0].m128i_i32[2] = 0;
  v10.m128i_i64[0] = 0;
  v10.m128i_i32[2] = 0;
  v8 = *(_QWORD *)(v7 + 8);
  v11[0].m128i_i64[0] = 0;
  if ( !(unsigned __int8)sub_2016240(a1, a2, *(_BYTE *)v7, v8, 1u, 0, 0) )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0xB:
        sub_211BC10(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
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
      case 0x51:
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
      case 0x56:
      case 0x57:
      case 0x58:
      case 0x59:
      case 0x5A:
      case 0x5B:
      case 0x5C:
      case 0x5D:
      case 0x5E:
      case 0x5F:
      case 0x60:
      case 0x61:
      case 0x62:
      case 0x64:
      case 0x66:
      case 0x67:
      case 0x68:
      case 0x69:
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
      case 0x87:
      case 0x89:
      case 0x8A:
      case 0x8B:
      case 0x8C:
      case 0x8D:
      case 0x8E:
      case 0x8F:
      case 0x90:
      case 0x91:
      case 0x94:
      case 0x95:
      case 0x96:
      case 0x97:
      case 0x98:
      case 0x99:
      case 0x9A:
      case 0x9B:
      case 0x9C:
      case 0x9F:
      case 0xA0:
      case 0xA1:
      case 0xB6:
      case 0xB7:
      case 0xB8:
      case 0xBA:
      case 0xBB:
      case 0xBC:
      case 0xBD:
      case 0xBE:
      case 0xBF:
      case 0xC0:
      case 0xC1:
      case 0xC2:
      case 0xC3:
      case 0xC4:
      case 0xC5:
      case 0xC6:
      case 0xC7:
      case 0xC8:
      case 0xC9:
      case 0xCA:
      case 0xCB:
      case 0xCC:
        sub_2144790(a1, a2, &v10, v11);
        break;
      case 0x30:
        sub_2147AE0(a1, a2, &v10, v11);
        break;
      case 0x31:
        sub_2143C70(a1, a2, &v10, v11);
        break;
      case 0x32:
        sub_2143C40(a1, a2, &v10, v11);
        break;
      case 0x33:
        sub_2143B90(a1, a2, (unsigned int)v6, &v10, v11);
        break;
      case 0x4C:
        sub_211C240(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0x4D:
        sub_211D2A0(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 0x4E:
        sub_211C9F0(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 0x4F:
        sub_211C440(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 0x50:
        sub_211D020(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0x63:
        sub_211C890(a1, a2, (__int64)&v10, (__int64)v11);
        break;
      case 0x65:
        sub_211C340(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0x6A:
        sub_2143D60(a1, a2, &v10, v11);
        break;
      case 0x86:
        sub_2146C90(a1, a2, &v10, v11);
        break;
      case 0x88:
        sub_2147770(a1, a2, &v10, v11);
        break;
      case 0x92:
      case 0x93:
        sub_211D780((__int64)a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, a5, a6);
        break;
      case 0x9D:
        sub_211CCD0(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 0x9E:
        sub_2147DE0(a1, a2, &v10, v11);
        break;
      case 0xA2:
        sub_211CBC0(
          (__int64)a1,
          a2,
          &v10,
          v11,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64);
        break;
      case 0xA3:
        sub_211BF80(
          (__int64)a1,
          a2,
          &v10,
          v11,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64);
        break;
      case 0xA4:
        sub_211D220(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xA5:
        sub_211D1A0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xA6:
        sub_211C3C0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xA7:
        sub_211CFA0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xA8:
        sub_211CF20(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xA9:
        sub_211C710(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAA:
        sub_211C790(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAB:
        sub_211C810(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAC:
        sub_211C590(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAD:
        sub_211C610(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAE:
        sub_211C2C0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xAF:
        sub_211D3F0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB0:
        sub_211D0A0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB1:
        sub_211CB40(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB2:
        sub_211D120(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB3:
        sub_211C690(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB4:
        sub_211C140(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB5:
        sub_211C1C0(a1, a2, (__int64)&v10, (__int64)v11, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 0xB9:
        sub_211D470(a1, a2, (__int64)&v10, (__int64)v11, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
    }
    if ( v10.m128i_i64[0] )
      sub_2016420(
        (__int64)a1,
        a2,
        v6,
        v10.m128i_i64[0],
        (__m128i *)v10.m128i_i64[1],
        v9,
        v11[0].m128i_u64[0],
        v11[0].m128i_i64[1]);
  }
}
