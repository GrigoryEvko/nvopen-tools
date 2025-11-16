// Function: sub_2029C10
// Address: 0x2029c10
//
void __fastcall sub_2029C10(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128 a5, __m128i a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  const __m128i *v9; // r9
  __m128i v10; // [rsp+8h] [rbp-40h] BYREF
  unsigned __int64 v11; // [rsp+18h] [rbp-30h] BYREF
  __int64 v12; // [rsp+20h] [rbp-28h]

  v6 = a3;
  v7 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v10.m128i_i32[2] = 0;
  LODWORD(v12) = 0;
  v8 = *(_QWORD *)(v7 + 8);
  v10.m128i_i64[0] = 0;
  v11 = 0;
  if ( !(unsigned __int8)sub_2016240(a1, a2, *(_BYTE *)v7, v8, 1u, 0, 0) )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0x30:
        sub_2147AE0(a1, a2, &v10, &v11);
        break;
      case 0x33:
        sub_2146BB0(a1, a2, (unsigned int)v6, &v10, &v11);
        break;
      case 0x34:
      case 0x35:
      case 0x36:
      case 0x37:
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
      case 0x50:
      case 0x70:
      case 0x71:
      case 0x72:
      case 0x73:
      case 0x74:
      case 0x75:
      case 0x76:
      case 0x77:
      case 0x78:
      case 0x7A:
      case 0x7B:
      case 0x7C:
      case 0xA8:
      case 0xB4:
      case 0xB5:
      case 0xB6:
      case 0xB7:
        sub_20230C0((__int64)a1, a2, (__int64)&v10, (__int64)&v11, *(double *)a4.m128i_i64, *(double *)a5.m128_u64, a6);
        break;
      case 0x51:
      case 0x52:
      case 0x53:
      case 0x54:
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
        sub_2025910((__int64)a1, a2, (__int64)&v10, (__int64)&v11, a4, a5, a6);
        break;
      case 0x63:
        sub_2023250((__int64)a1, a2, (__int64)&v10, (__int64)&v11, (__m128)a4, *(double *)a5.m128_u64, a6);
        break;
      case 0x65:
        sub_2024E20((__int64)a1, a2, (__int64)&v10, (__int64)&v11, *(double *)a4.m128i_i64, (__m128i)a5, a6);
        break;
      case 0x67:
      case 0x7F:
      case 0x80:
      case 0x81:
      case 0x82:
      case 0x83:
      case 0x84:
      case 0x85:
      case 0x91:
      case 0x92:
      case 0x93:
      case 0x98:
      case 0x99:
      case 0x9A:
      case 0x9D:
      case 0xA2:
      case 0xA3:
      case 0xA4:
      case 0xA5:
      case 0xA6:
      case 0xA9:
      case 0xAA:
      case 0xAB:
      case 0xAC:
      case 0xAD:
      case 0xAE:
      case 0xAF:
      case 0xB0:
      case 0xB1:
      case 0xB2:
      case 0xB3:
        sub_2028A10((__int64)a1, a2, (__int64)&v10, (__int64)&v11, a4, (__m128i)a5, a6);
        break;
      case 0x68:
        sub_2023B70((__int64)a1, a2, (__int64)&v10, (__int64)&v11, (__m128)a4, a5);
        break;
      case 0x69:
        sub_2025FA0(a1, a2, (unsigned __int64 *)&v10, (__int64)&v11);
        break;
      case 0x6B:
        sub_2023F80((__int64)a1, a2, (__int64)&v10, (__int64)&v11, (__m128)a4, a5, a6);
        break;
      case 0x6C:
        sub_2024660(
          (__int64)a1,
          a2,
          (unsigned __int64 *)&v10,
          &v11,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128_u64,
          a6);
        break;
      case 0x6D:
        sub_20243B0(a1, a2, (__int64)&v10, (__int64)&v11, *(double *)a4.m128i_i64, *(double *)a5.m128_u64, a6);
        break;
      case 0x6E:
        sub_20293A0(
          (__int64 **)a1,
          a2,
          (__int64)&v10,
          (__int64)&v11,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128_u64,
          a6);
        break;
      case 0x6F:
        sub_2026C30(
          (__int64)a1,
          a2,
          (__int64)&v10,
          (__int64)&v11,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128_u64,
          *(double *)a6.m128i_i64);
        break;
      case 0x86:
      case 0x87:
        sub_2146C90(a1, a2, &v10, &v11);
        break;
      case 0x88:
        sub_2147770(a1, a2, &v10, &v11);
        break;
      case 0x89:
        sub_2028530((__int64)a1, a2, (__int64)&v10, (__int64)&v11, a4, (__m128i)a5, a6);
        break;
      case 0x8E:
      case 0x8F:
      case 0x90:
        sub_2028DA0(a1, a2, (__int64)&v10, (__int64)&v11, a4, (__m128i)a5, a6);
        break;
      case 0x94:
      case 0x9C:
        sub_20251A0((__int64)a1, a2, (__int64)&v10, (__int64)&v11, *(double *)a4.m128i_i64, *(double *)a5.m128_u64, a6);
        break;
      case 0x95:
      case 0x96:
      case 0x97:
        sub_2025380((__int64)a1, a2, (__int64)&v10, (__int64)&v11, a4, (__m128i)a5, a6);
        break;
      case 0x9E:
        sub_2023450((__int64 **)a1, a2, &v10, &v11, a4);
        break;
      case 0xA7:
        sub_2024CF0((__int64)a1, a2, (__int64)&v10, (__int64)&v11, *(double *)a4.m128i_i64, *(double *)a5.m128_u64, a6);
        break;
      case 0xB9:
        sub_2026D80((__int64)a1, a2, (__int64)&v10, (__int64)&v11);
        break;
      case 0xEB:
        sub_20272D0(a1, a2, (__int64)&v10, (__int64)&v11);
        break;
      case 0xED:
        sub_2027BD0((__int64)a1, a2, (__int64)&v10, (__int64)&v11);
        break;
      default:
        sub_16BD130("Do not know how to split the result of this operator!\n", 1u);
    }
    if ( v10.m128i_i64[0] )
      sub_20167D0((__int64)a1, a2, v6, v10.m128i_i64[0], (__m128i *)v10.m128i_i64[1], v9, v11, v12);
  }
}
