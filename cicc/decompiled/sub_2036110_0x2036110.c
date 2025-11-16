// Function: sub_2036110
// Address: 0x2036110
//
unsigned __int64 __fastcall sub_2036110(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __m128 a7,
        __m128i a8,
        __m128i a9)
{
  unsigned int v9; // ebx
  unsigned __int64 result; // rax
  const __m128i *v11; // r9
  __int64 v12; // rcx
  unsigned int v13; // edx
  unsigned __int64 v14; // r8
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx

  v9 = a3;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0x30:
      result = (unsigned __int64)sub_2022FA0((__int64)a1, a2, a3, a4, a5, (__int64)a6);
      v12 = result;
      v14 = v33;
      break;
    case 0x33:
      v30 = sub_2013D30((__int64)a1, a2, a3, a4, a5, a6);
      result = sub_2032580((__int64)a1, v30, v31);
      v12 = result;
      v14 = v32;
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
    case 0x65:
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
      result = (unsigned __int64)sub_2032A50((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v15;
      break;
    case 0x63:
      result = (unsigned __int64)sub_2032B60((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v26;
      break;
    case 0x67:
    case 0x7F:
    case 0x80:
    case 0x81:
    case 0x82:
    case 0x83:
    case 0x84:
    case 0x85:
    case 0x8E:
    case 0x8F:
    case 0x90:
    case 0x91:
    case 0x92:
    case 0x93:
    case 0x98:
    case 0x99:
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
      result = sub_2033150(a1, a2, (__m128i)a7, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v13;
      break;
    case 0x68:
      result = sub_20224D0((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, *(double *)a9.m128i_i64);
      v12 = result;
      v14 = v17;
      break;
    case 0x69:
      result = sub_20228B0(
                 (__int64)a1,
                 a2,
                 *(double *)a7.m128_u64,
                 *(double *)a8.m128i_i64,
                 *(double *)a9.m128i_i64,
                 a3,
                 a4);
      v12 = result;
      v14 = v25;
      break;
    case 0x6D:
      result = (unsigned __int64)sub_2022740((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v24;
      break;
    case 0x6E:
      result = sub_2034940((__int64)a1, a2, a3, a4, a5, (__int64)a6);
      v12 = result;
      v14 = v23;
      break;
    case 0x6F:
      result = sub_2022DF0((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, *(double *)a9.m128i_i64);
      v12 = result;
      v14 = v35;
      break;
    case 0x86:
      result = (unsigned __int64)sub_2034680((__int64)a1, a2, a7, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v29;
      break;
    case 0x87:
      result = (unsigned __int64)sub_2033C80(a1, a2, *(double *)a7.m128_u64, a8, a9);
      v12 = result;
      v14 = v28;
      break;
    case 0x88:
      result = (unsigned __int64)sub_2034810((__int64)a1, a2);
      v12 = result;
      v14 = v27;
      break;
    case 0x89:
      result = sub_2034AD0((__int64 **)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v18;
      break;
    case 0x94:
    case 0x9C:
      result = (unsigned __int64)sub_2033510((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v22;
      break;
    case 0x95:
    case 0x96:
    case 0x97:
      result = sub_20337A0(a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v16;
      break;
    case 0x9A:
      result = (unsigned __int64)sub_2032ED0((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v34;
      break;
    case 0x9E:
      result = sub_2032C80(a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, *(double *)a9.m128i_i64, a3, a4);
      v12 = result;
      v14 = v21;
      break;
    case 0xA7:
      result = (unsigned __int64)sub_2033070((__int64)a1, a2, *(double *)a7.m128_u64, *(double *)a8.m128i_i64, a9);
      v12 = result;
      v14 = v20;
      break;
    case 0xB9:
      result = sub_2022A80((__int64)a1, a2, a3, a4, a5, (__int64)a6);
      v12 = result;
      v14 = v19;
      break;
    default:
      sub_16BD130("Do not know how to scalarize the result of this operator!\n", 1u);
  }
  if ( v12 )
    return sub_20150C0((__int64)a1, a2, v9, v12, (__m128i *)v14, v11);
  return result;
}
