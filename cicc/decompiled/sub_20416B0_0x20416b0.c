// Function: sub_20416B0
// Address: 0x20416b0
//
unsigned __int64 __fastcall sub_20416B0(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __m128i a4,
        __m128i a5,
        __m128i a6)
{
  __int64 v6; // rbx
  unsigned __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  const __m128i *v10; // r9
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
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned int v30; // edx
  unsigned int v31; // edx
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx
  unsigned int v36; // edx
  unsigned int v37; // edx
  unsigned int v38; // edx
  unsigned int v39; // edx
  unsigned int v40; // edx

  v6 = a3;
  result = sub_2015740(a1, a2, *(_BYTE *)(*(_QWORD *)(a2 + 40) + 16LL * a3));
  if ( !(_BYTE)result )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0x30:
        result = (unsigned __int64)sub_202F000(a1, a2);
        v12 = result;
        v14 = v21;
        break;
      case 0x31:
      case 0x32:
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
      case 0x6A:
      case 0x6C:
      case 0x79:
      case 0x7D:
      case 0x7E:
      case 0x84:
      case 0x85:
      case 0x8A:
      case 0x8B:
      case 0x8C:
      case 0x8D:
      case 0x9B:
      case 0x9F:
      case 0xA0:
      case 0xA1:
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
      case 0xCD:
      case 0xCE:
      case 0xCF:
      case 0xD0:
      case 0xD1:
      case 0xD2:
      case 0xD3:
      case 0xD4:
      case 0xD5:
      case 0xD6:
      case 0xD7:
      case 0xD8:
      case 0xD9:
      case 0xDA:
      case 0xDB:
      case 0xDC:
      case 0xDD:
      case 0xDE:
      case 0xDF:
      case 0xE0:
      case 0xE1:
      case 0xE2:
      case 0xE3:
      case 0xE4:
      case 0xE5:
      case 0xE6:
      case 0xE7:
      case 0xE8:
      case 0xE9:
      case 0xEA:
      case 0xEC:
      case 0xED:
        result = (unsigned __int64)sub_203A7B0(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v26;
        break;
      case 0x33:
        v22 = sub_2013D30((__int64)a1, a2, v6, v8, v9, v10);
        result = sub_20363F0((__int64)a1, v22, v23);
        v12 = result;
        v14 = v24;
        break;
      case 0x34:
      case 0x35:
      case 0x36:
      case 0x70:
      case 0x71:
      case 0x72:
      case 0x73:
      case 0x74:
      case 0x75:
      case 0x76:
      case 0x77:
      case 0x78:
      case 0xB4:
      case 0xB5:
      case 0xB6:
      case 0xB7:
        result = (unsigned __int64)sub_20369E0((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v15;
        break;
      case 0x37:
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
      case 0x50:
      case 0xA8:
        result = (unsigned __int64)sub_2036AE0((__int64 **)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v16;
        break;
      case 0x63:
        result = (unsigned __int64)sub_20368C0((__int64)a1, a2, (__m128)a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v25;
        break;
      case 0x65:
        result = (unsigned __int64)sub_2037F30((__int64 **)a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v39;
        break;
      case 0x68:
        result = (unsigned __int64)sub_202EA90(a1, a2, (__m128)a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v38;
        break;
      case 0x69:
        result = (unsigned __int64)sub_203A3B0((__int64)a1, a2, (__m128)a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v37;
        break;
      case 0x6B:
        result = sub_2040C70((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v36;
        break;
      case 0x6D:
        result = (unsigned __int64)sub_2039C70((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v35;
        break;
      case 0x6E:
        result = (unsigned __int64)sub_203BEF0(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v34;
        break;
      case 0x6F:
        result = sub_202EE60(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64);
        v12 = result;
        v14 = v33;
        break;
      case 0x7A:
      case 0x7B:
      case 0x7C:
        result = (unsigned __int64)sub_2039580(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v18;
        break;
      case 0x7F:
      case 0x80:
      case 0x81:
      case 0x82:
      case 0x83:
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
        result = sub_2039930(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64);
        v12 = result;
        v14 = v13;
        break;
      case 0x86:
      case 0x87:
        result = (unsigned __int64)sub_203BA50(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v20;
        break;
      case 0x88:
        result = (unsigned __int64)sub_203BDD0((__int64)a1, a2);
        v12 = result;
        v14 = v40;
        break;
      case 0x89:
        result = (unsigned __int64)sub_203C220(a1, a2, (__m128)a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v30;
        break;
      case 0x8E:
      case 0x8F:
      case 0x90:
      case 0x91:
      case 0x92:
      case 0x93:
      case 0x98:
      case 0x99:
      case 0x9A:
      case 0x9D:
        result = sub_2038010((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v17;
        break;
      case 0x94:
      case 0x9C:
        result = (unsigned __int64)sub_2039A20(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v31;
        break;
      case 0x95:
      case 0x96:
      case 0x97:
        result = sub_2038C10((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v19;
        break;
      case 0x9E:
        result = sub_2040260((__int64)a1, a2, a4, a5, a6);
        v12 = result;
        v14 = v32;
        break;
      case 0xA7:
        result = (unsigned __int64)sub_2039470(a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v29;
        break;
      case 0xB9:
        result = sub_2032070(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v28;
        break;
      case 0xEB:
        result = sub_203A490(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v12 = result;
        v14 = v27;
        break;
    }
    if ( v12 )
      return sub_2015400((__int64)a1, a2, v6, v12, (__m128i *)v14, v11);
  }
  return result;
}
