// Function: sub_2126090
// Address: 0x2126090
//
__int64 __fastcall sub_2126090(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  __int64 result; // rax
  __int64 v11; // rcx
  const __m128i *v12; // r9
  unsigned int v13; // edx
  unsigned __int64 v14; // r8
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // edx
  unsigned int v23; // edx
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
  unsigned int v41; // edx
  unsigned int v42; // edx
  unsigned int v43; // edx
  unsigned int v44; // edx
  unsigned int v45; // edx
  unsigned int v46; // edx
  unsigned int v47; // edx
  unsigned int v48; // edx
  unsigned int v49; // edx
  unsigned int v50; // edx
  unsigned int v51; // edx
  unsigned int v52; // edx
  unsigned int v53; // edx
  unsigned int v54; // edx
  unsigned int v55; // edx
  unsigned int v56; // edx

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 8:
    case 0x2E:
    case 0x2F:
      return 0;
    case 9:
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
    case 0x31:
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
    case 0x9B:
    case 0x9C:
    case 0x9F:
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
      v11 = (__int64)sub_211B770(a1, a2, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v14 = v34;
      goto LABEL_5;
    case 0xB:
      v11 = sub_211AA30(a1, (_QWORD *)a2, a3, a7, *(double *)a8.m128i_i64, a9);
      v14 = v25;
      goto LABEL_5;
    case 0x30:
      v11 = (__int64)sub_211B6E0(a1, a2);
      v14 = v24;
      goto LABEL_5;
    case 0x32:
      v11 = (__int64)sub_211A930(a1, a2, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v14 = v23;
      goto LABEL_5;
    case 0x33:
      v20 = sub_2013D30((__int64)a1, a2, a3, a4, a5, a6);
      v11 = sub_200D2A0(
              (__int64)a1,
              v20,
              v21,
              *(double *)a7.m128i_i64,
              *(double *)a8.m128i_i64,
              *(double *)a9.m128i_i64);
      v14 = v22;
      goto LABEL_5;
    case 0x4C:
      v11 = sub_2120CD0(a1, a2, a7, a8, a9);
      v14 = v19;
      goto LABEL_5;
    case 0x4D:
      v11 = sub_21231F0(a1, a2, a7, a8, a9);
      v14 = v18;
      goto LABEL_5;
    case 0x4E:
      v11 = sub_2122360(a1, a2, a7, a8, a9);
      v14 = v17;
      goto LABEL_5;
    case 0x4F:
      v11 = sub_2121850(a1, a2, a7, a8, a9);
      v14 = v16;
      goto LABEL_5;
    case 0x50:
      v11 = sub_2122B30(a1, a2, a7, a8, a9);
      v14 = v15;
      goto LABEL_5;
    case 0x63:
      v11 = sub_21221B0(a1, a2, a7, a8, a9);
      v14 = v40;
      goto LABEL_5;
    case 0x65:
      v11 = (__int64)sub_2120FA0(a1, a2, a3, a7, *(double *)a8.m128i_i64, a9);
      v14 = v39;
      goto LABEL_5;
    case 0x6A:
      v11 = (__int64)sub_211AD90(a1, a2, a3, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9);
      v14 = v38;
      goto LABEL_5;
    case 0x86:
      v11 = (__int64)sub_2123540(a1, a2, a3, (__m128)a7, *(double *)a8.m128i_i64, a9);
      v14 = v37;
      goto LABEL_5;
    case 0x88:
      v11 = (__int64)sub_2123790(a1, (__int64 *)a2, a3);
      v14 = v36;
      goto LABEL_5;
    case 0x92:
    case 0x93:
      v11 = sub_211B890((__int64)a1, a2, a7, a8, a9);
      v14 = v13;
      goto LABEL_5;
    case 0x9A:
      v11 = sub_211B1D0(a1, a2, *(double *)a7.m128i_i64, a8, a9);
      v14 = v35;
      goto LABEL_5;
    case 0x9D:
      v11 = sub_2125D70((__int64)a1, a2, *(double *)a7.m128i_i64, a8, a9);
      v14 = v48;
      goto LABEL_5;
    case 0x9E:
      v11 = sub_211A870(a1, a2, a3, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, *(double *)a9.m128i_i64);
      v14 = v56;
      goto LABEL_5;
    case 0xA0:
      v11 = sub_211AFD0(a1, a2, *(double *)a7.m128i_i64, a8, a9);
      v14 = v55;
      goto LABEL_5;
    case 0xA2:
      v11 = sub_2122630(a1, a2, a3, *(double *)a7.m128i_i64, a8, a9);
      v14 = v54;
      goto LABEL_5;
    case 0xA3:
      v11 = (__int64)sub_2120760(a1, (__int64 *)a2, a3, a7, *(double *)a8.m128i_i64, a9);
      v14 = v53;
      goto LABEL_5;
    case 0xA4:
      v11 = sub_21230A0(a1, a2, a7, a8, a9);
      v14 = v52;
      goto LABEL_5;
    case 0xA5:
      v11 = sub_2122F50(a1, a2, a7, a8, a9);
      v14 = v51;
      goto LABEL_5;
    case 0xA6:
      v11 = sub_2121700(a1, a2, a7, a8, a9);
      v14 = v50;
      goto LABEL_5;
    case 0xA7:
      v11 = sub_21229C0(a1, a2, *(double *)a7.m128i_i64, a8, a9);
      v14 = v49;
      goto LABEL_5;
    case 0xA8:
      v11 = sub_2122840(a1, a2, a7, a8, a9);
      v14 = v32;
      goto LABEL_5;
    case 0xA9:
      v11 = sub_2121DC0(a1, a2, a7, a8, a9);
      v14 = v31;
      goto LABEL_5;
    case 0xAA:
      v11 = sub_2121F10(a1, a2, a7, a8, a9);
      v14 = v30;
      goto LABEL_5;
    case 0xAB:
      v11 = sub_2122060(a1, a2, a7, a8, a9);
      v14 = v29;
      goto LABEL_5;
    case 0xAC:
      v11 = sub_21219D0(a1, a2, a7, a8, a9);
      v14 = v28;
      goto LABEL_5;
    case 0xAD:
      v11 = sub_2121B20(a1, a2, a7, a8, a9);
      v14 = v27;
      goto LABEL_5;
    case 0xAE:
      v11 = sub_2120E50(a1, a2, a7, a8, a9);
      v14 = v26;
      goto LABEL_5;
    case 0xAF:
      v11 = sub_2123370(a1, a2, a7, a8, a9);
      v14 = v33;
      goto LABEL_5;
    case 0xB0:
      v11 = sub_2122CB0(a1, a2, a7, a8, a9);
      v14 = v44;
      goto LABEL_5;
    case 0xB1:
      v11 = sub_21224E0(a1, a2, a7, a8, a9);
      v14 = v43;
      goto LABEL_5;
    case 0xB2:
      v11 = sub_2122E00(a1, a2, a7, a8, a9);
      v14 = v42;
      goto LABEL_5;
    case 0xB3:
      v11 = sub_2121C70(a1, a2, a7, a8, a9);
      v14 = v41;
      goto LABEL_5;
    case 0xB4:
      v11 = sub_21209D0(a1, a2, a7, a8, a9);
      v14 = v46;
      goto LABEL_5;
    case 0xB5:
      v11 = sub_2120B50(a1, a2, a7, a8, a9);
      v14 = v45;
      goto LABEL_5;
    case 0xB9:
      v11 = sub_211B370((__int64)a1, a2, a3, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, *(double *)a9.m128i_i64);
      v14 = v47;
LABEL_5:
      result = 0;
      if ( v11 )
      {
        if ( v11 != a2 )
        {
          sub_2014D80((__int64)a1, a2, a3, v11, (__m128i *)v14, v12);
          return 1;
        }
      }
      return result;
  }
}
