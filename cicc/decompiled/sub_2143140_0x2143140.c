// Function: sub_2143140
// Address: 0x2143140
//
__int64 __fastcall sub_2143140(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  const __m128i *v12; // r9
  __int64 v13; // rcx
  int v14; // eax
  const __m128i *v15; // r9
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  __int64 *v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rsi
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  __int64 *v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rsi
  __int64 *v32; // rax
  unsigned int v33; // edx
  __int64 v34; // rsi
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
  unsigned int v57; // edx
  unsigned int v58; // edx
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  unsigned int v61; // edx
  unsigned int v62; // edx
  unsigned int v63; // edx
  unsigned int v64; // edx
  unsigned int v65; // edx
  unsigned int v66; // edx
  unsigned int v67; // edx
  unsigned int v68; // edx
  unsigned int v69; // edx
  __int64 v70; // [rsp-8h] [rbp-340h]
  __int64 v71; // [rsp-8h] [rbp-340h]
  __int64 v72; // [rsp+308h] [rbp-30h] BYREF
  __m128i *v73; // [rsp+310h] [rbp-28h]

  v6 = a3;
  v8 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  LODWORD(v73) = 0;
  v72 = 0;
  result = sub_2016240(a1, a2, *(_BYTE *)v8, *(_QWORD *)(v8 + 8), 1u, a3, (__int64)&v72);
  if ( (_BYTE)result )
  {
    v13 = v72;
    if ( !v72 )
      return result;
    v14 = *(_DWORD *)(a2 + 64);
    goto LABEL_5;
  }
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 3:
      v72 = (__int64)sub_2140210((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v68;
      LODWORD(v73) = v68;
      break;
    case 4:
      v72 = (__int64)sub_21393C0((__int64)a1, a2, a4, a5, a6);
      result = v67;
      LODWORD(v73) = v67;
      break;
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
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
    case 0x31:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
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
    case 0x63:
    case 0x64:
    case 0x65:
    case 0x66:
    case 0x67:
    case 0x6C:
    case 0x70:
    case 0x71:
    case 0x79:
    case 0x7D:
    case 0x7E:
    case 0x8A:
    case 0x8B:
    case 0x8C:
    case 0x8D:
    case 0x92:
    case 0x93:
    case 0x9A:
    case 0x9B:
    case 0x9C:
    case 0x9D:
    case 0x9F:
    case 0xA0:
    case 0xA2:
    case 0xA3:
    case 0xA4:
    case 0xA5:
    case 0xA6:
    case 0xA7:
    case 0xA8:
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
    case 0xB4:
    case 0xB5:
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
    case 0xDC:
    case 0xEC:
    case 0xED:
      v72 = (__int64)sub_213B7D0(a1, a2);
      result = v69;
      LODWORD(v73) = v69;
      break;
    case 0xA:
      v72 = sub_2127C00(a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      result = v66;
      LODWORD(v73) = v66;
      break;
    case 0x30:
      v72 = (__int64)sub_2128640(a1, a2);
      result = v65;
      LODWORD(v73) = v65;
      break;
    case 0x32:
      v72 = sub_2127B20(a1, a2, a4, a5, a6);
      result = v64;
      LODWORD(v73) = v64;
      break;
    case 0x33:
      v59 = sub_2013D30((__int64)a1, a2, v6, v70, v11, v12);
      v72 = sub_2138AD0((__int64)a1, v59, v60);
      result = v61;
      LODWORD(v73) = v61;
      break;
    case 0x34:
    case 0x35:
    case 0x36:
    case 0x76:
    case 0x77:
    case 0x78:
      v72 = (__int64)sub_213BFE0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v17;
      LODWORD(v73) = v17;
      break;
    case 0x37:
    case 0x39:
    case 0x72:
    case 0x73:
      v72 = (__int64)sub_2140650((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v23;
      LODWORD(v73) = v23;
      break;
    case 0x38:
    case 0x3A:
    case 0x74:
    case 0x75:
      v72 = (__int64)sub_2139910((__int64)a1, a2, a4, a5, a6);
      result = v24;
      LODWORD(v73) = v24;
      break;
    case 0x44:
    case 0x45:
      if ( (_DWORD)v6 == 1 )
        v20 = sub_2128280(a1, a2, *(double *)a4.m128i_i64, a5, a6);
      else
        v20 = sub_2140900((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      v22 = (__int64)v20;
      result = v21;
      v72 = v22;
      LODWORD(v73) = v21;
      break;
    case 0x46:
    case 0x48:
      if ( (_DWORD)v6 == 1 )
        v32 = sub_2128280(a1, a2, *(double *)a4.m128i_i64, a5, a6);
      else
        v32 = sub_2140C50((__int64)a1, a2, (__m128)a4, a5, a6);
      v34 = (__int64)v32;
      result = v33;
      v72 = v34;
      LODWORD(v73) = v33;
      break;
    case 0x47:
    case 0x49:
      if ( (_DWORD)v6 == 1 )
        v29 = sub_2128280(a1, a2, *(double *)a4.m128i_i64, a5, a6);
      else
        v29 = sub_2139BA0((__int64)a1, a2, a4, a5, a6);
      v31 = (__int64)v29;
      result = v30;
      v72 = v31;
      LODWORD(v73) = v30;
      break;
    case 0x4A:
    case 0x4B:
      v72 = (__int64)sub_2139D90(a1, a2, v6, *(double *)a4.m128i_i64, a5, a6);
      result = v35;
      LODWORD(v73) = v35;
      break;
    case 0x68:
      v72 = (__int64)sub_2137500((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v45;
      LODWORD(v73) = v45;
      break;
    case 0x69:
      v72 = (__int64)sub_213ECF0((__int64)a1, a2, (__m128)a4, a5, a6);
      result = v44;
      LODWORD(v73) = v44;
      break;
    case 0x6A:
      v72 = (__int64)sub_2127D20((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v43;
      LODWORD(v73) = v43;
      break;
    case 0x6B:
      v72 = (__int64)sub_213E420((__int64)a1, a2, a4, a5, a6);
      result = v42;
      LODWORD(v73) = v42;
      break;
    case 0x6D:
      v72 = (__int64)sub_2136F40((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v41;
      LODWORD(v73) = v41;
      break;
    case 0x6E:
      v72 = (__int64)sub_213E300((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v40;
      LODWORD(v73) = v40;
      break;
    case 0x6F:
      v72 = sub_21378F0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      result = v38;
      LODWORD(v73) = v38;
      break;
    case 0x7A:
      v72 = (__int64)sub_213BD70(a1, a2, a4, a5, a6);
      result = v47;
      LODWORD(v73) = v47;
      break;
    case 0x7B:
      v72 = (__int64)sub_21403F0(a1, a2, a4, a5, a6);
      result = v49;
      LODWORD(v73) = v49;
      break;
    case 0x7C:
      v72 = (__int64)sub_2139A10(a1, a2, a4, a5, a6);
      result = v48;
      LODWORD(v73) = v48;
      break;
    case 0x7F:
      v72 = (__int64)sub_213A7B0((__int64)a1, a2, a4, a5, a6);
      result = v46;
      LODWORD(v73) = v46;
      break;
    case 0x80:
    case 0x84:
      v72 = sub_213AF50((__int64)a1, a2, a4, a5, a6);
      result = v28;
      LODWORD(v73) = v28;
      break;
    case 0x81:
    case 0x85:
      v72 = (__int64)sub_21394A0((__int64)a1, a2, a4, a5, a6);
      result = v27;
      LODWORD(v73) = v27;
      break;
    case 0x82:
      v72 = sub_2139850((__int64)a1, a2, a4, a5, a6);
      result = v37;
      LODWORD(v73) = v37;
      break;
    case 0x83:
      v72 = (__int64)sub_213AB80((__int64)a1, a2, a4, a5, a6);
      result = v53;
      LODWORD(v73) = v53;
      break;
    case 0x86:
      v72 = (__int64)sub_213B9B0((__int64)a1, a2, (__m128)a4, a5, a6);
      result = v52;
      LODWORD(v73) = v52;
      break;
    case 0x87:
      v72 = (__int64)sub_213BB40((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v51;
      LODWORD(v73) = v51;
      break;
    case 0x88:
      v72 = (__int64)sub_213BC50((__int64)a1, a2);
      result = v50;
      LODWORD(v73) = v50;
      break;
    case 0x89:
      v72 = sub_2128400((__int64)a1, a2, (__m128)a4, a5, a6);
      result = v58;
      LODWORD(v73) = v58;
      break;
    case 0x8E:
    case 0x8F:
    case 0x90:
      v72 = sub_213B3C0((__int64)a1, a2, a4, a5, a6);
      result = v18;
      LODWORD(v73) = v18;
      break;
    case 0x91:
      v72 = sub_213F940((__int64)a1, a2, a4, a5, a6);
      result = v55;
      LODWORD(v73) = v55;
      break;
    case 0x94:
      v72 = (__int64)sub_213BF00((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v54;
      LODWORD(v73) = v54;
      break;
    case 0x95:
    case 0x96:
    case 0x97:
      v72 = sub_213EAD0((__int64)a1, a2, a4, a5, a6);
      result = v19;
      LODWORD(v73) = v19;
      break;
    case 0x98:
    case 0x99:
      v72 = (__int64)sub_2127DE0((__int64)a1, a2, *(double *)a4.m128i_i64, a5, a6);
      result = v25;
      LODWORD(v73) = v25;
      break;
    case 0x9E:
      v72 = sub_21426C0((__int64)a1, a2, a4, a5, a6);
      result = v63;
      LODWORD(v73) = v63;
      break;
    case 0xA1:
      v72 = sub_2128090(a1, a2, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
      result = v62;
      LODWORD(v73) = v62;
      break;
    case 0xB9:
      v72 = sub_2128150(a1, a2);
      result = v57;
      LODWORD(v73) = v57;
      break;
    case 0xCC:
      v72 = sub_21286D0((__int64)a1, a2, a4, a5, a6, v10, v70, v11, (int)v12);
      result = v56;
      LODWORD(v73) = v56;
      break;
    case 0xDB:
      v72 = (__int64)sub_2127A00(a1, a2);
      result = v36;
      LODWORD(v73) = v36;
      break;
    case 0xDD:
    case 0xDE:
      v72 = (__int64)sub_213A410(a1, a2, v6);
      result = v26;
      LODWORD(v73) = v26;
      break;
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
      v72 = (__int64)sub_213A300((__int64)a1, a2);
      result = v16;
      LODWORD(v73) = v16;
      break;
    case 0xEB:
      v72 = sub_213B6A0(a1, a2);
      result = v39;
      LODWORD(v73) = v39;
      break;
  }
  if ( v72 )
  {
    sub_2010C50((__int64)a1, a2, v6, v72, v73, v15);
    v14 = *(_DWORD *)(a2 + 64);
    v13 = v72;
LABEL_5:
    *(_DWORD *)(v13 + 64) = v14;
    sub_1D306C0(a1[1], a2, v6, v13, (int)v73, 0, 0, 1);
    return v71;
  }
  return result;
}
