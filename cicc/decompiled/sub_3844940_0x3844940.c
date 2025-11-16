// Function: sub_3844940
// Address: 0x3844940
//
void __fastcall sub_3844940(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 v11; // r8
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned __int8 *v20; // rax
  unsigned int v21; // edx
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
  unsigned int v57; // edx
  unsigned int v58; // edx
  unsigned int v59; // edx
  unsigned int v60; // edx
  unsigned int v61; // edx
  unsigned int v62; // edx
  unsigned int v63; // edx
  unsigned int v64; // edx
  unsigned int v65; // edx
  unsigned int v66; // edx
  unsigned int v67; // edx
  unsigned int v68; // edx
  unsigned __int64 v69; // rax
  __int64 v70; // rdx
  unsigned int v71; // edx
  unsigned int v72; // edx
  unsigned int v73; // edx
  unsigned int v74; // edx
  unsigned int v75; // edx
  unsigned int v76; // edx
  unsigned int v77; // edx
  unsigned int v78; // edx
  unsigned int v79; // edx
  unsigned int v80; // edx
  unsigned int v81; // edx
  unsigned int v82; // edx
  unsigned int v83; // edx
  unsigned int v84; // edx
  unsigned int v85; // edx
  unsigned int v86; // edx

  v4 = a3;
  if ( !(unsigned __int8)sub_3761870(
                           a1,
                           a2,
                           *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3),
                           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * a3 + 8),
                           1) )
  {
    switch ( *(_DWORD *)(a2 + 24) )
    {
      case 3:
        v9 = (unsigned __int64)sub_38446B0((__int64)a1, a2);
        v11 = v55;
        goto LABEL_6;
      case 4:
        v9 = (unsigned __int64)sub_3843990((__int64)a1, a2, a4);
        v11 = v56;
        goto LABEL_6;
      case 0xB:
        v9 = (unsigned __int64)sub_38147C0(a1, a2, a4);
        v11 = v57;
        goto LABEL_6;
      case 0x33:
        v9 = (unsigned __int64)sub_3816490(a1, a2);
        v11 = v58;
        goto LABEL_6;
      case 0x34:
        v9 = (unsigned __int64)sub_38381F0((__int64)a1, a2, a4);
        v11 = v59;
        goto LABEL_6;
      case 0x36:
        v9 = (unsigned __int64)sub_38146A0(a1, a2, a4);
        v11 = v60;
        goto LABEL_6;
      case 0x37:
        v69 = sub_3761980((__int64)a1, a2, v4);
        v9 = sub_37AE0F0((__int64)a1, v69, v70);
        v11 = v71;
        goto LABEL_6;
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0xBA:
      case 0xBB:
      case 0xBC:
      case 0x18B:
      case 0x18C:
      case 0x18F:
      case 0x190:
      case 0x194:
      case 0x197:
        v9 = (unsigned __int64)sub_3844130((__int64)a1, a2);
        v11 = v10;
        goto LABEL_6;
      case 0x3B:
      case 0x3D:
      case 0xAE:
      case 0xB0:
      case 0xB2:
      case 0xB4:
      case 0xB5:
      case 0x191:
      case 0x193:
      case 0x198:
      case 0x199:
        v9 = (unsigned __int64)sub_383DFD0((__int64)a1, a2, a4, v5, v6, v7, v8);
        v11 = v15;
        goto LABEL_6;
      case 0x3C:
      case 0x3E:
      case 0xAF:
      case 0xB1:
      case 0xB3:
      case 0x195:
      case 0x196:
      case 0x19A:
      case 0x19B:
        v9 = (unsigned __int64)sub_3837640((__int64)a1, a2, a4);
        v11 = v16;
        goto LABEL_6;
      case 0x46:
      case 0x47:
      case 0x48:
      case 0x49:
        if ( (_DWORD)v4 == 1 )
          goto LABEL_35;
        v20 = sub_383C440((__int64)a1, a2);
        goto LABEL_18;
      case 0x4A:
      case 0x4B:
        goto LABEL_35;
      case 0x4C:
      case 0x4E:
        if ( (_DWORD)v4 == 1 )
          goto LABEL_35;
        v20 = sub_383C140((__int64)a1, a2);
        goto LABEL_18;
      case 0x4D:
      case 0x4F:
        if ( (_DWORD)v4 == 1 )
LABEL_35:
          v20 = sub_38159C0(a1, a2);
        else
          v20 = sub_3831CB0((__int64)a1, a2, a4);
LABEL_18:
        v9 = (unsigned __int64)v20;
        v11 = v21;
        goto LABEL_6;
      case 0x50:
      case 0x51:
        v9 = (unsigned __int64)sub_383C920((__int64)a1, a2, v4);
        v11 = v32;
        goto LABEL_6;
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
      case 0x56:
      case 0x57:
        v9 = (unsigned __int64)sub_383EE00(a1, a2);
        v11 = v18;
        goto LABEL_6;
      case 0x58:
      case 0x59:
      case 0x5A:
      case 0x5B:
        v9 = sub_383B570((__int64)a1, a2, a4);
        v11 = v24;
        goto LABEL_6;
      case 0x5C:
      case 0x5D:
      case 0x5E:
      case 0x5F:
        v9 = (unsigned __int64)sub_383BB10(a1, a2, a4);
        v11 = v25;
        goto LABEL_6;
      case 0x8D:
      case 0x8E:
      case 0xE2:
      case 0xE3:
      case 0x1C4:
      case 0x1C5:
        v9 = (unsigned __int64)sub_3814AA0(a1, a2, a4);
        v11 = v17;
        goto LABEL_6;
      case 0x93:
      case 0x94:
      case 0xD0:
        v9 = (unsigned __int64)sub_3815DC0(a1, a2, a4);
        v11 = v31;
        goto LABEL_6;
      case 0x9B:
        v9 = (unsigned __int64)sub_38161E0(a1, a2);
        v11 = v85;
        goto LABEL_6;
      case 0x9C:
        v9 = (unsigned __int64)sub_382A2C0(a1, a2);
        v11 = v86;
        goto LABEL_6;
      case 0x9D:
        v9 = sub_3843BE0((__int64)a1, a2, a4, v5, v6, v7, v8);
        v11 = v65;
        goto LABEL_6;
      case 0x9E:
        v9 = (unsigned __int64)sub_382DCF0(a1, a2);
        v11 = v66;
        goto LABEL_6;
      case 0x9F:
        v9 = (unsigned __int64)sub_3830830(a1, a2, a4);
        v11 = v63;
        goto LABEL_6;
      case 0xA0:
        v9 = sub_38302F0(a1, a2);
        v11 = v64;
        goto LABEL_6;
      case 0xA1:
        v9 = (unsigned __int64)sub_3833800(a1, a2);
        v11 = v53;
        goto LABEL_6;
      case 0xA2:
      case 0xA3:
        sub_38355B0((__int64)a1, a2, v5, v6, v7, v8);
        return;
      case 0xA4:
        v9 = (unsigned __int64)sub_38383A0((__int64)a1, a2, a4);
        v11 = v61;
        goto LABEL_6;
      case 0xA5:
        v9 = (unsigned __int64)sub_3830630((__int64)a1, a2, a4);
        v11 = v62;
        goto LABEL_6;
      case 0xA6:
        v9 = sub_383A670((__int64)a1, a2);
        v11 = v54;
        goto LABEL_6;
      case 0xA7:
      case 0xA8:
      case 0x1EC:
        v9 = sub_382A8A0(a1, a2, a4);
        v11 = v30;
        goto LABEL_6;
      case 0xAA:
        v9 = (unsigned __int64)sub_382AA40(a1, a2, a4);
        v11 = v67;
        goto LABEL_6;
      case 0xAB:
        v9 = sub_383AA10((__int64)a1, a2);
        v11 = v68;
        goto LABEL_6;
      case 0xB6:
      case 0xB7:
        v9 = (unsigned __int64)sub_383EC80(a1, a2);
        v11 = v40;
        goto LABEL_6;
      case 0xB8:
      case 0xB9:
        v9 = (unsigned __int64)sub_3815CB0(a1, a2);
        v11 = v41;
        goto LABEL_6;
      case 0xBD:
        v9 = (unsigned __int64)sub_383C680((__int64)a1, a2, a4);
        v11 = v83;
        goto LABEL_6;
      case 0xBE:
      case 0x192:
        v9 = (unsigned __int64)sub_3839100(a1, a2, a4);
        v11 = v42;
        goto LABEL_6;
      case 0xBF:
      case 0x18D:
        v9 = (unsigned __int64)sub_383E1F0(a1, a2, a4, v5, v6, v7, v8);
        v11 = v43;
        goto LABEL_6;
      case 0xC0:
      case 0x18E:
        v9 = (unsigned __int64)sub_3838700((__int64)a1, a2);
        v11 = v44;
        goto LABEL_6;
      case 0xC1:
      case 0xC2:
        v9 = sub_3816450((__int64)a1, a2);
        v11 = v45;
        goto LABEL_6;
      case 0xC3:
      case 0xC4:
        v9 = sub_38379A0(a1, a2);
        v11 = v46;
        goto LABEL_6;
      case 0xC5:
      case 0x19D:
        v9 = (unsigned __int64)sub_382CF50(a1, a2, a4);
        v11 = v47;
        goto LABEL_6;
      case 0xC6:
      case 0xCB:
      case 0x1A2:
      case 0x1A3:
        v9 = sub_382D870(a1, a2, a4);
        v11 = v26;
        goto LABEL_6;
      case 0xC7:
      case 0xCC:
      case 0x1A0:
      case 0x1A1:
        v9 = sub_3838AB0(a1, a2, a4);
        v11 = v22;
        goto LABEL_6;
      case 0xC8:
      case 0xCA:
      case 0x19F:
        v9 = (unsigned __int64)sub_38371B0(a1, a2, a4);
        v11 = v29;
        goto LABEL_6;
      case 0xC9:
      case 0x19E:
        v9 = (unsigned __int64)sub_382D3F0(a1, a2, a4);
        v11 = v35;
        goto LABEL_6;
      case 0xCD:
      case 0xCE:
      case 0x1E8:
      case 0x1E9:
        v9 = sub_38443E0((__int64)a1, a2);
        v11 = v27;
        goto LABEL_6;
      case 0xCF:
        v9 = (unsigned __int64)sub_383B180((__int64)a1, a2);
        v11 = v52;
        goto LABEL_6;
      case 0xD5:
      case 0xD6:
      case 0xD7:
      case 0x1CB:
      case 0x1CC:
        v9 = sub_382E080(a1, a2, a4);
        v11 = v19;
        goto LABEL_6;
      case 0xD8:
      case 0x1CA:
        v9 = sub_3835B00(a1, a2, a4);
        v11 = v33;
        goto LABEL_6;
      case 0xDE:
        v9 = (unsigned __int64)sub_3839D50((__int64)a1, a2);
        v11 = v80;
        goto LABEL_6;
      case 0xDF:
      case 0xE0:
      case 0xE1:
        v9 = (unsigned __int64)sub_3842130(a1, a2, a4);
        v11 = v28;
        goto LABEL_6;
      case 0xE4:
      case 0xE5:
        v9 = (unsigned __int64)sub_3815150(a1, a2);
        v11 = v34;
        goto LABEL_6;
      case 0xE7:
        v9 = (unsigned __int64)sub_38155A0(a1, a2);
        v11 = v79;
        goto LABEL_6;
      case 0xEA:
        v9 = (unsigned __int64)sub_38325E0((__int64)a1, (_QWORD *)a2, a4);
        v11 = v84;
        goto LABEL_6;
      case 0xED:
      case 0xF1:
        v9 = (unsigned __int64)sub_3815250(a1, a2, a4);
        v11 = v48;
        goto LABEL_6;
      case 0xEF:
      case 0xF3:
        v9 = (unsigned __int64)sub_3815350(a1, a2);
        v11 = v49;
        goto LABEL_6;
      case 0x105:
        v9 = (unsigned __int64)sub_3816300(a1, a2);
        v11 = v81;
        goto LABEL_6;
      case 0x115:
      case 0x116:
        v9 = (unsigned __int64)sub_38154A0(a1, a2, a4);
        v11 = v50;
        goto LABEL_6;
      case 0x12A:
        v9 = (unsigned __int64)sub_38156E0(a1, a2);
        v11 = v82;
        goto LABEL_6;
      case 0x13D:
        v9 = sub_3816720((__int64 **)a1, a2, a4);
        v11 = v73;
        goto LABEL_6;
      case 0x152:
        v9 = (unsigned __int64)sub_38144A0(a1, a2);
        v11 = v74;
        goto LABEL_6;
      case 0x154:
      case 0x155:
        v9 = (unsigned __int64)sub_383DA20(a1, a2, v4, a4);
        v11 = v36;
        goto LABEL_6;
      case 0x156:
      case 0x157:
      case 0x158:
      case 0x159:
      case 0x15A:
      case 0x15B:
      case 0x15C:
      case 0x15D:
      case 0x15E:
      case 0x15F:
      case 0x160:
      case 0x161:
        v9 = (unsigned __int64)sub_3842830((__int64)a1, a2);
        v11 = v12;
        goto LABEL_6;
      case 0x16A:
        v9 = (unsigned __int64)sub_3843E80(a1, a2);
        v11 = v75;
        goto LABEL_6;
      case 0x16C:
        v9 = (unsigned __int64)sub_382E330(a1, a2);
        v11 = v76;
        goto LABEL_6;
      case 0x175:
        v9 = (unsigned __int64)sub_3816560(a1, a2, a4);
        v11 = v77;
        goto LABEL_6;
      case 0x17E:
      case 0x17F:
      case 0x180:
      case 0x181:
      case 0x182:
      case 0x183:
      case 0x184:
      case 0x185:
      case 0x186:
        v9 = (unsigned __int64)sub_382AD60(a1, a2, a4);
        v11 = v13;
        goto LABEL_6;
      case 0x187:
      case 0x188:
        v9 = sub_3842EC0((__int64)a1, a2);
        v11 = v37;
        goto LABEL_6;
      case 0x18A:
        v9 = (unsigned __int64)sub_382AE60(a1, a2);
        v11 = v72;
        goto LABEL_6;
      case 0x1A4:
      case 0x1A5:
        v9 = (unsigned __int64)sub_38149A0(a1, a2, a4);
        v11 = v38;
        goto LABEL_6;
      case 0x1A6:
      case 0x1A7:
        v9 = (unsigned __int64)sub_38393F0(a1, a2);
        v11 = v39;
        goto LABEL_6;
      case 0x1A8:
      case 0x1A9:
      case 0x1AA:
      case 0x1AB:
        v9 = sub_383FAD0(a1, a2);
        v11 = v23;
        goto LABEL_6;
      case 0x1D4:
        v9 = (unsigned __int64)sub_3815850(a1, a2);
        v11 = v51;
        goto LABEL_6;
      case 0x1D7:
      case 0x1D8:
      case 0x1D9:
      case 0x1DA:
      case 0x1DB:
      case 0x1DC:
      case 0x1DD:
      case 0x1DE:
      case 0x1DF:
        v9 = (unsigned __int64)sub_3841890((__int64)a1, a2, a4);
        v11 = v14;
        goto LABEL_6;
      case 0x1F2:
        v9 = (unsigned __int64)sub_382AC40(a1, a2, a4);
        v11 = v78;
LABEL_6:
        if ( v9 )
          sub_375F010((__int64)a1, a2, v4, v9, v11);
        break;
      default:
        sub_C64ED0("Do not know how to promote this operator!", 1u);
    }
  }
}
