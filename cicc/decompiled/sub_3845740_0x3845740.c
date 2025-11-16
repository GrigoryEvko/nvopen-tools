// Function: sub_3845740
// Address: 0x3845740
//
__int64 __fastcall sub_3845740(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int *v5; // rax
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r13d
  unsigned __int64 v14; // r15
  unsigned int v15; // edx
  __int64 v16; // rbx
  int v17; // eax
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
  unsigned __int8 *v41; // r8
  __int64 v42; // rdx
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

  v5 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
  v7 = sub_3761870((_QWORD *)a1, a2, *(_WORD *)v6, *(_QWORD *)(v6 + 8), 0);
  if ( !(_BYTE)v7 )
  {
    v12 = v7;
    switch ( *(_DWORD *)(a2 + 24) )
    {
      case 0x16:
      case 0x17:
        v14 = (unsigned __int64)sub_383A4C0(a1, (__int64 *)a2, a4);
        v16 = v28;
        break;
      case 0x36:
        v14 = (unsigned __int64)sub_3832280(a1, a2, a4);
        v16 = v64;
        break;
      case 0x58:
      case 0x59:
      case 0x5A:
      case 0x5B:
      case 0x5C:
      case 0x5D:
      case 0x5E:
      case 0x5F:
        v14 = (unsigned __int64)sub_383AFB0(a1, a2, a4);
        v16 = v19;
        break;
      case 0x6D:
      case 0x6E:
      case 0x102:
      case 0x103:
        v14 = (unsigned __int64)sub_383D030((_QWORD *)a1, a2, a4);
        v16 = v22;
        break;
      case 0x8F:
        v14 = (unsigned __int64)sub_38430E0(a1, a2);
        v16 = v63;
        break;
      case 0x90:
      case 0xBE:
      case 0xBF:
      case 0xC0:
      case 0xC1:
      case 0xC2:
      case 0xEE:
        v14 = (unsigned __int64)sub_383A840(a1, a2, a4);
        v16 = v20;
        break;
      case 0x9C:
        v14 = (unsigned __int64)sub_3835280(a1, (__int64 *)a2);
        v16 = v61;
        break;
      case 0x9D:
        v14 = (unsigned __int64)sub_382EC80((__int64 *)a1, a2, a3, a4);
        v16 = v62;
        break;
      case 0x9E:
        v14 = (unsigned __int64)sub_3831730((__int64 *)a1, a2, a4);
        v16 = v49;
        break;
      case 0x9F:
        v14 = sub_3836600(a1, a2, a4, v8, v9, v10, v11);
        v16 = v37;
        break;
      case 0xA0:
        v14 = (unsigned __int64)sub_38319D0(a1, a2);
        v16 = v38;
        break;
      case 0xA1:
        v14 = (unsigned __int64)sub_38454A0(a1, a2, a4);
        v16 = v39;
        break;
      case 0xA7:
      case 0xA8:
      case 0x1EC:
        v14 = (unsigned __int64)sub_3836D60(a1, a2);
        v16 = v23;
        break;
      case 0xAB:
        v14 = sub_38174D0(a1, a2);
        v16 = v59;
        break;
      case 0xB8:
      case 0xB9:
        v14 = (unsigned __int64)sub_383ED50(a1, a2);
        v16 = v29;
        break;
      case 0xC3:
      case 0xC4:
        v14 = (unsigned __int64)sub_383ADD0(a1, a2, a4);
        v16 = v32;
        break;
      case 0xCD:
      case 0xCE:
        v14 = (unsigned __int64)sub_38170E0(a1, a2, a4);
        v16 = v25;
        break;
      case 0xCF:
        v14 = (unsigned __int64)sub_3842370(a1, (__int64 *)a2);
        v16 = v40;
        break;
      case 0xD0:
      case 0x1CF:
        v14 = (unsigned __int64)sub_3842530((__int64 *)a1, a2);
        v16 = v26;
        break;
      case 0xD5:
        v14 = (unsigned __int64)sub_3843520(a1, a2, a4);
        v16 = v35;
        break;
      case 0xD6:
        v14 = (unsigned __int64)sub_3842650(a1, a2, a4);
        v16 = v33;
        break;
      case 0xD7:
        v14 = (unsigned __int64)sub_3836EB0(a1, a2, a4);
        v16 = v34;
        break;
      case 0xD8:
      case 0x1CA:
        v14 = sub_3843750(a1, a2, a4);
        v16 = v27;
        break;
      case 0xDC:
      case 0x1C7:
        v14 = (unsigned __int64)sub_383CE80(a1, a2);
        v16 = v31;
        break;
      case 0xDD:
      case 0xEC:
      case 0xF0:
      case 0x1C6:
        v14 = (unsigned __int64)sub_3837060(a1, a2, a4);
        v16 = v21;
        break;
      case 0xE8:
        v14 = (unsigned __int64)sub_383AC00(a1, a2, a4);
        v16 = v55;
        break;
      case 0xEA:
        v14 = (unsigned __int64)sub_382E910((__int64 *)a1, a2, a4);
        v16 = v56;
        break;
      case 0x12B:
        v14 = (unsigned __int64)sub_3839F10(a1, a2);
        v16 = v65;
        break;
      case 0x131:
        v16 = 0;
        v41 = sub_375B580(
                a1,
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                a4,
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                1,
                0);
        v14 = (unsigned __int64)sub_33EC3B0(
                                  *(_QWORD **)(a1 + 8),
                                  (__int64 *)a2,
                                  **(_QWORD **)(a2 + 40),
                                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                  (__int64)v41,
                                  v42,
                                  *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
        break;
      case 0x132:
        v14 = (unsigned __int64)sub_3842450(a1, a2);
        v16 = v43;
        break;
      case 0x153:
        v14 = (unsigned __int64)sub_383A2C0(a1, a2);
        v16 = v44;
        break;
      case 0x16A:
        v14 = (unsigned __int64)sub_38172E0(a1, a2, a3, a4);
        v16 = v45;
        break;
      case 0x16B:
        v14 = (unsigned __int64)sub_382F1E0(a1, a2, a3, a4);
        v16 = v46;
        break;
      case 0x16C:
        v14 = sub_3840BB0(a1, a2, a3, a4, v9, v10, v11);
        v16 = v47;
        break;
      case 0x16D:
        v14 = (unsigned __int64)sub_3840E60(a1, a2, a3, a4, v9, v10);
        v16 = v48;
        break;
      case 0x170:
        v14 = (unsigned __int64)sub_382B110((__int64 *)a1, a2);
        v16 = v57;
        break;
      case 0x17E:
      case 0x17F:
      case 0x180:
      case 0x181:
      case 0x182:
      case 0x183:
      case 0x184:
      case 0x185:
      case 0x186:
        v14 = (unsigned __int64)sub_3841970((__int64 *)a1, a2, a4);
        v16 = v15;
        break;
      case 0x187:
      case 0x188:
        v14 = (unsigned __int64)sub_383D760(a1, a2, a4, v8, v9, v10, v11);
        v16 = v30;
        break;
      case 0x189:
        v14 = (unsigned __int64)sub_38175D0((__int64 *)a1, a2, a3, a4, v9, v10, v11);
        v16 = v58;
        break;
      case 0x18A:
        v14 = (unsigned __int64)sub_3817820((__int64 *)a1, a2, a3, a4, v9, v10, v11);
        v16 = v36;
        break;
      case 0x1CB:
        v14 = (unsigned __int64)sub_3842CA0(a1, a2, a4);
        v16 = v50;
        break;
      case 0x1CC:
        v14 = (unsigned __int64)sub_382EE50(a1, a2, a4);
        v16 = v51;
        break;
      case 0x1D1:
        v14 = (unsigned __int64)sub_383A0C0(a1, a2);
        v16 = v52;
        break;
      case 0x1D2:
      case 0x1D5:
        v14 = (unsigned __int64)sub_3831FE0(a1, a2, a3, v9, v10, v11);
        v16 = v24;
        break;
      case 0x1D7:
      case 0x1D8:
      case 0x1D9:
      case 0x1DA:
      case 0x1DB:
      case 0x1DC:
      case 0x1DD:
      case 0x1DE:
      case 0x1DF:
        v14 = (unsigned __int64)sub_38412E0(a1, a2, a3, a4);
        v16 = v18;
        break;
      case 0x1EA:
        v14 = (unsigned __int64)sub_383D470(a1, a2, a3, a4);
        v16 = v53;
        break;
      case 0x1F1:
        v14 = (unsigned __int64)sub_3842A50(a1, a2, a3, v9, v10, v11);
        v16 = v54;
        break;
      case 0x1F2:
        v14 = (unsigned __int64)sub_3843300(a1, a2, a3);
        v16 = v60;
        break;
      default:
        sub_C64ED0("Do not know how to promote this operator's operand!", 1u);
    }
    if ( v14 )
    {
      if ( a2 == v14 )
        return 1;
      v17 = *(_DWORD *)(a2 + 24);
      if ( v17 > 239 )
      {
        if ( (unsigned int)(v17 - 242) > 1 )
        {
LABEL_12:
          sub_3760E70(a1, a2, 0, v14, v16);
          return v12;
        }
      }
      else if ( v17 <= 237 && (unsigned int)(v17 - 101) > 0x2F )
      {
        goto LABEL_12;
      }
      sub_3760E70(a1, a2, 0, v14, v16);
      sub_3760E70(a1, a2, 1, v14, 1);
      return v12;
    }
  }
  return 0;
}
