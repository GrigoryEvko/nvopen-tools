// Function: sub_37ADC40
// Address: 0x37adc40
//
__int64 __fastcall sub_37ADC40(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int *v5; // rax
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r13d
  unsigned __int64 v14; // rcx
  unsigned int v15; // edx
  __int64 v16; // r8
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

  v5 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
  v7 = sub_3761870((_QWORD *)a1, a2, *(_WORD *)v6, *(_QWORD *)(v6 + 8), 0);
  if ( (_BYTE)v7 )
    return 0;
  v12 = v7;
  switch ( *(_DWORD *)(a2 + 24) )
  {
    case 0x8D:
    case 0x8E:
    case 0x8F:
    case 0x90:
    case 0x91:
    case 0x92:
    case 0xD8:
    case 0xDC:
    case 0xDD:
    case 0xE2:
    case 0xE3:
    case 0xE6:
    case 0xE9:
      v14 = (unsigned __int64)sub_37A59E0(a1, a2, a4, v8, v9, v10);
      v16 = v18;
      break;
    case 0x93:
    case 0x94:
      v14 = (unsigned __int64)sub_37AA9B0(a1, a2, a4);
      v16 = v24;
      break;
    case 0x98:
    case 0x103:
    case 0x113:
    case 0x114:
    case 0x115:
    case 0x116:
      v14 = (unsigned __int64)sub_3412A00(*(_QWORD **)(a1 + 8), a2, 0, v9, v10, v11, a4);
      v16 = v19;
      break;
    case 0x9B:
      v14 = (unsigned __int64)sub_37A5460((__int64 *)a1, a2, a4);
      v16 = v27;
      break;
    case 0x9E:
      v14 = (unsigned __int64)sub_37A8940(a1, a2);
      v16 = v40;
      break;
    case 0x9F:
      v14 = sub_37A7A50((__int64 *)a1, a2, v8, v9, v10);
      v16 = v41;
      break;
    case 0xA0:
      v14 = sub_37A8020((__int64 *)a1, a2);
      v16 = v42;
      break;
    case 0xA1:
      v14 = (unsigned __int64)sub_37A8860(a1, a2);
      v16 = v43;
      break;
    case 0xB8:
    case 0xB9:
      v14 = (unsigned __int64)sub_37A5180(a1, a2, a4);
      v16 = v22;
      break;
    case 0xCE:
      v14 = (unsigned __int64)sub_37AC610(a1, a2, a4);
      v16 = v38;
      break;
    case 0xD0:
      v14 = (unsigned __int64)sub_37AA430((_QWORD *)a1, a2, a4);
      v16 = v39;
      break;
    case 0xD5:
    case 0xD6:
    case 0xD7:
      v14 = (unsigned __int64)sub_37A67D0((__int64 *)a1, a2);
      v16 = v20;
      break;
    case 0xDF:
    case 0xE0:
    case 0xE1:
      v14 = (unsigned __int64)sub_37A8A20(a1, a2, a4);
      v16 = v21;
      break;
    case 0xE4:
    case 0xE5:
      v14 = (unsigned __int64)sub_37A70B0((__int64 *)a1, a2, a4);
      v16 = v23;
      break;
    case 0xEA:
      v14 = (unsigned __int64)sub_37A7420((_QWORD *)a1, a2, a4);
      v16 = v36;
      break;
    case 0x12B:
      v14 = (unsigned __int64)sub_37ADB10((__int64 *)a1, a2, a4);
      v16 = v37;
      break;
    case 0x16B:
      v14 = (unsigned __int64)sub_37A8F80((__int64 *)a1, a2, a3);
      v16 = v28;
      break;
    case 0x16C:
      v14 = sub_37A9500(a1, a2);
      v16 = v29;
      break;
    case 0x16D:
      v14 = (unsigned __int64)sub_37A96C0(a1, a2, a3, a4);
      v16 = v30;
      break;
    case 0x170:
      v14 = (unsigned __int64)sub_37A79B0(a1, a2);
      v16 = v31;
      break;
    case 0x176:
    case 0x177:
      v14 = (unsigned __int64)sub_37ABC10((__int64 *)a1, a2);
      v16 = v26;
      break;
    case 0x178:
    case 0x179:
    case 0x17A:
    case 0x17B:
    case 0x17C:
    case 0x17D:
    case 0x17E:
    case 0x17F:
    case 0x180:
    case 0x181:
    case 0x182:
    case 0x183:
    case 0x184:
    case 0x185:
    case 0x186:
      v14 = (unsigned __int64)sub_37AB210((_QWORD *)a1, a2, a4);
      v16 = v17;
      break;
    case 0x1A4:
    case 0x1A5:
      v14 = (unsigned __int64)sub_37AC7E0(a1, a2);
      v16 = v25;
      break;
    case 0x1D1:
      v14 = (unsigned __int64)sub_37A8BC0(a1, a2, a3);
      v16 = v34;
      break;
    case 0x1D2:
      v14 = (unsigned __int64)sub_37A8E00(a1, a2);
      v16 = v32;
      break;
    case 0x1D3:
      v14 = (unsigned __int64)sub_37A9DB0(a1, a2, a3);
      v16 = v33;
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
    case 0x1E0:
    case 0x1E1:
    case 0x1E2:
    case 0x1E3:
    case 0x1E4:
    case 0x1E5:
    case 0x1E6:
    case 0x1E7:
      v14 = (unsigned __int64)sub_37AC4A0(a1, a2);
      v16 = v15;
      break;
    case 0x1EC:
      v14 = sub_37A8AE0(a1, a2);
      v16 = v35;
      break;
    default:
      sub_C64ED0("Do not know how to widen this operator's operand!", 1u);
  }
  if ( !v14 )
  {
    return 0;
  }
  else if ( a2 == v14 )
  {
    return 1;
  }
  else
  {
    sub_3760E70(a1, a2, 0, v14, v16);
  }
  return v12;
}
