// Function: sub_380A350
// Address: 0x380a350
//
__int64 __fastcall sub_380A350(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  unsigned __int64 v3; // rcx
  unsigned int v4; // edx
  __int64 v5; // r8
  __int64 result; // rax
  unsigned int v7; // edx
  unsigned int v8; // edx
  unsigned int v9; // edx
  unsigned int v10; // edx
  unsigned int v11; // edx
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx

  switch ( *(_DWORD *)(a2 + 24) )
  {
    case 0x87:
    case 0x113:
      v3 = sub_380A0D0(a1, a2);
      v5 = v10;
      break;
    case 0x88:
    case 0x114:
      v3 = sub_380A170(a1, a2);
      v5 = v11;
      break;
    case 0x89:
    case 0x115:
      v3 = sub_380A210(a1, a2);
      v5 = v12;
      break;
    case 0x8A:
    case 0x116:
      v3 = sub_380A2B0(a1, a2);
      v5 = v13;
      break;
    case 0x8D:
    case 0x8E:
    case 0xE2:
    case 0xE3:
      v3 = (unsigned __int64)sub_3809310(a1, a2);
      v5 = v7;
      break;
    case 0x91:
    case 0xE6:
    case 0xED:
    case 0xEF:
    case 0xF1:
    case 0xF3:
      v3 = sub_3808DE0(a1, a2);
      v5 = v4;
      break;
    case 0x93:
    case 0x94:
    case 0xD0:
      v3 = (unsigned __int64)sub_3809850(a1, a2);
      v5 = v8;
      break;
    case 0x98:
      v3 = (unsigned __int64)sub_37FDAF0(a1, a2);
      v5 = v17;
      break;
    case 0xCF:
      v3 = (unsigned __int64)sub_3809620(a1, a2);
      v5 = v16;
      break;
    case 0xE4:
    case 0xE5:
      v3 = (unsigned __int64)sub_346B4B0((_BYTE *)*a1, a2, a1[1]);
      v5 = v9;
      break;
    case 0xEA:
      v3 = (unsigned __int64)sub_3808D10((__int64)a1, a2, a3);
      v5 = v14;
      break;
    case 0x12B:
      v3 = (unsigned __int64)sub_3809BA0((__int64)a1, a2, a3);
      v5 = v15;
      break;
    case 0x132:
      v3 = (unsigned __int64)sub_38090D0(a1, a2);
      v5 = v18;
      break;
    case 0x153:
      v3 = (unsigned __int64)sub_3809CF0((__int64)a1, a2);
      v5 = v19;
      break;
    default:
      sub_C64ED0("Do not know how to soften this operator's operand!", 1u);
  }
  if ( !v3 )
    return 0;
  result = 1;
  if ( a2 != v3 )
  {
    sub_3760E70((__int64)a1, a2, 0, v3, v5);
    return 0;
  }
  return result;
}
