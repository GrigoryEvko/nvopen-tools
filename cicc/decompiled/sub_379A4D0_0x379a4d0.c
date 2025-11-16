// Function: sub_379A4D0
// Address: 0x379a4d0
//
__int64 __fastcall sub_379A4D0(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  int v6; // eax
  unsigned __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 v9; // r8
  __int64 result; // rax
  unsigned int v11; // edx
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
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

  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 <= 299 )
  {
    if ( v6 > 140 )
    {
      switch ( v6 )
      {
        case 141:
        case 142:
        case 143:
        case 144:
          v7 = sub_3798B40((__int64)a1, a2);
          v9 = v13;
          goto LABEL_8;
        case 145:
          v7 = sub_3799A80((__int64)a1, a2);
          v9 = v18;
          goto LABEL_8;
        case 146:
          v7 = sub_3799EA0((__int64)a1, a2);
          v9 = v24;
          goto LABEL_8;
        case 158:
          v7 = sub_3799120((__int64)a1, a2, a6);
          v9 = v25;
          goto LABEL_8;
        case 159:
          v7 = (unsigned __int64)sub_3798E20((__int64)a1, a2, a3, a4, a5);
          v9 = v26;
          goto LABEL_8;
        case 160:
          v7 = sub_3799020((__int64)a1, a2);
          v9 = v16;
          goto LABEL_8;
        case 184:
        case 185:
          v7 = (unsigned __int64)sub_379A310((__int64)a1, a2, a6, a3, a4);
          v9 = v15;
          goto LABEL_8;
        case 206:
          v7 = sub_3799300((__int64)a1, a2);
          v9 = v17;
          goto LABEL_8;
        case 208:
          v7 = (unsigned __int64)sub_37993E0(a1, a2, a6);
          v9 = v23;
          goto LABEL_8;
        case 213:
        case 214:
        case 215:
        case 216:
        case 220:
        case 221:
        case 226:
        case 227:
        case 275:
        case 276:
        case 277:
        case 278:
          v7 = (unsigned __int64)sub_3798780((__int64)a1, a2, a6);
          v9 = v12;
          goto LABEL_8;
        case 228:
        case 229:
          v7 = (unsigned __int64)sub_3798950((__int64)a1, a2, a6);
          v9 = v14;
          goto LABEL_8;
        case 230:
          v7 = (unsigned __int64)sub_37998C0((__int64)a1, a2, a6, a3, a4);
          v9 = v19;
          goto LABEL_8;
        case 233:
          v7 = (unsigned __int64)sub_3799D00((__int64)a1, a2, a6, a3, a4);
          v9 = v20;
          goto LABEL_8;
        case 234:
          v7 = (unsigned __int64)sub_3798610((__int64)a1, a2, a6);
          v9 = v21;
          goto LABEL_8;
        case 299:
          v7 = (unsigned __int64)sub_37996A0((__int64)a1, a2, a3, a4);
          v9 = v22;
          goto LABEL_8;
        default:
          break;
      }
    }
LABEL_31:
    sub_C64ED0("Do not know how to scalarize this operator's operand!\n", 1u);
  }
  if ( v6 > 375 )
  {
    if ( (unsigned int)(v6 - 376) > 0xE )
      goto LABEL_31;
    v7 = sub_379A100((__int64)a1, a2, a6);
    v9 = v11;
  }
  else if ( v6 > 373 )
  {
    v7 = (unsigned __int64)sub_379A210((__int64)a1, a2, a6);
    v9 = v27;
  }
  else
  {
    if ( v6 != 368 )
      goto LABEL_31;
    v7 = (unsigned __int64)sub_37986E0((__int64)a1, a2);
    v9 = v8;
  }
LABEL_8:
  if ( !v7 )
    return 0;
  result = 1;
  if ( a2 != v7 )
  {
    sub_3760E70((__int64)a1, a2, 0, v7, v9);
    return 0;
  }
  return result;
}
