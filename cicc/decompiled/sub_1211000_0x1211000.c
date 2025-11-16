// Function: sub_1211000
// Address: 0x1211000
//
__int64 __fastcall sub_1211000(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r13
  int i; // eax
  unsigned __int64 v5; // rsi
  int v6; // [rsp+Ch] [rbp-54h] BYREF
  const char *v7; // [rsp+10h] [rbp-50h] BYREF
  char v8; // [rsp+30h] [rbp-30h]
  char v9; // [rsp+31h] [rbp-2Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in funcFlags")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in funcFlags") )
  {
    return 1;
  }
  for ( i = *(_DWORD *)(a1 + 240); ; *(_DWORD *)(a1 + 240) = i )
  {
    v6 = 0;
    switch ( i )
    {
      case 429:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = v6 & 1 | *a2 & 0xFE;
        break;
      case 430:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = (2 * (v6 & 1)) | *a2 & 0xFD;
        break;
      case 431:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = (4 * (v6 & 1)) | *a2 & 0xFB;
        break;
      case 432:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = (8 * (v6 & 1)) | *a2 & 0xF7;
        break;
      case 433:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = (16 * (v6 & 1)) | *a2 & 0xEF;
        break;
      case 434:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = (32 * (v6 & 1)) | *a2 & 0xDF;
        break;
      case 435:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = ((v6 & 1) << 6) | *a2 & 0xBF;
        break;
      case 436:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        *a2 = ((_BYTE)v6 << 7) | *a2 & 0x7F;
        break;
      case 437:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        a2[1] = v6 & 1 | a2[1] & 0xFE;
        break;
      case 438:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
          return 1;
        a2[1] = (2 * (v6 & 1)) | a2[1] & 0xFD;
        break;
      default:
        v9 = 1;
        v5 = *(_QWORD *)(a1 + 232);
        v8 = 3;
        v7 = "expected function flag type";
        sub_11FD800(v2, v5, (__int64)&v7, 1);
        return 1;
    }
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    i = sub_1205200(v2);
  }
  return sub_120AFE0(a1, 13, "expected ')' in funcFlags");
}
