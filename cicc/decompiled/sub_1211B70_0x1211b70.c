// Function: sub_1211B70
// Address: 0x1211b70
//
__int64 __fastcall sub_1211B70(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r13
  int i; // eax
  int v5; // eax
  unsigned __int64 v6; // rsi
  int v7; // [rsp+Ch] [rbp-54h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+30h] [rbp-30h]
  char v10; // [rsp+31h] [rbp-2Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  for ( i = *(_DWORD *)(a1 + 240); ; *(_DWORD *)(a1 + 240) = i )
  {
    v7 = 0;
    switch ( i )
    {
      case 417:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
          return 1;
        *a2 = sub_1205440(*(_DWORD *)(a1 + 240), v8) & 0xF | *a2 & 0xF0;
        v5 = sub_1205200(v2);
        *(_DWORD *)(a1 + 240) = v5;
        break;
      case 418:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
          return 1;
        sub_120C3D0(a1, &v7);
        *a2 = (16 * (v7 & 3)) | *a2 & 0xCF;
        v5 = *(_DWORD *)(a1 + 240);
        break;
      case 419:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v7) )
          return 1;
        *a2 = ((v7 & 1) << 6) | *a2 & 0xBF;
        v5 = *(_DWORD *)(a1 + 240);
        break;
      case 420:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v7) )
          return 1;
        *a2 = ((_BYTE)v7 << 7) | *a2 & 0x7F;
        v5 = *(_DWORD *)(a1 + 240);
        break;
      case 421:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v7) )
          return 1;
        a2[1] = v7 & 1 | a2[1] & 0xFE;
        v5 = *(_DWORD *)(a1 + 240);
        break;
      case 422:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v7) )
          return 1;
        a2[1] = (2 * (v7 & 1)) | a2[1] & 0xFD;
        v5 = *(_DWORD *)(a1 + 240);
        break;
      case 423:
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
          || (unsigned __int8)sub_120C430(a1, *(_DWORD *)(a1 + 240), v8) )
        {
          return 1;
        }
        a2[1] = (4 * (v8[0] & 1)) | a2[1] & 0xFB;
        v5 = sub_1205200(v2);
        *(_DWORD *)(a1 + 240) = v5;
        break;
      default:
        v10 = 1;
        v6 = *(_QWORD *)(a1 + 232);
        v9 = 3;
        v8[0] = "expected gv flag type";
        sub_11FD800(v2, v6, (__int64)v8, 1);
        return 1;
    }
    if ( v5 != 4 )
      break;
    i = sub_1205200(v2);
  }
  return sub_120AFE0(a1, 13, "expected ')' here");
}
