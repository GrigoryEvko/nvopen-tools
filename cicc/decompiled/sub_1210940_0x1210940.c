// Function: sub_1210940
// Address: 0x1210940
//
__int64 __fastcall sub_1210940(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  unsigned __int64 v4; // rsi
  unsigned int v5; // eax
  _QWORD v6[4]; // [rsp+0h] [rbp-50h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  if ( !(unsigned __int8)sub_120AFE0(a1, 464, "expected 'typeTestRes' here")
    && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    && !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
    && !(unsigned __int8)sub_120AFE0(a1, 465, "expected 'kind' here")
    && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
  {
    switch ( *(_DWORD *)(a1 + 240) )
    {
      case 0x1BC:
        *(_DWORD *)a2 = 5;
        goto LABEL_11;
      case 0x1D2:
        *(_DWORD *)a2 = 0;
        goto LABEL_11;
      case 0x1D3:
        *(_DWORD *)a2 = 1;
        goto LABEL_11;
      case 0x1D4:
        *(_DWORD *)a2 = 2;
        goto LABEL_11;
      case 0x1D5:
        *(_DWORD *)a2 = 3;
        goto LABEL_11;
      case 0x1D6:
        *(_DWORD *)a2 = 4;
LABEL_11:
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
          || (unsigned __int8)sub_120AFE0(a1, 471, "expected 'sizeM1BitWidth' here")
          || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
          || (unsigned __int8)sub_120BD00(a1, (_DWORD *)(a2 + 4)) )
        {
          return 1;
        }
        if ( *(_DWORD *)(a1 + 240) != 4 )
          return sub_120AFE0(a1, 13, "expected ')' here");
        break;
      default:
        v8 = 1;
        v3 = "unexpected TypeTestResolution kind";
        goto LABEL_9;
    }
    while ( 1 )
    {
      v5 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v5;
      if ( v5 == 474 )
      {
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_120BD00(a1, v6) )
          return 1;
        *(_BYTE *)(a2 + 24) = v6[0];
      }
      else if ( v5 > 0x1DA )
      {
        if ( v5 != 475 )
        {
LABEL_37:
          v8 = 1;
          v3 = "expected optional TypeTestResolution field";
LABEL_9:
          v6[0] = v3;
          v4 = *(_QWORD *)(a1 + 232);
          v7 = 3;
          sub_11FD800(a1 + 176, v4, (__int64)v6, 1);
          return 1;
        }
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
          || (unsigned __int8)sub_120C050(a1, (__int64 *)(a2 + 32)) )
        {
          return 1;
        }
      }
      else if ( v5 == 472 )
      {
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
          || (unsigned __int8)sub_120C050(a1, (__int64 *)(a2 + 8)) )
        {
          return 1;
        }
      }
      else
      {
        if ( v5 != 473 )
          goto LABEL_37;
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
          || (unsigned __int8)sub_120C050(a1, (__int64 *)(a2 + 16)) )
        {
          return 1;
        }
      }
      if ( *(_DWORD *)(a1 + 240) != 4 )
        return sub_120AFE0(a1, 13, "expected ')' here");
    }
  }
  return 1;
}
