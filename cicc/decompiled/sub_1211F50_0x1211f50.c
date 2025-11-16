// Function: sub_1211F50
// Address: 0x1211f50
//
__int64 __fastcall sub_1211F50(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r13
  unsigned int i; // eax
  unsigned __int64 v5; // rsi
  int v6; // [rsp+Ch] [rbp-54h] BYREF
  const char *v7; // [rsp+10h] [rbp-50h] BYREF
  char v8; // [rsp+30h] [rbp-30h]
  char v9; // [rsp+31h] [rbp-2Fh]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  for ( i = *(_DWORD *)(a1 + 240); ; *(_DWORD *)(a1 + 240) = i )
  {
    v6 = 0;
    if ( i == 243 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
        return 1;
      *a2 = (2 * (v6 & 1)) | *a2 & 0xFD;
    }
    else if ( i > 0xF3 )
    {
      if ( i != 476 )
      {
LABEL_16:
        v9 = 1;
        v5 = *(_QWORD *)(a1 + 232);
        v8 = 3;
        v7 = "expected gvar flag type";
        sub_11FD800(v2, v5, (__int64)&v7, 1);
        return 1;
      }
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
        return 1;
      *a2 = (8 * (v6 & 3)) | *a2 & 0xE7;
    }
    else if ( i == 25 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
        return 1;
      *a2 = (4 * (v6 & 1)) | *a2 & 0xFB;
    }
    else
    {
      if ( i != 216 )
        goto LABEL_16;
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v6) )
        return 1;
      *a2 = v6 & 1 | *a2 & 0xFE;
    }
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    i = sub_1205200(v2);
  }
  return sub_120AFE0(a1, 13, "expected ')' here");
}
