// Function: sub_120DC80
// Address: 0x120dc80
//
__int64 __fastcall sub_120DC80(__int64 a1)
{
  __int64 v1; // r12
  int v2; // eax
  const char *v3; // rax
  unsigned __int64 v4; // rsi
  unsigned int v5; // r13d
  unsigned int v7; // r13d
  int v8; // eax
  int v9; // eax
  int v10; // eax
  __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  const char *v12; // [rsp+10h] [rbp-50h] BYREF
  char v13; // [rsp+30h] [rbp-30h]
  char v14; // [rsp+31h] [rbp-2Fh]

  v1 = a1 + 176;
  v2 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v2;
  if ( v2 == 12 )
  {
    v7 = 0;
    v8 = sub_1205200(v1);
    *(_DWORD *)(a1 + 240) = v8;
    while ( 1 )
    {
      v11 = 0;
      if ( v8 != 78 )
        break;
      v9 = 4;
LABEL_12:
      v7 |= v9;
      v8 = sub_1205200(v1);
      *(_DWORD *)(a1 + 240) = v8;
      if ( v8 == 13 )
      {
        *(_DWORD *)(a1 + 240) = sub_1205200(v1);
        return v7;
      }
    }
    switch ( v8 )
    {
      case 278:
        v9 = 1023;
        goto LABEL_12;
      case 279:
        v9 = 3;
        goto LABEL_12;
      case 280:
        v9 = 1;
        goto LABEL_12;
      case 281:
        v9 = 2;
        goto LABEL_12;
      case 282:
        v9 = 516;
        goto LABEL_12;
      case 283:
        v9 = 512;
        goto LABEL_12;
      case 284:
        v9 = 264;
        goto LABEL_12;
      case 285:
        v9 = 8;
        goto LABEL_12;
      case 286:
        v9 = 256;
        goto LABEL_12;
      case 287:
        v9 = 16;
        goto LABEL_12;
      case 288:
        v9 = 128;
        goto LABEL_12;
      case 289:
        v9 = 96;
        goto LABEL_12;
      case 290:
        v9 = 32;
        goto LABEL_12;
      case 291:
        v9 = 64;
        goto LABEL_12;
      case 335:
        v9 = 144;
        goto LABEL_12;
      default:
        if ( v8 != 529 || v7 || (unsigned __int8)sub_120C050(a1, &v11) )
        {
          v14 = 1;
          v3 = "expected nofpclass test mask";
          goto LABEL_3;
        }
        if ( !v11 || (v11 & 0xFFFFFC00) != 0 )
        {
          v14 = 1;
          v3 = "invalid mask value for 'nofpclass'";
          goto LABEL_3;
        }
        if ( *(_DWORD *)(a1 + 240) != 13 )
        {
          v14 = 1;
          v3 = "expected ')'";
          goto LABEL_3;
        }
        v10 = sub_1205200(v1);
        v5 = v11;
        *(_DWORD *)(a1 + 240) = v10;
        break;
    }
  }
  else
  {
    v14 = 1;
    v3 = "expected '('";
LABEL_3:
    v12 = v3;
    v4 = *(_QWORD *)(a1 + 232);
    v13 = 3;
    v5 = 0;
    sub_11FD800(v1, v4, (__int64)&v12, 1);
  }
  return v5;
}
