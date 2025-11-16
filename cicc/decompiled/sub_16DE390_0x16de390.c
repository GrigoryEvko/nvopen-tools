// Function: sub_16DE390
// Address: 0x16de390
//
__int64 __fastcall sub_16DE390(__int64 a1, unsigned __int64 a2)
{
  bool v2; // r11
  bool v3; // r10
  bool v4; // r9
  char v5; // cl
  unsigned int v6; // r8d
  char v7; // dl
  char v8; // bl
  char v10; // al
  char v11; // r10
  char v12; // al
  char v13; // al
  unsigned int v14; // r8d

  v2 = a2 > 8;
  if ( a2 <= 5 )
  {
    if ( a2 <= 3 )
    {
      v3 = 0;
      v5 = 0;
      v10 = 0;
LABEL_39:
      v4 = a2 > 6;
      if ( !v10 )
        goto LABEL_40;
      v6 = 9;
LABEL_25:
      v7 = 0;
      v8 = 1;
      goto LABEL_26;
    }
  }
  else if ( *(_DWORD *)a1 == 1768055141 && *(_WORD *)(a1 + 4) == 26216 )
  {
    v5 = 1;
    v10 = 1;
    v3 = a2 > 7;
    goto LABEL_39;
  }
  v3 = a2 > 7;
  if ( *(_DWORD *)a1 == 1768055141 )
  {
    v6 = 8;
    v5 = 1;
    v4 = a2 > 6;
    goto LABEL_25;
  }
  if ( a2 > 8 )
  {
    v4 = a2 > 6;
    if ( *(_QWORD *)a1 == 0x336E696261756E67LL )
    {
      v6 = 2;
      if ( *(_BYTE *)(a1 + 8) == 50 )
        goto LABEL_24;
    }
    if ( *(_QWORD *)a1 == 0x3436696261756E67LL )
    {
LABEL_23:
      v6 = 3;
LABEL_24:
      v5 = 1;
      goto LABEL_25;
    }
    if ( *(_QWORD *)a1 == 0x6869626165756E67LL )
    {
      v6 = 5;
      v5 = 1;
      if ( *(_BYTE *)(a1 + 8) == 102 )
        goto LABEL_25;
    }
LABEL_7:
    if ( *(_DWORD *)a1 == 1702194791 && *(_WORD *)(a1 + 4) == 25185 && *(_BYTE *)(a1 + 6) == 105 )
    {
      v7 = 0;
      v8 = 1;
      v6 = 4;
      v5 = 1;
      goto LABEL_26;
    }
    v5 = 0;
LABEL_9:
    if ( *(_DWORD *)a1 == 2020961895 && (v6 = 6, *(_WORD *)(a1 + 4) == 12851)
      || *(_DWORD *)a1 == 1701080931 && (v6 = 7, *(_WORD *)(a1 + 4) == 13873) )
    {
      v7 = 0;
      v8 = 1;
      v5 = 1;
      goto LABEL_45;
    }
LABEL_42:
    if ( *(_WORD *)a1 == 28263 )
    {
      v6 = 1;
      if ( *(_BYTE *)(a1 + 2) == 117 )
        goto LABEL_31;
    }
    v8 = v5;
    v6 = 9;
    v7 = v5 ^ 1;
    if ( (v4 & ((unsigned __int8)v5 ^ 1)) == 0 )
      goto LABEL_44;
    goto LABEL_27;
  }
  if ( a2 > 7 )
  {
    v3 = 1;
    v4 = a2 > 6;
    if ( *(_QWORD *)a1 != 0x3436696261756E67LL )
      goto LABEL_7;
    goto LABEL_23;
  }
  v5 = 0;
  v4 = a2 > 6;
  if ( a2 > 6 )
  {
    v4 = 1;
    goto LABEL_7;
  }
LABEL_40:
  if ( a2 > 5 )
    goto LABEL_9;
  if ( a2 > 2 )
    goto LABEL_42;
  v7 = 1;
  v8 = 0;
  v6 = 9;
LABEL_26:
  if ( ((unsigned __int8)v7 & v4) == 0 )
    goto LABEL_44;
LABEL_27:
  if ( *(_DWORD *)a1 == 1919184481 && *(_WORD *)(a1 + 4) == 26991 && *(_BYTE *)(a1 + 6) == 100 )
  {
    v6 = 10;
    goto LABEL_31;
  }
LABEL_44:
  if ( ((unsigned __int8)v7 & (a2 > 9)) != 0 )
  {
    if ( *(_QWORD *)a1 != 0x696261656C73756DLL || *(_WORD *)(a1 + 8) != 26216 )
    {
      if ( !v7 || !v3 )
        goto LABEL_16;
      goto LABEL_47;
    }
    v6 = 13;
LABEL_31:
    v11 = 1;
    v12 = 0;
LABEL_32:
    v13 = v2 & v12;
    goto LABEL_33;
  }
LABEL_45:
  if ( !v7 || !v3 )
  {
    if ( !v8 )
      goto LABEL_16;
    goto LABEL_31;
  }
LABEL_47:
  if ( *(_QWORD *)a1 == 0x696261656C73756DLL )
    return 12;
LABEL_16:
  if ( a2 > 3 )
  {
    if ( *(_DWORD *)a1 == 1819506029 )
      return 11;
    if ( *(_DWORD *)a1 == 1668707181 )
      return 14;
  }
  v11 = v5;
  v12 = v5 ^ 1;
  if ( (v4 & ((unsigned __int8)v5 ^ 1)) != 0 )
  {
    if ( *(_DWORD *)a1 == 1851880553 && *(_WORD *)(a1 + 4) == 30057 && *(_BYTE *)(a1 + 6) == 109 )
      return 15;
    if ( *(_DWORD *)a1 != 1852275043 )
      goto LABEL_53;
    goto LABEL_86;
  }
  if ( a2 <= 5 || !v12 )
    goto LABEL_32;
  if ( *(_DWORD *)a1 == 1852275043 )
  {
LABEL_86:
    if ( *(_WORD *)(a1 + 4) == 29557 )
      return 16;
  }
LABEL_53:
  v11 = v5;
  if ( v5 != 1 && v4 )
  {
    if ( *(_DWORD *)a1 == 1701998435 && *(_WORD *)(a1 + 4) == 27747 )
    {
      v14 = 17;
      if ( *(_BYTE *)(a1 + 6) == 114 )
        return v14;
    }
    if ( a2 <= 8 )
      return 0;
LABEL_59:
    if ( *(_QWORD *)a1 == 0x6F74616C756D6973LL )
    {
      v14 = 18;
      if ( *(_BYTE *)(a1 + 8) == 114 )
        return v14;
    }
    return 0;
  }
  v13 = (v5 ^ 1) & v2;
LABEL_33:
  if ( v13 )
    goto LABEL_59;
  if ( !v11 )
    return 0;
  return v6;
}
