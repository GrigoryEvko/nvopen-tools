// Function: sub_14E7070
// Address: 0x14e7070
//
__int64 __fastcall sub_14E7070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  bool v5; // r10
  char v6; // r9
  char v7; // al
  char v8; // dl
  char v10; // cl

  v5 = a2 == 20;
  switch ( a2 )
  {
    case 14LL:
      if ( *(_QWORD *)a1 == 0x615F4554415F5744LL && *(_DWORD *)(a1 + 8) == 1701995620 && *(_WORD *)(a1 + 12) == 29555 )
      {
        v7 = 1;
        v6 = 0;
        a5 = 1;
      }
      else if ( *(_QWORD *)a1 == 0x625F4554415F5744LL
             && *(_DWORD *)(a1 + 8) == 1701605231
             && *(_WORD *)(a1 + 12) == 28257 )
      {
        v7 = 1;
        v6 = 0;
        a5 = 2;
      }
      else
      {
        v7 = 0;
        v6 = 1;
      }
      break;
    case 20LL:
      a5 = 3;
      if ( *(_QWORD *)a1 ^ 0x635F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x665F78656C706D6FLL
        || (v6 = 0, v7 = 1, *(_DWORD *)(a1 + 16) != 1952542572) )
      {
LABEL_25:
        v6 = 1;
        v7 = 0;
      }
      break;
    case 12LL:
      if ( *(_QWORD *)a1 == 0x665F4554415F5744LL && *(_DWORD *)(a1 + 8) == 1952542572 )
      {
        v6 = 0;
        a5 = 4;
        v7 = 1;
      }
      else
      {
        v6 = 1;
        v7 = 0;
      }
LABEL_27:
      if ( a2 == 15 && v6 )
      {
        if ( *(_QWORD *)a1 == 0x755F4554415F5744LL
          && *(_DWORD *)(a1 + 8) == 1734964078
          && *(_WORD *)(a1 + 12) == 25966
          && *(_BYTE *)(a1 + 14) == 100 )
        {
          a5 = 7;
          goto LABEL_10;
        }
        goto LABEL_30;
      }
      goto LABEL_29;
    case 13LL:
      if ( *(_QWORD *)a1 != 0x735F4554415F5744LL || *(_DWORD *)(a1 + 8) != 1701734249 || *(_BYTE *)(a1 + 12) != 100 )
      {
        v6 = 1;
        v7 = 0;
        goto LABEL_30;
      }
      v7 = 1;
      v6 = 0;
      a5 = 5;
      goto LABEL_29;
    default:
      goto LABEL_25;
  }
  if ( ((unsigned __int8)v6 & (a2 == 18)) == 0 )
    goto LABEL_27;
  if ( !(*(_QWORD *)a1 ^ 0x735F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x68635F64656E6769LL)
    && *(_WORD *)(a1 + 16) == 29281 )
  {
    v7 = v6 & (a2 == 18);
    v6 = 0;
    a5 = 6;
LABEL_30:
    if ( ((unsigned __int8)v6 & (a2 == 22)) != 0 )
    {
      if ( *(_QWORD *)a1 ^ 0x695F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7972616E6967616DLL
        || *(_DWORD *)(a1 + 16) != 1869375071
        || *(_WORD *)(a1 + 20) != 29793 )
      {
        goto LABEL_33;
      }
      a5 = 9;
      goto LABEL_10;
    }
    goto LABEL_31;
  }
LABEL_29:
  if ( ((unsigned __int8)v6 & v5) == 0 )
    goto LABEL_30;
  if ( !(*(_QWORD *)a1 ^ 0x755F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F64656E6769736ELL)
    && *(_DWORD *)(a1 + 16) == 1918986339 )
  {
    a5 = 8;
LABEL_10:
    v7 = 0;
    goto LABEL_11;
  }
LABEL_31:
  if ( v7 )
    goto LABEL_10;
  v6 = 1;
  if ( a2 == 21 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x705F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65645F64656B6361LL)
      && *(_DWORD *)(a1 + 16) == 1634560355
      && *(_BYTE *)(a1 + 20) == 108 )
    {
      a5 = 10;
    }
    else
    {
      if ( *(_QWORD *)a1 ^ 0x6E5F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x735F636972656D75LL
        || *(_DWORD *)(a1 + 16) != 1852404340
        || *(_BYTE *)(a1 + 20) != 103 )
      {
        v8 = 0;
        v7 = 1;
        goto LABEL_12;
      }
      a5 = 11;
    }
LABEL_11:
    v8 = 1;
    goto LABEL_12;
  }
LABEL_33:
  v8 = v6 & (a2 == 13);
  if ( v8 )
  {
    if ( *(_QWORD *)a1 == 0x655F4554415F5744LL && *(_DWORD *)(a1 + 8) == 1702127972 && *(_BYTE *)(a1 + 12) == 100 )
    {
      a5 = 12;
      v7 = 0;
    }
    else
    {
      v10 = v7;
      v7 = v6 & (a2 == 13);
      v8 = v10;
    }
LABEL_14:
    if ( ((unsigned __int8)v7 & v5) == 0 )
      goto LABEL_15;
LABEL_43:
    if ( !(*(_QWORD *)a1 ^ 0x645F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x665F6C616D696365LL) )
    {
      a5 = 15;
      if ( *(_DWORD *)(a1 + 16) == 1952542572 )
        return a5;
    }
    goto LABEL_17;
  }
  if ( ((unsigned __int8)v6 & (a2 == 19)) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x735F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x69665F64656E6769LL)
      && *(_WORD *)(a1 + 16) == 25976
      && *(_BYTE *)(a1 + 18) == 100 )
    {
      return 13;
    }
    v8 = v7;
    v7 = v6 & (a2 == 19);
    if ( a2 == 20 )
      goto LABEL_43;
    goto LABEL_15;
  }
  v8 = v7;
  v7 = v6;
LABEL_12:
  if ( a2 != 21 || !v7 )
    goto LABEL_14;
  if ( !(*(_QWORD *)a1 ^ 0x755F4554415F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F64656E6769736ELL)
    && *(_DWORD *)(a1 + 16) == 1702390118
    && *(_BYTE *)(a1 + 20) == 100 )
  {
    return 14;
  }
LABEL_15:
  if ( v8 )
    return a5;
  v7 = 1;
  if ( a2 == 10 )
  {
    if ( *(_QWORD *)a1 != 0x555F4554415F5744LL || (a5 = 16, *(_WORD *)(a1 + 8) != 18004) )
    {
      if ( *(_QWORD *)a1 != 0x555F4554415F5744LL )
        return 0;
      a5 = 17;
      if ( *(_WORD *)(a1 + 8) != 21315 )
        return 0;
    }
    return a5;
  }
LABEL_17:
  if ( a2 != 12 )
    return 0;
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)a1 != 0x415F4554415F5744LL )
    return 0;
  a5 = 18;
  if ( *(_DWORD *)(a1 + 8) != 1229538131 )
    return 0;
  return a5;
}
