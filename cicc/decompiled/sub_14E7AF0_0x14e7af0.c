// Function: sub_14E7AF0
// Address: 0x14e7af0
//
__int64 __fastcall sub_14E7AF0(__int64 a1, __int64 a2)
{
  bool v2; // r10
  unsigned int v3; // r12d
  char v4; // cl
  char v5; // al
  char v6; // r14
  char v7; // r9
  char v8; // dl
  bool v9; // r11
  char v10; // r8
  char v11; // r8
  bool v12; // r13
  char v13; // r15
  bool v14; // r8
  char v15; // bl
  char v16; // al
  unsigned __int8 v17; // al
  char v18; // r9
  char v20; // r14
  char v21; // r10

  v2 = a2 == 13;
  switch ( a2 )
  {
    case 11LL:
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 14403 )
      {
        v4 = 1;
        v3 = 1;
        if ( *(_BYTE *)(a1 + 10) == 57 )
          goto LABEL_174;
      }
      v4 = 0;
      v3 = 1;
      goto LABEL_11;
    case 9LL:
      v3 = 2;
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_BYTE *)(a1 + 8) != 67 )
        goto LABEL_8;
      goto LABEL_51;
    case 13LL:
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_DWORD *)(a1 + 8) != 945906753 || *(_BYTE *)(a1 + 12) != 51 )
      {
LABEL_7:
        v3 = 1;
LABEL_8:
        v4 = 0;
        goto LABEL_11;
      }
      v3 = 3;
LABEL_51:
      v4 = 1;
LABEL_174:
      v7 = a2 == 15;
LABEL_175:
      v6 = 1;
      v8 = a2 == 12;
      v5 = 0;
      goto LABEL_59;
    case 19LL:
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
        && *(_WORD *)(a1 + 16) == 30060 )
      {
        v3 = 4;
        v4 = 1;
        if ( *(_BYTE *)(a1 + 18) == 115 )
          goto LABEL_174;
      }
      goto LABEL_7;
  }
  v3 = 1;
  v4 = 0;
  if ( a2 == 15 )
  {
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
      && *(_DWORD *)(a1 + 8) == 1868721987
      && *(_WORD *)(a1 + 12) == 14188
      && *(_BYTE *)(a1 + 14) == 52 )
    {
      v3 = 5;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL
        || *(_DWORD *)(a1 + 8) != 1868721987
        || *(_WORD *)(a1 + 12) != 14444
        || *(_BYTE *)(a1 + 14) != 53 )
      {
        v7 = 1;
        v5 = 1;
LABEL_105:
        v8 = a2 == 12;
        v6 = v7 & 1;
        if ( (v7 & 1) != 0 )
        {
          v9 = a2 == 11;
          if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
            && *(_DWORD *)(a1 + 8) == 1969516365
            && *(_WORD *)(a1 + 12) == 24940
            && *(_BYTE *)(a1 + 14) == 50 )
          {
            v7 &= 1u;
            v5 = 0;
            v3 = 10;
            v4 = 1;
          }
          else
          {
            v10 = a2 == 11;
            v5 = v7 & 1;
            v7 &= 1u;
            v6 = 0;
            if ( a2 == 11 )
              goto LABEL_108;
          }
LABEL_15:
          v11 = v2 & v5;
          v12 = a2 == 17;
          if ( (v2 & (unsigned __int8)v5) == 0 )
            goto LABEL_16;
          goto LABEL_62;
        }
        goto LABEL_59;
      }
      v3 = 6;
    }
    v7 = 1;
    v4 = 1;
    goto LABEL_175;
  }
  if ( a2 != 17 )
  {
LABEL_11:
    v5 = 1;
    v6 = a2 == 16;
    v7 = a2 == 15;
    if ( a2 == 16 )
    {
      if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x33386C6163736150LL )
      {
        v5 = a2 == 16;
        v9 = 0;
        v8 = 0;
        v6 = 0;
      }
      else
      {
        v8 = 0;
        v9 = 0;
        v3 = 9;
        v4 = 1;
        v5 = 0;
      }
LABEL_14:
      v10 = v5 & v9;
      if ( ((unsigned __int8)v5 & v9) == 0 )
        goto LABEL_15;
LABEL_108:
      v12 = a2 == 17;
      if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 14659 && *(_BYTE *)(a1 + 10) == 57 )
      {
        v6 = v10;
        v5 = 0;
        v3 = 12;
        v4 = 1;
        goto LABEL_17;
      }
      goto LABEL_16;
    }
    goto LABEL_105;
  }
  if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x376E617274726F46LL) && *(_BYTE *)(a1 + 16) == 55 )
  {
    v3 = 7;
  }
  else
  {
    if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x396E617274726F46LL || *(_BYTE *)(a1 + 16) != 48 )
    {
      v6 = 0;
      v5 = 1;
      v8 = 0;
      v7 = 0;
      goto LABEL_59;
    }
    v3 = 8;
  }
  v6 = 1;
  v8 = 0;
  v7 = 0;
  v5 = 0;
  v4 = 1;
LABEL_59:
  v9 = a2 == 11;
  if ( ((unsigned __int8)v5 & (unsigned __int8)v8) == 0 )
    goto LABEL_14;
  if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1635148106 )
  {
    v6 = v5 & v8;
    v3 = 11;
    v4 = 1;
    v12 = a2 == 17;
    v5 = 0;
LABEL_16:
    if ( ((unsigned __int8)v5 & v12) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x396E617274726F46LL)
        && *(_BYTE *)(a1 + 16) == 53 )
      {
        v6 = v5 & v12;
        v3 = 14;
        v4 = 1;
        v14 = a2 == 22;
        v5 = 0;
      }
      else
      {
        v13 = v8 & v5;
        v14 = a2 == 22;
        if ( ((unsigned __int8)v8 & (unsigned __int8)v5) != 0 )
          goto LABEL_166;
      }
LABEL_19:
      if ( (v14 & (unsigned __int8)v5) == 0 )
        goto LABEL_20;
LABEL_159:
      if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x756C705F436A624FLL)
        && *(_DWORD *)(a1 + 16) == 1819303795
        && *(_WORD *)(a1 + 20) == 29557 )
      {
        v3 = 17;
        v4 = 1;
        goto LABEL_72;
      }
      goto LABEL_21;
    }
    goto LABEL_17;
  }
  v11 = v5 & v2;
  v12 = a2 == 17;
  if ( ((unsigned __int8)v5 & v2) == 0 )
    goto LABEL_16;
LABEL_62:
  if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 962683969 && *(_BYTE *)(a1 + 12) == 53 )
  {
    v6 = v11;
    v5 = 0;
    v3 = 13;
    v4 = 1;
LABEL_18:
    v13 = v5 & v8;
    v14 = a2 == 22;
    if ( ((unsigned __int8)v5 & (unsigned __int8)v8) == 0 )
      goto LABEL_19;
LABEL_166:
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1131045455 )
    {
      v6 = v13;
      v3 = 16;
      v5 = 0;
      v4 = 1;
      goto LABEL_21;
    }
    goto LABEL_20;
  }
LABEL_17:
  if ( (v9 & (unsigned __int8)v5) == 0 )
    goto LABEL_18;
  v14 = a2 == 22;
  if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 19536 && *(_BYTE *)(a1 + 10) == 73 )
  {
    v6 = v9 & v5;
    v3 = 15;
    v4 = 1;
    v5 = 0;
  }
  else if ( ((unsigned __int8)v5 & v14) != 0 )
  {
    goto LABEL_159;
  }
LABEL_20:
  if ( (v9 & (unsigned __int8)v5) != 0 )
  {
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 20565 && *(_BYTE *)(a1 + 10) == 67 )
    {
      v3 = 18;
      v4 = 1;
      goto LABEL_72;
    }
    goto LABEL_22;
  }
LABEL_21:
  if ( ((unsigned __int8)v5 & (a2 == 9)) != 0 )
  {
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_BYTE *)(a1 + 8) == 68 )
    {
      v3 = 19;
      v4 = 1;
      goto LABEL_72;
    }
    goto LABEL_24;
  }
LABEL_22:
  if ( v6 )
  {
LABEL_72:
    v16 = 0;
    v15 = 1;
LABEL_73:
    v20 = v8 & v16;
    goto LABEL_74;
  }
  v5 = 1;
  if ( a2 == 14 )
  {
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1752463696 && *(_WORD *)(a1 + 12) == 28271 )
    {
      v3 = 20;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL || *(_DWORD *)(a1 + 8) != 1852141647 || *(_WORD *)(a1 + 12) != 19523 )
        goto LABEL_26;
      v3 = 21;
    }
LABEL_71:
    v4 = 1;
    goto LABEL_72;
  }
LABEL_24:
  if ( a2 != 10 || !v5 )
  {
LABEL_26:
    if ( !v7 )
      goto LABEL_27;
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
      && *(_DWORD *)(a1 + 8) == 1969516365
      && *(_WORD *)(a1 + 12) == 24940
      && *(_BYTE *)(a1 + 14) == 51 )
    {
      v3 = 23;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x5F474E414C5F5744LL
        || *(_DWORD *)(a1 + 8) != 1802723656
        || *(_WORD *)(a1 + 12) != 27749
        || *(_BYTE *)(a1 + 14) != 108 )
      {
LABEL_28:
        v15 = v4;
        v16 = v4 ^ 1;
        if ( v4 != 1 && a2 == 13 )
        {
          if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1835090767 && *(_BYTE *)(a1 + 12) == 108 )
          {
            v3 = 27;
            goto LABEL_89;
          }
          if ( !v9 )
          {
LABEL_32:
            if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
              && *(_DWORD *)(a1 + 8) == 1718187859
              && *(_BYTE *)(a1 + 12) == 116 )
            {
              v3 = 30;
              goto LABEL_89;
            }
            if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1768715594 && *(_BYTE *)(a1 + 12) == 97 )
            {
              v3 = 31;
              goto LABEL_89;
            }
            if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL
              && *(_DWORD *)(a1 + 8) == 1634498884
              && *(_BYTE *)(a1 + 12) == 110 )
            {
              v3 = 32;
              goto LABEL_89;
            }
LABEL_35:
            if ( v14 )
              goto LABEL_36;
            goto LABEL_80;
          }
          v15 = 0;
          goto LABEL_87;
        }
        goto LABEL_73;
      }
      v3 = 24;
    }
    goto LABEL_71;
  }
  if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 28487 )
  {
    v3 = 22;
    v4 = 1;
    goto LABEL_72;
  }
LABEL_27:
  if ( !v14 )
    goto LABEL_28;
  if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
    && *(_DWORD *)(a1 + 16) == 1601402220
    && *(_WORD *)(a1 + 20) == 13104 )
  {
    v3 = 25;
    goto LABEL_206;
  }
  if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
    && *(_DWORD *)(a1 + 16) == 1601402220
    && *(_WORD *)(a1 + 20) == 12593 )
  {
    v3 = 26;
LABEL_206:
    v15 = v14;
    v4 = 1;
    goto LABEL_77;
  }
  v15 = v4;
  v16 = v4 ^ 1;
  v20 = v8 & (v4 ^ 1);
LABEL_74:
  if ( v20 )
  {
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1953723730 )
    {
      v3 = 28;
      goto LABEL_89;
    }
  }
  else if ( v16 && v9 )
  {
LABEL_87:
    if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_WORD *)(a1 + 8) == 12611 && *(_BYTE *)(a1 + 10) == 49 )
    {
      v3 = 29;
    }
    else if ( !v15 )
    {
      goto LABEL_35;
    }
    goto LABEL_89;
  }
LABEL_77:
  if ( !v15 )
  {
    if ( a2 != 13 )
    {
      if ( v14 )
      {
LABEL_36:
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F73756C705F43LL)
          && *(_DWORD *)(a1 + 16) == 1601402220
          && *(_WORD *)(a1 + 20) == 13361 )
        {
          v14 = 1;
          v4 = 1;
          v3 = 33;
          v17 = 0;
          goto LABEL_90;
        }
        v12 = 1;
LABEL_39:
        v17 = v4 ^ 1;
        v14 = v12;
        if ( (((unsigned __int8)v4 ^ 1) & (a2 == 20)) != 0 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x63537265646E6552LL)
            && *(_DWORD *)(a1 + 16) == 1953524082 )
          {
            v14 = v12;
            v4 = (v4 ^ 1) & (a2 == 20);
            v3 = 36;
            v17 = 0;
          }
          else
          {
            v17 = (v4 ^ 1) & (a2 == 20);
            v14 = v12;
            v4 = 0;
            if ( v8 )
            {
LABEL_94:
              if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1280527431 )
              {
                v4 = v8;
                v3 = 52;
                v17 = 0;
              }
              else if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1280527432 )
              {
                return 54;
              }
LABEL_45:
              if ( ((a2 == 27) & v17) != 0
                && !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x525F454C474F4F47LL)
                && *(_QWORD *)(a1 + 16) == 0x7263537265646E65LL
                && *(_WORD *)(a1 + 24) == 28777
                && *(_BYTE *)(a1 + 26) == 116 )
              {
                return 36439;
              }
              else if ( !v4 )
              {
                return 0;
              }
              return v3;
            }
          }
          goto LABEL_43;
        }
        goto LABEL_90;
      }
LABEL_80:
      if ( v12 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x306E617274726F46LL)
          && *(_BYTE *)(a1 + 16) == 51 )
        {
          v3 = 34;
        }
        else
        {
          if ( *(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x306E617274726F46LL
            || *(_BYTE *)(a1 + 16) != 56 )
          {
            v14 = 0;
            v17 = v4 ^ 1;
            v21 = (v4 ^ 1) & v2;
            goto LABEL_91;
          }
          v3 = 35;
        }
        v17 = 0;
        v4 = v12;
        v14 = 0;
        v8 = 0;
LABEL_93:
        if ( v8 )
          goto LABEL_94;
        goto LABEL_43;
      }
      goto LABEL_39;
    }
    goto LABEL_32;
  }
LABEL_89:
  v4 = 1;
  v17 = 0;
LABEL_90:
  v21 = v17 & v2;
LABEL_91:
  if ( !v21 )
  {
    v8 &= v17;
    goto LABEL_93;
  }
  if ( *(_QWORD *)a1 == 0x5F474E414C5F5744LL && *(_DWORD *)(a1 + 8) == 1397312578 && *(_BYTE *)(a1 + 12) == 83 )
  {
    v4 = v21;
    v17 = 0;
    v3 = 37;
    goto LABEL_44;
  }
LABEL_43:
  v18 = v17 & v7;
  if ( v18
    && *(_QWORD *)a1 == 0x5F474E414C5F5744LL
    && *(_DWORD *)(a1 + 8) == 1280527431
    && *(_WORD *)(a1 + 12) == 17759
    && *(_BYTE *)(a1 + 14) == 83 )
  {
    v4 = v18;
    v3 = 53;
    v17 = 0;
    goto LABEL_45;
  }
LABEL_44:
  if ( (v17 & v14) == 0 )
    goto LABEL_45;
  if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7373415F7370694DLL)
    && *(_DWORD *)(a1 + 16) == 1818389861 )
  {
    v3 = 32769;
    if ( *(_WORD *)(a1 + 20) == 29285 )
      return v3;
  }
  if ( !(*(_QWORD *)a1 ^ 0x5F474E414C5F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F444E414C524F42LL)
    && *(_DWORD *)(a1 + 16) == 1886152004 )
  {
    v3 = 45056;
    if ( *(_WORD *)(a1 + 20) == 26984 )
      return v3;
  }
  return 0;
}
