// Function: sub_14C8690
// Address: 0x14c8690
//
__int64 __fastcall sub_14C8690(__int64 a1, __int64 a2)
{
  bool v2; // cl
  bool v3; // r11
  bool v4; // r8
  bool v5; // r10
  char v6; // al
  char v7; // bl
  bool v8; // r9
  unsigned int v9; // r12d
  char v10; // dl
  char v11; // dl
  bool v12; // dl
  bool v13; // r14
  char v14; // r9
  unsigned __int8 v15; // al
  bool v16; // bl
  bool v17; // r13
  char v18; // r13
  char v19; // al
  char v20; // r14
  char v21; // bl
  char v22; // r8
  char v23; // bl
  char v24; // al
  char v25; // si
  char v26; // dl
  char v27; // r10
  char v28; // al
  bool v30; // [rsp+1h] [rbp-2Bh]
  char v31; // [rsp+2h] [rbp-2Ah]
  bool v32; // [rsp+3h] [rbp-29h]

  v2 = a2 == 10;
  v3 = a2 == 9;
  v30 = a2 == 12;
  if ( a2 == 5 )
  {
    if ( *(_DWORD *)a1 == 1734962273 && *(_BYTE *)(a1 + 4) == 110 )
    {
      v31 = 1;
      v6 = 0;
      v7 = 1;
    }
    else
    {
      v31 = 0;
      v6 = 1;
      v7 = 0;
    }
    v4 = 0;
    v9 = 1;
    goto LABEL_10;
  }
  if ( a2 == 9 )
  {
    if ( *(_QWORD *)a1 == 0x7A6973636F6C6C61LL && *(_BYTE *)(a1 + 8) == 101 )
    {
      v31 = 1;
      v9 = 2;
      v6 = 0;
      v4 = 0;
      v7 = 1;
    }
    else
    {
      v31 = 0;
      v4 = 0;
      v9 = 2;
      v7 = 0;
      v6 = 1;
    }
LABEL_10:
    v10 = v6 & v4;
    goto LABEL_11;
  }
  if ( a2 != 12 )
  {
    v4 = a2 == 7;
    if ( a2 == 10 )
    {
      if ( *(_QWORD *)a1 == 0x6E6F6D656D677261LL && *(_WORD *)(a1 + 8) == 31084 )
      {
        v31 = 1;
        v6 = 0;
        v7 = 1;
        v8 = 0;
        v5 = 0;
        v9 = 4;
      }
      else
      {
        v31 = 0;
        v5 = 0;
        v6 = 1;
        v7 = 0;
        v8 = 0;
        v9 = 1;
      }
      goto LABEL_13;
    }
    v31 = 0;
    v9 = 1;
    v6 = 1;
    v7 = 0;
    goto LABEL_10;
  }
  v10 = 0;
  v4 = 0;
  if ( *(_QWORD *)a1 == 0x6E69737961776C61LL && *(_DWORD *)(a1 + 8) == 1701734764 )
  {
    v31 = 1;
    v9 = 3;
    v6 = 0;
    v8 = 0;
    v7 = 1;
    goto LABEL_12;
  }
  v31 = 0;
  v7 = 0;
  v6 = 1;
  v9 = 1;
LABEL_11:
  v8 = a2 == 5;
  if ( v10 )
  {
    v5 = a2 == 4;
    if ( *(_DWORD *)a1 != 1818850658 || *(_WORD *)(a1 + 4) != 26996 || *(_BYTE *)(a1 + 6) != 110 )
    {
      v11 = v6 & v5;
LABEL_14:
      if ( v11 )
      {
        v32 = a2 == 15;
        if ( *(_DWORD *)a1 == 1684828003 )
        {
          v31 = 1;
          v7 = v11;
          v6 = 0;
          v9 = 7;
          goto LABEL_17;
        }
        goto LABEL_16;
      }
      goto LABEL_15;
    }
    v31 = 1;
    v7 = v10;
    v6 = 0;
    v9 = 5;
    goto LABEL_15;
  }
LABEL_12:
  v5 = a2 == 4;
  if ( (v8 & (unsigned __int8)v6) == 0 )
  {
LABEL_13:
    v11 = v5 & v6;
    goto LABEL_14;
  }
  if ( *(_DWORD *)a1 == 1635154274 && *(_BYTE *)(a1 + 4) == 108 )
  {
    v31 = 1;
    v7 = v8 & v6;
    v9 = 6;
    v32 = a2 == 15;
    v6 = 0;
    goto LABEL_16;
  }
LABEL_15:
  v32 = a2 == 15;
  if ( ((unsigned __int8)v6 & v2) != 0 )
  {
    if ( *(_QWORD *)a1 == 0x65677265766E6F63LL && *(_WORD *)(a1 + 8) == 29806 )
    {
      v31 = 1;
      v7 = v6 & v2;
      v9 = 8;
      v12 = a2 == 8;
      v6 = 0;
      goto LABEL_18;
    }
    goto LABEL_17;
  }
LABEL_16:
  if ( ((unsigned __int8)v6 & v32) != 0 )
  {
    v12 = a2 == 8;
    if ( *(_QWORD *)a1 != 0x6572656665726564LL
      || *(_DWORD *)(a1 + 8) != 1634034542
      || *(_WORD *)(a1 + 12) != 27746
      || *(_BYTE *)(a1 + 14) != 101 )
    {
      goto LABEL_18;
    }
    v31 = 1;
    v7 = v6 & v32;
    v6 = 0;
    v9 = 9;
LABEL_19:
    if ( (v8 & (unsigned __int8)v6) != 0 )
    {
      if ( *(_DWORD *)a1 == 1701998185 && *(_BYTE *)(a1 + 4) == 103 )
      {
        v31 = 1;
        v7 = v8 & v6;
        v9 = 12;
        v6 = 0;
        goto LABEL_22;
      }
      goto LABEL_21;
    }
    goto LABEL_20;
  }
LABEL_17:
  v12 = a2 == 8;
  if ( ((unsigned __int8)v6 & (a2 == 23)) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x6572656665726564LL | *(_QWORD *)(a1 + 8) ^ 0x5F656C626165636ELL)
      && *(_DWORD *)(a1 + 16) == 1851748975
      && *(_WORD *)(a1 + 20) == 27765
      && *(_BYTE *)(a1 + 22) == 108 )
    {
      v31 = 1;
      v7 = v6 & (a2 == 23);
      v9 = 10;
      v6 = 0;
      goto LABEL_20;
    }
    goto LABEL_19;
  }
LABEL_18:
  if ( ((unsigned __int8)v6 & v12) == 0 )
    goto LABEL_19;
  if ( *(_QWORD *)a1 == 0x61636F6C6C616E69LL )
  {
    v31 = 1;
    v7 = v6 & v12;
    v9 = 11;
    v6 = 0;
    goto LABEL_21;
  }
LABEL_20:
  if ( ((unsigned __int8)v6 & (a2 == 19)) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x7373656363616E69LL | *(_QWORD *)(a1 + 8) ^ 0x6F6D656D656C6269LL)
      && *(_WORD *)(a1 + 16) == 27758
      && *(_BYTE *)(a1 + 18) == 121 )
    {
      v31 = 1;
      v7 = v6 & (a2 == 19);
      v6 = 0;
      v9 = 13;
      goto LABEL_23;
    }
    goto LABEL_22;
  }
LABEL_21:
  if ( ((unsigned __int8)v6 & (a2 == 29)) != 0 )
  {
    if ( *(_QWORD *)a1 ^ 0x7373656363616E69LL | *(_QWORD *)(a1 + 8) ^ 0x5F6D656D656C6269LL
      || *(_QWORD *)(a1 + 16) != 0x656D6772615F726FLL
      || *(_DWORD *)(a1 + 24) != 1819176813
      || *(_BYTE *)(a1 + 28) != 121 )
    {
      goto LABEL_23;
    }
    v31 = 1;
    v7 = v6 & (a2 == 29);
    v9 = 14;
    v6 = 0;
LABEL_24:
    if ( (v4 & (unsigned __int8)v6) != 0 )
    {
      if ( *(_DWORD *)a1 == 1936615789 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101 )
      {
        v31 = 1;
        v7 = v4 & v6;
        v6 = 0;
        v9 = 17;
        goto LABEL_27;
      }
      goto LABEL_26;
    }
    goto LABEL_25;
  }
LABEL_22:
  if ( ((unsigned __int8)v6 & v2) != 0 )
  {
    if ( *(_QWORD *)a1 == 0x6968656E696C6E69LL && *(_WORD *)(a1 + 8) == 29806 )
    {
      v31 = 1;
      v7 = v6 & v2;
      v9 = 15;
      v6 = 0;
      goto LABEL_25;
    }
    goto LABEL_24;
  }
LABEL_23:
  if ( ((unsigned __int8)v6 & v3) == 0 )
    goto LABEL_24;
  if ( *(_QWORD *)a1 == 0x6C626174706D756ALL && *(_BYTE *)(a1 + 8) == 101 )
  {
    v31 = 1;
    v7 = v6 & v3;
    v6 = 0;
    v9 = 16;
    goto LABEL_26;
  }
LABEL_25:
  if ( ((unsigned __int8)v6 & v8) != 0 )
  {
    if ( *(_DWORD *)a1 == 1701536110 && *(_BYTE *)(a1 + 4) == 100 )
    {
      v31 = 1;
      v9 = 18;
      goto LABEL_47;
    }
    goto LABEL_27;
  }
LABEL_26:
  if ( (v5 & (unsigned __int8)v6) != 0 )
  {
    if ( *(_DWORD *)a1 == 1953719662 )
    {
      v31 = 1;
      v9 = 19;
      goto LABEL_47;
    }
    goto LABEL_28;
  }
LABEL_27:
  if ( (v4 & (unsigned __int8)v6) != 0 )
  {
    if ( *(_DWORD *)a1 != 1818324846 || *(_WORD *)(a1 + 4) != 24937 || *(_BYTE *)(a1 + 6) != 115 )
      goto LABEL_30;
    v31 = 1;
    v9 = 20;
LABEL_47:
    v14 = 1;
    v7 = a2 == 11;
    v6 = 0;
    goto LABEL_48;
  }
LABEL_28:
  if ( v7 )
    goto LABEL_47;
  v6 = 1;
  if ( a2 == 9 )
  {
    if ( *(_QWORD *)a1 == 0x69746C6975626F6ELL && *(_BYTE *)(a1 + 8) == 110 )
    {
      v9 = 21;
    }
    else
    {
      if ( *(_QWORD *)a1 != 0x7275747061636F6ELL || *(_BYTE *)(a1 + 8) != 101 )
      {
        v14 = 0;
        v6 = 1;
        goto LABEL_48;
      }
      v9 = 22;
    }
    v31 = 1;
    v6 = 0;
    v14 = 1;
LABEL_48:
    if ( ((unsigned __int8)v6 & v32) == 0 )
      goto LABEL_33;
LABEL_49:
    if ( *(_QWORD *)a1 == 0x63696C706D696F6ELL
      && *(_DWORD *)(a1 + 8) == 1818653801
      && *(_WORD *)(a1 + 12) == 24943
      && *(_BYTE *)(a1 + 14) == 116 )
    {
      v31 = 1;
      v9 = 25;
      goto LABEL_54;
    }
LABEL_34:
    if ( !v14 )
    {
      if ( a2 != 9 )
        goto LABEL_36;
      if ( *(_QWORD *)a1 == 0x7372756365726F6ELL && *(_BYTE *)(a1 + 8) == 101 )
      {
        v9 = 27;
      }
      else
      {
        if ( *(_QWORD *)a1 != 0x6E6F7A6465726F6ELL || *(_BYTE *)(a1 + 8) != 101 )
        {
LABEL_37:
          v15 = v14 ^ 1;
          if ( v14 != 1 && v7 )
          {
            if ( *(_QWORD *)a1 == 0x62797A616C6E6F6ELL && *(_WORD *)(a1 + 8) == 28265 && *(_BYTE *)(a1 + 10) == 100 )
            {
              v9 = 31;
              goto LABEL_107;
            }
            v16 = a2 == 13;
            v14 = 0;
            if ( a2 == 13 )
            {
LABEL_41:
              if ( *(_QWORD *)a1 == 0x7566726F6674706FLL
                && *(_DWORD *)(a1 + 8) == 1852406394
                && *(_BYTE *)(a1 + 12) == 103 )
              {
                v9 = 33;
                goto LABEL_107;
              }
              v17 = v4;
              v4 = v14;
              if ( v12 )
                goto LABEL_43;
              goto LABEL_61;
            }
            goto LABEL_58;
          }
          goto LABEL_55;
        }
        v9 = 28;
      }
      v31 = 1;
      v14 = 1;
      v15 = 0;
LABEL_55:
      v18 = v4 & v15;
      goto LABEL_56;
    }
LABEL_54:
    v14 = 1;
    v15 = 0;
    goto LABEL_55;
  }
LABEL_30:
  v13 = a2 == 11;
  v14 = v6 & v2;
  if ( ((unsigned __int8)v6 & v2) == 0 )
  {
    if ( ((unsigned __int8)v6 & v13) != 0 )
    {
      if ( *(_QWORD *)a1 == 0x63696C7075646F6ELL && *(_WORD *)(a1 + 8) == 29793 && *(_BYTE *)(a1 + 10) == 101 )
      {
        v31 = 1;
        v9 = 24;
        goto LABEL_54;
      }
      v14 = v7;
      v7 = v6 & v13;
      if ( !v12 )
        goto LABEL_34;
      goto LABEL_113;
    }
    v14 = v7;
    v7 = a2 == 11;
    goto LABEL_48;
  }
  if ( *(_QWORD *)a1 != 0x6568635F66636F6ELL || *(_WORD *)(a1 + 8) != 27491 )
  {
    v6 &= v2;
    v14 = v7;
    v7 = a2 == 11;
    if ( !v32 )
      goto LABEL_33;
    goto LABEL_49;
  }
  v31 = 1;
  v7 = a2 == 11;
  v9 = 23;
  v6 = 0;
LABEL_33:
  if ( (v12 & (unsigned __int8)v6) == 0 )
    goto LABEL_34;
LABEL_113:
  if ( *(_QWORD *)a1 == 0x656E696C6E696F6ELL )
  {
    v31 = 1;
    v9 = 26;
    goto LABEL_54;
  }
LABEL_36:
  v14 = v31;
  if ( !v12 )
    goto LABEL_37;
  if ( *(_QWORD *)a1 == 0x6E72757465726F6ELL )
  {
    v9 = 29;
LABEL_133:
    v31 = 1;
    v14 = v12;
    v16 = a2 == 13;
    v15 = 0;
LABEL_57:
    if ( (v16 & v15) != 0 )
      goto LABEL_41;
    goto LABEL_58;
  }
  if ( *(_QWORD *)a1 == 0x646E69776E756F6ELL )
  {
    v9 = 30;
    goto LABEL_133;
  }
  v14 = v31;
  v15 = v31 ^ 1;
  v18 = v4 & (v31 ^ 1);
LABEL_56:
  v16 = a2 == 13;
  if ( !v18 )
    goto LABEL_57;
  if ( *(_DWORD *)a1 == 1852731246 && *(_WORD *)(a1 + 4) == 27765 && *(_BYTE *)(a1 + 6) == 108 )
  {
    v9 = 32;
    goto LABEL_107;
  }
LABEL_58:
  if ( v14 )
  {
LABEL_107:
    v17 = v4;
    goto LABEL_108;
  }
  v17 = 0;
  if ( v4 )
  {
    if ( *(_DWORD *)a1 == 1937010799 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101 )
    {
      v9 = 34;
      goto LABEL_107;
    }
    if ( *(_DWORD *)a1 == 1853124719 && *(_WORD *)(a1 + 4) == 28271 && *(_BYTE *)(a1 + 6) == 101 )
    {
      v9 = 35;
      goto LABEL_107;
    }
    v17 = v4;
    v4 = v12;
    goto LABEL_70;
  }
  if ( v12 )
  {
LABEL_43:
    switch ( *(_QWORD *)a1 )
    {
      case 0x656E6F6E64616572LL:
        v9 = 36;
LABEL_108:
        v14 = 1;
        v19 = 0;
LABEL_72:
        if ( ((unsigned __int8)v19 & v3) == 0 )
          goto LABEL_73;
LABEL_147:
        if ( *(_QWORD *)a1 == 0x6361747365666173LL && *(_BYTE *)(a1 + 8) == 107 )
        {
          v9 = 41;
          goto LABEL_87;
        }
        goto LABEL_74;
      case 0x796C6E6F64616572LL:
        v9 = 37;
        goto LABEL_108;
      case 0x64656E7275746572LL:
        v9 = 38;
        goto LABEL_108;
    }
    v14 = v31;
    v4 = v12;
    goto LABEL_70;
  }
LABEL_61:
  v14 = v31;
  if ( v4 )
  {
    v19 = v31 ^ 1;
    v20 = v17 & (v31 ^ 1);
    if ( !v20 )
      goto LABEL_72;
LABEL_63:
    if ( *(_DWORD *)a1 == 1852270963 && *(_WORD *)(a1 + 4) == 30821 && *(_BYTE *)(a1 + 6) == 116 )
    {
      v14 = v20;
      v19 = 0;
      v9 = 40;
      goto LABEL_74;
    }
    goto LABEL_73;
  }
LABEL_70:
  v12 = v4;
  v19 = v14 ^ 1;
  v20 = v17 & (v14 ^ 1);
  v21 = (v14 ^ 1) & v16;
  if ( !v21 )
  {
    if ( !v20 )
      goto LABEL_72;
    goto LABEL_63;
  }
  if ( *(_QWORD *)a1 == 0x5F736E7275746572LL && *(_DWORD *)(a1 + 8) == 1667856244 && *(_BYTE *)(a1 + 12) == 101 )
  {
    v14 = v21;
    v19 = 0;
    v9 = 39;
  }
  else
  {
    v19 = v21;
    v12 = v4;
    v14 = 0;
    if ( a2 == 9 )
      goto LABEL_147;
  }
LABEL_73:
  if ( ((unsigned __int8)v19 & (a2 == 16)) == 0 )
  {
LABEL_74:
    if ( ((unsigned __int8)v19 & (a2 == 18)) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x657264646177685FLL)
        && *(_WORD *)(a1 + 16) == 29555 )
      {
        v9 = 43;
        goto LABEL_87;
      }
      if ( v14 )
      {
        v22 = v19 & v2;
        goto LABEL_78;
      }
      goto LABEL_77;
    }
    goto LABEL_75;
  }
  if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x737365726464615FLL) )
  {
    v9 = 42;
    goto LABEL_87;
  }
LABEL_75:
  if ( v14 )
  {
LABEL_87:
    v14 = 0;
    v22 = 1;
    goto LABEL_88;
  }
  v19 = 1;
  if ( !v32 )
    goto LABEL_77;
  if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
    && *(_DWORD *)(a1 + 8) == 1835363679
    && *(_WORD *)(a1 + 12) == 29295
    && *(_BYTE *)(a1 + 14) == 121 )
  {
    v9 = 44;
    goto LABEL_87;
  }
  if ( *(_QWORD *)a1 == 0x657A6974696E6173LL
    && *(_DWORD *)(a1 + 8) == 1919448159
    && *(_WORD *)(a1 + 12) == 24933
    && *(_BYTE *)(a1 + 14) == 100 )
  {
    v9 = 45;
    goto LABEL_87;
  }
  if ( *(_QWORD *)a1 == 0x6163776F64616873LL
    && *(_DWORD *)(a1 + 8) == 1953721452
    && *(_WORD *)(a1 + 12) == 25441
    && *(_BYTE *)(a1 + 14) == 107 )
  {
    v9 = 46;
    goto LABEL_87;
  }
  v19 = v32;
LABEL_77:
  v22 = v19 & v2;
  v14 = v19 & v30;
  if ( ((unsigned __int8)v19 & v30) != 0 )
  {
    if ( *(_QWORD *)a1 == 0x74616C7563657073LL && *(_DWORD *)(a1 + 8) == 1701601889 )
    {
      v22 = v19 & v30;
      v9 = 47;
      v14 = 0;
    }
    else
    {
      v22 = 0;
    }
    goto LABEL_88;
  }
LABEL_78:
  if ( v22 )
  {
    if ( *(_QWORD *)a1 == 0x6174736E67696C61LL && *(_WORD *)(a1 + 8) == 27491 )
    {
      v9 = 48;
      v14 = 0;
    }
    else
    {
      v22 = v14;
      v14 = v19;
      v25 = v19 & (a2 == 6);
      if ( v25 )
        goto LABEL_203;
    }
LABEL_89:
    v24 = v14 & v3;
    if ( ((unsigned __int8)v14 & v3) == 0 )
      goto LABEL_90;
LABEL_82:
    if ( *(_QWORD *)a1 == 0x6E6F727473707373LL && *(_BYTE *)(a1 + 8) == 103 )
    {
      v22 = v24;
      v9 = 51;
      v14 = 0;
      goto LABEL_92;
    }
LABEL_91:
    v27 = v14 & v5;
    if ( v27 )
    {
      if ( *(_DWORD *)a1 == 1952805491 )
      {
        v22 = v27;
        v14 = 0;
        v9 = 53;
        goto LABEL_95;
      }
      v28 = v3 & v14;
      goto LABEL_94;
    }
    goto LABEL_92;
  }
  v23 = v19 & (a2 == 3);
  if ( v23 )
  {
    if ( *(_WORD *)a1 == 29555 && *(_BYTE *)(a1 + 2) == 112 )
    {
      v14 = 0;
      v22 = v19 & (a2 == 3);
      v9 = 49;
      goto LABEL_90;
    }
    v24 = a2 == 9;
    v22 = v14;
    v14 = v23;
    if ( a2 != 9 )
      goto LABEL_90;
    goto LABEL_82;
  }
  v22 = v14;
  v14 = v19;
LABEL_88:
  v25 = v14 & (a2 == 6);
  if ( !v25 )
    goto LABEL_89;
LABEL_203:
  if ( *(_DWORD *)a1 == 1919972211 && *(_WORD *)(a1 + 4) == 29029 )
  {
    v22 = v25;
    v14 = 0;
    v9 = 50;
    goto LABEL_91;
  }
LABEL_90:
  v26 = v14 & v12;
  if ( !v26 )
    goto LABEL_91;
  if ( *(_QWORD *)a1 != 0x7066746369727473LL )
  {
LABEL_92:
    if ( ((unsigned __int8)v14 & v2) != 0 )
    {
      if ( *(_QWORD *)a1 == 0x7272657466697773LL && *(_WORD *)(a1 + 8) == 29295 )
        return 54;
      goto LABEL_95;
    }
    goto LABEL_93;
  }
  v22 = v26;
  v14 = 0;
  v9 = 52;
LABEL_93:
  v28 = v14 & v3;
LABEL_94:
  if ( v28 )
  {
    if ( *(_QWORD *)a1 == 0x6C65737466697773LL && *(_BYTE *)(a1 + 8) == 102 )
      return 55;
    goto LABEL_118;
  }
LABEL_95:
  if ( ((unsigned __int8)v14 & v17) == 0 )
  {
    if ( !v3 || !v14 )
      goto LABEL_98;
LABEL_118:
    if ( *(_QWORD *)a1 != 0x6C6E6F6574697277LL || *(_BYTE *)(a1 + 8) != 121 )
    {
LABEL_98:
      if ( !v22 )
        return 0;
      return v9;
    }
    return 57;
  }
  if ( *(_DWORD *)a1 == 1635022709 && *(_WORD *)(a1 + 4) == 27746 )
  {
    v9 = 56;
    if ( *(_BYTE *)(a1 + 6) == 101 )
      return v9;
  }
  if ( *(_DWORD *)a1 == 1869768058 && *(_WORD *)(a1 + 4) == 30821 )
  {
    v9 = 58;
    if ( *(_BYTE *)(a1 + 6) == 116 )
      return v9;
  }
  return 0;
}
