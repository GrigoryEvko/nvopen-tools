// Function: sub_E07820
// Address: 0xe07820
//
__int64 __fastcall sub_E07820(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v4; // [rsp+10h] [rbp-30h] BYREF
  __int64 v5; // [rsp+18h] [rbp-28h]
  unsigned int v6; // [rsp+20h] [rbp-20h]
  int v7; // [rsp+24h] [rbp-1Ch]

  v4 = a1;
  v5 = a2;
  v7 = 0;
  v3 = 3;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_addr", 10);
  v3 = 6;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_deref", 11);
  v3 = 8;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_const1u", 13);
  v3 = 9;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_const1s", 13);
  if ( !(_BYTE)v7
    && v5 == 13
    && *(_QWORD *)v4 == 0x6F635F504F5F5744LL
    && *(_DWORD *)(v4 + 8) == 846492526
    && *(_BYTE *)(v4 + 12) == 117 )
  {
    v6 = 10;
    LOBYTE(v7) = 1;
  }
  v3 = 11;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_const2s", 13);
  v3 = 12;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_const4u", 13);
  if ( !(_BYTE)v7 && v5 == 13 )
  {
    if ( *(_QWORD *)v4 == 0x6F635F504F5F5744LL && *(_DWORD *)(v4 + 8) == 880046958 && *(_BYTE *)(v4 + 12) == 115 )
    {
      v6 = 13;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x6F635F504F5F5744LL && *(_DWORD *)(v4 + 8) == 947155822 && *(_BYTE *)(v4 + 12) == 117 )
    {
      v6 = 14;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x6F635F504F5F5744LL && *(_DWORD *)(v4 + 8) == 947155822 && *(_BYTE *)(v4 + 12) == 115 )
    {
      v6 = 15;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 16;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_constu", 12);
  if ( !(_BYTE)v7 )
  {
    switch ( v5 )
    {
      case 12LL:
        if ( *(_QWORD *)v4 == 0x6F635F504F5F5744LL && *(_DWORD *)(v4 + 8) == 1937011566 )
        {
          v6 = 17;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x64785F504F5F5744LL && *(_DWORD *)(v4 + 8) == 1717924453 )
        {
          v6 = 24;
          LOBYTE(v7) = 1;
        }
        break;
      case 9LL:
        if ( *(_QWORD *)v4 == 0x75645F504F5F5744LL && *(_BYTE *)(v4 + 8) == 112 )
        {
          v6 = 18;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x6F725F504F5F5744LL && *(_BYTE *)(v4 + 8) == 116 )
        {
          v6 = 23;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x62615F504F5F5744LL && *(_BYTE *)(v4 + 8) == 115 )
        {
          v6 = 25;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x6E615F504F5F5744LL && *(_BYTE *)(v4 + 8) == 100 )
        {
          v6 = 26;
          LOBYTE(v7) = 1;
        }
        break;
      case 10LL:
        if ( *(_QWORD *)v4 == 0x72645F504F5F5744LL && *(_WORD *)(v4 + 8) == 28783 )
        {
          v6 = 19;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x766F5F504F5F5744LL && *(_WORD *)(v4 + 8) == 29285 )
        {
          v6 = 20;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x69705F504F5F5744LL && *(_WORD *)(v4 + 8) == 27491 )
        {
          v6 = 21;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x77735F504F5F5744LL && *(_WORD *)(v4 + 8) == 28769 )
        {
          v6 = 22;
          LOBYTE(v7) = 1;
        }
        break;
    }
  }
  v3 = 27;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_div", 9);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 11 )
    {
      if ( *(_QWORD *)v4 == 0x696D5F504F5F5744LL && *(_WORD *)(v4 + 8) == 30062 && *(_BYTE *)(v4 + 10) == 115 )
      {
        v6 = 28;
        LOBYTE(v7) = 1;
      }
    }
    else if ( v5 == 9 )
    {
      if ( *(_QWORD *)v4 == 0x6F6D5F504F5F5744LL && *(_BYTE *)(v4 + 8) == 100 )
      {
        v6 = 29;
        LOBYTE(v7) = 1;
      }
      else if ( *(_QWORD *)v4 == 0x756D5F504F5F5744LL && *(_BYTE *)(v4 + 8) == 108 )
      {
        v6 = 30;
        LOBYTE(v7) = 1;
      }
      else if ( *(_QWORD *)v4 == 0x656E5F504F5F5744LL && *(_BYTE *)(v4 + 8) == 103 )
      {
        v6 = 31;
        LOBYTE(v7) = 1;
      }
    }
  }
  v3 = 32;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_not", 9);
  v3 = 33;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_or", 8);
  if ( !(_BYTE)v7 )
  {
    switch ( v5 )
    {
      case 10LL:
        if ( *(_QWORD *)v4 == 0x6C705F504F5F5744LL && *(_WORD *)(v4 + 8) == 29557 )
        {
          v6 = 34;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x68735F504F5F5744LL && *(_WORD *)(v4 + 8) == 24946 )
        {
          v6 = 38;
          LOBYTE(v7) = 1;
        }
        break;
      case 17LL:
        if ( !(*(_QWORD *)v4 ^ 0x6C705F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x736E6F63755F7375LL)
          && *(_BYTE *)(v4 + 16) == 116 )
        {
          v6 = 35;
          LOBYTE(v7) = 1;
        }
        break;
      case 9LL:
        if ( *(_QWORD *)v4 == 0x68735F504F5F5744LL && *(_BYTE *)(v4 + 8) == 108 )
        {
          v6 = 36;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x68735F504F5F5744LL && *(_BYTE *)(v4 + 8) == 114 )
        {
          v6 = 37;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x6F785F504F5F5744LL && *(_BYTE *)(v4 + 8) == 114 )
        {
          v6 = 39;
          LOBYTE(v7) = 1;
        }
        else if ( *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_BYTE *)(v4 + 8) == 97 )
        {
          v6 = 40;
          LOBYTE(v7) = 1;
        }
        break;
      case 8LL:
        switch ( *(_QWORD *)v4 )
        {
          case 0x71655F504F5F5744LL:
            v6 = 41;
            LOBYTE(v7) = 1;
            break;
          case 0x65675F504F5F5744LL:
            v6 = 42;
            LOBYTE(v7) = 1;
            break;
          case 0x74675F504F5F5744LL:
            v6 = 43;
            LOBYTE(v7) = 1;
            break;
          case 0x656C5F504F5F5744LL:
            v6 = 44;
            LOBYTE(v7) = 1;
            break;
          case 0x746C5F504F5F5744LL:
            v6 = 45;
            LOBYTE(v7) = 1;
            break;
        }
        break;
    }
  }
  v3 = 46;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_ne", 8);
  if ( !(_BYTE)v7 && v5 == 10 )
  {
    if ( *(_QWORD *)v4 == 0x6B735F504F5F5744LL && *(_WORD *)(v4 + 8) == 28777 )
    {
      v6 = 47;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12404 )
    {
      v6 = 48;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 49;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit1", 10);
  if ( !(_BYTE)v7 && v5 == 10 )
  {
    if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12916 )
    {
      v6 = 50;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 13172 )
    {
      v6 = 51;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 13428 )
    {
      v6 = 52;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 53;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit5", 10);
  v3 = 54;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit6", 10);
  if ( !(_BYTE)v7 && v5 == 10 && *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 14196 )
  {
    v6 = 55;
    LOBYTE(v7) = 1;
  }
  v3 = 56;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit8", 10);
  if ( !(_BYTE)v7 && v5 == 10 && *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 14708 )
  {
    v6 = 57;
    LOBYTE(v7) = 1;
  }
  v3 = 58;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit10", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12660
    && *(_BYTE *)(v4 + 10) == 49 )
  {
    v6 = 59;
    LOBYTE(v7) = 1;
  }
  v3 = 60;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit12", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12660
    && *(_BYTE *)(v4 + 10) == 51 )
  {
    v6 = 61;
    LOBYTE(v7) = 1;
  }
  v3 = 62;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit14", 11);
  v3 = 63;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit15", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12660
    && *(_BYTE *)(v4 + 10) == 54 )
  {
    v6 = 64;
    LOBYTE(v7) = 1;
  }
  v3 = 65;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit17", 11);
  if ( !(_BYTE)v7 && v5 == 11 )
  {
    if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12660 && *(_BYTE *)(v4 + 10) == 56 )
    {
      v6 = 66;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12660 && *(_BYTE *)(v4 + 10) == 57 )
    {
      v6 = 67;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 68;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit20", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12916
    && *(_BYTE *)(v4 + 10) == 49 )
  {
    v6 = 69;
    LOBYTE(v7) = 1;
  }
  v3 = 70;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit22", 11);
  if ( !(_BYTE)v7 && v5 == 11 )
  {
    if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12916 && *(_BYTE *)(v4 + 10) == 51 )
    {
      v6 = 71;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12916 && *(_BYTE *)(v4 + 10) == 52 )
    {
      v6 = 72;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x696C5F504F5F5744LL && *(_WORD *)(v4 + 8) == 12916 && *(_BYTE *)(v4 + 10) == 53 )
    {
      v6 = 73;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 74;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit26", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12916
    && *(_BYTE *)(v4 + 10) == 55 )
  {
    v6 = 75;
    LOBYTE(v7) = 1;
  }
  v3 = 76;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit28", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x696C5F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12916
    && *(_BYTE *)(v4 + 10) == 57 )
  {
    v6 = 77;
    LOBYTE(v7) = 1;
  }
  v3 = 78;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit30", 11);
  v3 = 79;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_lit31", 11);
  if ( !(_BYTE)v7 && v5 == 10 )
  {
    if ( *(_QWORD *)v4 == 0x65725F504F5F5744LL && *(_WORD *)(v4 + 8) == 12391 )
    {
      v6 = 80;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x65725F504F5F5744LL && *(_WORD *)(v4 + 8) == 12647 )
    {
      v6 = 81;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 82;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg2", 10);
  v3 = 83;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg3", 10);
  v3 = 84;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg4", 10);
  v3 = 85;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg5", 10);
  v3 = 86;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg6", 10);
  v3 = 87;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg7", 10);
  v3 = 88;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg8", 10);
  v3 = 89;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg9", 10);
  v3 = 90;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg10", 11);
  v3 = 91;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg11", 11);
  v3 = 92;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg12", 11);
  v3 = 93;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg13", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x65725F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12647
    && *(_BYTE *)(v4 + 10) == 52 )
  {
    v6 = 94;
    LOBYTE(v7) = 1;
  }
  v3 = 95;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg15", 11);
  v3 = 96;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg16", 11);
  if ( !(_BYTE)v7 && v5 == 11 )
  {
    if ( *(_QWORD *)v4 == 0x65725F504F5F5744LL && *(_WORD *)(v4 + 8) == 12647 && *(_BYTE *)(v4 + 10) == 55 )
    {
      v6 = 97;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x65725F504F5F5744LL && *(_WORD *)(v4 + 8) == 12647 && *(_BYTE *)(v4 + 10) == 56 )
    {
      v6 = 98;
      LOBYTE(v7) = 1;
    }
    else if ( *(_QWORD *)v4 == 0x65725F504F5F5744LL && *(_WORD *)(v4 + 8) == 12647 && *(_BYTE *)(v4 + 10) == 57 )
    {
      v6 = 99;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 100;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg20", 11);
  v3 = 101;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg21", 11);
  v3 = 102;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg22", 11);
  v3 = 103;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg23", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x65725F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 12903
    && *(_BYTE *)(v4 + 10) == 52 )
  {
    v6 = 104;
    LOBYTE(v7) = 1;
  }
  v3 = 105;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg25", 11);
  v3 = 106;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg26", 11);
  v3 = 107;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg27", 11);
  v3 = 108;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg28", 11);
  v3 = 109;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg29", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x65725F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 13159
    && *(_BYTE *)(v4 + 10) == 48 )
  {
    v6 = 110;
    LOBYTE(v7) = 1;
  }
  v3 = 111;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_reg31", 11);
  v3 = 112;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg0", 11);
  v3 = 113;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg1", 11);
  v3 = 114;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg2", 11);
  v3 = 115;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg3", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x72625F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 26469
    && *(_BYTE *)(v4 + 10) == 52 )
  {
    v6 = 116;
    LOBYTE(v7) = 1;
  }
  v3 = 117;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg5", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x72625F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 26469
    && *(_BYTE *)(v4 + 10) == 54 )
  {
    v6 = 118;
    LOBYTE(v7) = 1;
  }
  v3 = 119;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg7", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x72625F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 26469
    && *(_BYTE *)(v4 + 10) == 56 )
  {
    v6 = 120;
    LOBYTE(v7) = 1;
  }
  v3 = 121;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg9", 11);
  v3 = 122;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg10", 12);
  if ( !(_BYTE)v7 && v5 == 12 && *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_DWORD *)(v4 + 8) == 825321317 )
  {
    v6 = 123;
    LOBYTE(v7) = 1;
  }
  v3 = 124;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg12", 12);
  v3 = 125;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg13", 12);
  v3 = 126;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg14", 12);
  v3 = 127;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg15", 12);
  if ( !(_BYTE)v7 && v5 == 12 && *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_DWORD *)(v4 + 8) == 909207397 )
  {
    v6 = 128;
    LOBYTE(v7) = 1;
  }
  v3 = 129;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg17", 12);
  v3 = 130;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg18", 12);
  v3 = 131;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg19", 12);
  v3 = 132;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg20", 12);
  if ( !(_BYTE)v7 && v5 == 12 && *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_DWORD *)(v4 + 8) == 825386853 )
  {
    v6 = 133;
    LOBYTE(v7) = 1;
  }
  v3 = 134;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg22", 12);
  v3 = 135;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg23", 12);
  v3 = 136;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg24", 12);
  v3 = 137;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg25", 12);
  if ( !(_BYTE)v7 && v5 == 12 && *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_DWORD *)(v4 + 8) == 909272933 )
  {
    v6 = 138;
    LOBYTE(v7) = 1;
  }
  v3 = 139;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg27", 12);
  v3 = 140;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg28", 12);
  v3 = 141;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg29", 12);
  v3 = 142;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg30", 12);
  v3 = 143;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_breg31", 12);
  v3 = 144;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_regx", 10);
  v3 = 145;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_fbreg", 11);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 11 && *(_QWORD *)v4 == 0x72625F504F5F5744LL && *(_WORD *)(v4 + 8) == 26469 && *(_BYTE *)(v4 + 10) == 120 )
    {
      v6 = 146;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 11
           && *(_QWORD *)v4 == 0x69705F504F5F5744LL
           && *(_WORD *)(v4 + 8) == 25445
           && *(_BYTE *)(v4 + 10) == 101 )
    {
      v6 = 147;
      LOBYTE(v7) = 1;
    }
    else if ( v5 != 16 || *(_QWORD *)v4 ^ 0x65645F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x657A69735F666572LL )
    {
      if ( v5 == 17
        && !(*(_QWORD *)v4 ^ 0x64785F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x7A69735F66657265LL)
        && *(_BYTE *)(v4 + 16) == 101 )
      {
        v6 = 149;
        LOBYTE(v7) = 1;
      }
    }
    else
    {
      v6 = 148;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 150;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_nop", 9);
  if ( !(_BYTE)v7
    && v5 == 25
    && !(*(_QWORD *)v4 ^ 0x75705F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x63656A626F5F6873LL)
    && *(_QWORD *)(v4 + 16) == 0x7365726464615F74LL
    && *(_BYTE *)(v4 + 24) == 115 )
  {
    v6 = 151;
    LOBYTE(v7) = 1;
  }
  v3 = 152;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_call2", 11);
  if ( !(_BYTE)v7
    && v5 == 11
    && *(_QWORD *)v4 == 0x61635F504F5F5744LL
    && *(_WORD *)(v4 + 8) == 27756
    && *(_BYTE *)(v4 + 10) == 52 )
  {
    v6 = 153;
    LOBYTE(v7) = 1;
  }
  v3 = 154;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_call_ref", 14);
  v3 = 155;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_form_tls_address", 22);
  if ( !(_BYTE)v7
    && v5 == 20
    && !(*(_QWORD *)v4 ^ 0x61635F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x656D6172665F6C6CLL)
    && *(_DWORD *)(v4 + 16) == 1634100063 )
  {
    v6 = 156;
    LOBYTE(v7) = 1;
  }
  v3 = 157;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_bit_piece", 15);
  v3 = 158;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_implicit_value", 20);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 17
      && !(*(_QWORD *)v4 ^ 0x74735F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x756C61765F6B6361LL)
      && *(_BYTE *)(v4 + 16) == 101 )
    {
      v6 = 159;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 22
           && !(*(_QWORD *)v4 ^ 0x6D695F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x705F746963696C70LL)
           && *(_DWORD *)(v4 + 16) == 1953393007
           && *(_WORD *)(v4 + 20) == 29285 )
    {
      v6 = 160;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 11
           && *(_QWORD *)v4 == 0x64615F504F5F5744LL
           && *(_WORD *)(v4 + 8) == 29284
           && *(_BYTE *)(v4 + 10) == 120 )
    {
      v6 = 161;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 12 && *(_QWORD *)v4 == 0x6F635F504F5F5744LL && *(_DWORD *)(v4 + 8) == 2020897646 )
    {
      v6 = 162;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 17
           && !(*(_QWORD *)v4 ^ 0x6E655F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x756C61765F797274LL)
           && *(_BYTE *)(v4 + 16) == 101 )
    {
      v6 = 163;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 16 && !(*(_QWORD *)v4 ^ 0x6F635F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x657079745F74736ELL) )
    {
      v6 = 164;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 17
           && !(*(_QWORD *)v4 ^ 0x65725F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x7079745F6C617667LL)
           && *(_BYTE *)(v4 + 16) == 101 )
    {
      v6 = 165;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 166;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_deref_type", 16);
  v3 = 167;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_xderef_type", 17);
  v3 = 168;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_convert", 13);
  if ( !(_BYTE)v7
    && v5 == 17
    && !(*(_QWORD *)v4 ^ 0x65725F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x6572707265746E69LL)
    && *(_BYTE *)(v4 + 16) == 116 )
  {
    v6 = 169;
    LOBYTE(v7) = 1;
  }
  v3 = 224;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_GNU_push_tls_address", 26);
  v3 = 225;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_HP_is_value", 17);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 18
      && !(*(_QWORD *)v4 ^ 0x50485F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x736E6F63746C665FLL)
      && *(_WORD *)(v4 + 16) == 13428 )
    {
      v6 = 226;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 18
           && !(*(_QWORD *)v4 ^ 0x50485F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x736E6F63746C665FLL)
           && *(_WORD *)(v4 + 16) == 14452 )
    {
      v6 = 227;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 228;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_HP_mod_range", 18);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 20
      && !(*(_QWORD *)v4 ^ 0x50485F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x725F646F6D6E755FLL)
      && *(_DWORD *)(v4 + 16) == 1701277281 )
    {
      v6 = 229;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 12 && *(_QWORD *)v4 == 0x50485F504F5F5744LL && *(_DWORD *)(v4 + 8) == 1936487519 )
    {
      v6 = 230;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 21
           && !(*(_QWORD *)v4 ^ 0x4E495F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x5F7469625F4C4554LL)
           && *(_DWORD *)(v4 + 16) == 1667590512
           && *(_BYTE *)(v4 + 20) == 101 )
    {
      v6 = 232;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 19
           && !(*(_QWORD *)v4 ^ 0x41575F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x7461636F6C5F4D53LL)
           && *(_WORD *)(v4 + 16) == 28521
           && *(_BYTE *)(v4 + 18) == 110 )
    {
      v6 = 237;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 238;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_WASM_location_int", 23);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 18
      && !(*(_QWORD *)v4 ^ 0x50415F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x6E696E755F454C50LL)
      && *(_WORD *)(v4 + 16) == 29801 )
    {
      v6 = 240;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 21
           && !(*(_QWORD *)v4 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x5F7972746E655F55LL)
           && *(_DWORD *)(v4 + 16) == 1970037110
           && *(_BYTE *)(v4 + 20) == 101 )
    {
      v6 = 243;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 24
           && !(*(_QWORD *)v4 ^ 0x47505F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x68745F706D6F5F49LL)
           && *(_QWORD *)(v4 + 16) == 0x6D756E5F64616572LL )
    {
      v6 = 248;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 20
           && !(*(_QWORD *)v4 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x695F726464615F55LL)
           && *(_DWORD *)(v4 + 16) == 2019910766 )
    {
      v6 = 251;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 21
           && !(*(_QWORD *)v4 ^ 0x4E475F504F5F5744LL | *(_QWORD *)(v4 + 8) ^ 0x5F74736E6F635F55LL)
           && *(_DWORD *)(v4 + 16) == 1701080681
           && *(_BYTE *)(v4 + 20) == 120 )
    {
      v6 = 252;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 15
           && *(_QWORD *)v4 == 0x4C4C5F504F5F5744LL
           && *(_DWORD *)(v4 + 8) == 1969179990
           && *(_WORD *)(v4 + 12) == 25971
           && *(_BYTE *)(v4 + 14) == 114 )
    {
      v6 = 233;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 4097;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_convert", 18);
  v3 = 4096;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_fragment", 19);
  v3 = 4098;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_tag_offset", 21);
  v3 = 4099;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_entry_value", 22);
  v3 = 4100;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_implicit_pointer", 27);
  if ( !(_BYTE)v7
    && v5 == 14
    && *(_QWORD *)v4 == 0x4C4C5F504F5F5744LL
    && *(_DWORD *)(v4 + 8) == 1633635670
    && *(_WORD *)(v4 + 12) == 26482 )
  {
    v6 = 4101;
    LOBYTE(v7) = 1;
  }
  v3 = 4102;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_extract_bits_sext", 28);
  v3 = 4103;
  sub_E02B30((__int64)&v4, &v3, "DW_OP_LLVM_extract_bits_zext", 28);
  result = 0;
  if ( (_BYTE)v7 )
    return v6;
  return result;
}
