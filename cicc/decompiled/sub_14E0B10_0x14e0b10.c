// Function: sub_14E0B10
// Address: 0x14e0b10
//
__int64 __fastcall sub_14E0B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  bool v7; // r13
  char v8; // cl
  char v9; // bl
  bool v10; // al
  bool v11; // r15
  char v12; // dl
  bool v13; // si
  bool v14; // dl
  char v15; // si
  char v16; // si
  bool v17; // r8
  bool v18; // r14
  char v19; // si
  char v20; // al
  char v21; // al
  char v22; // dl
  char v23; // r10
  char v24; // al
  char v25; // al
  char v26; // r8
  char v27; // dl
  bool v28; // al
  char v29; // r13
  char v30; // r14
  char v31; // r15
  char v32; // al
  int v34; // eax
  unsigned int v35; // [rsp+8h] [rbp-48h]
  char v36; // [rsp+Eh] [rbp-42h]
  bool v37; // [rsp+Fh] [rbp-41h]
  char v38; // [rsp+18h] [rbp-38h]
  bool v39; // [rsp+1Ah] [rbp-36h]
  bool v40; // [rsp+1Bh] [rbp-35h]
  bool v41; // [rsp+1Ch] [rbp-34h]
  bool v42; // [rsp+1Dh] [rbp-33h]
  char v43; // [rsp+1Eh] [rbp-32h]
  bool v44; // [rsp+1Fh] [rbp-31h]

  v7 = a2 == 18;
  switch ( a2 )
  {
    case 11LL:
      if ( *(_QWORD *)a1 == 0x6E5F4741545F5744LL && *(_WORD *)(a1 + 8) == 27765 && *(_BYTE *)(a1 + 10) == 108 )
      {
        a6 = 0;
        goto LABEL_366;
      }
      goto LABEL_9;
    case 17LL:
      if ( !(*(_QWORD *)a1 ^ 0x615F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F79617272LL)
        && *(_BYTE *)(a1 + 16) == 101 )
      {
        a6 = 1;
        goto LABEL_366;
      }
      if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F7373616CLL)
        && *(_BYTE *)(a1 + 16) == 101 )
      {
        a6 = 2;
LABEL_366:
        v43 = 1;
        v8 = a2 == 20;
        v9 = 0;
        goto LABEL_97;
      }
      goto LABEL_9;
    case 18LL:
      if ( !(*(_QWORD *)a1 ^ 0x655F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x696F705F7972746ELL)
        && *(_WORD *)(a1 + 16) == 29806 )
      {
        a6 = 3;
        goto LABEL_366;
      }
LABEL_9:
      v8 = a2 == 20;
      if ( a2 == 12 )
      {
        if ( *(_QWORD *)a1 == 0x6C5F4741545F5744LL && *(_DWORD *)(a1 + 8) == 1818583649 )
        {
          v44 = 0;
          a6 = 10;
          v9 = 0;
          v10 = 0;
          v43 = 1;
        }
        else
        {
          v44 = 0;
          v9 = 1;
          v10 = 0;
          v43 = 0;
        }
LABEL_12:
        v11 = a2 == 21;
        if ( ((unsigned __int8)v9 & v44) != 0 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x705F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x745F7265746E696FLL)
            && *(_WORD *)(a1 + 16) == 28793
            && *(_BYTE *)(a1 + 18) == 101 )
          {
            v43 = v9 & v44;
            a6 = 15;
            v9 = 0;
            goto LABEL_16;
          }
          if ( *(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x755F656C69706D6FLL
            || *(_WORD *)(a1 + 16) != 26990
            || *(_BYTE *)(a1 + 18) != 116 )
          {
            goto LABEL_16;
          }
          v43 = v9 & v44;
          a6 = 17;
          v41 = a2 == 22;
          v9 = 0;
LABEL_17:
          v13 = a2 == 14;
          if ( ((unsigned __int8)v9 & v41) != 0 )
          {
            v14 = a2 == 17;
            if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E6974756F726275LL)
              && *(_DWORD *)(a1 + 16) == 2037669733
              && *(_WORD *)(a1 + 20) == 25968 )
            {
              v43 = v9 & v41;
              a6 = 21;
              v39 = a2 == 29;
              v9 = 0;
              goto LABEL_20;
            }
            goto LABEL_19;
          }
          goto LABEL_18;
        }
        goto LABEL_13;
      }
      v43 = 0;
      v9 = 1;
      goto LABEL_97;
  }
  if ( a2 != 23 )
  {
    if ( a2 == 27 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F646574726F706DLL)
        && *(_QWORD *)(a1 + 16) == 0x746172616C636564LL
        && *(_WORD *)(a1 + 24) == 28521
        && *(_BYTE *)(a1 + 26) == 110 )
      {
        v43 = 1;
        v10 = 0;
        v8 = 0;
        a6 = 8;
        v9 = 0;
      }
      else
      {
        v8 = 0;
        v43 = 0;
        v9 = 1;
        v10 = 0;
      }
      goto LABEL_98;
    }
    goto LABEL_9;
  }
  if ( !(*(_QWORD *)a1 ^ 0x655F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x69746172656D756ELL)
    && *(_DWORD *)(a1 + 16) == 1952411247
    && *(_WORD *)(a1 + 20) == 28793
    && *(_BYTE *)(a1 + 22) == 101 )
  {
    a6 = 4;
  }
  else
  {
    if ( *(_QWORD *)a1 ^ 0x665F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x61705F6C616D726FLL
      || *(_DWORD *)(a1 + 16) != 1701667186
      || *(_WORD *)(a1 + 20) != 25972
      || *(_BYTE *)(a1 + 22) != 114 )
    {
      v43 = 0;
      v8 = 0;
      v9 = 1;
      goto LABEL_97;
    }
    a6 = 5;
  }
  v43 = 1;
  v8 = 0;
  v9 = 0;
LABEL_97:
  v10 = a2 == 13;
  if ( ((unsigned __int8)v9 & (unsigned __int8)v8) != 0 )
  {
    v44 = a2 == 19;
    if ( !(*(_QWORD *)a1 ^ 0x6C5F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x625F6C6163697865LL)
      && *(_DWORD *)(a1 + 16) == 1801678700 )
    {
      v43 = v9 & v8;
      a6 = 11;
      v11 = a2 == 21;
      v9 = 0;
      goto LABEL_13;
    }
    goto LABEL_12;
  }
LABEL_98:
  v44 = a2 == 19;
  if ( ((unsigned __int8)v9 & v10) == 0 )
    goto LABEL_12;
  v11 = a2 == 21;
  if ( *(_QWORD *)a1 == 0x6D5F4741545F5744LL && *(_DWORD *)(a1 + 8) == 1700949349 && *(_BYTE *)(a1 + 12) == 114 )
  {
    v43 = v9 & v10;
    a6 = 13;
    v9 = 0;
    goto LABEL_14;
  }
LABEL_13:
  if ( (v11 & (unsigned __int8)v9) == 0 )
  {
LABEL_14:
    v12 = v9 & v7;
    goto LABEL_15;
  }
  if ( !(*(_QWORD *)a1 ^ 0x725F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65636E6572656665LL)
    && *(_DWORD *)(a1 + 16) == 1887007839
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    v43 = v11 & v9;
    a6 = 16;
    v9 = 0;
    goto LABEL_16;
  }
  v12 = v7 & v9;
LABEL_15:
  if ( v12 )
  {
    v41 = a2 == 22;
    if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x79745F676E697274LL)
      && *(_WORD *)(a1 + 16) == 25968 )
    {
      v43 = v12;
      a6 = 18;
      v13 = a2 == 14;
      v9 = 0;
      goto LABEL_18;
    }
    goto LABEL_17;
  }
LABEL_16:
  v41 = a2 == 22;
  if ( (v11 & (unsigned __int8)v9) == 0 )
    goto LABEL_17;
  v13 = a2 == 14;
  if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6572757463757274LL)
    && *(_DWORD *)(a1 + 16) == 1887007839
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    v43 = v11 & v9;
    a6 = 19;
    v14 = a2 == 17;
    v9 = 0;
    goto LABEL_19;
  }
LABEL_18:
  v14 = a2 == 17;
  if ( (v13 & (unsigned __int8)v9) != 0 )
  {
    v39 = a2 == 29;
    if ( *(_QWORD *)a1 != 0x745F4741545F5744LL || *(_DWORD *)(a1 + 8) != 1684369529 || *(_WORD *)(a1 + 12) != 26213 )
      goto LABEL_20;
    v43 = v13 & v9;
    v9 = 0;
    a6 = 22;
    goto LABEL_21;
  }
LABEL_19:
  v39 = a2 == 29;
  if ( (v14 & (unsigned __int8)v9) == 0 )
  {
LABEL_20:
    if ( ((unsigned __int8)v9 & v39) == 0 )
      goto LABEL_21;
    if ( !(*(_QWORD *)a1 ^ 0x755F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x696669636570736ELL)
      && *(_QWORD *)(a1 + 16) == 0x6D617261705F6465LL
      && *(_DWORD *)(a1 + 24) == 1919251557
      && *(_BYTE *)(a1 + 28) == 115 )
    {
      v43 = v9 & v39;
      a6 = 24;
      v42 = a2 == 23;
      v9 = 0;
      goto LABEL_23;
    }
LABEL_22:
    v42 = a2 == 23;
    if ( ((unsigned __int8)v9 & v44) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6C625F6E6F6D6D6FLL)
        && *(_WORD *)(a1 + 16) == 25455
        && *(_BYTE *)(a1 + 18) == 107 )
      {
        v43 = v9 & v44;
        a6 = 26;
        v40 = a2 == 25;
        v9 = 0;
        goto LABEL_26;
      }
      v16 = v7 & v9;
      goto LABEL_25;
    }
    goto LABEL_23;
  }
  if ( !(*(_QWORD *)a1 ^ 0x755F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F6E6F696ELL)
    && *(_BYTE *)(a1 + 16) == 101 )
  {
    v43 = v14 & v9;
    a6 = 23;
    v9 = 0;
    goto LABEL_22;
  }
LABEL_21:
  v15 = v9 & v13;
  if ( !v15 )
    goto LABEL_22;
  v42 = a2 == 23;
  if ( *(_QWORD *)a1 != 0x765F4741545F5744LL || *(_DWORD *)(a1 + 8) != 1634300513 || *(_WORD *)(a1 + 12) != 29806 )
  {
LABEL_23:
    if ( ((unsigned __int8)v9 & v42) != 0 )
    {
      v40 = a2 == 25;
      if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E695F6E6F6D6D6FLL)
        && *(_DWORD *)(a1 + 16) == 1937075299
        && *(_WORD *)(a1 + 20) == 28521
        && *(_BYTE *)(a1 + 22) == 110 )
      {
        v43 = v9 & v42;
        v9 = 0;
        a6 = 27;
        goto LABEL_27;
      }
      goto LABEL_26;
    }
    goto LABEL_24;
  }
  v43 = v15;
  a6 = 25;
  v9 = 0;
LABEL_24:
  v16 = v9 & v7;
LABEL_25:
  v40 = a2 == 25;
  if ( v16 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E6174697265686ELL)
      && *(_WORD *)(a1 + 16) == 25955 )
    {
      v43 = v16;
      a6 = 28;
      v17 = a2 == 15;
      v9 = 0;
LABEL_28:
      if ( (v17 & (unsigned __int8)v9) != 0 )
      {
        v18 = a2 == 16;
        if ( *(_QWORD *)a1 == 0x735F4741545F5744LL
          && *(_DWORD *)(a1 + 8) == 1952412773
          && *(_WORD *)(a1 + 12) == 28793
          && *(_BYTE *)(a1 + 14) == 101 )
        {
          v43 = v17 & v9;
          a6 = 32;
          v9 = 0;
          goto LABEL_31;
        }
        goto LABEL_30;
      }
      goto LABEL_29;
    }
LABEL_27:
    v17 = a2 == 15;
    if ( (v10 & (unsigned __int8)v9) != 0
      && *(_QWORD *)a1 == 0x6D5F4741545F5744LL
      && *(_DWORD *)(a1 + 8) == 1819632751
      && *(_BYTE *)(a1 + 12) == 101 )
    {
      v43 = v10 & v9;
      v9 = 0;
      a6 = 30;
      goto LABEL_29;
    }
    goto LABEL_28;
  }
LABEL_26:
  if ( ((unsigned __int8)v9 & v40) == 0 )
    goto LABEL_27;
  if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x735F64656E696C6ELL)
    && *(_QWORD *)(a1 + 16) == 0x6E6974756F726275LL
    && *(_BYTE *)(a1 + 24) == 101 )
  {
    v43 = v9 & v40;
    a6 = 29;
    v17 = a2 == 15;
    v9 = 0;
  }
  else
  {
    v17 = a2 == 15;
    if ( !(*(_QWORD *)a1 ^ 0x705F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x656D5F6F745F7274LL)
      && *(_QWORD *)(a1 + 16) == 0x7079745F7265626DLL
      && *(_BYTE *)(a1 + 24) == 101 )
    {
      v43 = v9 & v40;
      a6 = 31;
      v18 = a2 == 16;
      v9 = 0;
LABEL_30:
      if ( (v18 & (unsigned __int8)v9) != 0 )
      {
        if ( *(_QWORD *)a1 ^ 0x775F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x746D74735F687469LL )
        {
          if ( !(*(_QWORD *)a1 ^ 0x625F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657079745F657361LL) )
          {
            v43 = v18 & v9;
            v9 = 0;
            a6 = 36;
            goto LABEL_35;
          }
        }
        else
        {
          v43 = v18 & v9;
          a6 = 34;
          v9 = 0;
        }
        goto LABEL_34;
      }
      goto LABEL_31;
    }
  }
LABEL_29:
  v18 = a2 == 16;
  if ( ((unsigned __int8)v8 & (unsigned __int8)v9) == 0 )
    goto LABEL_30;
  if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F65676E61726275LL)
    && *(_DWORD *)(a1 + 16) == 1701869940 )
  {
    v43 = v8 & v9;
    v9 = 0;
    a6 = 33;
    goto LABEL_32;
  }
LABEL_31:
  if ( ((unsigned __int8)v9 & v40) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x615F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65645F7373656363LL)
      && *(_QWORD *)(a1 + 16) == 0x6F69746172616C63LL
      && *(_BYTE *)(a1 + 24) == 110 )
    {
      v43 = v9 & v40;
      a6 = 35;
      v9 = 0;
      goto LABEL_34;
    }
    v19 = v7 & v9;
    goto LABEL_33;
  }
LABEL_32:
  v19 = v9 & v7;
LABEL_33:
  if ( v19 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F6C625F68637461LL)
      && *(_WORD *)(a1 + 16) == 27491 )
    {
      v43 = v19;
      a6 = 37;
      v9 = 0;
LABEL_36:
      if ( (v18 & (unsigned __int8)v9) == 0 )
        goto LABEL_37;
      if ( !(*(_QWORD *)a1 ^ 0x665F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657079745F656C69LL) )
      {
        v43 = v18 & v9;
        v9 = 0;
        a6 = 41;
        goto LABEL_39;
      }
LABEL_38:
      if ( (v17 & (unsigned __int8)v9) != 0 )
      {
        if ( *(_QWORD *)a1 == 0x6E5F4741545F5744LL
          && *(_DWORD *)(a1 + 8) == 1818586465
          && *(_WORD *)(a1 + 12) == 29545
          && *(_BYTE *)(a1 + 14) == 116 )
        {
          v43 = v17 & v9;
          v9 = 0;
          a6 = 43;
          goto LABEL_42;
        }
        v21 = v7 & v9;
        goto LABEL_41;
      }
      goto LABEL_39;
    }
LABEL_35:
    if ( (v17 & (unsigned __int8)v9) != 0
      && *(_QWORD *)a1 == 0x635F4741545F5744LL
      && *(_DWORD *)(a1 + 8) == 1953721967
      && *(_WORD *)(a1 + 12) == 28257
      && *(_BYTE *)(a1 + 14) == 116 )
    {
      v43 = v17 & v9;
      v9 = 0;
      a6 = 39;
      goto LABEL_37;
    }
    goto LABEL_36;
  }
LABEL_34:
  if ( (v14 & (unsigned __int8)v9) == 0 )
    goto LABEL_35;
  if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x7079745F74736E6FLL)
    && *(_BYTE *)(a1 + 16) == 101 )
  {
    v43 = v14 & v9;
    v9 = 0;
    a6 = 38;
  }
  else if ( !(*(_QWORD *)a1 ^ 0x655F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F746172656D756ELL)
         && *(_BYTE *)(a1 + 16) == 114 )
  {
    v43 = v14 & v9;
    v9 = 0;
    a6 = 40;
    goto LABEL_38;
  }
LABEL_37:
  v20 = v9 & v10;
  if ( !v20 )
    goto LABEL_38;
  if ( *(_QWORD *)a1 != 0x665F4741545F5744LL || *(_DWORD *)(a1 + 8) != 1852139890 || *(_BYTE *)(a1 + 12) != 100 )
  {
LABEL_39:
    if ( ((unsigned __int8)v8 & (unsigned __int8)v9) != 0 )
    {
      if ( *(_QWORD *)a1 ^ 0x6E5F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F7473696C656D61LL
        || *(_DWORD *)(a1 + 16) != 1835365481 )
      {
        goto LABEL_42;
      }
      v43 = v8 & v9;
      a6 = 44;
      v23 = a2 == 30;
      v9 = 0;
      goto LABEL_43;
    }
    goto LABEL_40;
  }
  v43 = v20;
  v9 = 0;
  a6 = 42;
LABEL_40:
  v21 = v9 & v7;
LABEL_41:
  if ( !v21 )
  {
LABEL_42:
    v22 = v9 & v14;
    v23 = a2 == 30;
    if ( !v22 )
      goto LABEL_43;
    if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6172676F72706275LL)
      && *(_BYTE *)(a1 + 16) == 109 )
    {
      v43 = v22;
      v9 = 0;
      a6 = 46;
LABEL_45:
      v24 = v9 & v7;
      goto LABEL_46;
    }
LABEL_44:
    if ( ((unsigned __int8)v9 & (a2 == 31)) != 0 )
    {
      v36 = v23;
      v37 = v17;
      v38 = v8;
      v35 = a6;
      v34 = memcmp((const void *)a1, "DW_TAG_template_value_parameter", 0x1Fu);
      v8 = v38;
      v17 = v37;
      v23 = v36;
      if ( !v34 )
      {
        v43 = v9 & (a2 == 31);
        v9 = 0;
        a6 = 48;
        goto LABEL_49;
      }
      a6 = v35;
      v25 = v18 & v9;
      goto LABEL_48;
    }
    goto LABEL_45;
  }
  v23 = a2 == 30;
  if ( !(*(_QWORD *)a1 ^ 0x705F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x79745F64656B6361LL)
    && *(_WORD *)(a1 + 16) == 25968 )
  {
    v43 = v21;
    v9 = 0;
    a6 = 45;
    goto LABEL_44;
  }
LABEL_43:
  if ( ((unsigned __int8)v23 & (unsigned __int8)v9) == 0 )
    goto LABEL_44;
  if ( *(_QWORD *)a1 ^ 0x745F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6574616C706D65LL
    || *(_QWORD *)(a1 + 16) != 0x7261705F65707974LL
    || *(_DWORD *)(a1 + 24) != 1952804193
    || *(_WORD *)(a1 + 28) != 29285 )
  {
    v24 = v7 & v9;
LABEL_46:
    if ( v24 )
    {
      if ( *(_QWORD *)a1 ^ 0x745F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x79745F6E776F7268LL
        || *(_WORD *)(a1 + 16) != 25968 )
      {
        goto LABEL_49;
      }
      v43 = v24;
      v9 = 0;
      a6 = 49;
      goto LABEL_50;
    }
    goto LABEL_47;
  }
  v43 = v23 & v9;
  a6 = 47;
  v9 = 0;
LABEL_47:
  v25 = v18 & v9;
LABEL_48:
  if ( !v25 )
  {
LABEL_49:
    if ( ((unsigned __int8)v9 & v44) == 0 )
      goto LABEL_50;
    if ( !(*(_QWORD *)a1 ^ 0x765F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F746E61697261LL)
      && *(_WORD *)(a1 + 16) == 29281
      && *(_BYTE *)(a1 + 18) == 116 )
    {
      v43 = v9 & v44;
      v9 = 0;
      a6 = 51;
      goto LABEL_52;
    }
LABEL_51:
    if ( ((unsigned __int8)v8 & (unsigned __int8)v9) != 0 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x765F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F656C6974616C6FLL)
        && *(_DWORD *)(a1 + 16) == 1701869940 )
      {
        v43 = v8 & v9;
        v9 = 0;
        a6 = 53;
      }
      else if ( !(*(_QWORD *)a1 ^ 0x725F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F74636972747365LL)
             && *(_DWORD *)(a1 + 16) == 1701869940 )
      {
        v43 = v8 & v9;
        v9 = 0;
        a6 = 55;
        goto LABEL_55;
      }
LABEL_54:
      if ( (v18 & (unsigned __int8)v9) != 0 )
      {
        if ( !(*(_QWORD *)a1 ^ 0x6E5F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6563617073656D61LL) )
        {
          v43 = v18 & v9;
          v9 = 0;
          a6 = 57;
          goto LABEL_57;
        }
        goto LABEL_56;
      }
      goto LABEL_55;
    }
    goto LABEL_52;
  }
  if ( !(*(_QWORD *)a1 ^ 0x745F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6B636F6C625F7972LL) )
  {
    v43 = v25;
    a6 = 50;
    v9 = 0;
    goto LABEL_51;
  }
LABEL_50:
  v26 = v9 & v17;
  if ( !v26 )
    goto LABEL_51;
  if ( *(_QWORD *)a1 != 0x765F4741545F5744LL
    || *(_DWORD *)(a1 + 8) != 1634300513
    || *(_WORD *)(a1 + 12) != 27746
    || *(_BYTE *)(a1 + 14) != 101 )
  {
LABEL_52:
    if ( ((unsigned __int8)v9 & v41) != 0
      && !(*(_QWORD *)a1 ^ 0x645F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F72705F66726177LL)
      && *(_DWORD *)(a1 + 16) == 1969513827
      && *(_WORD *)(a1 + 20) == 25970 )
    {
      v43 = v9 & v41;
      v9 = 0;
      a6 = 54;
      goto LABEL_54;
    }
    goto LABEL_53;
  }
  v43 = v26;
  a6 = 52;
  v9 = 0;
LABEL_53:
  if ( (v11 & (unsigned __int8)v9) == 0 )
    goto LABEL_54;
  if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x656361667265746ELL)
    && *(_DWORD *)(a1 + 16) == 1887007839
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    v43 = v11 & v9;
    a6 = 56;
    v9 = 0;
    goto LABEL_56;
  }
LABEL_55:
  if ( ((unsigned __int8)v9 & v41) != 0 )
  {
    if ( *(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F646574726F706DLL
      || *(_DWORD *)(a1 + 16) != 1969516397
      || *(_WORD *)(a1 + 20) != 25964 )
    {
      goto LABEL_57;
    }
    v43 = v9 & v41;
    v9 = 0;
    a6 = 58;
    goto LABEL_58;
  }
LABEL_56:
  if ( ((unsigned __int8)v9 & v42) == 0 )
  {
LABEL_57:
    if ( ((unsigned __int8)v9 & v44) == 0 )
      goto LABEL_58;
    if ( !(*(_QWORD *)a1 ^ 0x705F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x755F6C6169747261LL)
      && *(_WORD *)(a1 + 16) == 26990
      && *(_BYTE *)(a1 + 18) == 116 )
    {
      v43 = v9 & v44;
      v9 = 0;
      a6 = 60;
LABEL_60:
      v27 = v9 & v7;
      goto LABEL_61;
    }
LABEL_59:
    if ( (v18 & (unsigned __int8)v9) != 0 )
    {
      if ( *(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6E6F697469646E6FLL )
      {
        v28 = a2 == 28;
        if ( !(*(_QWORD *)a1 ^ 0x745F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x74696E755F657079LL) )
        {
          v43 = v18 & v9;
          v9 = 0;
          a6 = 65;
          goto LABEL_64;
        }
      }
      else
      {
        v43 = v18 & v9;
        a6 = 63;
        v28 = a2 == 28;
        v9 = 0;
      }
      goto LABEL_63;
    }
    goto LABEL_60;
  }
  if ( !(*(_QWORD *)a1 ^ 0x755F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x696669636570736ELL)
    && *(_DWORD *)(a1 + 16) == 1952408677
    && *(_WORD *)(a1 + 20) == 28793
    && *(_BYTE *)(a1 + 22) == 101 )
  {
    v43 = v9 & v42;
    a6 = 59;
    v9 = 0;
    goto LABEL_59;
  }
LABEL_58:
  if ( ((unsigned __int8)v8 & (unsigned __int8)v9) == 0 )
    goto LABEL_59;
  if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F646574726F706DLL)
    && *(_DWORD *)(a1 + 16) == 1953066613 )
  {
    v43 = v8 & v9;
    a6 = 61;
    v28 = a2 == 28;
    v9 = 0;
    goto LABEL_62;
  }
  v27 = v7 & v9;
LABEL_61:
  v28 = a2 == 28;
  if ( v27
    && !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x79745F6465726168LL)
    && *(_WORD *)(a1 + 16) == 25968 )
  {
    v43 = v27;
    v9 = 0;
    a6 = 64;
    goto LABEL_63;
  }
LABEL_62:
  if ( (v28 & (unsigned __int8)v9) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x725F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x65725F65756C6176LL)
      && *(_QWORD *)(a1 + 16) == 0x5F65636E65726566LL
      && *(_DWORD *)(a1 + 24) == 1701869940 )
    {
      v43 = v28 & v9;
      v9 = 0;
      a6 = 66;
      goto LABEL_65;
    }
    goto LABEL_64;
  }
LABEL_63:
  if ( (v11 & (unsigned __int8)v9) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x745F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6574616C706D65LL)
      && *(_DWORD *)(a1 + 16) == 1634298977
      && *(_BYTE *)(a1 + 20) == 115 )
    {
      v43 = v11 & v9;
      a6 = 67;
      v9 = 0;
      goto LABEL_66;
    }
LABEL_65:
    if ( ((unsigned __int8)v9 & v42) != 0
      && !(*(_QWORD *)a1 ^ 0x675F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x735F636972656E65LL)
      && *(_DWORD *)(a1 + 16) == 1634886261
      && *(_WORD *)(a1 + 20) == 26478
      && *(_BYTE *)(a1 + 22) == 101 )
    {
      v43 = v9 & v42;
      v9 = 0;
      a6 = 69;
      goto LABEL_67;
    }
LABEL_66:
    v29 = v9 & v7;
    if ( v29 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x615F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x79745F63696D6F74LL)
        && *(_WORD *)(a1 + 16) == 25968 )
      {
        v43 = v29;
        v9 = 0;
        a6 = 71;
        goto LABEL_69;
      }
      goto LABEL_68;
    }
    goto LABEL_67;
  }
LABEL_64:
  if ( ((unsigned __int8)v9 & v44) == 0 )
    goto LABEL_65;
  if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x745F79617272616FLL)
    && *(_WORD *)(a1 + 16) == 28793
    && *(_BYTE *)(a1 + 18) == 101 )
  {
    v43 = v9 & v44;
    v9 = 0;
    a6 = 68;
  }
  else if ( !(*(_QWORD *)a1 ^ 0x645F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x745F63696D616E79LL)
         && *(_WORD *)(a1 + 16) == 28793
         && *(_BYTE *)(a1 + 18) == 101 )
  {
    v43 = v9 & v44;
    v9 = 0;
    a6 = 70;
    goto LABEL_68;
  }
LABEL_67:
  if ( (v18 & (unsigned __int8)v9) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657469735F6C6C61LL) )
    {
      v43 = v18 & v9;
      v9 = 0;
      a6 = 72;
      goto LABEL_70;
    }
    goto LABEL_69;
  }
LABEL_68:
  if ( ((unsigned __int8)v9 & (a2 == 26)) != 0 )
  {
    if ( *(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x657469735F6C6C61LL
      || *(_QWORD *)(a1 + 16) != 0x74656D617261705FLL
      || *(_WORD *)(a1 + 24) != 29285 )
    {
      goto LABEL_70;
    }
    v43 = v9 & (a2 == 26);
    v9 = 0;
    a6 = 73;
LABEL_71:
    v30 = v9 & v18;
    if ( v30 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x4D5F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x706F6F6C5F535049LL) )
      {
        v43 = v30;
        v9 = 0;
        a6 = 16513;
        goto LABEL_74;
      }
      goto LABEL_73;
    }
    goto LABEL_72;
  }
LABEL_69:
  if ( ((unsigned __int8)v8 & (unsigned __int8)v9) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x735F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6E6F74656C656BLL)
      && *(_DWORD *)(a1 + 16) == 1953066613 )
    {
      v43 = v8 & v9;
      v9 = 0;
      a6 = 74;
      goto LABEL_72;
    }
    goto LABEL_71;
  }
LABEL_70:
  if ( (v11 & (unsigned __int8)v9) == 0 )
    goto LABEL_71;
  if ( !(*(_QWORD *)a1 ^ 0x695F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x656C626174756D6DLL)
    && *(_DWORD *)(a1 + 16) == 1887007839
    && *(_BYTE *)(a1 + 20) == 101 )
  {
    v43 = v11 & v9;
    v9 = 0;
    a6 = 75;
    goto LABEL_73;
  }
LABEL_72:
  if ( ((unsigned __int8)v9 & v44) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x665F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x616C5F74616D726FLL)
      && *(_WORD *)(a1 + 16) == 25954
      && *(_BYTE *)(a1 + 18) == 108 )
    {
      a6 = 16641;
      goto LABEL_106;
    }
    goto LABEL_74;
  }
LABEL_73:
  if ( ((unsigned __int8)v9 & (a2 == 24)) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x665F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6E6F6974636E75LL)
      && *(_QWORD *)(a1 + 16) == 0x6574616C706D6574LL )
    {
      a6 = 16642;
      goto LABEL_106;
    }
    goto LABEL_75;
  }
LABEL_74:
  if ( (v11 & (unsigned __int8)v9) != 0 )
  {
    if ( *(_QWORD *)a1 ^ 0x635F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6D65745F7373616CLL
      || *(_DWORD *)(a1 + 16) != 1952541808
      || *(_BYTE *)(a1 + 20) != 101 )
    {
      goto LABEL_77;
    }
    a6 = 16643;
LABEL_106:
    v43 = 1;
    v9 = 0;
    goto LABEL_107;
  }
LABEL_75:
  if ( v43 )
    goto LABEL_106;
  v9 = 1;
  if ( a2 == 34 )
  {
    if ( *(_QWORD *)a1 ^ 0x475F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6C706D65745F554ELL
      || *(_QWORD *)(a1 + 16) ^ 0x706D65745F657461LL | *(_QWORD *)(a1 + 24) ^ 0x7261705F6574616CLL
      || *(_WORD *)(a1 + 32) != 28001 )
    {
      if ( *(_QWORD *)a1 ^ 0x475F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6C706D65745F554ELL
        || *(_QWORD *)(a1 + 16) ^ 0x617261705F657461LL | *(_QWORD *)(a1 + 24) ^ 0x61705F726574656DLL
        || *(_WORD *)(a1 + 32) != 27491 )
      {
        v9 = 1;
LABEL_108:
        if ( v8 )
        {
          if ( !(*(_QWORD *)a1 ^ 0x475F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6C6C61635F554ELL)
            && *(_DWORD *)(a1 + 16) == 1702127987 )
          {
            v43 = v8;
            a6 = 16649;
            v9 = 0;
            goto LABEL_82;
          }
          goto LABEL_81;
        }
        goto LABEL_109;
      }
      a6 = 16647;
    }
    else
    {
      a6 = 16646;
    }
    v43 = 1;
    v9 = 0;
LABEL_109:
    v23 &= v9;
    if ( v23 )
      goto LABEL_110;
    goto LABEL_81;
  }
LABEL_77:
  if ( ((unsigned __int8)v9 & (a2 == 32)) == 0 )
  {
LABEL_107:
    v8 &= v9;
    goto LABEL_108;
  }
  if ( *(_QWORD *)a1 ^ 0x475F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x616D726F665F554ELL
    || *(_QWORD *)(a1 + 16) ^ 0x656D617261705F6CLL | *(_QWORD *)(a1 + 24) ^ 0x6B6361705F726574LL )
  {
    v9 &= a2 == 32;
    if ( !v23 )
      goto LABEL_81;
LABEL_110:
    if ( *(_QWORD *)a1 ^ 0x475F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x5F6C6C61635F554ELL
      || *(_QWORD *)(a1 + 16) != 0x7261705F65746973LL
      || *(_DWORD *)(a1 + 24) != 1952804193
      || *(_WORD *)(a1 + 28) != 29285 )
    {
      goto LABEL_82;
    }
    v43 = v23;
    a6 = 16650;
    v9 = 0;
LABEL_83:
    v32 = v9 & v28;
    if ( v32 )
    {
      if ( !(*(_QWORD *)a1 ^ 0x425F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x445F444E414C524FLL)
        && *(_QWORD *)(a1 + 16) == 0x74735F6968706C65LL
        && *(_DWORD *)(a1 + 24) == 1735289202 )
      {
        v43 = v32;
        a6 = 45057;
        v9 = 0;
        goto LABEL_86;
      }
      goto LABEL_85;
    }
    goto LABEL_84;
  }
  v43 = v9 & (a2 == 32);
  a6 = 16648;
  v9 = 0;
LABEL_81:
  v31 = v9 & v11;
  if ( v31 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x415F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F72705F454C5050LL)
      && *(_DWORD *)(a1 + 16) == 1953654128
      && *(_BYTE *)(a1 + 20) == 121 )
    {
      v43 = v31;
      v9 = 0;
      a6 = 16896;
      goto LABEL_84;
    }
    goto LABEL_83;
  }
LABEL_82:
  if ( ((unsigned __int8)v9 & v42) == 0 )
    goto LABEL_83;
  if ( !(*(_QWORD *)a1 ^ 0x425F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x705F444E414C524FLL)
    && *(_DWORD *)(a1 + 16) == 1701867378
    && *(_WORD *)(a1 + 20) == 29810
    && *(_BYTE *)(a1 + 22) == 121 )
  {
    v43 = v9 & v42;
    v9 = 0;
    a6 = 45056;
    goto LABEL_85;
  }
LABEL_84:
  if ( ((unsigned __int8)v9 & (a2 == 35)) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x425F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x445F444E414C524FLL)
      && !(*(_QWORD *)(a1 + 16) ^ 0x79645F6968706C65LL | *(_QWORD *)(a1 + 24) ^ 0x72615F63696D616ELL)
      && *(_WORD *)(a1 + 32) == 24946
      && *(_BYTE *)(a1 + 34) == 121 )
    {
      return 45058;
    }
    goto LABEL_86;
  }
LABEL_85:
  if ( ((unsigned __int8)v9 & v40) != 0 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x425F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x445F444E414C524FLL)
      && *(_QWORD *)(a1 + 16) == 0x65735F6968706C65LL
      && *(_BYTE *)(a1 + 24) == 116 )
    {
      return 45059;
    }
    goto LABEL_87;
  }
LABEL_86:
  if ( (v39 & (unsigned __int8)v9) == 0 )
  {
LABEL_87:
    if ( !v43 )
      return (unsigned int)-1;
    return a6;
  }
  if ( *(_QWORD *)a1 ^ 0x425F4741545F5744LL | *(_QWORD *)(a1 + 8) ^ 0x445F444E414C524FLL )
    return (unsigned int)-1;
  if ( *(_QWORD *)(a1 + 16) != 0x61765F6968706C65LL )
    return (unsigned int)-1;
  if ( *(_DWORD *)(a1 + 24) != 1851877746 )
    return (unsigned int)-1;
  a6 = 45060;
  if ( *(_BYTE *)(a1 + 28) != 116 )
    return (unsigned int)-1;
  return a6;
}
