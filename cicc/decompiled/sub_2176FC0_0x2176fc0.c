// Function: sub_2176FC0
// Address: 0x2176fc0
//
__int64 __fastcall sub_2176FC0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdx
  __int16 v26; // ax
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rdx

  if ( a3 > 0x1052 )
  {
    if ( a3 > 0x1190 )
    {
      if ( a3 - 5304 <= 0x8F )
      {
        *(_DWORD *)a1 = 44;
        *(_DWORD *)(a1 + 32) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        v10 = *(_QWORD *)a2;
        v11 = sub_15F2050(a2);
        v12 = sub_1632FA0(v11);
        LOBYTE(v13) = sub_21719A0(v12, v10);
        *(_DWORD *)(a1 + 40) = 0;
        *(_DWORD *)(a1 + 8) = v13;
        *(_WORD *)(a1 + 44) = 5;
        *(_QWORD *)(a1 + 16) = v14;
        return 1;
      }
    }
    else
    {
      if ( a3 > 0x118E )
      {
        *(_DWORD *)a1 = 44;
        *(_BYTE *)(a1 + 8) = 5;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        *(_DWORD *)(a1 + 32) = 0;
        *(_WORD *)(a1 + 44) = 1;
        *(_DWORD *)(a1 + 40) = 0;
        return 1;
      }
      if ( a3 == 4434 || (a3 & 0xFFFFFFDF) == 0x1148 )
      {
        *(_DWORD *)a1 = 44;
        v5 = **(_QWORD **)(*(_QWORD *)a2 + 16LL);
        v6 = sub_15F2050(a2);
        v7 = sub_1632FA0(v6);
        LOBYTE(v8) = sub_21719A0(v7, v5);
        *(_QWORD *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 16) = v9;
        *(_DWORD *)(a1 + 8) = v8;
        *(_DWORD *)(a1 + 32) = 0;
        *(_WORD *)(a1 + 44) = 1;
        *(_DWORD *)(a1 + 40) = 16;
        return 1;
      }
    }
    return 0;
  }
  if ( a3 <= 0xF79 )
  {
    switch ( a3 )
    {
      case 0xE55u:
      case 0xE74u:
        *(_DWORD *)a1 = 44;
        *(_BYTE *)(a1 + 8) = 7;
        goto LABEL_17;
      case 0xE56u:
        *(_DWORD *)a1 = 44;
        *(_BYTE *)(a1 + 8) = 4;
        goto LABEL_17;
      case 0xE57u:
      case 0xE75u:
        goto LABEL_20;
      case 0xE58u:
      case 0xE76u:
        goto LABEL_16;
      case 0xE72u:
        goto LABEL_23;
      case 0xE73u:
        goto LABEL_24;
      case 0xE79u:
      case 0xE7Au:
        *(_DWORD *)a1 = 44;
        *(_BYTE *)(a1 + 8) = (a3 != 3705) + 90;
        goto LABEL_17;
      case 0xEA5u:
      case 0xEA6u:
      case 0xEA7u:
        goto LABEL_13;
      case 0xEA9u:
        goto LABEL_15;
      default:
        return 0;
    }
  }
  switch ( a3 )
  {
    case 0xF7Au:
    case 0xF7Bu:
    case 0xF7Cu:
    case 0xF7Du:
    case 0xF88u:
    case 0xF89u:
    case 0xF8Au:
    case 0xF8Bu:
    case 0xF92u:
    case 0xF93u:
    case 0xF94u:
    case 0xF95u:
    case 0xFA6u:
    case 0xFA7u:
    case 0xFA8u:
    case 0xFA9u:
    case 0xFAAu:
    case 0xFAEu:
    case 0xFAFu:
    case 0xFB0u:
    case 0xFB1u:
    case 0xFB2u:
    case 0xFB6u:
    case 0xFB7u:
    case 0xFB8u:
    case 0xFB9u:
    case 0xFBAu:
    case 0xFBEu:
    case 0xFBFu:
    case 0xFC0u:
    case 0xFC1u:
    case 0xFC2u:
LABEL_13:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 5;
      *(_QWORD *)(a1 + 16) = 0;
      v15 = -3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      goto LABEL_14;
    case 0xF82u:
    case 0xF83u:
    case 0xF90u:
    case 0xF91u:
    case 0xF9Au:
    case 0xF9Bu:
    case 0xFADu:
    case 0xFB5u:
    case 0xFBDu:
    case 0xFC5u:
LABEL_15:
      *(_DWORD *)a1 = 45;
      *(_BYTE *)(a1 + 8) = 5;
      *(_QWORD *)(a1 + 16) = 0;
      v17 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      *(_DWORD *)(a1 + 32) = 0;
      *(_WORD *)(a1 + 44) = 2;
      *(_QWORD *)(a1 + 24) = v17;
      *(_DWORD *)(a1 + 40) = 0;
      return 1;
    case 0xFC6u:
LABEL_23:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 9;
      goto LABEL_17;
    case 0xFC7u:
LABEL_24:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 10;
      goto LABEL_17;
    case 0xFC8u:
LABEL_20:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 5;
      goto LABEL_17;
    case 0xFC9u:
LABEL_16:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 6;
LABEL_17:
      *(_QWORD *)(a1 + 16) = 0;
      v18 = 3 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      goto LABEL_18;
    case 0xFFFu:
    case 0x1000u:
    case 0x1001u:
      *(_DWORD *)a1 = 44;
      *(_BYTE *)(a1 + 8) = 5;
      *(_QWORD *)(a1 + 16) = 0;
      v15 = 3 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
LABEL_14:
      v16 = *(_QWORD *)(a2 + 8 * v15);
      *(_DWORD *)(a1 + 32) = 0;
      *(_WORD *)(a1 + 44) = 1;
      *(_QWORD *)(a1 + 24) = v16;
      *(_DWORD *)(a1 + 40) = 0;
      return 1;
    case 0x1043u:
    case 0x1046u:
      *(_DWORD *)a1 = 44;
      v27 = *(_QWORD *)(**(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) + 24LL);
      v28 = sub_15F2050(a2);
      v29 = sub_1632FA0(v28);
      LOBYTE(v30) = sub_21719A0(v29, v27);
      *(_DWORD *)(a1 + 8) = v30;
      *(_QWORD *)(a1 + 16) = v31;
      v18 = 3 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
LABEL_18:
      v19 = *(_QWORD *)(a2 + 8 * v18);
      *(_DWORD *)(a1 + 32) = 0;
      *(_WORD *)(a1 + 44) = 3;
      *(_QWORD *)(a1 + 24) = v19;
      *(_DWORD *)(a1 + 40) = 0;
      result = 1;
      break;
    case 0x104Fu:
    case 0x1052u:
      *(_DWORD *)a1 = 44;
      *(_DWORD *)(a1 + 32) = 0;
      if ( a3 == 4175 )
      {
        *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        v32 = *(_QWORD *)a2;
        v33 = sub_15F2050(a2);
        v34 = sub_1632FA0(v33);
        LOBYTE(v35) = sub_21719A0(v34, v32);
        v24 = 1;
        *(_DWORD *)(a1 + 8) = v35;
        v26 = 1;
        *(_QWORD *)(a1 + 16) = v36;
      }
      else
      {
        *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        v20 = **(_QWORD **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        v21 = sub_15F2050(a2);
        v22 = sub_1632FA0(v21);
        LOBYTE(v23) = sub_21719A0(v22, v20);
        v24 = 2;
        *(_QWORD *)(a1 + 16) = v25;
        *(_DWORD *)(a1 + 8) = v23;
        v26 = 2;
      }
      *(_WORD *)(a1 + 44) = v26;
      *(_DWORD *)(a1 + 40) = sub_15603A0((_QWORD *)(a2 + 56), v24);
      result = 1;
      break;
    default:
      return 0;
  }
  return result;
}
