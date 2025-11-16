// Function: sub_1D19A30
// Address: 0x1d19a30
//
__int64 __fastcall sub_1D19A30(__int64 a1, __int64 a2, _QWORD *a3)
{
  char v5; // al
  __int64 v6; // r14
  unsigned __int8 v7; // cl
  char v8; // al
  __int64 v9; // rdx
  __int64 v11; // rdx
  unsigned __int8 v12; // si
  unsigned __int8 v13; // di
  __int64 v14; // rbx
  unsigned int v15; // r15d
  int v16; // ecx
  __int64 v17; // rcx
  unsigned int v18; // [rsp+8h] [rbp-58h]
  unsigned int v19; // [rsp+8h] [rbp-58h]
  _BYTE v20[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+20h] [rbp-40h]

  v5 = *(_BYTE *)a3;
  v6 = *(_QWORD *)(a2 + 48);
  if ( *(_BYTE *)a3 )
  {
    v7 = v5 - 14;
    if ( (unsigned __int8)(v5 - 14) > 0x5Fu )
    {
LABEL_3:
      sub_1F40D10(v20, *(_QWORD *)(a2 + 16), v6, *a3, a3[1]);
      v8 = v20[8];
      v9 = v21;
      goto LABEL_4;
    }
    switch ( v5 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v13 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v13 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v13 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v13 = 6;
        break;
      case 55:
        v13 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v13 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v13 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v13 = 10;
        break;
      default:
        v13 = 2;
        break;
    }
    v11 = 0;
  }
  else
  {
    if ( !(unsigned __int8)sub_1F58D20(a3) )
      goto LABEL_3;
    v12 = sub_1F596B0(a3);
    v13 = v12;
    v5 = *(_BYTE *)a3;
    v14 = v11;
    if ( !*(_BYTE *)a3 )
    {
      v15 = v12;
      v16 = (unsigned int)sub_1F58D30(a3) >> 1;
      v13 = v12;
LABEL_9:
      v18 = v16;
      v8 = sub_1D15020(v13, v16);
      v17 = v18;
      goto LABEL_10;
    }
    v7 = v5 - 14;
  }
  v14 = v11;
  v15 = v13;
  v16 = word_42E7700[v7] >> 1;
  if ( (unsigned __int8)(v5 - 56) > 0x1Du && (unsigned __int8)(v5 - 98) > 0xBu )
    goto LABEL_9;
  v19 = v16;
  v8 = sub_1D154A0(v13, v16);
  v17 = v19;
LABEL_10:
  v9 = 0;
  if ( !v8 )
    v8 = sub_1F593D0(v6, v15, v14, v17);
LABEL_4:
  *(_BYTE *)a1 = v8;
  *(_BYTE *)(a1 + 16) = v8;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 24) = v9;
  return a1;
}
