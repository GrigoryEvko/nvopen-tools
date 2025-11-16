// Function: sub_1D1AF30
// Address: 0x1d1af30
//
__int64 __fastcall sub_1D1AF30(__int64 a1, unsigned int a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rsi
  char v9; // di
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int v16; // eax
  unsigned int v17; // r13d
  unsigned __int64 v18; // rbx
  unsigned int v20; // eax
  char v21[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v22; // [rsp+8h] [rbp-38h]
  char v23[8]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v24; // [rsp+18h] [rbp-28h]

  v6 = sub_1D1ADA0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 48LL), a3, a4, a5, a6);
  if ( !v6 )
    return 0;
  v7 = *(_QWORD *)(v6 + 88);
  v8 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v11 = v7 + 24;
  v21[0] = v9;
  v22 = v10;
  if ( v9 )
  {
    if ( (unsigned __int8)(v9 - 14) <= 0x5Fu )
    {
      switch ( v9 )
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
          v9 = 3;
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
          v9 = 4;
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
          v9 = 5;
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
          v9 = 6;
          break;
        case 55:
          v9 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v9 = 8;
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
          v9 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v9 = 10;
          break;
        default:
          v9 = 2;
          break;
      }
    }
  }
  else
  {
    if ( !(unsigned __int8)sub_1F58D20(v21) )
    {
      v23[0] = 0;
      v24 = v10;
      goto LABEL_5;
    }
    v23[0] = sub_1F596B0(v21);
    v9 = v23[0];
    v24 = v12;
    if ( !v23[0] )
    {
LABEL_5:
      v16 = sub_1F58D40(v23, v8, v12, v13, v14, v15);
      v17 = *(_DWORD *)(v7 + 32);
      v18 = v16;
      if ( v17 <= 0x40 )
        goto LABEL_6;
LABEL_11:
      if ( v17 - (unsigned int)sub_16A57B0(v7 + 24) > 0x40 || v18 <= **(_QWORD **)(v7 + 24) )
        return 0;
      return v11;
    }
  }
  v20 = sub_1D13440(v9);
  v17 = *(_DWORD *)(v7 + 32);
  v18 = v20;
  if ( v17 > 0x40 )
    goto LABEL_11;
LABEL_6:
  if ( v18 <= *(_QWORD *)(v7 + 24) )
    return 0;
  return v11;
}
