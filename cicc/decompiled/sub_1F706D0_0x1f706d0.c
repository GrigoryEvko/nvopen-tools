// Function: sub_1F706D0
// Address: 0x1f706d0
//
__int64 __fastcall sub_1F706D0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  char v4; // di
  __int64 v5; // rbx
  __int64 v6; // rdx
  int v7; // ecx
  int v8; // ebx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rdi
  unsigned int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned int v15; // r13d
  __int64 v17; // rdx
  char v18[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h]
  char v20[8]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v21; // [rsp+18h] [rbp-28h]

  v3 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v4 = *(_BYTE *)v3;
  v5 = *(_QWORD *)(v3 + 8);
  v18[0] = v4;
  v19 = v5;
  if ( v4 )
  {
    if ( (unsigned __int8)(v4 - 14) <= 0x5Fu )
    {
      switch ( v4 )
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
          v4 = 3;
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
          v4 = 4;
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
          v4 = 5;
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
          v4 = 6;
          break;
        case 55:
          v4 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v4 = 8;
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
          v4 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v4 = 10;
          break;
        default:
          v4 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  if ( sub_1F58D20((__int64)v18) )
  {
    v20[0] = sub_1F596B0((__int64)v18);
    v4 = v20[0];
    v21 = v17;
    if ( v20[0] )
    {
LABEL_3:
      v8 = sub_1F6C8D0(v4);
      goto LABEL_4;
    }
  }
  else
  {
    v20[0] = 0;
    v21 = v5;
  }
  v8 = sub_1F58D40((__int64)v20);
LABEL_4:
  v11 = a1;
  v12 = 0;
  v13 = sub_1D1ADA0(v11, a2, v6, v7, v9, v10);
  if ( !v13 )
    return v12;
  v14 = *(_QWORD *)(v13 + 88);
  v15 = *(_DWORD *)(v14 + 32);
  if ( v15 > 0x40 )
  {
    if ( (unsigned int)sub_16A57B0(v14 + 24) == v15 - 1 )
LABEL_7:
      LOBYTE(v12) = v8 == v15;
    return v12;
  }
  if ( *(_QWORD *)(v14 + 24) == 1 )
    goto LABEL_7;
  return 0;
}
