// Function: sub_1F709E0
// Address: 0x1f709e0
//
__int64 __fastcall sub_1F709E0(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  _DWORD *v3; // rax
  __int64 v4; // rax
  char v5; // di
  __int64 v6; // r13
  __int64 v7; // rdx
  int v8; // ecx
  int v9; // r8d
  int v10; // r9d
  int v11; // r13d
  __int64 v12; // rdi
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // ebx
  __int64 v18; // rdx
  char v19[8]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+8h] [rbp-48h]
  char v21[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]

  v2 = a1;
  if ( *(_WORD *)(a1 + 24) == 158 )
  {
    do
    {
      v3 = *(_DWORD **)(v2 + 32);
      v2 = *(_QWORD *)v3;
    }
    while ( *(_WORD *)(*(_QWORD *)v3 + 24LL) == 158 );
    a2 = v3[2];
  }
  v4 = *(_QWORD *)(v2 + 40) + 16LL * a2;
  v5 = *(_BYTE *)v4;
  v6 = *(_QWORD *)(v4 + 8);
  v19[0] = v5;
  v20 = v6;
  if ( v5 )
  {
    if ( (unsigned __int8)(v5 - 14) <= 0x5Fu )
    {
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
          v5 = 3;
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
          v5 = 4;
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
          v5 = 5;
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
          v5 = 6;
          break;
        case 55:
          v5 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v5 = 8;
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
          v5 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v5 = 10;
          break;
        default:
          v5 = 2;
          break;
      }
    }
    goto LABEL_6;
  }
  if ( sub_1F58D20((__int64)v19) )
  {
    v21[0] = sub_1F596B0((__int64)v19);
    v5 = v21[0];
    v22 = v18;
    if ( v21[0] )
    {
LABEL_6:
      v11 = sub_1F6C8D0(v5);
      goto LABEL_7;
    }
  }
  else
  {
    v21[0] = 0;
    v22 = v6;
  }
  v11 = sub_1F58D40((__int64)v21);
LABEL_7:
  v12 = v2;
  v13 = 0;
  v14 = sub_1D1ADA0(v12, a2, v7, v8, v9, v10);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 88);
    v16 = *(_DWORD *)(v15 + 32);
    if ( v16 <= 0x40 )
    {
      if ( *(_QWORD *)(v15 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) )
        return v13;
      goto LABEL_10;
    }
    if ( v16 == (unsigned int)sub_16A58F0(v15 + 24) )
LABEL_10:
      LOBYTE(v13) = v11 == v16;
  }
  return v13;
}
