// Function: sub_2034940
// Address: 0x2034940
//
__int64 __fastcall sub_2034940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned int v10; // r13d
  bool v11; // al
  char *v13; // rdx
  _QWORD *v14; // r12
  char v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rdx
  __int64 v19; // r8
  _QWORD *v20; // r12
  _BYTE v21[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  int v24; // [rsp+18h] [rbp-28h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = **(_QWORD **)(*(_QWORD *)(v7 + 80) + 32LL);
  if ( *(_WORD *)(v8 + 24) == 48 )
  {
    v13 = *(char **)(a2 + 40);
    v14 = *(_QWORD **)(a1 + 8);
    v15 = *v13;
    v16 = *((_QWORD *)v13 + 1);
    v21[0] = v15;
    v22 = v16;
    if ( v15 )
    {
      switch ( v15 )
      {
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 56:
        case 57:
        case 58:
        case 59:
        case 60:
        case 61:
          v17 = 2;
          break;
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
          v17 = 3;
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
          v17 = 4;
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
          v17 = 5;
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
          v17 = 6;
          break;
        case 55:
          v17 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v17 = 8;
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
          v17 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v17 = 10;
          break;
      }
      v19 = 0;
    }
    else
    {
      v17 = sub_1F596B0((__int64)v21);
      v19 = v18;
    }
    v23 = 0;
    v24 = 0;
    v20 = sub_1D2B300(v14, 0x30u, (__int64)&v23, v17, v19, a6);
    if ( v23 )
      sub_161E7C0((__int64)&v23, v23);
    return (__int64)v20;
  }
  else
  {
    v9 = *(_QWORD *)(v8 + 88);
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 <= 0x40 )
      v11 = *(_QWORD *)(v9 + 24) == 0;
    else
      v11 = v10 == (unsigned int)sub_16A57B0(v9 + 24);
    return sub_2032580(a1, *(_QWORD *)(v7 + 40LL * !v11), *(_QWORD *)(v7 + 40LL * !v11 + 8));
  }
}
