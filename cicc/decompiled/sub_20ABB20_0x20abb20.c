// Function: sub_20ABB20
// Address: 0x20abb20
//
__int64 __fastcall sub_20ABB20(_DWORD *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  _QWORD *v6; // r12
  __int16 v7; // ax
  __int64 v8; // rax
  unsigned int v9; // ecx
  char *v10; // rdx
  char v11; // al
  __int64 v12; // rdx
  bool v13; // bl
  int v14; // eax
  unsigned int v15; // ebx
  int v16; // eax
  _QWORD **v17; // rdi
  __int64 v19; // r14
  char *v20; // rdx
  char v21; // al
  __int64 v22; // r12
  unsigned int v23; // r15d
  unsigned int v24; // eax
  unsigned int v25; // ebx
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rdx
  _QWORD **v29; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+8h] [rbp-58h]
  _BYTE v31[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v32; // [rsp+18h] [rbp-48h]
  _QWORD **v33; // [rsp+20h] [rbp-40h] BYREF
  __int64 v34; // [rsp+28h] [rbp-38h]

  if ( !a2 )
  {
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  v7 = *(_WORD *)(a2 + 24);
  v30 = 1;
  v29 = 0;
  LOBYTE(v6) = v7 == 32 || v7 == 10;
  if ( (_BYTE)v6 )
  {
    v8 = *(_QWORD *)(a2 + 88);
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      v30 = *(_DWORD *)(v8 + 32);
      v29 = (_QWORD **)(*(_QWORD *)(v8 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9));
    }
    else
    {
      sub_16A51C0((__int64)&v29, v8 + 24);
    }
    goto LABEL_5;
  }
  if ( v7 != 104 )
    return (unsigned int)v6;
  LOBYTE(a3) = 0;
  v19 = sub_1D1AD70(a2, 0, a3, a4, a5, a6);
  if ( !v19 )
  {
    v24 = v30;
    goto LABEL_26;
  }
  v20 = *(char **)(a2 + 40);
  v21 = *v20;
  v22 = *((_QWORD *)v20 + 1);
  v31[0] = v21;
  v32 = v22;
  if ( v21 )
  {
    if ( (unsigned __int8)(v21 - 14) <= 0x5Fu )
    {
      switch ( v21 )
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
          LOBYTE(v33) = 3;
          v34 = 0;
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
          LOBYTE(v33) = 4;
          v34 = 0;
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
          LOBYTE(v33) = 5;
          v34 = 0;
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
          LOBYTE(v33) = 6;
          v34 = 0;
          break;
        case 55:
          LOBYTE(v33) = 7;
          v34 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v33) = 8;
          v34 = 0;
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
          LOBYTE(v33) = 9;
          v34 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v33) = 10;
          v34 = 0;
          break;
        default:
          LOBYTE(v33) = 2;
          v34 = 0;
          break;
      }
    }
    else
    {
      LOBYTE(v33) = v21;
      v34 = v22;
    }
    goto LABEL_18;
  }
  if ( !sub_1F58D20((__int64)v31) )
  {
    LOBYTE(v33) = 0;
    v34 = v22;
LABEL_36:
    v6 = &v33;
    v23 = sub_1F58D40((__int64)&v33);
    goto LABEL_37;
  }
  LOBYTE(v33) = sub_1F596B0((__int64)v31);
  v34 = v28;
  if ( !(_BYTE)v33 )
    goto LABEL_36;
LABEL_18:
  v6 = &v33;
  v23 = sub_1F3E310(&v33);
LABEL_37:
  v26 = *(_QWORD *)(v19 + 88);
  v27 = *(_DWORD *)(v26 + 32);
  if ( v27 <= 0x40 )
  {
    v30 = *(_DWORD *)(v26 + 32);
    v29 = (_QWORD **)(*(_QWORD *)(v26 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27));
  }
  else
  {
    sub_16A51C0((__int64)&v29, v26 + 24);
    v27 = v30;
  }
  if ( v23 < v27 )
  {
    sub_16A5A50((__int64)&v33, (__int64 *)&v29, v23);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    v29 = v33;
    v30 = v34;
  }
LABEL_5:
  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOBYTE(v33) = v11;
  v34 = v12;
  if ( !v11 )
  {
    v6 = &v33;
    v13 = sub_1F58CD0((__int64)&v33);
    if ( !sub_1F58D20((__int64)&v33) )
      goto LABEL_8;
LABEL_20:
    v14 = a1[17];
    goto LABEL_21;
  }
  if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
    goto LABEL_20;
  v13 = (unsigned __int8)(v11 - 86) <= 0x17u || (unsigned __int8)(v11 - 8) <= 5u;
LABEL_8:
  if ( v13 )
  {
    v14 = a1[16];
    if ( v14 == 1 )
      goto LABEL_10;
    goto LABEL_22;
  }
  v14 = a1[15];
LABEL_21:
  if ( v14 != 1 )
  {
LABEL_22:
    if ( v14 == 2 )
    {
      v25 = v30;
      if ( v30 <= 0x40 )
      {
        LOBYTE(v6) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v30) == (_QWORD)v29;
        return (unsigned int)v6;
      }
      LOBYTE(v6) = v25 == (unsigned int)sub_16A58F0((__int64)&v29);
LABEL_27:
      v17 = v29;
      if ( !v29 )
        return (unsigned int)v6;
LABEL_28:
      j_j___libc_free_0_0(v17);
      return (unsigned int)v6;
    }
    v24 = v30;
    LOBYTE(v6) = (_BYTE)v29;
    if ( v30 > 0x40 )
      v6 = *v29;
    LODWORD(v6) = (unsigned __int8)v6 & 1;
LABEL_26:
    if ( v24 <= 0x40 )
      return (unsigned int)v6;
    goto LABEL_27;
  }
LABEL_10:
  v15 = v30;
  if ( v30 <= 0x40 )
  {
    LOBYTE(v6) = v29 == (_QWORD **)1;
    return (unsigned int)v6;
  }
  v16 = sub_16A57B0((__int64)&v29);
  v17 = v29;
  LOBYTE(v6) = v15 - 1 == v16;
  if ( v29 )
    goto LABEL_28;
  return (unsigned int)v6;
}
