// Function: sub_1D16BF0
// Address: 0x1d16bf0
//
char __fastcall sub_1D16BF0(__int64 a1, unsigned int a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // rsi
  __int64 v6; // rcx
  char v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  int v11; // eax
  char result; // al
  __int64 v13; // rdx
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // r15
  _QWORD *v17; // r9
  __int64 v18; // rcx
  int v19; // eax
  unsigned __int8 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r10
  char v23; // al
  char v24; // al
  __int64 v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-78h]
  _QWORD *v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+28h] [rbp-58h] BYREF
  __int64 v31; // [rsp+30h] [rbp-50h] BYREF
  __int64 v32; // [rsp+38h] [rbp-48h] BYREF
  _QWORD v33[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v6 = *(_QWORD *)(a3 + 40) + 16LL * a4;
  v7 = *(_BYTE *)v5;
  if ( *(_BYTE *)v5 != *(_BYTE *)v6 )
    return 0;
  v8 = *(_QWORD *)(v5 + 8);
  v9 = a1;
  if ( *(_QWORD *)(v6 + 8) != v8 && !v7 )
    return 0;
  v11 = *(unsigned __int16 *)(a1 + 24);
  if ( v11 == 32 || v11 == 10 )
  {
    v13 = *(unsigned __int16 *)(a3 + 24);
    if ( (_DWORD)v13 == 32 || (_DWORD)v13 == 10 )
    {
      v14 = *(_QWORD *)(a5 + 16) == 0;
      v30 = a1;
      v31 = a3;
      if ( v14 )
LABEL_45:
        sub_4263D6(a1, v5, v13);
      return (*(__int64 (__fastcall **)(__int64, __int64 *, __int64 *))(a5 + 24))(a5, &v30, &v31);
    }
  }
  if ( v11 != 104 || *(_WORD *)(a3 + 24) != 104 )
    return 0;
  LOBYTE(v33[0]) = *(_BYTE *)v5;
  v33[1] = v8;
  if ( v7 )
  {
    if ( (unsigned __int8)(v7 - 14) <= 0x5Fu )
    {
      switch ( v7 )
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
          v7 = 3;
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
          v7 = 4;
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
          v7 = 5;
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
          v7 = 6;
          break;
        case 55:
          v7 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v7 = 8;
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
          v7 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v7 = 10;
          break;
        default:
          v7 = 2;
          break;
      }
      v8 = 0;
    }
  }
  else
  {
    v29 = a5;
    v23 = sub_1F58D20(v33);
    a5 = v29;
    if ( v23 )
    {
      v24 = sub_1F596B0(v33);
      a5 = v29;
      v7 = v24;
      v8 = v25;
    }
  }
  v15 = *(unsigned int *)(a1 + 56);
  if ( !(_DWORD)v15 )
    return 1;
  v16 = 0;
  v17 = v33;
  v18 = 40 * v15;
  while ( 1 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + v16);
    result = *(_WORD *)(v13 + 24) == 10 || *(_WORD *)(v13 + 24) == 32;
    if ( !result )
      return result;
    a1 = *(_QWORD *)(v16 + *(_QWORD *)(a3 + 32));
    v19 = *(unsigned __int16 *)(a1 + 24);
    if ( v19 != 10 && v19 != 32 )
      return 0;
    v20 = *(unsigned __int8 **)(v13 + 40);
    v5 = *v20;
    if ( v7 != (_BYTE)v5 )
      return 0;
    v21 = *((_QWORD *)v20 + 1);
    if ( v8 != v21 && !v7 )
      return 0;
    v22 = *(_QWORD *)(a1 + 40);
    if ( (_BYTE)v5 != *(_BYTE *)v22 || *(_QWORD *)(v22 + 8) != v21 && !v7 )
      return 0;
    v14 = *(_QWORD *)(a5 + 16) == 0;
    v26 = v18;
    v32 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + v16);
    v33[0] = a1;
    if ( v14 )
      goto LABEL_45;
    v27 = v17;
    v28 = a5;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *, _QWORD *))(a5 + 24))(a5, &v32, v17) )
      return 0;
    v18 = v26;
    v16 += 40;
    a5 = v28;
    v17 = v27;
    if ( v16 == v26 )
      return 1;
  }
}
