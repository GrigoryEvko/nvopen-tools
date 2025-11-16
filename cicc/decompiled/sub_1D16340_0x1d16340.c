// Function: sub_1D16340
// Address: 0x1d16340
//
__int64 __fastcall sub_1D16340(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int i; // eax
  int v4; // r15d
  _DWORD *v5; // rax
  int v6; // r12d
  __int64 v7; // r14
  __int16 v8; // r13
  __int64 v9; // rax
  char v10; // di
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  void *v18; // rax
  __int64 *v19; // rsi
  unsigned int v20; // r13d
  __int64 v21; // rax
  bool v24; // zf
  unsigned int v25; // eax
  unsigned int v26; // r12d
  _DWORD *v27; // rdx
  char v28; // al
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  unsigned int v33; // [rsp+10h] [rbp-60h]
  int v34; // [rsp+1Ch] [rbp-54h]
  char v35[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h] BYREF
  __int64 v38; // [rsp+38h] [rbp-38h]

  v2 = a1;
  for ( i = *(unsigned __int16 *)(a1 + 24); i == 158; i = *(unsigned __int16 *)(v2 + 24) )
    v2 = **(_QWORD **)(v2 + 32);
  if ( i != 104 )
    return 0;
  v4 = *(_DWORD *)(v2 + 56);
  if ( !v4 )
    return 0;
  v5 = *(_DWORD **)(v2 + 32);
  v6 = 0;
  while ( 1 )
  {
    v7 = *(_QWORD *)v5;
    v8 = *(_WORD *)(*(_QWORD *)v5 + 24LL);
    if ( v8 != 48 )
      break;
    ++v6;
    v5 += 10;
    if ( v4 == v6 )
      return 0;
  }
  v34 = v5[2];
  v9 = *(_QWORD *)(v2 + 40);
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v35[0] = v10;
  v36 = v11;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) <= 0x5Fu )
    {
      switch ( v10 )
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
          v10 = 3;
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
          v10 = 4;
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
          v10 = 5;
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
          v10 = 6;
          break;
        case 55:
          v10 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v10 = 8;
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
          v10 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v10 = 10;
          break;
        default:
          v10 = 2;
          break;
      }
    }
    goto LABEL_18;
  }
  v31 = v11;
  v12 = sub_1F58D20(v35);
  v16 = v31;
  if ( v12 )
  {
    v28 = sub_1F596B0(v35);
    v8 = *(_WORD *)(v7 + 24);
    v10 = v28;
    LOBYTE(v37) = v28;
    v38 = v16;
    if ( !v28 )
      goto LABEL_11;
LABEL_18:
    v33 = sub_1D13440(v10);
    goto LABEL_12;
  }
  LOBYTE(v37) = 0;
  v38 = v31;
LABEL_11:
  v33 = sub_1F58D40(&v37, a2, v16, v13, v14, v15);
LABEL_12:
  if ( v8 == 32 || v8 == 10 )
  {
    v21 = *(_QWORD *)(v7 + 88);
    if ( *(_DWORD *)(v21 + 32) > 0x40u )
    {
      v25 = sub_16A58F0(v21 + 24);
    }
    else
    {
      _RAX = ~*(_QWORD *)(v21 + 24);
      __asm { tzcnt   rdx, rax }
      v24 = _RAX == 0;
      v25 = 64;
      if ( !v24 )
        v25 = _RDX;
    }
    if ( v33 > v25 )
      return 0;
  }
  else
  {
    if ( v8 != 11 && v8 != 33 )
      return 0;
    v18 = sub_16982C0();
    v19 = (__int64 *)(*(_QWORD *)(v7 + 88) + 32LL);
    if ( (void *)*v19 == v18 )
      sub_169D930((__int64)&v37, (__int64)v19);
    else
      sub_169D7E0((__int64)&v37, v19);
    if ( (unsigned int)v38 <= 0x40 )
    {
      _R9 = ~v37;
      v20 = 64;
      __asm { tzcnt   rax, r9 }
      if ( v37 != -1 )
        v20 = _RAX;
    }
    else
    {
      v32 = v37;
      v20 = sub_16A58F0((__int64)&v37);
      if ( v32 )
        j_j___libc_free_0_0(v32);
    }
    if ( v33 > v20 )
      return 0;
  }
  v26 = v6 + 1;
  if ( v4 != v26 )
  {
    while ( 1 )
    {
      v27 = (_DWORD *)(*(_QWORD *)(v2 + 32) + 40LL * v26);
      if ( (v7 != *(_QWORD *)v27 || v27[2] != v34) && *(_WORD *)(*(_QWORD *)v27 + 24LL) != 48 )
        break;
      if ( v4 == ++v26 )
        return 1;
    }
    return 0;
  }
  return 1;
}
