// Function: sub_1D16620
// Address: 0x1d16620
//
__int64 __fastcall sub_1D16620(__int64 a1, __int64 *_RSI)
{
  __int64 v2; // rbx
  int i; // eax
  __int64 *v4; // r12
  __int64 *v5; // r13
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // rdx
  unsigned int v9; // r15d
  int v10; // eax
  unsigned int v12; // eax
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // eax
  __int64 v22; // rdi
  unsigned int v24; // edx
  __int64 v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned int v28; // [rsp+8h] [rbp-58h]
  char v29[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h] BYREF
  __int64 v32; // [rsp+28h] [rbp-38h]

  v2 = a1;
  for ( i = *(unsigned __int16 *)(a1 + 24); i == 158; i = *(unsigned __int16 *)(v2 + 24) )
    v2 = **(_QWORD **)(v2 + 32);
  if ( i != 104 )
    return 0;
  v4 = *(__int64 **)(v2 + 32);
  v5 = &v4[5 * *(unsigned int *)(v2 + 56)];
  if ( v4 == v5 )
    return 0;
  v6 = 1;
  do
  {
    if ( *(_WORD *)(*v4 + 24) == 48 )
      goto LABEL_20;
    v14 = *(_QWORD *)(v2 + 40);
    v15 = *(_BYTE *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    v29[0] = v15;
    v30 = v16;
    if ( v15 )
    {
      if ( (unsigned __int8)(v15 - 14) <= 0x5Fu )
      {
        switch ( v15 )
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
            v15 = 3;
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
            v15 = 4;
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
            v15 = 5;
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
            v15 = 6;
            break;
          case 55:
            v15 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v15 = 8;
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
            v15 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v15 = 10;
            break;
          default:
            v15 = 2;
            break;
        }
      }
    }
    else
    {
      if ( !(unsigned __int8)sub_1F58D20(v29) )
      {
        LOBYTE(v31) = 0;
        v32 = v16;
LABEL_25:
        v21 = sub_1F58D40(&v31, _RSI, v17, v18, v19, v20);
        v8 = *v4;
        v9 = v21;
        v10 = *(unsigned __int16 *)(*v4 + 24);
        if ( v10 == 32 )
          goto LABEL_26;
        goto LABEL_8;
      }
      LOBYTE(v31) = sub_1F596B0(v29);
      v15 = v31;
      v32 = v17;
      if ( !(_BYTE)v31 )
        goto LABEL_25;
    }
    v7 = sub_1D13440(v15);
    v8 = *v4;
    v9 = v7;
    v10 = *(unsigned __int16 *)(*v4 + 24);
    if ( v10 == 32 )
      goto LABEL_26;
LABEL_8:
    if ( v10 != 10 )
    {
      if ( v10 != 33 && v10 != 11 )
        return 0;
      _RSI = (__int64 *)(*(_QWORD *)(v8 + 88) + 32LL);
      if ( (void *)*_RSI == sub_16982C0() )
        sub_169D930((__int64)&v31, (__int64)_RSI);
      else
        sub_169D7E0((__int64)&v31, _RSI);
      _R8 = v31;
      if ( (unsigned int)v32 > 0x40 )
      {
        v27 = v31;
        v12 = sub_16A58A0((__int64)&v31);
        if ( v27 )
        {
          v26 = v27;
          v28 = v12;
          j_j___libc_free_0_0(v26);
          v12 = v28;
        }
      }
      else
      {
        v12 = 64;
        __asm { tzcnt   rcx, r8 }
        if ( v31 )
          v12 = _RCX;
        if ( v12 > (unsigned int)v32 )
          v12 = v32;
      }
      goto LABEL_18;
    }
LABEL_26:
    v22 = *(_QWORD *)(v8 + 88);
    v12 = *(_DWORD *)(v22 + 32);
    if ( v12 > 0x40 )
    {
      v12 = sub_16A58A0(v22 + 24);
    }
    else
    {
      _RCX = *(_QWORD *)(v22 + 24);
      v24 = 64;
      __asm { tzcnt   rsi, rcx }
      if ( _RCX )
        v24 = (unsigned int)_RSI;
      if ( v12 > v24 )
        v12 = v24;
    }
LABEL_18:
    if ( v9 > v12 )
      return 0;
    v6 = 0;
LABEL_20:
    v4 += 5;
  }
  while ( v5 != v4 );
  return v6 ^ 1u;
}
