// Function: sub_2177670
// Address: 0x2177670
//
char __fastcall sub_2177670(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4)
{
  _QWORD *v5; // r12
  int v6; // ecx
  __int64 *v7; // rax
  int v8; // ecx
  char result; // al
  __int64 *v10; // rax
  __int64 v11; // rbx
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  __int64 v15; // rsi
  char v16; // al
  __int64 v17; // rdx
  char v18; // di
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // r14
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdx
  _QWORD v28[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v29[6]; // [rsp+20h] [rbp-30h] BYREF

  while ( 1 )
  {
    v5 = a4;
    v6 = *(__int16 *)(a1 + 24);
    if ( *(__int16 *)(a1 + 24) < 0 )
      break;
    if ( (_WORD)v6 == 118 )
    {
      v10 = *(__int64 **)(a1 + 32);
      v11 = *v10;
      v12 = *(unsigned __int16 *)(*v10 + 24);
      if ( v12 == 32 || v12 == 10 )
      {
        v14 = v10[5];
        v15 = *((unsigned int *)v10 + 12);
      }
      else
      {
        v11 = v10[5];
        v13 = *(unsigned __int16 *)(v11 + 24);
        if ( v13 != 10 && v13 != 32 )
          return 0;
        v14 = *v10;
        v15 = *((unsigned int *)v10 + 2);
      }
      v24 = *v5;
      if ( (unsigned __int8)sub_2177670(v14, v15, a3, v5) )
      {
        v25 = *(_QWORD *)(v11 + 88);
        v26 = *(_QWORD *)(v25 + 24);
        if ( *(_DWORD *)(v25 + 32) > 0x40u )
          v26 = *(_QWORD *)v26;
        v27 = v26 + 1;
        if ( v26 != -1 && (v26 & v27) == 0 )
        {
          _BitScanReverse64(&v27, v27);
          return *v5 - v24 == 63 - ((unsigned int)v27 ^ 0x3F);
        }
      }
      return 0;
    }
    if ( (unsigned __int16)(v6 - 142) > 2u )
    {
      if ( (unsigned int)(v6 - 659) > 5 )
        return 0;
      if ( *a3 )
      {
        if ( *a3 != a1 )
          return 0;
      }
      else
      {
        *a3 = a1;
      }
      v16 = *(_BYTE *)(a1 + 88);
      v17 = *(_QWORD *)(a1 + 96);
      LOBYTE(v28[0]) = v16;
      v28[1] = v17;
      if ( v16 )
      {
        switch ( v16 )
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
            v18 = 2;
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
            v18 = 3;
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
            v18 = 4;
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
            v18 = 5;
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
            v18 = 6;
            break;
          case 55:
            v18 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v18 = 8;
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
            v18 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v18 = 10;
            break;
        }
      }
      else
      {
        LOBYTE(v29[0]) = sub_1F596B0((__int64)v28);
        v18 = v29[0];
        v29[1] = v19;
        if ( !LOBYTE(v29[0]) )
        {
          v20 = sub_1F58D40((__int64)v29);
          goto LABEL_29;
        }
      }
      v20 = sub_216FFF0(v18);
LABEL_29:
      v21 = v20 * (unsigned int)a2;
      if ( v21 == *v5 )
      {
        *v5 = v20 + v21;
        return 1;
      }
      return 0;
    }
LABEL_4:
    v7 = *(__int64 **)(a1 + 32);
    a4 = v5;
    a1 = *v7;
    a2 = v7[1];
  }
  v8 = ~v6;
  if ( v8 > 316 )
  {
    if ( v8 > 620 )
    {
      result = 0;
      if ( v8 != 3243 )
        return result;
LABEL_8:
      if ( (unsigned __int8)sub_216FEC0(a1, v28, v29) )
      {
        v22 = *v5;
        if ( (unsigned __int8)sub_2177670(
                                *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a1 + 32) + 48LL),
                                a3,
                                v5) )
        {
          v23 = v28[0] + v22;
          if ( *v5 == v23 )
          {
            if ( (unsigned __int8)sub_2177670(**(_QWORD **)(a1 + 32), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL), a3, v5) )
              return v29[0] + v23 == *v5;
          }
        }
      }
      return 0;
    }
    if ( v8 <= 612 || ((1LL << ((unsigned __int8)v8 - 101)) & 0xB3) == 0 )
      return 0;
    goto LABEL_4;
  }
  if ( v8 > 253 )
  {
    switch ( v8 )
    {
      case 254:
      case 257:
      case 265:
      case 266:
      case 268:
      case 302:
      case 305:
      case 313:
      case 314:
      case 316:
        goto LABEL_4;
      default:
        return 0;
    }
  }
  result = 0;
  if ( (unsigned int)(v8 - 164) <= 1 )
    goto LABEL_8;
  return result;
}
