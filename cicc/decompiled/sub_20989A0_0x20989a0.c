// Function: sub_20989A0
// Address: 0x20989a0
//
_QWORD *__fastcall sub_20989A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r14
  __int64 v7; // r12
  __int64 v8; // r10
  unsigned __int64 v9; // r8
  char v10; // di
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // r9
  unsigned int v13; // eax
  __int64 v14; // r14
  unsigned __int64 v15; // r11
  __int64 v16; // rdx
  int v17; // r9d
  _QWORD *v18; // rax
  __int64 v19; // rcx
  int v20; // r9d
  int v21; // r8d
  _QWORD *v22; // r14
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  int v27; // [rsp+Ch] [rbp-44h]
  __int64 v28; // [rsp+10h] [rbp-40h] BYREF
  __int64 v29; // [rsp+18h] [rbp-38h]

  v6 = *(_QWORD **)(a4 + 552);
  v29 = a3;
  v28 = a2;
  v7 = *(_QWORD *)(v6[4] + 56LL);
  if ( (_BYTE)a2 )
  {
    switch ( (char)a2 )
    {
      case 0:
      case 1:
      case 111:
      case 112:
      case 113:
      case 114:
        v8 = 0;
        break;
      case 2:
      case 3:
      case 14:
      case 15:
      case 16:
      case 17:
      case 24:
      case 56:
      case 57:
      case 58:
      case 59:
      case 62:
        v8 = 1;
        break;
      case 4:
      case 8:
      case 18:
      case 25:
      case 33:
      case 60:
      case 63:
      case 68:
        v8 = 2;
        break;
      case 5:
      case 9:
      case 19:
      case 26:
      case 34:
      case 41:
      case 61:
      case 64:
      case 69:
      case 74:
      case 86:
      case 89:
      case 98:
      case 101:
        v8 = 4;
        break;
      case 6:
      case 10:
      case 20:
      case 27:
      case 35:
      case 42:
      case 49:
      case 65:
      case 70:
      case 75:
      case 80:
      case 87:
      case 90:
      case 94:
      case 99:
      case 102:
      case 106:
      case 110:
        v8 = 8;
        break;
      case 7:
      case 12:
      case 13:
      case 21:
      case 28:
      case 36:
      case 43:
      case 50:
      case 55:
      case 66:
      case 71:
      case 76:
      case 81:
      case 88:
      case 91:
      case 95:
      case 100:
      case 103:
      case 107:
        v8 = 16;
        break;
      case 11:
        v8 = 10;
        break;
      case 22:
      case 30:
      case 38:
      case 45:
      case 52:
      case 73:
      case 78:
      case 83:
      case 93:
      case 97:
      case 105:
      case 109:
        v8 = 64;
        break;
      case 23:
      case 31:
      case 39:
      case 46:
      case 53:
      case 79:
      case 84:
        v8 = 128;
        break;
      case 29:
      case 37:
      case 44:
      case 51:
      case 67:
      case 72:
      case 77:
      case 82:
      case 92:
      case 96:
      case 104:
      case 108:
        v8 = 32;
        break;
      case 32:
      case 40:
      case 47:
      case 54:
      case 85:
        v8 = 256;
        break;
      case 48:
        v8 = 512;
        break;
    }
  }
  else
  {
    v8 = ((unsigned int)sub_1F58D40((__int64)&v28) + 7) >> 3;
  }
  v9 = *(_QWORD *)(a1 + 32);
  v10 = v9 & 1;
  if ( (*(_QWORD *)(a1 + 32) & 1) != 0 )
    v11 = v9 >> 58;
  else
    v11 = *(unsigned int *)(v9 + 16);
  v12 = *(unsigned int *)(a1 + 40);
  v13 = *(_DWORD *)(a1 + 40);
  if ( v11 <= v12 )
  {
LABEL_15:
    v18 = sub_1D29C20(v6, (unsigned int)v28, v29, 1, v9, v12);
    v21 = *((_DWORD *)v18 + 21);
    v22 = v18;
    *(_BYTE *)(*(_QWORD *)(v7 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v7 + 32) + v21) + 22) = 1;
    v23 = *(_QWORD *)(a4 + 712);
    v24 = *(unsigned int *)(v23 + 576);
    if ( (unsigned int)v24 >= *(_DWORD *)(v23 + 580) )
    {
      v27 = v21;
      sub_16CD150(v23 + 568, (const void *)(v23 + 584), 0, 4, v21, v20);
      v24 = *(unsigned int *)(v23 + 576);
      v21 = v27;
    }
    *(_DWORD *)(*(_QWORD *)(v23 + 568) + 4 * v24) = v21;
    ++*(_DWORD *)(v23 + 576);
    v25 = *(_QWORD *)(a1 + 32);
    if ( (v25 & 1) != 0 )
      v25 >>= 58;
    else
      LODWORD(v25) = *(_DWORD *)(v25 + 16);
    sub_13A5100((unsigned __int64 *)(a1 + 32), v25 + 1, 1u, v19, v21, v20);
    return v22;
  }
  else
  {
    v14 = ~(-1LL << (v9 >> 58));
    v15 = v14 & (v9 >> 1);
    while ( 1 )
    {
      v16 = v10 ? (v15 >> v13) & 1 : (*(_QWORD *)(*(_QWORD *)v9 + 8LL * (v13 >> 6)) >> v13) & 1LL;
      if ( !(_BYTE)v16 )
      {
        v17 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 712) + 568LL) + 4 * v12);
        if ( v8 == *(_QWORD *)(*(_QWORD *)(v7 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v7 + 32) + v17) + 8) )
          break;
      }
      v12 = v13 + 1;
      *(_DWORD *)(a1 + 40) = v12;
      ++v13;
      if ( v12 >= v11 )
      {
        v6 = *(_QWORD **)(a4 + 552);
        goto LABEL_15;
      }
    }
    if ( v10 )
      *(_QWORD *)(a1 + 32) = 2 * ((v15 | (1LL << v13)) & v14 | (v9 >> 58 << 57)) + 1;
    else
      *(_QWORD *)(*(_QWORD *)v9 + 8LL * (v13 >> 6)) |= 1LL << v13;
    return sub_1D299D0(*(_QWORD **)(a4 + 552), v17, v28, v29, 0);
  }
}
