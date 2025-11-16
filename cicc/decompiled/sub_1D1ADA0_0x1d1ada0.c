// Function: sub_1D1ADA0
// Address: 0x1d1ada0
//
__int64 __fastcall sub_1D1ADA0(__int64 a1, unsigned int a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 v13; // rsi
  char v14; // bl
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  char v18[8]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v19; // [rsp+8h] [rbp-48h]
  _QWORD v20[2]; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+20h] [rbp-30h]

  v6 = a1;
  v7 = *(unsigned __int16 *)(a1 + 24);
  if ( (_WORD)v7 != 32 && (_DWORD)v7 != 10 )
  {
    v6 = 0;
    if ( (_WORD)v7 == 104 )
    {
      v20[0] = 0;
      v20[1] = 0;
      v21 = 0;
      v9 = sub_1D1AD70(a1, (__int64)v20, v7, a4, a5, a6);
      v10 = v20[0];
      v6 = v9;
      if ( !v9 )
        goto LABEL_9;
      if ( (unsigned int)(v21 + 63) >> 6 )
      {
        v10 = v20[0];
        v11 = (_QWORD *)v20[0];
        while ( !*v11 )
        {
          if ( (_QWORD *)(v20[0] + 8LL * (((unsigned int)(v21 + 63) >> 6) - 1) + 8) == ++v11 )
            goto LABEL_11;
        }
        goto LABEL_9;
      }
LABEL_11:
      v13 = *(_QWORD *)(a1 + 40) + 16LL * a2;
      v14 = *(_BYTE *)v13;
      v15 = *(_QWORD *)(v13 + 8);
      v18[0] = v14;
      v19 = v15;
      if ( v14 )
      {
        if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
        {
          v10 = v20[0];
          if ( v14 == **(_BYTE **)(v6 + 40) )
          {
LABEL_14:
            _libc_free(v10);
            return v6;
          }
LABEL_9:
          _libc_free(v10);
          return 0;
        }
        switch ( v14 )
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
            v14 = 3;
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
            v14 = 4;
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
            v14 = 5;
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
            v14 = 6;
            break;
          case 55:
            v14 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v14 = 8;
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
            v14 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v14 = 10;
            break;
          default:
            v14 = 2;
            break;
        }
        v15 = 0;
      }
      else if ( (unsigned __int8)sub_1F58D20(v18) )
      {
        v14 = sub_1F596B0(v18);
        v15 = v17;
      }
      v16 = *(_QWORD *)(v6 + 40);
      v10 = v20[0];
      if ( *(_BYTE *)v16 == v14 && (*(_QWORD *)(v16 + 8) == v15 || v14) )
        goto LABEL_14;
      goto LABEL_9;
    }
  }
  return v6;
}
