// Function: sub_1F5A910
// Address: 0x1f5a910
//
__int64 __fastcall sub_1F5A910(__int64 a1)
{
  unsigned int v1; // ebx
  _QWORD **v3; // r14
  char v4; // di
  _QWORD *v5; // r12
  unsigned int v6; // eax
  bool v7; // cc
  unsigned __int8 v8; // r14
  char v9; // al
  unsigned int v10; // r13d
  unsigned int v11; // eax
  __int64 v13; // rdx
  _QWORD **v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-48h]
  char v16[8]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD **v17; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD ***)(a1 + 8);
  v4 = *(_BYTE *)a1;
  v5 = *v3;
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
LABEL_3:
    v6 = sub_1F58BF0(v4);
    v7 = v6 <= 0x20;
    if ( v6 != 32 )
      goto LABEL_4;
LABEL_21:
    v8 = 5;
    goto LABEL_7;
  }
  if ( sub_1F58D20(a1) )
  {
    v16[0] = sub_1F596B0(a1);
    v4 = v16[0];
    v17 = v14;
    if ( v16[0] )
      goto LABEL_3;
  }
  else
  {
    v16[0] = 0;
    v17 = v3;
  }
  v6 = sub_1F58D40((__int64)v16);
  v7 = v6 <= 0x20;
  if ( v6 == 32 )
    goto LABEL_21;
LABEL_4:
  if ( v7 )
  {
    if ( v6 == 8 )
    {
      v8 = 3;
    }
    else
    {
      v8 = 4;
      if ( v6 != 16 )
      {
        v8 = 2;
        if ( v6 != 1 )
        {
LABEL_16:
          v8 = sub_1F58CC0(v5, v6);
          v9 = *(_BYTE *)a1;
          v15 = v13;
          if ( *(_BYTE *)a1 )
            goto LABEL_17;
LABEL_8:
          v10 = sub_1F58D30(a1);
          goto LABEL_9;
        }
      }
    }
  }
  else if ( v6 == 64 )
  {
    v8 = 6;
  }
  else
  {
    if ( v6 != 128 )
      goto LABEL_16;
    v8 = 7;
  }
LABEL_7:
  v9 = *(_BYTE *)a1;
  v15 = 0;
  if ( !*(_BYTE *)a1 )
    goto LABEL_8;
LABEL_17:
  v10 = word_42F4D80[(unsigned __int8)(v9 - 14)];
LABEL_9:
  LOBYTE(v11) = sub_1D15020(v8, v10);
  if ( !(_BYTE)v11 )
  {
    v11 = sub_1F593D0(v5, v8, v15, v10);
    v1 = v11;
  }
  LOBYTE(v1) = v11;
  return v1;
}
