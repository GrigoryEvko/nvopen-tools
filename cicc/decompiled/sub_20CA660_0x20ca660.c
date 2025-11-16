// Function: sub_20CA660
// Address: 0x20ca660
//
__int64 __fastcall sub_20CA660(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  char v7; // al
  _QWORD *v8; // rsi
  __int64 v9; // r13
  unsigned int v10; // eax
  _QWORD *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // r12
  unsigned int v14; // r15d
  char v15; // al
  __int64 v16; // rdx
  char v17[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h]

  v2 = *(_BYTE *)(a1 + 8);
  if ( v2 == 15 )
  {
    v3 = 8 * sub_15A9520(a2, *(_DWORD *)(a1 + 8) >> 8);
    if ( v3 == 32 )
    {
      v17[0] = 5;
      v18 = 0;
LABEL_36:
      v3 = 32;
    }
    else
    {
      if ( v3 <= 0x20 )
      {
        if ( v3 == 8 )
        {
          v17[0] = 3;
          v18 = 0;
          return sub_1644900(*(_QWORD **)a1, v3);
        }
        if ( v3 == 16 )
        {
          v17[0] = 4;
          v18 = 0;
LABEL_7:
          v3 = 16;
          return sub_1644900(*(_QWORD **)a1, v3);
        }
        goto LABEL_33;
      }
      if ( v3 == 64 )
      {
        v17[0] = 6;
        v18 = 0;
LABEL_32:
        v3 = 64;
      }
      else
      {
        if ( v3 != 128 )
        {
LABEL_33:
          v17[0] = 0;
          v18 = 0;
LABEL_23:
          v3 = (sub_1F58D40((__int64)v17) + 7) & 0xFFFFFFF8;
          return sub_1644900(*(_QWORD **)a1, v3);
        }
        v17[0] = 7;
        v18 = 0;
LABEL_12:
        v3 = 128;
      }
    }
  }
  else
  {
    if ( v2 == 16 )
    {
      v5 = *(_QWORD *)(a1 + 24);
      if ( *(_BYTE *)(v5 + 8) == 15 )
      {
        v6 = 8 * sub_15A9520(a2, *(_DWORD *)(v5 + 8) >> 8);
        if ( v6 == 32 )
        {
          v7 = 5;
        }
        else if ( v6 > 0x20 )
        {
          v7 = 6;
          if ( v6 != 64 )
          {
            v7 = 0;
            if ( v6 == 128 )
              v7 = 7;
          }
        }
        else
        {
          v7 = 3;
          if ( v6 != 8 )
            v7 = 4 * (v6 == 16);
        }
        v8 = *(_QWORD **)a1;
        v17[0] = v7;
        v18 = 0;
        v5 = sub_1F58E60((__int64)v17, v8);
      }
      v9 = *(_QWORD *)(a1 + 32);
      LOBYTE(v10) = sub_1F59570(v5);
      v11 = *(_QWORD **)a1;
      v13 = v12;
      v14 = v10;
      v15 = sub_1D15020(v10, v9);
      v16 = 0;
      if ( !v15 )
        v15 = sub_1F593D0(v11, v14, v13, v9);
    }
    else
    {
      v15 = sub_1F59570(a1);
    }
    v17[0] = v15;
    v18 = v16;
    if ( !v15 )
      goto LABEL_23;
    switch ( v15 )
    {
      case 0:
      case 1:
      case 111:
      case 112:
      case 113:
      case 114:
        v3 = 0;
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
        v3 = 8;
        break;
      case 4:
      case 8:
      case 18:
      case 25:
      case 33:
      case 60:
      case 63:
      case 68:
        goto LABEL_7;
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
        goto LABEL_36;
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
        goto LABEL_32;
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
        goto LABEL_12;
      case 11:
        v3 = 80;
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
        v3 = 512;
        break;
      case 23:
      case 31:
      case 39:
      case 46:
      case 53:
      case 79:
      case 84:
        v3 = 1024;
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
        v3 = 256;
        break;
      case 32:
      case 40:
      case 47:
      case 54:
      case 85:
        v3 = 2048;
        break;
      case 48:
        v3 = 4096;
        break;
    }
  }
  return sub_1644900(*(_QWORD **)a1, v3);
}
