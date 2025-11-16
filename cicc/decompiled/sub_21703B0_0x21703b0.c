// Function: sub_21703B0
// Address: 0x21703b0
//
__int64 __fastcall sub_21703B0(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v8; // r14
  char v9; // r15
  __int64 v10; // rdx
  unsigned __int8 v11; // r15
  __int64 *v12; // r10
  const void **v13; // r8
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rsi
  bool v18; // al
  char v19; // al
  const void **v20; // rdx
  __int128 v21; // [rsp-10h] [rbp-70h]
  const void **v22; // [rsp+10h] [rbp-50h]
  __int64 *v23; // [rsp+18h] [rbp-48h]
  __int64 *v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  int v26; // [rsp+28h] [rbp-38h]

  v8 = a1[1];
  v9 = *(_BYTE *)v8;
  if ( !*(_BYTE *)v8 )
  {
    if ( sub_1F58D20(a1[1]) )
    {
      v9 = sub_1F596B0(v8);
      goto LABEL_4;
    }
LABEL_3:
    v10 = *(_QWORD *)(v8 + 8);
LABEL_4:
    if ( *(_BYTE *)*a1 == v9 && (v9 || *(_QWORD *)(*a1 + 8) == v10) )
      return a2;
    v8 = a1[1];
LABEL_6:
    v11 = *(_BYTE *)v8;
    v12 = (__int64 *)a1[2];
    if ( *(_BYTE *)v8 )
    {
      if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
      {
        switch ( v11 )
        {
          case 0x18u:
          case 0x19u:
          case 0x1Au:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x1Eu:
          case 0x1Fu:
          case 0x20u:
          case 0x3Eu:
          case 0x3Fu:
          case 0x40u:
          case 0x41u:
          case 0x42u:
          case 0x43u:
            v11 = 3;
            break;
          case 0x21u:
          case 0x22u:
          case 0x23u:
          case 0x24u:
          case 0x25u:
          case 0x26u:
          case 0x27u:
          case 0x28u:
          case 0x44u:
          case 0x45u:
          case 0x46u:
          case 0x47u:
          case 0x48u:
          case 0x49u:
            v11 = 4;
            break;
          case 0x29u:
          case 0x2Au:
          case 0x2Bu:
          case 0x2Cu:
          case 0x2Du:
          case 0x2Eu:
          case 0x2Fu:
          case 0x30u:
          case 0x4Au:
          case 0x4Bu:
          case 0x4Cu:
          case 0x4Du:
          case 0x4Eu:
          case 0x4Fu:
            v11 = 5;
            break;
          case 0x31u:
          case 0x32u:
          case 0x33u:
          case 0x34u:
          case 0x35u:
          case 0x36u:
          case 0x50u:
          case 0x51u:
          case 0x52u:
          case 0x53u:
          case 0x54u:
          case 0x55u:
            v11 = 6;
            break;
          case 0x37u:
            v11 = 7;
            break;
          case 0x56u:
          case 0x57u:
          case 0x58u:
          case 0x62u:
          case 0x63u:
          case 0x64u:
            v11 = 8;
            break;
          case 0x59u:
          case 0x5Au:
          case 0x5Bu:
          case 0x5Cu:
          case 0x5Du:
          case 0x65u:
          case 0x66u:
          case 0x67u:
          case 0x68u:
          case 0x69u:
            v11 = 9;
            break;
          case 0x5Eu:
          case 0x5Fu:
          case 0x60u:
          case 0x61u:
          case 0x6Au:
          case 0x6Bu:
          case 0x6Cu:
          case 0x6Du:
            v11 = 10;
            break;
          default:
            v11 = 2;
            break;
        }
        v13 = 0;
        goto LABEL_9;
      }
    }
    else
    {
      v24 = (__int64 *)a1[2];
      v18 = sub_1F58D20(v8);
      v12 = v24;
      if ( v18 )
      {
        v19 = sub_1F596B0(v8);
        v12 = v24;
        v11 = v19;
        v13 = v20;
        goto LABEL_9;
      }
    }
    v13 = *(const void ***)(v8 + 8);
LABEL_9:
    v14 = a1[3];
    v15 = v11;
    v16 = *(_QWORD *)(v14 + 72);
    v25 = v16;
    if ( v16 )
    {
      v22 = v13;
      v23 = v12;
      sub_1623A60((__int64)&v25, v16, 2);
      v15 = v11;
      v13 = v22;
      v12 = v23;
    }
    *((_QWORD *)&v21 + 1) = a3;
    *(_QWORD *)&v21 = a2;
    v26 = *(_DWORD *)(v14 + 64);
    a2 = sub_1D309E0(v12, 145, (__int64)&v25, v15, v13, 0, a4, a5, a6, v21);
    if ( v25 )
      sub_161E7C0((__int64)&v25, v25);
    return a2;
  }
  if ( (unsigned __int8)(v9 - 14) > 0x5Fu )
    goto LABEL_3;
  switch ( v9 )
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
      if ( *(_BYTE *)*a1 != 3 )
        goto LABEL_6;
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
      if ( *(_BYTE *)*a1 != 4 )
        goto LABEL_6;
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
      if ( *(_BYTE *)*a1 != 5 )
        goto LABEL_6;
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
      if ( *(_BYTE *)*a1 != 6 )
        goto LABEL_6;
      break;
    case 55:
      if ( *(_BYTE *)*a1 != 7 )
        goto LABEL_6;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      if ( *(_BYTE *)*a1 != 8 )
        goto LABEL_6;
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
      if ( *(_BYTE *)*a1 != 9 )
        goto LABEL_6;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      if ( *(_BYTE *)*a1 != 10 )
        goto LABEL_6;
      break;
    default:
      if ( *(_BYTE *)*a1 != 2 )
        goto LABEL_6;
      break;
  }
  return a2;
}
