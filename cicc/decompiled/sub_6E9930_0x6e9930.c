// Function: sub_6E9930
// Address: 0x6e9930
//
__int64 __fastcall sub_6E9930(int a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // edi
  unsigned int v4; // r8d

  while ( 1 )
  {
    v2 = *(_BYTE *)(a2 + 140);
    if ( v2 != 12 )
      break;
    a2 = *(_QWORD *)(a2 + 160);
  }
  v3 = a1 - 33;
  switch ( (__int16)v3 )
  {
    case 0:
      v4 = 55;
      break;
    case 1:
      v4 = 41;
      break;
    case 2:
      v4 = 50;
      if ( v2 == 6 )
        return v4;
      v4 = 39;
      break;
    case 3:
      v4 = 51;
      if ( v2 == 6 )
        return v4;
      v4 = 40;
      break;
    case 6:
      v4 = 42;
      break;
    case 7:
      v4 = 43;
      break;
    case 8:
      v4 = 53;
      break;
    case 9:
      v4 = 54;
      break;
    case 10:
      v4 = 68;
      if ( v2 == 15 )
        return v4;
      v4 = 61;
      break;
    case 11:
      v4 = 67;
      if ( v2 == 15 )
        return v4;
      v4 = 60;
      break;
    case 12:
      v4 = 70;
      if ( v2 == 15 )
        return v4;
      v4 = 63;
      break;
    case 13:
      v4 = 69;
      if ( v2 == 15 )
        return v4;
      v4 = 62;
      break;
    case 14:
      v4 = 65;
      if ( v2 == 15 )
        return v4;
      v4 = 58;
      break;
    case 15:
      v4 = 66;
      if ( v2 == 15 )
        return v4;
      v4 = 59;
      break;
    case 16:
      v4 = 64;
      break;
    case 17:
      v4 = 57;
      break;
    case 18:
      v4 = 56;
      break;
    case 19:
      v4 = 89;
      if ( v2 == 15 )
        return v4;
      v4 = 87;
      break;
    case 20:
      v4 = 90;
      if ( v2 == 15 )
        return v4;
      v4 = 88;
      break;
    case 23:
      v4 = 73;
      break;
    case 24:
      v4 = 76;
      break;
    case 25:
      v4 = 77;
      break;
    case 26:
      v4 = 78;
      break;
    case 27:
      v4 = 84;
      if ( v2 == 6 )
        return v4;
      v4 = 74;
      break;
    case 28:
      v4 = 85;
      if ( v2 == 6 )
        return v4;
      v4 = 75;
      break;
    case 29:
      v4 = 79;
      break;
    case 30:
      v4 = 80;
      break;
    case 31:
      v4 = 81;
      break;
    case 32:
      v4 = 83;
      break;
    case 33:
      v4 = 82;
      break;
    case 37:
      v4 = 71;
      break;
    case 38:
      v4 = 72;
      break;
    default:
      sub_721090(v3);
  }
  if ( !v2 )
    return 119;
  return v4;
}
