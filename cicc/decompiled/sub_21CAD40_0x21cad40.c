// Function: sub_21CAD40
// Address: 0x21cad40
//
__int64 __fastcall sub_21CAD40(__int64 a1, __int64 a2, __int64 a3)
{
  __int8 v3; // bl
  int v5; // eax
  __m128i v6; // [rsp+0h] [rbp-30h] BYREF
  __m128i v7[2]; // [rsp+10h] [rbp-20h] BYREF

  v3 = a2;
  v6.m128i_i64[0] = a2;
  v6.m128i_i64[1] = a3;
  if ( !(_BYTE)a2 )
  {
    if ( (unsigned int)sub_1F58D30((__int64)&v6) == 1 )
    {
      v7[0] = _mm_load_si128(&v6);
    }
    else
    {
      if ( sub_1F58D20((__int64)&v6) )
      {
        if ( sub_1F596B0((__int64)&v6) == 2 )
          return 6;
        v3 = v6.m128i_i8[0];
LABEL_16:
        if ( v3 == 86 )
          return 0;
      }
      v7[0] = _mm_load_si128(&v6);
      if ( v3 )
        goto LABEL_8;
    }
    v5 = sub_1F58D30((__int64)v7);
    goto LABEL_13;
  }
  if ( word_435D740[(unsigned __int8)(a2 - 14)] != 1 )
  {
    switch ( (char)a2 )
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
        return 6;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
      case 86:
      case 87:
      case 88:
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 94:
      case 95:
      case 96:
      case 97:
      case 98:
      case 99:
      case 100:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
      case 106:
      case 107:
      case 108:
      case 109:
        goto LABEL_16;
    }
  }
  if ( (_BYTE)a2 == 86 )
    return 0;
LABEL_8:
  v5 = (unsigned __int16)word_435D740[(unsigned __int8)(v3 - 14)];
LABEL_13:
  LOBYTE(v5) = v5 == 1;
  return (unsigned int)(4 * v5 + 1);
}
