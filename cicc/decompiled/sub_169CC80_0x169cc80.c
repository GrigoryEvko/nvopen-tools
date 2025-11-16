// Function: sub_169CC80
// Address: 0x169cc80
//
__int64 __fastcall sub_169CC80(_BYTE *a1, _BYTE *a2, char a3)
{
  __int64 result; // rax
  char v5; // al
  char v6; // dl

  if ( (unsigned __int8)sub_169CC60(a1) || (unsigned __int8)sub_169CC60(a2) )
  {
LABEL_5:
    sub_16986F0(a1, 0, 0, 0);
    return 1;
  }
  else
  {
    switch ( (a2[18] & 7) + 4 * (a1[18] & 7) )
    {
      case 0:
        if ( a3 != (((a2[18] ^ a1[18]) & 8) != 0) )
          goto LABEL_5;
        goto LABEL_7;
      case 1:
      case 9:
      case 0xD:
        a1[18] = a1[18] & 0xF0 | (8 * (a3 ^ ((a2[18] & 8) != 0)) + 1) & 0xF;
        sub_16985E0((__int64)a1, (__int64)a2);
        result = 0;
        break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 0xB:
      case 0xF:
LABEL_7:
        result = 0;
        break;
      case 8:
      case 0xC:
        v5 = a1[18] & 0xF0;
        a1[18] &= 0xF8u;
        v6 = v5 | (8 * ((a3 ^ (a2[18] >> 3)) & 1));
        result = 0;
        a1[18] = v6;
        break;
      case 0xA:
        result = 2;
        break;
      case 0xE:
        sub_1698630((__int64)a1, (__int64)a2);
        a1[18] = a1[18] & 0xF7 | (8 * ((a3 ^ (a2[18] >> 3)) & 1));
        result = 0;
        break;
    }
  }
  return result;
}
