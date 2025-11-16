// Function: sub_388C2C0
// Address: 0x388c2c0
//
__int64 __fastcall sub_388C2C0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  switch ( *(_DWORD *)(a1 + 64) )
  {
    case 0x66:
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      return sub_388BA90(a1, a2);
    case 0x67:
      *a2 = 0;
      goto LABEL_3;
    case 0x68:
      *a2 = 8;
      goto LABEL_3;
    case 0x69:
      *a2 = 9;
      goto LABEL_3;
    case 0x6A:
      *a2 = 77;
      goto LABEL_3;
    case 0x6B:
      *a2 = 64;
      goto LABEL_3;
    case 0x6C:
      *a2 = 65;
      goto LABEL_3;
    case 0x6D:
      *a2 = 70;
      goto LABEL_3;
    case 0x6E:
      *a2 = 80;
      goto LABEL_3;
    case 0x6F:
      *a2 = 92;
      goto LABEL_3;
    case 0x70:
      *a2 = 66;
      goto LABEL_3;
    case 0x71:
      *a2 = 67;
      goto LABEL_3;
    case 0x72:
      *a2 = 68;
      goto LABEL_3;
    case 0x73:
      *a2 = 69;
      goto LABEL_3;
    case 0x74:
      *a2 = 84;
      goto LABEL_3;
    case 0x75:
      *a2 = 85;
      goto LABEL_3;
    case 0x76:
      *a2 = 71;
      goto LABEL_3;
    case 0x77:
      *a2 = 72;
      goto LABEL_3;
    case 0x78:
      *a2 = 76;
      goto LABEL_3;
    case 0x79:
      *a2 = 75;
      goto LABEL_3;
    case 0x7A:
      *a2 = 78;
      goto LABEL_3;
    case 0x7B:
      *a2 = 79;
      goto LABEL_3;
    case 0x7C:
      *a2 = 12;
      goto LABEL_3;
    case 0x7D:
      *a2 = 13;
      goto LABEL_3;
    case 0x7E:
      *a2 = 16;
      goto LABEL_3;
    case 0x7F:
      *a2 = 14;
      goto LABEL_3;
    case 0x80:
      *a2 = 15;
      goto LABEL_3;
    case 0x81:
      *a2 = 10;
      goto LABEL_3;
    case 0x82:
      *a2 = 83;
      goto LABEL_3;
    case 0x83:
      *a2 = 81;
      goto LABEL_3;
    case 0x84:
      *a2 = 82;
      goto LABEL_3;
    case 0x85:
      *a2 = 17;
      goto LABEL_3;
    case 0x86:
      *a2 = 87;
      goto LABEL_3;
    case 0x87:
      *a2 = 95;
      goto LABEL_3;
    case 0x88:
      *a2 = 93;
      goto LABEL_3;
    case 0x89:
      *a2 = 96;
      goto LABEL_3;
    case 0x8A:
      *a2 = 88;
      goto LABEL_3;
    case 0x8B:
      *a2 = 89;
      goto LABEL_3;
    case 0x8C:
      *a2 = 90;
      goto LABEL_3;
    case 0x8D:
      *a2 = 91;
LABEL_3:
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      result = 0;
      break;
    default:
      *a2 = 0;
      result = 0;
      break;
  }
  return result;
}
