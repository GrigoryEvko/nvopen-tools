// Function: sub_6A98C0
// Address: 0x6a98c0
//
__int64 __fastcall sub_6A98C0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r13d
  int v3; // ebx
  __int64 v4; // rax
  __int64 result; // rax
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v11; // rax

  if ( !a1 )
  {
    switch ( word_4F06418[0] )
    {
      case 0xC3u:
        v2 = 1;
        v3 = 1;
        goto LABEL_3;
      case 0xC4u:
        v8 = sub_68AFD0(2u);
        sub_6A9320(0, 2, v8, 1, 0, 0, a2);
        goto LABEL_12;
      case 0xC5u:
        v6 = 3;
        goto LABEL_11;
      case 0xC6u:
        v6 = 4;
        goto LABEL_11;
      case 0xC7u:
        v6 = 5;
        goto LABEL_11;
      case 0xC8u:
        v6 = 6;
        goto LABEL_11;
      case 0xC9u:
        v6 = 7;
        goto LABEL_11;
      case 0xCAu:
        v6 = 8;
        goto LABEL_11;
      case 0xCBu:
        v6 = 9;
        goto LABEL_11;
      case 0xCCu:
        v6 = 10;
        goto LABEL_11;
      case 0xCDu:
        v6 = 11;
        goto LABEL_11;
      case 0xCEu:
        v6 = 12;
        goto LABEL_11;
      case 0xD0u:
        v6 = 14;
        goto LABEL_11;
      case 0xD4u:
        v6 = 16;
        goto LABEL_11;
      case 0xD5u:
        v6 = 17;
        goto LABEL_11;
      case 0xD6u:
        v6 = 114;
        goto LABEL_11;
      case 0xD7u:
        v6 = 18;
        goto LABEL_11;
      case 0xD8u:
        v6 = 19;
        goto LABEL_11;
      case 0xD9u:
        v6 = 20;
        goto LABEL_11;
      case 0xDAu:
        v6 = 23;
        goto LABEL_11;
      case 0xDBu:
        v6 = 24;
        goto LABEL_11;
      case 0xDCu:
        v6 = 25;
        goto LABEL_11;
      case 0xDDu:
        v6 = 26;
        goto LABEL_11;
      case 0xDEu:
        v6 = 27;
        goto LABEL_11;
      case 0xDFu:
        v6 = 28;
        goto LABEL_11;
      case 0xE0u:
        v6 = 29;
        goto LABEL_11;
      case 0xF2u:
        v6 = 40;
        goto LABEL_11;
      case 0x11Du:
        v6 = 64;
        goto LABEL_11;
      case 0x11Eu:
        v6 = 65;
        goto LABEL_11;
      case 0x125u:
        v6 = 69;
        goto LABEL_11;
      case 0x128u:
        v6 = 70;
        goto LABEL_11;
      case 0x129u:
        v6 = 71;
        goto LABEL_11;
      case 0x131u:
        v6 = 79;
        goto LABEL_11;
      case 0x134u:
        v6 = 82;
        goto LABEL_11;
      case 0x135u:
        v6 = 83;
        goto LABEL_11;
      case 0x136u:
        v6 = 84;
        goto LABEL_11;
      case 0x137u:
        v6 = 85;
        goto LABEL_11;
      case 0x138u:
        v6 = 86;
        goto LABEL_11;
      case 0x139u:
        v6 = 87;
        goto LABEL_11;
      case 0x13Au:
        v6 = 88;
        goto LABEL_11;
      case 0x13Bu:
        v6 = 89;
        goto LABEL_11;
      case 0x13Cu:
        v6 = 90;
        goto LABEL_11;
      case 0x13Du:
        v6 = 91;
        goto LABEL_11;
      case 0x13Eu:
        v6 = 92;
        goto LABEL_11;
      case 0x13Fu:
        v6 = 93;
        goto LABEL_11;
      case 0x140u:
        v6 = 94;
        goto LABEL_11;
      case 0x141u:
        v6 = 95;
        goto LABEL_11;
      case 0x142u:
        v6 = 96;
        goto LABEL_11;
      case 0x143u:
        v6 = 97;
        goto LABEL_11;
      case 0x144u:
        v6 = 98;
        goto LABEL_11;
      case 0x145u:
        v6 = 99;
        goto LABEL_11;
      case 0x146u:
        v6 = 100;
        goto LABEL_11;
      case 0x147u:
        v6 = 101;
        goto LABEL_11;
      case 0x148u:
        v10 = 105;
        goto LABEL_71;
      case 0x149u:
        v10 = 106;
LABEL_71:
        v11 = sub_68AFD0(v10);
        return sub_6A9320(0, v10, v11, 1, 0, 0, a2);
      case 0x14Au:
        v9 = sub_68AFD0(0x6Bu);
        return sub_6A9320(0, 107, v9, 1, 0, 0, a2);
      case 0x150u:
        v6 = 102;
        goto LABEL_11;
      case 0x151u:
        v6 = 103;
        goto LABEL_11;
      case 0x152u:
        v6 = 104;
        goto LABEL_11;
      case 0x163u:
        v6 = 115;
        goto LABEL_11;
      case 0x164u:
        v6 = 116;
LABEL_11:
        v7 = sub_68AFD0(v6);
        sub_6A9320(0, v6, v7, 1, 0, 0, a2);
LABEL_12:
        result = (__int64)&dword_4D044B0;
        if ( dword_4D044B0 )
          return result;
        return sub_6E6840(a2);
      default:
        sub_721090(0);
    }
  }
  v2 = *(unsigned __int8 *)(*a1 + 56LL);
  v3 = v2;
LABEL_3:
  v4 = sub_68AFD0(v2);
  sub_6A9320(a1, v2, v4, 1, 0, 0, a2);
  result = (__int64)&dword_4D044B0;
  if ( !dword_4D044B0 )
  {
    result = (unsigned int)(v3 - 105);
    if ( (unsigned __int8)(v3 - 105) > 1u && (_BYTE)v3 != 107 )
      return sub_6E6840(a2);
  }
  return result;
}
