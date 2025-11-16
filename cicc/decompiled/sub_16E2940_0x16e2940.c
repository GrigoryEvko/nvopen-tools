// Function: sub_16E2940
// Address: 0x16e2940
//
__int64 *__fastcall sub_16E2940(__int64 *a1, __int64 a2)
{
  *a1 = (__int64)(a1 + 2);
  sub_16DDF30(a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  a1[4] = *(_QWORD *)(a2 + 32);
  a1[5] = *(_QWORD *)(a2 + 40);
  a1[6] = *(_QWORD *)(a2 + 48);
  switch ( *(_DWORD *)(a2 + 32) )
  {
    case 0:
    case 5:
    case 6:
    case 9:
    case 0xE:
    case 0xF:
    case 0x13:
    case 0x19:
    case 0x1B:
    case 0x1C:
    case 0x21:
    case 0x2C:
    case 0x2D:
    case 0x2E:
      sub_16E28D0(a1, 0);
      return a1;
    case 1:
    case 0x1D:
      sub_16E28D0(a1, 3);
      return a1;
    case 2:
    case 0x1E:
      sub_16E28D0(a1, 4);
      return a1;
    case 0xA:
      sub_16E28D0(a1, 12);
      return a1;
    case 0xB:
      sub_16E28D0(a1, 13);
      return a1;
    case 0x10:
      sub_16E28D0(a1, 17);
      return a1;
    case 0x15:
      sub_16E28D0(a1, 22);
      return a1;
    case 0x17:
      sub_16E28D0(a1, 24);
      return a1;
    case 0x1F:
      sub_16E28D0(a1, 32);
      return a1;
    case 0x22:
      sub_16E28D0(a1, 35);
      return a1;
    case 0x24:
      sub_16E28D0(a1, 37);
      return a1;
    case 0x26:
      sub_16E28D0(a1, 39);
      return a1;
    case 0x28:
      sub_16E28D0(a1, 41);
      return a1;
    case 0x2A:
      sub_16E28D0(a1, 43);
      return a1;
    case 0x2F:
      sub_16E28D0(a1, 48);
      return a1;
    case 0x32:
      sub_16E28D0(a1, 51);
      return a1;
    default:
      return a1;
  }
}
