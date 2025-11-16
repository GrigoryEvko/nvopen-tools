// Function: sub_ACFDF0
// Address: 0xacfdf0
//
__int64 __fastcall sub_ACFDF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 i; // rax

  switch ( *(_BYTE *)a1 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
      sub_4089E0();
    case 4:
      sub_AC4080((__int64)a1);
      break;
    case 5:
      sub_ACFD40((__int64)a1);
      break;
    case 6:
      sub_AC4200((__int64)a1, a2, a3);
      break;
    case 7:
      sub_AC4320((__int64)a1, a2, a3);
      break;
    case 8:
      sub_AC7D40(a1);
      break;
    case 9:
      sub_AC72C0((__int64)a1);
      break;
    case 0xA:
      sub_AC75A0((__int64)a1);
      break;
    case 0xB:
      sub_AC7880((__int64)a1);
      break;
    case 0xC:
      sub_AC3C50((unsigned __int8 *)a1, a2, a3);
    case 0xD:
      sub_AC3DD0((__int64)a1, a2, a3);
      break;
    case 0xE:
      sub_AC39A0((__int64)a1, a2, a3);
      break;
    case 0xF:
    case 0x10:
      sub_AC5BE0((__int64)a1);
      break;
    case 0x11:
      sub_4086E4();
    case 0x12:
      sub_4086E6();
    case 0x13:
      sub_AC3B90((__int64)a1, a2, a3);
      break;
    case 0x14:
      sub_AC3AD0((__int64)a1, a2, a3);
      break;
    case 0x15:
      sub_4086E8();
    default:
      BUG();
  }
  for ( i = a1[2]; i; i = a1[2] )
    sub_ACFDF0(*(_QWORD *)(i + 24));
  return sub_AC70E0((__int64)a1, a2);
}
