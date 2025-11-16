// Function: sub_E405D0
// Address: 0xe405d0
//
void *__fastcall sub_E405D0(__int64 a1, char *a2, __int64 a3)
{
  unsigned __int8 v4; // r8
  _QWORD v6[12]; // [rsp+0h] [rbp-60h] BYREF

  v6[5] = 0x100000000LL;
  v6[6] = a1;
  v6[1] = 2;
  memset(&v6[2], 0, 24);
  v6[0] = &unk_49DD288;
  sub_CB5980((__int64)v6, 0, 0, 0);
  switch ( *(_DWORD *)(a3 + 24) )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
      v4 = 0;
      break;
    case 2:
    case 4:
      v4 = 95;
      break;
    default:
      BUG();
  }
  sub_E401D0((__int64)v6, a2, 0, a3, v4);
  v6[0] = &unk_49DD388;
  return sub_CB5840((__int64)v6);
}
