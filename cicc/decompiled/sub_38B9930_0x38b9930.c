// Function: sub_38B9930
// Address: 0x38b9930
//
void *__fastcall sub_38B9930(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r8
  _QWORD v6[4]; // [rsp+0h] [rbp-50h] BYREF
  int v7; // [rsp+20h] [rbp-30h]
  __int64 v8; // [rsp+28h] [rbp-28h]

  v8 = a1;
  v7 = 1;
  v6[0] = &unk_49EFC48;
  memset(&v6[1], 0, 24);
  sub_16E7A40((__int64)v6, 0, 0, 0);
  switch ( *(_DWORD *)(a3 + 16) )
  {
    case 0:
    case 1:
    case 3:
    case 5:
      v4 = 0;
      break;
    case 2:
    case 4:
      v4 = 95;
      break;
  }
  sub_38B95A0((__int64)v6, a2, 0, a3, v4);
  v6[0] = &unk_49EFD28;
  return sub_16E7960((__int64)v6);
}
