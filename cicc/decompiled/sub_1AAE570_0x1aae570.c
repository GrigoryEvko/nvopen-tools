// Function: sub_1AAE570
// Address: 0x1aae570
//
__int64 __fastcall sub_1AAE570(__int64 a1, int a2)
{
  char v2; // r8
  __int64 result; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v4[0] = *(_QWORD *)(a1 + 112);
  v2 = sub_1560290(v4, a2, 22);
  result = 0;
  if ( !v2 )
  {
    sub_15E0DF0(a1, a2, 22);
    return 1;
  }
  return result;
}
