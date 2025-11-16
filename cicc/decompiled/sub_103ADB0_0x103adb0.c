// Function: sub_103ADB0
// Address: 0x103adb0
//
__int64 __fastcall sub_103ADB0(__int64 *a1, __int64 a2)
{
  __m128i v3[5]; // [rsp+0h] [rbp-50h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 6) )
    return 1;
  sub_D665A0(v3, a2);
  return (((unsigned __int8)sub_CF4FA0(*a1, (__int64)v3, (__int64)(a1 + 1), 0) >> 1) ^ 1) & 1;
}
