// Function: sub_25390B0
// Address: 0x25390b0
//
__int64 __fastcall sub_25390B0(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  char v4; // [rsp+16h] [rbp-32h] BYREF
  char v5; // [rsp+17h] [rbp-31h] BYREF
  _QWORD v6[6]; // [rsp+18h] [rbp-30h] BYREF

  v4 = 0;
  LODWORD(v6[0]) = 1;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2535030,
    (__int64)&v5,
    a1,
    (int *)v6,
    1,
    &v4,
    0,
    0);
  v6[0] = a2;
  v6[1] = a1;
  v2 = sub_2523890(a2, (__int64 (__fastcall *)(__int64, __int64 *))sub_254E410, (__int64)v6, a1, 1u, &v4);
  result = 1;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return result;
}
