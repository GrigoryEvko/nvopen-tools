// Function: sub_259ADE0
// Address: 0x259ade0
//
__int64 __fastcall sub_259ADE0(__int64 a1, __int64 a2)
{
  char v3; // [rsp+16h] [rbp-22h] BYREF
  char v4; // [rsp+17h] [rbp-21h] BYREF
  _QWORD v5[4]; // [rsp+18h] [rbp-20h] BYREF

  if ( (unsigned __int8)sub_259AB90(a2, a1, (__m128i *)(a1 + 72), 1, &v3, 0, 0) )
  {
    if ( v3 )
      *(_BYTE *)(a1 + 96) = *(_BYTE *)(a1 + 97);
    return 1;
  }
  v5[0] = a2;
  v5[1] = a1;
  v4 = 1;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_2595220,
                          (__int64)v5,
                          a1,
                          1u,
                          &v4) )
    return 1;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
