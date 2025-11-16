// Function: sub_25357D0
// Address: 0x25357d0
//
_BOOL8 __fastcall sub_25357D0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // r12
  char v4; // [rsp+Fh] [rbp-21h] BYREF
  _QWORD v5[4]; // [rsp+10h] [rbp-20h] BYREF

  v5[0] = a2;
  v2 = *(_BYTE *)(a1 + 97);
  v5[1] = a1;
  v4 = 0;
  if ( (unsigned __int8)sub_25264B0(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2598270,
                          (__int64)v5,
                          a1,
                          &v4) )
    return *(_BYTE *)(a1 + 97) == v2;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
