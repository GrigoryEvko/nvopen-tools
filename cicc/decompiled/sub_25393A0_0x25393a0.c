// Function: sub_25393A0
// Address: 0x25393a0
//
_BOOL8 __fastcall sub_25393A0(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // dl
  __int64 v5; // [rsp+8h] [rbp-58h] BYREF
  __int128 v6; // [rsp+10h] [rbp-50h] BYREF
  __int64 v7; // [rsp+20h] [rbp-40h]
  _QWORD v8[5]; // [rsp+30h] [rbp-30h] BYREF

  v8[0] = &v5;
  v8[1] = a2;
  v5 = 0;
  v7 = 0;
  v8[2] = a1;
  v8[3] = &v6;
  v6 = 0;
  if ( (unsigned __int8)sub_2527330(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_258FC10,
                          (__int64)v8,
                          a1,
                          1u,
                          0)
    && (!(_BYTE)v7 || BYTE9(v6)) )
  {
    return 1;
  }
  v3 = *(_BYTE *)(a1 + 96);
  v4 = *(_BYTE *)(a1 + 97);
  *(_BYTE *)(a1 + 97) = v3;
  return v4 == v3;
}
