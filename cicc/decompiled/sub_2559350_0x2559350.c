// Function: sub_2559350
// Address: 0x2559350
//
_BOOL8 __fastcall sub_2559350(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // dl
  char v5; // [rsp+Bh] [rbp-55h] BYREF
  int v6; // [rsp+Ch] [rbp-54h] BYREF
  __int128 v7; // [rsp+10h] [rbp-50h] BYREF
  __int64 v8; // [rsp+20h] [rbp-40h]
  _QWORD v9[6]; // [rsp+30h] [rbp-30h] BYREF

  v8 = 0;
  v7 = 0;
  v6 = sub_250CB50((__int64 *)(a1 + 72), 0);
  v9[0] = &v6;
  v9[1] = a2;
  v9[2] = a1;
  v9[3] = &v7;
  v5 = 0;
  if ( (unsigned __int8)sub_2523890(
                          a2,
                          (__int64 (__fastcall *)(__int64, __int64 *))sub_25970B0,
                          (__int64)v9,
                          a1,
                          1u,
                          &v5)
    && (!(_BYTE)v8 || BYTE9(v7)) )
  {
    return 1;
  }
  v3 = *(_BYTE *)(a1 + 96);
  v4 = *(_BYTE *)(a1 + 97);
  *(_BYTE *)(a1 + 97) = v3;
  return v4 == v3;
}
