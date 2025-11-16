// Function: sub_254DB60
// Address: 0x254db60
//
__int64 __fastcall sub_254DB60(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r8
  char v4; // dl
  unsigned __int64 v5; // r8
  char v7; // [rsp+Bh] [rbp-45h] BYREF
  unsigned int v8; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v9[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = sub_2509800((_QWORD *)(a1 + 72));
  v3 = *(_QWORD *)(a1 + 72);
  v7 = v2;
  v4 = v3;
  v5 = v3 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v4 & 3) == 3 )
    v5 = *(_QWORD *)(v5 + 24);
  v9[1] = v5;
  v9[0] = &v7;
  v9[4] = &v8;
  v8 = 1;
  v9[2] = a2;
  v9[3] = a1;
  v9[5] = a1 + 88;
  if ( (unsigned __int8)sub_251CC40(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64 *, __int64))sub_25A1F90,
                          (__int64)v9,
                          a1,
                          v5) )
    return v8;
  *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  return 0;
}
