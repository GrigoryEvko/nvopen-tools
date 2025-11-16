// Function: sub_254D900
// Address: 0x254d900
//
__int64 __fastcall sub_254D900(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r8
  char v4; // dl
  unsigned __int64 v5; // r8
  char v6; // r8
  __int64 result; // rax
  char v8; // [rsp+Bh] [rbp-45h] BYREF
  unsigned int v9; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v10[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = sub_2509800((_QWORD *)(a1 + 72));
  v3 = *(_QWORD *)(a1 + 72);
  v8 = v2;
  v4 = v3;
  v5 = v3 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v4 & 3) == 3 )
    v5 = *(_QWORD *)(v5 + 24);
  v10[1] = v5;
  v10[0] = &v8;
  v10[4] = &v9;
  v9 = 1;
  v10[2] = a2;
  v10[3] = a1;
  v10[5] = a1 + 88;
  v6 = sub_251CC40(a2, (__int64 (__fastcall *)(__int64, unsigned __int64 *, __int64))sub_258E5F0, (__int64)v10, a1, v5);
  result = v9;
  if ( !v6 )
  {
    *(_DWORD *)(a1 + 108) = *(_DWORD *)(a1 + 104);
    *(_BYTE *)(a1 + 169) = *(_BYTE *)(a1 + 168);
    return 0;
  }
  return result;
}
