// Function: sub_254F350
// Address: 0x254f350
//
_BOOL8 __fastcall sub_254F350(__int64 a1, __int64 a2)
{
  char v2; // r8
  int v3; // eax
  int v4; // edx
  int v5; // eax
  char v7; // [rsp+Bh] [rbp-55h] BYREF
  int v8; // [rsp+Ch] [rbp-54h] BYREF
  __int128 v9; // [rsp+10h] [rbp-50h] BYREF
  __int64 v10; // [rsp+20h] [rbp-40h]
  _QWORD v11[6]; // [rsp+30h] [rbp-30h] BYREF

  v9 = 0;
  v10 = 0;
  v8 = sub_250CB50((__int64 *)(a1 + 72), 0);
  v11[0] = &v8;
  v11[1] = a2;
  v11[2] = a1;
  v11[3] = &v9;
  v7 = 0;
  v2 = sub_2523890(a2, (__int64 (__fastcall *)(__int64, __int64 *))sub_257D4A0, (__int64)v11, a1, 1u, &v7);
  v3 = 0;
  if ( v2 )
  {
    v3 = 1023;
    if ( (_BYTE)v10 )
      v3 = WORD6(v9) & 0x3FF;
  }
  v4 = *(_DWORD *)(a1 + 100);
  v5 = *(_DWORD *)(a1 + 96) | v4 & v3;
  *(_DWORD *)(a1 + 100) = v5;
  return v4 == v5;
}
