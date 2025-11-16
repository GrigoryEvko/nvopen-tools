// Function: sub_2539B10
// Address: 0x2539b10
//
_BOOL8 __fastcall sub_2539B10(__int64 a1, __int64 a2)
{
  char v2; // r8
  int v3; // eax
  int v4; // edx
  int v5; // eax
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  __int128 v8; // [rsp+10h] [rbp-50h] BYREF
  __int64 v9; // [rsp+20h] [rbp-40h]
  _QWORD v10[5]; // [rsp+30h] [rbp-30h] BYREF

  v10[0] = &v7;
  v10[1] = a2;
  v7 = 0;
  v9 = 0;
  v10[2] = a1;
  v10[3] = &v8;
  v8 = 0;
  v2 = sub_2527330(a2, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_257D6A0, (__int64)v10, a1, 1u, 0);
  v3 = 0;
  if ( v2 )
  {
    v3 = 1023;
    if ( (_BYTE)v9 )
      v3 = WORD6(v8) & 0x3FF;
  }
  v4 = *(_DWORD *)(a1 + 100);
  v5 = *(_DWORD *)(a1 + 96) | v4 & v3;
  *(_DWORD *)(a1 + 100) = v5;
  return v4 == v5;
}
