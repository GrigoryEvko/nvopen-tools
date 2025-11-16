// Function: sub_254E8F0
// Address: 0x254e8f0
//
__int64 __fastcall sub_254E8F0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 *v3; // r14
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  char v6; // dl
  __int64 **v7; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  unsigned __int64 v10; // rax
  bool v12; // [rsp+6h] [rbp-6Ah] BYREF
  unsigned __int8 v13; // [rsp+7h] [rbp-69h] BYREF
  unsigned __int64 v14; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v17[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(_DWORD *)(a1 + 100);
  if ( v2 == -1 )
    return 1;
  v3 = (__int64 *)(a1 + 72);
  v4 = sub_250D180((__int64 *)(a1 + 72), a2);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( *(_DWORD *)(v4 + 8) >> 8 == v2 )
    return 1;
  v17[0] = sub_250ED40(*(_QWORD *)(a2 + 208));
  if ( !BYTE4(v17[0]) )
    abort();
  v5 = sub_250D070(v3);
  v14 = v5;
  v6 = *(_BYTE *)v5;
  if ( *(_BYTE *)v5 <= 0x1Cu )
  {
    if ( v6 == 5 && *(_WORD *)(v5 + 2) == 50 )
      v5 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  }
  else if ( v6 == 79 )
  {
    v5 = *(_QWORD *)(v5 - 32);
  }
  v15 = v5;
  v7 = (__int64 **)sub_250D180(v3, a2);
  v16 = sub_BCE3C0(*v7, v2);
  v8 = *(_QWORD *)(v15 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = *(_DWORD *)(v8 + 8);
  v17[1] = a2;
  v13 = 0;
  v17[0] = &v14;
  v17[2] = &v13;
  v17[3] = &v15;
  v17[4] = &v16;
  v12 = v9 >> 8 == v2;
  v17[5] = &v12;
  v10 = sub_250D070(v3);
  sub_252FFB0(
    a2,
    (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_256E940,
    (__int64)v17,
    a1,
    v10,
    1,
    1,
    1,
    0,
    0);
  return v13 ^ 1u;
}
