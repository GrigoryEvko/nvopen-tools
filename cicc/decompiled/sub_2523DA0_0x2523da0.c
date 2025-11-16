// Function: sub_2523DA0
// Address: 0x2523da0
//
__int64 __fastcall sub_2523DA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v5; // rax
  char v6; // [rsp-3Eh] [rbp-3Eh] BYREF
  _BYTE v7[5]; // [rsp-3Dh] [rbp-3Dh] BYREF
  __int64 v8; // [rsp-38h] [rbp-38h] BYREF
  __int64 v9[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( !*(_BYTE *)(a1 + 4298) )
    return 0;
  v2 = *(_QWORD *)(a2 + 24);
  v3 = *(_QWORD *)(v2 + 24);
  v8 = v2;
  if ( *(_DWORD *)(v3 + 8) >> 8 )
    return 0;
  v9[0] = *(_QWORD *)(v2 + 120);
  if ( (unsigned __int8)sub_A74390(v9, 21, 0) )
    return 0;
  if ( (unsigned __int8)sub_A74390(v9, 85, 0) )
    return 0;
  if ( (unsigned __int8)sub_A74390(v9, 83, 0) )
    return 0;
  if ( (unsigned __int8)sub_A74390(v9, 84, 0) )
    return 0;
  v6 = 0;
  if ( !(unsigned __int8)sub_25230B0(
                           a1,
                           (__int64 (__fastcall *)(__int64, __int64 *))sub_2509520,
                           (__int64)&v8,
                           v2,
                           1,
                           0,
                           &v6,
                           1) )
    return 0;
  v5 = sub_251B1C0(*(_QWORD *)(a1 + 208), v2);
  *(_DWORD *)&v7[1] = 56;
  return sub_2522A30(
           0,
           v5,
           (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2505F00,
           (__int64)v7,
           0,
           0,
           (int *)&v7[1],
           1,
           &v6,
           0,
           0);
}
