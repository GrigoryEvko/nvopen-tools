// Function: sub_12A1DF0
// Address: 0x12a1df0
//
__int64 __fastcall sub_12A1DF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  int v4; // r14d
  int v5; // r13d
  int v6; // eax
  __int64 v7; // rsi
  int v8; // r9d
  __int64 v9; // r12
  _QWORD *v11; // [rsp+10h] [rbp-60h]
  __int64 v12; // [rsp+18h] [rbp-58h]
  _QWORD v13[10]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a2 + 160);
  v3 = 8LL * *(_QWORD *)(v2 + 128);
  if ( *(char *)(v2 + 142) >= 0 && *(_BYTE *)(v2 + 140) == 12 )
    v4 = 8 * sub_8D4AB0(v2);
  else
    v4 = 8 * *(_DWORD *)(v2 + 136);
  v5 = sub_12A0C10(a1, v2);
  v11 = v13;
  v12 = 0x400000001LL;
  v13[0] = sub_15A6850(a1 + 16, 0, 1);
  v6 = sub_15A5DC0(a1 + 16, v13, 1);
  v7 = v3;
  v9 = sub_15A6E90((int)a1 + 16, v3, v4, v5, v6, v8, (__int64)byte_3F871B3, 0);
  if ( v11 != v13 )
    _libc_free(v11, v7);
  return v9;
}
