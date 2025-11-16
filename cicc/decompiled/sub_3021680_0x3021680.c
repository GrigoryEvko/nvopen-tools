// Function: sub_3021680
// Address: 0x3021680
//
__int64 __fastcall sub_3021680(__int64 a1)
{
  __int64 v1; // rdx
  unsigned int v2; // eax
  __int64 v3; // rdi
  __int64 v4; // r12
  const char *v6; // [rsp+0h] [rbp-100h] BYREF
  __int64 v7; // [rsp+8h] [rbp-F8h]
  __int64 v8; // [rsp+10h] [rbp-F0h]
  __int64 v9; // [rsp+18h] [rbp-E8h]
  __int64 v10; // [rsp+20h] [rbp-E0h]
  __int64 v11; // [rsp+28h] [rbp-D8h]
  const char **v12; // [rsp+30h] [rbp-D0h]
  const char *v13; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+48h] [rbp-B8h]
  __int64 v15; // [rsp+50h] [rbp-B0h]
  _BYTE v16[168]; // [rsp+58h] [rbp-A8h] BYREF

  v11 = 0x100000000LL;
  v12 = &v13;
  v13 = v16;
  v6 = (const char *)&unk_49DD288;
  v14 = 0;
  v15 = 128;
  v7 = 2;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  sub_CB5980((__int64)&v6, 0, 0, 0);
  v1 = v10;
  if ( (unsigned __int64)(v9 - v10) <= 0xC )
  {
    sub_CB6200((__int64)&v6, "__local_depot", 0xDu);
  }
  else
  {
    *(_DWORD *)(v10 + 8) = 1869636964;
    *(_QWORD *)v1 = 0x5F6C61636F6C5F5FLL;
    *(_BYTE *)(v1 + 12) = 116;
    v10 += 13;
  }
  v2 = sub_31DA6A0(a1);
  sub_CB59D0((__int64)&v6, v2);
  v6 = (const char *)&unk_49DD388;
  sub_CB5840((__int64)&v6);
  v3 = *(_QWORD *)(a1 + 216);
  LOWORD(v10) = 261;
  v6 = v13;
  v7 = v14;
  v4 = sub_E6C460(v3, &v6);
  if ( v13 != v16 )
    _libc_free((unsigned __int64)v13);
  return v4;
}
