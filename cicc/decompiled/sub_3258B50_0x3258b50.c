// Function: sub_3258B50
// Address: 0x3258b50
//
__int64 __fastcall sub_3258B50(__int64 a1)
{
  __int64 *v2; // r12
  const char *v3; // rax
  __int64 v4; // rdx
  const char *v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rax
  bool v9; // zf
  char *v10; // rdx
  int v11; // r8d
  _QWORD v13[4]; // [rsp-138h] [rbp-138h] BYREF
  __int16 v14; // [rsp-118h] [rbp-118h]
  _QWORD v15[4]; // [rsp-108h] [rbp-108h] BYREF
  __int16 v16; // [rsp-E8h] [rbp-E8h]
  _QWORD *v17; // [rsp-D8h] [rbp-D8h] BYREF
  int v18; // [rsp-C8h] [rbp-C8h]
  __int16 v19; // [rsp-B8h] [rbp-B8h]
  _QWORD v20[4]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v21; // [rsp-88h] [rbp-88h]
  _QWORD v22[4]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v23; // [rsp-58h] [rbp-58h]
  const char *v24[4]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v25; // [rsp-28h] [rbp-28h]

  if ( !a1 )
    return 0;
  v2 = *(__int64 **)(a1 + 32);
  v3 = sub_BD5D20(*v2);
  v5 = v3;
  v6 = v4;
  if ( v4 && *v3 == 1 )
  {
    v6 = v4 - 1;
    v5 = v3 + 1;
  }
  v7 = v2[3];
  v8 = (*(_BYTE *)(a1 + 236) == 0) + 4LL;
  v22[2] = v5;
  v9 = *(_BYTE *)(a1 + 236) == 0;
  v22[3] = v6;
  v10 = "dtor";
  if ( v9 )
    v10 = "catch";
  v11 = *(_DWORD *)(a1 + 24);
  v13[3] = v8;
  v15[0] = v13;
  v15[2] = "$";
  v17 = v15;
  v20[0] = &v17;
  v20[2] = "@?0?";
  v22[0] = v20;
  v24[0] = (const char *)v22;
  v13[0] = "?";
  v24[2] = "@4HA";
  v21 = 770;
  v23 = 1282;
  v14 = 1283;
  v13[2] = v10;
  v16 = 770;
  v18 = v11;
  v19 = 2562;
  v25 = 770;
  return sub_E6C460(v7, v24);
}
