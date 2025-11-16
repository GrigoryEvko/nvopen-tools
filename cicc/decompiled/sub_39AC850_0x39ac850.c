// Function: sub_39AC850
// Address: 0x39ac850
//
__int64 __fastcall sub_39AC850(__int64 a1)
{
  __int64 *v2; // r12
  const char *v3; // rax
  __int64 v4; // rdx
  bool v5; // zf
  const char *v6; // rdi
  __int64 v7; // r12
  int v8; // eax
  _QWORD v10[2]; // [rsp-118h] [rbp-118h] BYREF
  _QWORD v11[2]; // [rsp-108h] [rbp-108h] BYREF
  _QWORD v12[2]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v13; // [rsp-E8h] [rbp-E8h]
  _QWORD v14[2]; // [rsp-D8h] [rbp-D8h] BYREF
  __int16 v15; // [rsp-C8h] [rbp-C8h]
  __int64 v16; // [rsp-B8h] [rbp-B8h]
  _QWORD v17[2]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v18; // [rsp-88h] [rbp-88h]
  _QWORD v19[2]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v20; // [rsp-68h] [rbp-68h]
  _QWORD v21[2]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v22; // [rsp-48h] [rbp-48h]
  _QWORD v23[2]; // [rsp-38h] [rbp-38h] BYREF
  __int16 v24; // [rsp-28h] [rbp-28h]

  if ( !a1 )
    return 0;
  v2 = *(__int64 **)(a1 + 56);
  v3 = sub_1649960(*v2);
  if ( v4 && *v3 == 1 )
  {
    --v4;
    ++v3;
  }
  v5 = *(_BYTE *)(a1 + 184) == 0;
  v10[0] = v3;
  v6 = "dtor";
  if ( v5 )
    v6 = "catch";
  v10[1] = v4;
  v7 = v2[3];
  v11[0] = v6;
  v20 = 770;
  v11[1] = strlen(v6);
  v8 = *(_DWORD *)(a1 + 48);
  v22 = 1282;
  LODWORD(v16) = v8;
  v13 = 1283;
  v12[0] = "?";
  v12[1] = v11;
  v14[0] = v12;
  v14[1] = "$";
  v17[0] = v14;
  v15 = 770;
  v17[1] = v16;
  v19[0] = v17;
  v19[1] = "@?0?";
  v21[0] = v19;
  v21[1] = v10;
  v23[0] = v21;
  v18 = 2562;
  v23[1] = "@4HA";
  v24 = 770;
  return sub_38BF510(v7, (__int64)v23);
}
