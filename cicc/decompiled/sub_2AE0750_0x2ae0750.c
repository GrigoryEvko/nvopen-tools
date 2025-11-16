// Function: sub_2AE0750
// Address: 0x2ae0750
//
unsigned __int64 __fastcall sub_2AE0750(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // r8
  _QWORD *v8; // rdx
  int v9; // ecx
  __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  bool v13; // of
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r13
  _QWORD v18[3]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v19; // [rsp+28h] [rbp-C8h]
  __int64 v20; // [rsp+30h] [rbp-C0h]
  __int64 v21; // [rsp+38h] [rbp-B8h]
  _QWORD *v22; // [rsp+40h] [rbp-B0h]
  __int64 v23; // [rsp+48h] [rbp-A8h]
  __int64 v24; // [rsp+50h] [rbp-A0h]
  __int64 v25; // [rsp+58h] [rbp-98h]
  __int64 v26; // [rsp+60h] [rbp-90h]
  char *v27; // [rsp+68h] [rbp-88h]
  __int64 v28; // [rsp+70h] [rbp-80h]
  int v29; // [rsp+78h] [rbp-78h]
  char v30; // [rsp+7Ch] [rbp-74h]
  char v31; // [rsp+80h] [rbp-70h] BYREF
  int v32; // [rsp+C0h] [rbp-30h]

  v4 = a1[6];
  v18[2] = 0;
  v5 = a1[5];
  v6 = *(_QWORD *)(v4 + 456);
  v7 = *(_QWORD *)(v4 + 448);
  v19 = 0;
  v8 = *(_QWORD **)(v5 + 336);
  v9 = *(_DWORD *)(v4 + 992);
  v20 = 0;
  v18[0] = v7;
  v18[1] = v6;
  v21 = 0;
  v22 = v8;
  v10 = *v8;
  v25 = v4;
  v23 = v10;
  v24 = v10;
  v32 = v9;
  v26 = 0;
  v27 = &v31;
  v28 = 8;
  v29 = 0;
  v30 = 1;
  v11 = sub_2ADF280(a1, a2, a3, (__int64)v18);
  v12 = sub_2BF4230(a2, a3, v18);
  v13 = __OFADD__(v12, v11);
  v14 = v12 + v11;
  if ( v13 )
  {
    v15 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v12 <= 0 )
      v15 = 0x8000000000000000LL;
  }
  else
  {
    v15 = v14;
  }
  if ( !v30 )
    _libc_free((unsigned __int64)v27);
  sub_C7D6A0(v19, 16LL * (unsigned int)v21, 8);
  return v15;
}
