// Function: sub_DA40F0
// Address: 0xda40f0
//
_QWORD *__fastcall sub_DA40F0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // r15
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // [rsp+0h] [rbp-F0h]
  void *v15; // [rsp+10h] [rbp-E0h]
  __int64 v16; // [rsp+18h] [rbp-D8h]
  __int64 *v17; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-C0h] BYREF
  _DWORD v19[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-A8h]
  __int64 v21; // [rsp+50h] [rbp-A0h]

  v21 = a4;
  v19[1] = a2;
  v6 = v18;
  v20 = a3;
  v18[0] = v19;
  v18[1] = 0x2000000006LL;
  v19[0] = 1;
  v17 = 0;
  v7 = sub_C65B40(a1 + 1048, (__int64)v18, (__int64 *)&v17, (__int64)&off_49DEA60);
  if ( v7 )
  {
    v8 = v7 - 1;
  }
  else
  {
    v13 = a4;
    v15 = sub_C65D30((__int64)v18, (unsigned __int64 *)(a1 + 1064));
    v16 = v10;
    v11 = sub_A777F0(0x38u, (__int64 *)(a1 + 1064));
    v8 = (_QWORD *)v11;
    if ( v11 )
    {
      sub_D9AB30(v11, (__int64)v15, v16, a2, a3, v13);
      v12 = v17;
      v6 = v8 + 1;
    }
    else
    {
      v12 = v17;
      v6 = 0;
    }
    sub_C657C0((__int64 *)(a1 + 1048), v6, v12, (__int64)&off_49DEA60);
  }
  if ( (_DWORD *)v18[0] != v19 )
    _libc_free(v18[0], v6);
  return v8;
}
