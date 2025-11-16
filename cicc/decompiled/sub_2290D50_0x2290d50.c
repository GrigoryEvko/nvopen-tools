// Function: sub_2290D50
// Address: 0x2290d50
//
void __fastcall sub_2290D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r10
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 v12; // r10
  __int64 *v13; // r13
  __int64 *v14; // r12
  char v15; // al
  __int64 v16; // r10
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 *v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 *v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  _QWORD *v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+28h] [rbp-48h]
  _QWORD *v30; // [rsp+30h] [rbp-40h] BYREF
  __int64 v31; // [rsp+38h] [rbp-38h]

  v5 = 9LL * a5;
  v6 = 32LL * a5;
  v7 = a3 + v6;
  v8 = (__int64 *)(a2 + v6);
  v9 = (__int64 *)(a4 + 16 * v5);
  v10 = *v9;
  v9[8] = 0;
  v9[16] = 0;
  v24 = v10;
  if ( v10 )
  {
    v22 = a3 + v6;
    v23 = *(__int64 **)(a1 + 8);
    v30 = sub_DCC810(v23, v8[2], *(_QWORD *)(v7 + 8), 0, 0);
    v28 = &v30;
    v31 = v24;
    v29 = 0x200000002LL;
    v11 = sub_DC8BD0(v23, (__int64)&v28, 0, 0);
    v12 = v22;
    if ( v28 != &v30 )
    {
      v25 = v11;
      _libc_free((unsigned __int64)v28);
      v12 = v22;
      v11 = v25;
    }
    v9[16] = (__int64)v11;
    v13 = *(__int64 **)(a1 + 8);
    v26 = *v9;
    v30 = sub_DCC810(v13, v8[1], *(_QWORD *)(v12 + 16), 0, 0);
    v28 = &v30;
    v31 = v26;
    v29 = 0x200000002LL;
    v14 = sub_DC8BD0(v13, (__int64)&v28, 0, 0);
    if ( v28 != &v30 )
      _libc_free((unsigned __int64)v28);
    v9[8] = (__int64)v14;
  }
  else
  {
    v27 = a3 + v6;
    v15 = sub_228DFC0(a1, 0x20u, v8[2], *(_QWORD *)(v7 + 8));
    v16 = v27;
    if ( v15 )
    {
      v19 = *(_QWORD *)(a1 + 8);
      v20 = sub_D95540(*v8);
      v21 = sub_DA2C50(v19, v20, 0, 0);
      v16 = v27;
      v9[16] = (__int64)v21;
    }
    if ( sub_228DFC0(a1, 0x20u, v8[1], *(_QWORD *)(v16 + 16)) )
    {
      v17 = *(_QWORD *)(a1 + 8);
      v18 = sub_D95540(*v8);
      v9[8] = (__int64)sub_DA2C50(v17, v18, 0, 0);
    }
  }
}
