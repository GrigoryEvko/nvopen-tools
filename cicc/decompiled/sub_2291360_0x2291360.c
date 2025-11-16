// Function: sub_2291360
// Address: 0x2291360
//
void __fastcall sub_2291360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rdi
  __int64 *v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  unsigned __int64 v18; // rax
  __int64 *v19; // r12
  __int64 v20; // r13
  _QWORD *v21; // r12
  _QWORD *v22; // rax
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // [rsp+0h] [rbp-70h]
  __int64 *v27; // [rsp+8h] [rbp-68h]
  _QWORD *v28; // [rsp+8h] [rbp-68h]
  _QWORD *v29; // [rsp+18h] [rbp-58h]
  __int64 **v30; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+28h] [rbp-48h]
  __int64 *v32; // [rsp+30h] [rbp-40h] BYREF
  __int64 v33; // [rsp+38h] [rbp-38h]

  v6 = 32LL * a5;
  v7 = a3 + v6;
  v8 = (__int64 *)(a2 + v6);
  v9 = (__int64 *)(a4 + 144LL * a5);
  v10 = *v9;
  v9[13] = 0;
  v9[5] = 0;
  if ( v10 )
  {
    v11 = *(__int64 **)(a1 + 8);
    v12 = sub_D95540(v10);
    v13 = sub_DA2C50((__int64)v11, v12, 1, 0);
    v29 = sub_DCC810(v11, *v9, (__int64)v13, 0, 0);
    v14 = sub_DCC810(*(__int64 **)(a1 + 8), *v8, *(_QWORD *)(v7 + 8), 0, 0);
    v15 = sub_2290F70(a1, (__int64)v14);
    v26 = *v8;
    v27 = *(__int64 **)(a1 + 8);
    v32 = sub_DCA690(v27, v15, (__int64)v29, 0, 0);
    v30 = &v32;
    v33 = v26;
    v31 = 0x200000002LL;
    v16 = sub_DC7EB0(v27, (__int64)&v30, 0, 0);
    if ( v30 != &v32 )
    {
      v28 = v16;
      _libc_free((unsigned __int64)v30);
      v16 = v28;
    }
    v9[13] = (__int64)v16;
    v17 = sub_DCC810(*(__int64 **)(a1 + 8), *v8, *(_QWORD *)(v7 + 16), 0, 0);
    v18 = sub_2290F30(a1, (__int64)v17);
    v19 = *(__int64 **)(a1 + 8);
    v20 = *v8;
    v32 = sub_DCA690(v19, v18, (__int64)v29, 0, 0);
    v30 = &v32;
    v33 = v20;
    v31 = 0x200000002LL;
    v21 = sub_DC7EB0(v19, (__int64)&v30, 0, 0);
    if ( v30 != &v32 )
      _libc_free((unsigned __int64)v30);
    v9[5] = (__int64)v21;
  }
  else
  {
    v22 = sub_DCC810(*(__int64 **)(a1 + 8), *v8, *(_QWORD *)(v7 + 8), 0, 0);
    v23 = sub_2290F70(a1, (__int64)v22);
    if ( sub_D968A0(v23) )
      v9[13] = *v8;
    v24 = sub_DCC810(*(__int64 **)(a1 + 8), *v8, *(_QWORD *)(v7 + 16), 0, 0);
    v25 = sub_2290F30(a1, (__int64)v24);
    if ( sub_D968A0(v25) )
      v9[5] = *v8;
  }
}
