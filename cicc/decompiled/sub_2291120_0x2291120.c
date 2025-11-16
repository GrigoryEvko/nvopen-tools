// Function: sub_2291120
// Address: 0x2291120
//
char __fastcall sub_2291120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r14
  __int64 *v7; // r13
  __int64 *v8; // r12
  __int64 v9; // r14
  __int64 v10; // rdi
  __int64 *v11; // r15
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // r9
  __int64 v17; // rcx
  __int64 *v18; // rax
  __int64 *v19; // r9
  __int64 v20; // rsi
  _QWORD *v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // r13
  __int64 *v25; // r14
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  unsigned __int64 v28; // rax
  _QWORD *v29; // rax
  unsigned __int64 v30; // rax
  __int64 *v32; // [rsp+8h] [rbp-78h]
  __int64 *v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  _QWORD *v35; // [rsp+28h] [rbp-58h]
  unsigned __int64 *v36; // [rsp+30h] [rbp-50h] BYREF
  __int64 v37; // [rsp+38h] [rbp-48h]
  unsigned __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  _QWORD *v39; // [rsp+48h] [rbp-38h]

  v5 = 32LL * a5;
  v7 = (__int64 *)(a3 + v5);
  v8 = (__int64 *)(a4 + 144LL * a5);
  v9 = a2 + v5;
  v10 = *v8;
  v8[10] = 0;
  v8[2] = 0;
  if ( v10 )
  {
    v11 = *(__int64 **)(a1 + 8);
    v12 = sub_D95540(v10);
    v13 = sub_DA2C50((__int64)v11, v12, 1, 0);
    v35 = sub_DCC810(v11, *v8, (__int64)v13, 0, 0);
    v14 = sub_DCC810(*(__int64 **)(a1 + 8), *(_QWORD *)(v9 + 16), *v7, 0, 0);
    v15 = sub_2290F70(a1, (__int64)v14);
    v16 = *(__int64 **)(a1 + 8);
    v17 = *v7;
    v38 = v15;
    v34 = v17;
    v39 = v35;
    v33 = v16;
    v36 = &v38;
    v37 = 0x200000002LL;
    v18 = sub_DC8BD0(v16, (__int64)&v36, 0, 0);
    v19 = v33;
    v20 = (__int64)v18;
    if ( v36 != &v38 )
    {
      v32 = v18;
      _libc_free((unsigned __int64)v36);
      v20 = (__int64)v32;
      v19 = v33;
    }
    v8[10] = (__int64)sub_DCC810(v19, v20, v34, 0, 0);
    v21 = sub_DCC810(*(__int64 **)(a1 + 8), *(_QWORD *)(v9 + 8), *v7, 0, 0);
    v22 = sub_2290F30(a1, (__int64)v21);
    v23 = *(__int64 **)(a1 + 8);
    v38 = v22;
    v36 = &v38;
    v24 = *v7;
    v39 = v35;
    v37 = 0x200000002LL;
    v25 = sub_DC8BD0(v23, (__int64)&v36, 0, 0);
    if ( v36 != &v38 )
      _libc_free((unsigned __int64)v36);
    v26 = sub_DCC810(v23, (__int64)v25, v24, 0, 0);
    v8[2] = (__int64)v26;
  }
  else
  {
    v27 = sub_DCC810(*(__int64 **)(a1 + 8), *(_QWORD *)(v9 + 16), *v7, 0, 0);
    v28 = sub_2290F70(a1, (__int64)v27);
    if ( sub_D968A0(v28) )
      v8[10] = (__int64)sub_DCAF50(*(__int64 **)(a1 + 8), *v7, 0);
    v29 = sub_DCC810(*(__int64 **)(a1 + 8), *(_QWORD *)(v9 + 8), *v7, 0, 0);
    v30 = sub_2290F30(a1, (__int64)v29);
    LOBYTE(v26) = sub_D968A0(v30);
    if ( (_BYTE)v26 )
    {
      v26 = sub_DCAF50(*(__int64 **)(a1 + 8), *v7, 0);
      v8[2] = (__int64)v26;
    }
  }
  return (char)v26;
}
