// Function: sub_2290FB0
// Address: 0x2290fb0
//
void __fastcall sub_2290FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 *v8; // rbx
  bool v9; // zf
  __int64 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // r14
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdi
  __int64 *v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rdi
  __int64 v21; // r12
  _QWORD *v22; // r13
  unsigned __int64 v23; // r14
  __int64 *v24; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28; // [rsp+28h] [rbp-38h]

  v6 = 9LL * a5;
  v7 = 32LL * a5;
  v8 = (__int64 *)(a4 + 16 * v6);
  v9 = *v8 == 0;
  v8[11] = 0;
  v8[3] = 0;
  v10 = *(__int64 **)(a1 + 8);
  v11 = *(_QWORD *)(v7 + a3);
  v12 = *(_QWORD *)(v7 + a2);
  if ( !v9 )
  {
    v13 = sub_DCC810(v10, v12, v11, 0, 0);
    v14 = sub_2290F70(a1, (__int64)v13);
    v15 = *v8;
    v16 = *(__int64 **)(a1 + 8);
    v27 = v14;
    v28 = v15;
    v25 = &v27;
    v26 = 0x200000002LL;
    v17 = sub_DC8BD0(v16, (__int64)&v25, 0, 0);
    if ( v25 != &v27 )
    {
      v24 = v17;
      _libc_free((unsigned __int64)v25);
      v17 = v24;
    }
    v8[11] = (__int64)v17;
    v18 = sub_2290F30(a1, (__int64)v13);
    v19 = *v8;
    v20 = *(__int64 **)(a1 + 8);
    v27 = v18;
    v28 = v19;
    v25 = &v27;
    v26 = 0x200000002LL;
    v21 = (__int64)sub_DC8BD0(v20, (__int64)&v25, 0, 0);
    if ( v25 != &v27 )
      _libc_free((unsigned __int64)v25);
LABEL_6:
    v8[3] = v21;
    return;
  }
  v22 = sub_DCC810(v10, v12, v11, 0, 0);
  v23 = sub_2290F70(a1, (__int64)v22);
  if ( sub_D968A0(v23) )
    v8[11] = v23;
  v21 = sub_2290F30(a1, (__int64)v22);
  if ( sub_D968A0(v21) )
    goto LABEL_6;
}
