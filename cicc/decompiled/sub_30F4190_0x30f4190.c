// Function: sub_30F4190
// Address: 0x30f4190
//
__int64 __fastcall sub_30F4190(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 *v8; // rbx
  __int64 v9; // r13
  __int64 *v10; // r14
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // r13
  __int64 v19; // rax
  _BYTE *v20; // r12
  __int64 *v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 *v24; // [rsp+8h] [rbp-68h]
  unsigned __int64 v27[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v28[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 24);
  v6 = 8LL * *(unsigned int *)(a1 + 32);
  v7 = *(_QWORD *)(v5 + v6 - 8);
  v8 = (__int64 *)(v5 + v6);
  if ( v8 == (__int64 *)v5 )
  {
LABEL_6:
    v11 = sub_30F4150(a1, a2, (__int64)a3, v5, a5);
    v23 = *(_QWORD *)(a1 + 104);
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * *(unsigned int *)(a1 + 72) - 8);
    v13 = sub_D95540(v12);
    v14 = sub_D95540(v11);
    v15 = sub_D970B0(v23, v14, v13);
    v24 = *(__int64 **)(a1 + 104);
    v16 = sub_DD2D10((__int64)v24, v12, v15);
    v28[0] = sub_DD2D10(*(_QWORD *)(a1 + 104), v11, v15);
    v27[0] = (unsigned __int64)v28;
    v28[1] = v16;
    v27[1] = 0x200000002LL;
    v17 = sub_DC8BD0(v24, (__int64)v27, 0, 0);
    if ( (_QWORD *)v27[0] != v28 )
      _libc_free(v27[0]);
    *a3 = (__int64)v17;
    v18 = *(_QWORD *)(a1 + 104);
    v19 = sub_D95540((__int64)v17);
    v20 = sub_DA2C50(v18, v19, a4, 0);
    if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 104), *a3) )
      v21 = sub_DCAF50(*(__int64 **)(a1 + 104), *a3, 0);
    else
      v21 = (__int64 *)*a3;
    *a3 = (__int64)v21;
    return sub_DC3A60(*(_QWORD *)(a1 + 104), 36, v21, v20);
  }
  else
  {
    v9 = a2;
    v10 = *(__int64 **)(a1 + 24);
    while ( 1 )
    {
      a2 = *v10;
      if ( v7 != *v10 && !sub_30F4170(a1, a2, v9) )
        return 0;
      if ( v8 == ++v10 )
        goto LABEL_6;
    }
  }
}
