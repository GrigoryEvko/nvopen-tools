// Function: sub_3950250
// Address: 0x3950250
//
__int64 __fastcall sub_3950250(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v6; // rdi
  const char *v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r15
  __int64 v12; // r14
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  const char *v29; // [rsp+20h] [rbp-60h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+20h] [rbp-60h]
  __int64 v33[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v34; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v6 + 16) )
    v6 = 0;
  v7 = sub_1649960(v6);
  v27 = v8;
  v29 = v7;
  v9 = sub_15F2050(a2);
  v26 = sub_1632FA0(v9);
  v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v11 = *(_BYTE **)(a2 - 24 * v10);
  v12 = *(_QWORD *)(a2 + 24 * (1 - v10));
  v25 = *(_QWORD *)(a2 + 24 * (2 - v10));
  if ( a4 == 111 && *(_BYTE *)(a1 + 8) != 1 && (_BYTE *)v12 == v11 )
  {
    v20 = sub_1AB1960(v12, (__int64)a3, v26, *(__int64 **)a1);
    if ( v20 )
    {
      v23 = (_QWORD *)a3[3];
      v34 = 257;
      v24 = sub_1643330(v23);
      return sub_17CEC00(a3, v24, v11, v20, v33);
    }
    return v20;
  }
  if ( !sub_394FE80(a1, a2, 2u, 1u, 1) )
  {
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    v30 = sub_14AD030(v12, 8u);
    if ( !v30 )
      return 0;
    v16 = sub_16498A0(a2);
    v17 = sub_15A9620(v26, v16, 0);
    v18 = v30;
    v31 = v17;
    v28 = v18;
    v19 = sub_15A0680(v17, v18, 0);
    v20 = sub_1AB2330((__int64)v11, v12, v19, v25, (__int64)a3, v26, *(_QWORD *)a1);
    if ( v20 && a4 == 111 )
    {
      v34 = 257;
      v21 = sub_15A0680(v31, v28 - 1, 0);
      v22 = sub_1643330((_QWORD *)a3[3]);
      return sub_12815B0(a3, v22, v11, v21, (__int64)v33);
    }
    return v20;
  }
  v13 = v27;
  v14 = 0;
  if ( v27 > 1 )
  {
    v14 = v27 - 2;
    v13 = 2;
    if ( v27 - 2 > 5 )
      v14 = 6;
  }
  return sub_1AB1D50((__int64)v11, v12, (__int64)a3, *(__int64 **)a1, (__int64)&v29[v13], v14);
}
