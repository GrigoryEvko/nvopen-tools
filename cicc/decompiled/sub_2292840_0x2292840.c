// Function: sub_2292840
// Address: 0x2292840
//
__int64 __fastcall sub_2292840(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // r15
  __int64 v18; // rcx
  _QWORD *v20; // [rsp+8h] [rbp-78h]
  __int64 *v22; // [rsp+10h] [rbp-70h]
  __int64 *v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  _QWORD *v26; // [rsp+28h] [rbp-58h]
  __int64 *v27; // [rsp+28h] [rbp-58h]
  __int64 *v28; // [rsp+28h] [rbp-58h]
  _QWORD *v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h]
  _QWORD *v31; // [rsp+40h] [rbp-40h] BYREF
  __int64 v32; // [rsp+48h] [rbp-38h]

  v5 = sub_228CE20(a4);
  v20 = sub_2291EA0(a1, *a2, v5, v6, v7);
  v26 = sub_2291EA0(a1, *a3, v5, v8, v9);
  v24 = *(__int64 **)(a1 + 8);
  v32 = sub_228CDC0(a4);
  v31 = v20;
  v29 = &v31;
  v30 = 0x200000002LL;
  v10 = sub_DC8BD0(v24, (__int64)&v29, 0, 0);
  v11 = a4;
  v25 = (__int64)v10;
  if ( v29 != &v31 )
  {
    _libc_free((unsigned __int64)v29);
    v11 = a4;
  }
  v22 = *(__int64 **)(a1 + 8);
  v32 = sub_228CDD0(v11);
  v31 = v26;
  v29 = &v31;
  v30 = 0x200000002LL;
  v12 = sub_DC8BD0(v22, (__int64)&v29, 0, 0);
  v13 = (__int64)v12;
  if ( v29 != &v31 )
  {
    v27 = v12;
    _libc_free((unsigned __int64)v29);
    v13 = (__int64)v27;
  }
  v28 = *(__int64 **)(a1 + 8);
  v14 = sub_DCC810(v28, v25, v13, 0, 0);
  v15 = *a2;
  v32 = (__int64)v14;
  v31 = (_QWORD *)v15;
  v29 = &v31;
  v30 = 0x200000002LL;
  v17 = sub_DC7EB0(v28, (__int64)&v29, 0, 0);
  if ( v29 != &v31 )
    _libc_free((unsigned __int64)v29);
  *a2 = (__int64)v17;
  *a2 = (__int64)sub_2291F00(a1, (__int64)v17, v5, v16);
  *a3 = (__int64)sub_2291F00(a1, *a3, v5, v18);
  return 1;
}
