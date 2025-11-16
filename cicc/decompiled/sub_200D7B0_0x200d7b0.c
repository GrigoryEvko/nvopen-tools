// Function: sub_200D7B0
// Address: 0x200d7b0
//
__int64 __fastcall sub_200D7B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // r10
  __int64 v9; // rsi
  _QWORD *v10; // rdi
  unsigned __int8 *v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+20h] [rbp-80h] BYREF
  int v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h]
  __int64 v30; // [rsp+38h] [rbp-68h]
  __int64 v31; // [rsp+40h] [rbp-60h]
  __int64 v32; // [rsp+50h] [rbp-50h] BYREF
  __int64 v33; // [rsp+58h] [rbp-48h]
  __int64 v34; // [rsp+60h] [rbp-40h]

  v5 = a3;
  v6 = a5;
  v9 = *(_QWORD *)(a2 + 72);
  v27 = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)&v27, v9, 2);
    v6 = a5;
    v5 = a3;
  }
  v10 = *(_QWORD **)(a1 + 8);
  v24 = v6;
  v28 = *(_DWORD *)(a2 + 64);
  v11 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)v5);
  v22 = v5;
  v12 = sub_1D29D50(v10, *v11, *((_QWORD *)v11 + 1), a4, v6, v5);
  v13 = *(_QWORD **)(a1 + 8);
  v15 = v14;
  v16 = (__int64)v12;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v29 = 0;
  v17 = sub_1D2BF40(v13, (__int64)(v13 + 11), 0, (__int64)&v27, a2, v22, (__int64)v12, v14, 0, 0, 0, 0, (__int64)&v32);
  v18 = *(_QWORD **)(a1 + 8);
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v29 = 0;
  v20 = sub_1D2B730(v18, a4, v24, (__int64)&v27, v17, v19, v16, v15, 0, 0, 0, 0, (__int64)&v32, 0);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v20;
}
