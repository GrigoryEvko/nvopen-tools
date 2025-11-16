// Function: sub_DFD060
// Address: 0xdfd060
//
__int64 __fastcall sub_DFD060(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 (__fastcall *v6)(__int64, unsigned int, __int64, __int64); // rax
  unsigned int v7; // eax
  __int64 v8; // r9
  unsigned int v9; // r12d
  unsigned __int8 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int8 *v18; // rdi
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 v21; // r9
  unsigned int v22; // r13d
  unsigned __int8 *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // r9
  __int64 v27; // [rsp+18h] [rbp-38h] BYREF
  __int64 v28[6]; // [rsp+20h] [rbp-30h] BYREF

  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64))(*(_QWORD *)*a1 + 1200LL);
  if ( v6 != sub_DF7C30 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64))v6)(v5, a2, a3);
  if ( (_DWORD)a2 == 48 )
  {
    v20 = sub_BCB060(a4);
    v21 = *(_QWORD *)(v5 + 8);
    v22 = v20;
    v23 = *(unsigned __int8 **)(v21 + 32);
    v24 = *(_QWORD *)(v21 + 40);
    v28[0] = v20;
    return &v23[v24] == sub_DF6450(v23, (__int64)&v23[v24], v28) || v22 > (unsigned int)sub_AE43A0(v25, a3);
  }
  if ( (unsigned int)a2 > 0x30 )
    return (_DWORD)a2 != 49 || a3 != a4 && (*(_BYTE *)(a3 + 8) != 14 || *(_BYTE *)(a4 + 8) != 14);
  if ( (_DWORD)a2 == 38 )
  {
    v15 = sub_9208B0(*(_QWORD *)(v5 + 8), a3);
    v28[1] = v16;
    v28[0] = v15;
    if ( (_BYTE)v16 )
      return 1;
    v17 = *(_QWORD *)(v5 + 8);
    v18 = *(unsigned __int8 **)(v17 + 32);
    v19 = *(_QWORD *)(v17 + 40);
    v27 = v28[0];
    return &v18[v19] == sub_DF6450(v18, (__int64)&v18[v19], &v27);
  }
  if ( (_DWORD)a2 == 47 )
  {
    v7 = sub_BCB060(a3);
    v8 = *(_QWORD *)(v5 + 8);
    v9 = v7;
    v10 = *(unsigned __int8 **)(v8 + 32);
    v11 = *(_QWORD *)(v8 + 40);
    v28[0] = v7;
    if ( &v10[v11] != sub_DF6450(v10, (__int64)&v10[v11], v28) && v9 >= (unsigned int)sub_AE43A0(v12, a4) )
      return 0;
  }
  return 1;
}
